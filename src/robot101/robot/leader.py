from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

from lerobot.types import RobotObservation, RobotAction

from robot101.robot.gripper import GripperClampDetector


class SOLeaderController:
    """Wraps SO100Leader so it matches xboxController's connect / get_action / disconnect API."""

    gripper_error_threshold = 3.0
    _synced = False
    motor_names = ['shoulder_pan', 
                   'shoulder_lift', 
                   'elbow_flex',
                    'wrist_flex', 
                    'wrist_roll', 
                    'gripper']
    

    def __init__(self, config: SO100LeaderConfig):
        self._leader = SO100Leader(config)

        self.targets = {name: 0.0 for name in self.motor_names}
        # Silent here; recorders/runtimes run their own detector with prints.
        self.clamp = GripperClampDetector(verbose=False)
        # Raw leader gripper command from the last get_action (before clamping).
        self.last_leader_gripper = 0.0

    def connect(self) -> None:
        self._leader.connect(calibrate=True)

    def disconnect(self) -> None:
        self._leader.disconnect()

    # sync targets from observation
    def sync_from_observation(self, obs: RobotObservation) -> None:
        for name in self.motor_names:
            self.targets[name] = float(obs[f"{name}.pos"])
        self.clamp.reset(float(obs["gripper.pos"]))
        self._synced = True


    def get_action(self, dt: float, obs: RobotObservation) -> RobotAction:
        
        #--- SYNC (LAZY ONLY ONCE) -----
        if self._synced == False:
            self.sync_from_observation(obs)


        #---- GET ACTION ------
        action = self._leader.get_action()

        # Stall guard: 0 = closed, 100 = open. Pass the leader through while the
        # follower is still moving; only clamp after several frames of no motion.
        actual_gripper = float(obs["gripper.pos"])
        leader_gripper = float(action["gripper.pos"])
        self.last_leader_gripper = leader_gripper
        self.clamp.update(leader_gripper, actual_gripper)

        if self.clamp.stalling:
            action["gripper.pos"] = max(
                leader_gripper,
                actual_gripper - self.gripper_error_threshold,
            )
        return action
