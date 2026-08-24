from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

from lerobot.types import RobotObservation, RobotAction


class SOLeaderController:
    """Wraps SO100Leader so it matches xboxController's connect / get_action / disconnect API."""

    gripper_error_threshold = 3.0
    _stall_frames_needed = 3
    _stall_motion_eps = 0.5
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
        self._prev_gripper = 0.0
        self._stall_count = 0

    def connect(self) -> None:
        self._leader.connect(calibrate=True)

    def disconnect(self) -> None:
        self._leader.disconnect()

    # sync targets from observation
    def sync_from_observation(self, obs: RobotObservation) -> None:
        for name in self.motor_names:
            self.targets[name] = float(obs[f"{name}.pos"])
        self._prev_gripper = float(obs["gripper.pos"])
        self._stall_count = 0
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
        closing = leader_gripper < actual_gripper - 1.0
        moved_closed = (self._prev_gripper - actual_gripper) > self._stall_motion_eps

        if closing and not moved_closed:
            self._stall_count += 1
        else:
            self._stall_count = 0

        if self._stall_count >= self._stall_frames_needed:
            action["gripper.pos"] = max(
                leader_gripper,
                actual_gripper - self.gripper_error_threshold,
            )

        self._prev_gripper = actual_gripper
        return action
