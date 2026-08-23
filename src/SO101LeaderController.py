from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

from lerobot.types import RobotObservation, RobotAction


class SOLeaderController:
    """Wraps SO100Leader so it matches xboxController's connect / get_action / disconnect API."""

    def __init__(self, config: SO100LeaderConfig):
        self._leader = SO100Leader(config)

    def connect(self) -> None:
        self._leader.connect(calibrate=True)

    def disconnect(self) -> None:
        self._leader.disconnect()

    def sync_from_observation(self, obs: RobotObservation) -> None:
        return

    def get_action(self, dt: float, obs: RobotObservation) -> RobotAction:
        return self._leader.get_action()
