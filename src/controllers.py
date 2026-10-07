from enum import Enum

from lerobot.robots.so_follower import SO100Follower
from lerobot.teleoperators.so_leader import SO100LeaderConfig

from xboxController import MyTeleopConfig, xboxController

from SO101LeaderController import SOLeaderController




class ControllerType(Enum):
    XBOX = "xbox"
    SO101 = "so100_leader"


# Flip this to switch the teleop device used by teleoperate.py and record.py.
CONTROLLER = ControllerType.SO101


#--- SO100 Leader specific settings ---

LEADER_PORT = "COM4"
LEADER_ID = "my_awesome_leader_arm"




def make_controller(controller_type: ControllerType, robot: SO100Follower, leader_port=LEADER_PORT, leader_id=LEADER_ID):
    if controller_type is ControllerType.XBOX:
        return xboxController(MyTeleopConfig(id="xbox_controller"), robot)

    if controller_type is ControllerType.SO101:
        config = SO100LeaderConfig(
            port=leader_port,
            id=leader_id,
            use_degrees=True,
        )
        return SOLeaderController(config)

    raise ValueError(f"Unknown controller type: {controller_type}")
