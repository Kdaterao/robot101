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

LEADER_PORT = "/dev/tty.usbmodem5B610348321"
LEADER_ID = "my_awesome_leader_arm"




def make_controller(controller_type: ControllerType, robot: SO100Follower):
    if controller_type is ControllerType.XBOX:
        return xboxController(MyTeleopConfig(id="xbox_controller"), robot)

    if controller_type is ControllerType.SO101:
        config = SO100LeaderConfig(
            port=LEADER_PORT,
            id=LEADER_ID,
            use_degrees=True,
        )
        return SOLeaderController(config)

    raise ValueError(f"Unknown controller type: {controller_type}")
