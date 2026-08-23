import os

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import math
from dataclasses import dataclass

import pygame

from lerobot.robots.so_follower import SO100Follower
from lerobot.types import RobotAction, RobotObservation


@dataclass
class MyTeleopConfig:
    id: str = "xbox_controller"
    device_id: str = "/dev/input/js0"
    sensitivity: float = 1.0


# URDF limits converted to degrees (motors use use_degrees=True).
JOINT_LIMITS_DEG = {
    "shoulder_pan": (-math.degrees(1.91986), math.degrees(1.91986)),
    "shoulder_lift": (-math.degrees(1.74533), math.degrees(1.74533)),
    "elbow_flex": (-math.degrees(1.69), math.degrees(1.69)),
    "wrist_flex": (-math.degrees(1.65806), math.degrees(1.65806)),
    "wrist_roll": (math.degrees(-2.74385), math.degrees(2.84121)),
    "gripper": (0.0, 100.0),
}


class xboxController:
    def __init__(self, config: MyTeleopConfig, robot: SO100Follower):
        self.config = config
        self.device_id = config.device_id
        self.sensitivity = config.sensitivity
        self.controller = None
        self.motor_names = list(robot.bus.motors.keys())

        pygame.init()
        pygame.joystick.init()

        self.arm_speed = 45.0  # deg/s for pan / lift / elbow
        self.wrist_speed = math.degrees(2.0)  # ~115 deg/s
        self.gripper_speed = 40.0  # 0-100 units/s

        # If the close command is more than this below the measured jaw, loosen
        # and refuse further close (stalled on an object or hard stop).
        self.gripper_error_threshold = 3.0
        self.base_gripper_error_threshold = 3.0

        self.targets = {name: 0.0 for name in self.motor_names}
        self._synced = False
        self.pad = None

    def connect(self) -> None:
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        self.pad = None
        try:
            import pygame._sdl2.controller as sdlc

            if not sdlc.get_init():
                sdlc.init()
            self.pad = sdlc.Controller.from_joystick(self.controller)
        except (pygame.error, AttributeError, ValueError, TypeError):
            self.pad = None

    def disconnect(self) -> None:
        if self.pad is not None:
            self.pad.quit()
            self.pad = None
        if self.controller is not None:
            self.controller.quit()
            self.controller = None
        pygame.joystick.quit()

    def _any_button(self, *ids: int) -> bool:
        n = self.controller.get_numbuttons()
        return any(self.controller.get_button(i) for i in ids if i < n)

    def _bumper(self, left: bool) -> bool:
        """LB/RB indices differ by platform; prefer SDL names, then common raw ids."""
        if self.pad is not None:
            btn = (
                pygame.CONTROLLER_BUTTON_LEFTSHOULDER
                if left
                else pygame.CONTROLLER_BUTTON_RIGHTSHOULDER
            )
            if self.pad.get_button(btn):
                return True
        return self._any_button(*(4, 6, 9) if left else (5, 7, 10))







    #--------- POSITION UPDATES ------------


    # trigger mapping helper
    def trigger_01(self, axis: int) -> float:
        """Map an Xbox trigger from SDL's [-1, 1] (rest = -1) to [0, 1] (rest = 0)."""
        return min(1.0, max(0.0, (self.controller.get_axis(axis) + 1.0) / 2.0))

    # sync targets from observation
    def sync_from_observation(self, obs: RobotObservation) -> None:
        for name in self.motor_names:
            self.targets[name] = float(obs[f"{name}.pos"])
        self._synced = True

    def _clamp_joint(self, name: str) -> None:
        lo, hi = JOINT_LIMITS_DEG[name]
        self.targets[name] = max(lo, min(hi, self.targets[name]))

    def get_action(self, dt: float, obs: RobotObservation) -> RobotAction:
        pygame.event.pump()
        dt = min(max(dt, 0.0), 0.1)

        if not self._synced:
            self.sync_from_observation(obs)

        def deadzone(value, threshold=0.2):
            if abs(value) < threshold:
                return 0.0
            return value

        # Left stick: pan (X) and lift (Y, inverted so stick-up raises the arm)
        self.targets["shoulder_pan"] += deadzone(self.controller.get_axis(0)) * self.arm_speed * dt
        self.targets["shoulder_lift"] += deadzone(-self.controller.get_axis(1)) * self.arm_speed * dt
        # Right stick Y: elbow
        self.targets["elbow_flex"] += deadzone(-self.controller.get_axis(3)) * self.arm_speed * dt

        self._clamp_joint("shoulder_pan")
        self._clamp_joint("shoulder_lift")
        self._clamp_joint("elbow_flex")

        # LT / RT: wrist flex, or wrist roll while X is held
        lt = self.trigger_01(4)
        rt = self.trigger_01(5)
        wrist_delta = (rt - lt) * self.wrist_speed * dt
        if self.controller.get_button(2):
            self.targets["wrist_roll"] += wrist_delta
            self._clamp_joint("wrist_roll")
        else:
            self.targets["wrist_flex"] += wrist_delta
            self._clamp_joint("wrist_flex")



        # LB close / RB open. LeRobot gripper: 0 = closed, 100 = open.
        actual_gripper = float(obs["gripper.pos"])
        close_error = actual_gripper - self.targets["gripper"]


        #--- case 1: gripper is closing ---
        if self._bumper(left=False):

            if close_error > self.gripper_error_threshold:
                #--- case 1.1: gripper is stalled ---
                self.targets["gripper"] = actual_gripper - self.gripper_error_threshold
                print("Gripper could be stalled, stop closing")
            else:
                #--- case 1.2: gripper is not stalled ---
                self.targets["gripper"] -= self.gripper_speed * dt
                self.targets["gripper"] = max( self.targets["gripper"], actual_gripper - self.gripper_error_threshold) #--> kinda just prevents us from even closing too hard 

        #--- case 2: gripper is opening ---
        if self._bumper(left=True):
            if close_error > self.gripper_error_threshold:
                #--- case 2.1: gripper is stalled ---
                self.targets["gripper"] = actual_gripper + self.gripper_error_threshold
                print("Gripper could be stalled, stop opening")
            else:
                #--- case 2.2: gripper is not stalled ---
                self.targets["gripper"] += self.gripper_speed * dt
                

        #---- case 3: gripper cannot exert enough force -----
        a = self.controller.get_button(0)
        b = self.controller.get_button(1)

        if (a and b) or b:
            self.gripper_error_threshold = self.base_gripper_error_threshold
        elif a:
            self.gripper_error_threshold += 0.05
            self.targets["gripper"] -= self.gripper_speed * dt
            self.targets["gripper"] = max( self.targets["gripper"], actual_gripper - self.gripper_error_threshold) #--> kinda just prevents us from even closing too hard 
            print("tighening to:", self.gripper_error_threshold)
        

        self._clamp_joint("gripper")


    

        return {f"{name}.pos": self.targets[name] for name in self.motor_names}
