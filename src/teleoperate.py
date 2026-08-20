import os

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import time

import cv2
import numpy as np
import pygame
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from xboxController import MyTeleopConfig, xboxController

from utility import FPS, show_cameras, print_joint_angles, go_to_rest, ease_to_position, NEUTRAL_POS






def main():

    #------------------
    #   DEFINE CONTORLLER + FOLLOWER 
    #------------------

    #---- FOLLOWER ------
    camera_config = {"camera1": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
                     "camera2": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS)}
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5B3E0903291",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
        max_relative_target=20.0
    )

    robot = SO100Follower(robot_config)


    #----- CONTROLLER -------
    controller_config = MyTeleopConfig(id="xbox_controller")
    controller = xboxController(controller_config, robot)





    #-----------------
    #   CONNECT
    #----------------


    
    controller.connect()
    robot.connect(calibrate=True)



    #----------------#
    #    LOOP 
    #_---------------
    pygame.init()
    pygame.display.set_mode((320, 80))
    pygame.display.set_caption("teleop — press q to quit")

    time.sleep(0.3)
    prev_time = time.perf_counter()

    try:

        ease_to_position(robot, NEUTRAL_POS)
        while True:
            loop_start = time.perf_counter()
            obs = robot.get_observation()

            current_time = time.perf_counter()
            dt = current_time - prev_time

            action = controller.get_action(dt, obs)
            robot.send_action(action)
            show_cameras(obs)

            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_p]:
                print_joint_angles(robot)
                precise_sleep(0.3)
            if keys[pygame.K_q]:
                break

            if keys[pygame.K_j]:
                print_joint_angles(robot)

            prev_time = current_time
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - loop_start), 0.0))

    finally:
        try:
            print("Moving to rest pose...")
            go_to_rest(robot)
        except Exception as exc:
            print(f"Rest pose failed: {exc}")
        controller.disconnect()
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"Robot disconnect failed (motors may need a power cycle): {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
        main()

