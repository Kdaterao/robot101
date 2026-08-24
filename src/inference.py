import sys
import time

import cv2
import pygame
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

from utility import FPS, ask_question, go_to_rest, preview_camera, print_joint_angles


def main():
    pygame.init()
    screen = pygame.display.set_mode((500, 500))

    #------------------
    #  Confirm Cameras
    #------------------
    front_index = 0
    top_index = 1

    '''
    for i in range(2):
        print(i)
        preview_camera(i)

        whichCamera = ask_question(pygame, screen, "t = top, f = front, anything else = neither")

        if whichCamera == "t":
            top_index = i

        if whichCamera == "f":
            front_index = i

    if front_index == -1 or top_index == -1:
        print("both cameras not found")
        sys.exit()
    '''

    #------------------
    #   DEFINE POLICY + ROBOT
    #------------------

    camera_config = {
        # smolvla_base expects observation.images.camera1 / camera2 (camera3 is optional/padded)
        "camera1": OpenCVCameraConfig(index_or_path=front_index, width=640, height=480, fps=FPS),
        "camera2": OpenCVCameraConfig(index_or_path=top_index, width=640, height=480, fps=FPS),
    }
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5B3E0903291",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
    )

    robot = SO100Follower(robot_config)

    

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_id = "lerobot/smolvla_base"
    #model_id = "kdaterao/smolVLA_desk"
    
    model = SmolVLAPolicy.from_pretrained(model_id)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    #-----------------
    #   CONNECT
    #----------------

    robot.connect(calibrate=True)

    pygame.display.set_caption("inference — press q to quit")

    task = "pick up blue box"
    robot_type = "so101_follower"

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    i = 0
    try:
        while True:

            obs = robot.get_observation()

            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=task,
                robot_type=robot_type,
            )
            obs = preprocess(obs_frame)

            with torch.inference_mode():
                action = model.select_action(obs)

            action = postprocess(action)

            i += 1
            print(i)
            print(action)


            action = make_robot_action(action, dataset_features)
            robot.send_action(action)

            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_p]:
                print_joint_angles(robot)
                precise_sleep(0.3)
            if keys[pygame.K_q]:
                break

    finally:
        try:
            print("Moving to rest pose...")
            go_to_rest(robot)
        except Exception as exc:
            print(f"Rest pose failed: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"Robot disconnect failed (motors may need a power cycle): {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
