import os

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import random
import shutil
import time
import sys


import cv2
import numpy as np
import pygame
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.feature_utils import hw_to_dataset_features

from controllers import CONTROLLER, make_controller, ControllerType

from utility import FPS, MAX_RELATIVE_TARGET, show_cameras, print_joint_angles, go_to_rest, ease_to_position, NEUTRAL_POS, REST_POSE, preview_camera
from utility import RANDOM_START_POSES
from utility import features 
from utility import joint_names


# True: ease to a random RAND_POS* at the start of each episode.
# False: always ease to NEUTRAL_POS.
USE_RANDOM_START = False #--> only do this if we get a good baseline !


from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME



def main():
    pygame.init()

    #------------------
    #  Confirm Cameras 
    #------------------
    front_index = 0
    top_index = 1

    
    for i in range(2):
        print(i)
        preview_camera(i)

    #------------------
    #   DEFINE CONTORLLER + FOLLOWER 
    #------------------

    #---- FOLLOWER ------
    camera_config = {"camera1": OpenCVCameraConfig(index_or_path=front_index, width=640, height=480, fps=FPS),
                     "camera2": OpenCVCameraConfig(index_or_path=top_index, width=640, height=480, fps=FPS)
                     }
    
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5B3E0903291",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
        max_relative_target=MAX_RELATIVE_TARGET
    )

    robot = SO100Follower(robot_config)

    #---- debugging related ------
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    print(dataset_features)

    #----- CONTROLLER -------
    # Switch in controllers.py: ControllerType.XBOX or ControllerType.SO100_LEADER
    controller = make_controller(CONTROLLER, robot)



    #---------------------------
    #        DATASET
    #---------------------------

    repo_id = "kdaterao/so101_data2"
    dataset_root = HF_LEROBOT_HOME / repo_id

    
    if (dataset_root / "meta" / "tasks.parquet").exists():
        print(f"Resuming local dataset at {dataset_root}")
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=dataset_root)
    else:
        if dataset_root.exists():
            print('had to remove dataset')
            shutil.rmtree(dataset_root)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=dataset_root,
            fps=30,
            robot_type="so_follower",
            features=features,
            use_videos=True,
        )



    
    #-----------------
    #   CONNECT
    #----------------

    #---- controller + robot  connect -----
    controller.connect()
    robot.connect(calibrate=True)


    #-----------------
    #    Loop
    #-----------------

    #--- start timer -----
    time.sleep(0.3)
    prev_time = time.perf_counter()

    #--- logic ----
    try:
        episode_index = 1
        #------ EPISODE RECORD LOOP ------
        while True:

            answer = input("type y to record new episode: ").strip().lower()

            if answer == "y":
                episode_task = input("name for episode?: ").strip()

                pygame.display.init()
                pygame.display.set_mode((500, 500))
                pygame.display.set_caption("teleop — press q to quit")

                if CONTROLLER == ControllerType.SO101:
                        obs = robot.get_observation()
                        start_pos = controller.get_action(0, obs)
                else:
                        action = REST_POSE
                ease_to_position(robot, start_pos)
                # Re-seed controller joints after the ease, and reset dt so the
                # time spent in prompts isn't applied as one huge first step.
                obs = robot.get_observation()
                controller.sync_from_observation(obs)
                pygame.event.pump()
                prev_time = time.perf_counter()

                #----- RUN EPISODE -------
                while True:

                    # time update logic
                    loop_start = time.perf_counter()
                    current_time = time.perf_counter()
                    dt = current_time - prev_time



                    # observation + action + camera
                    obs = robot.get_observation()
                    action = controller.get_action(dt, obs)

                    state = np.array(
                        [obs[f"{name}.pos"] for name in joint_names],
                        dtype=np.float32,
                    )
                    action_vec = np.array(
                        [action[f"{name}.pos"] for name in joint_names],
                        dtype=np.float32,
                    )
                    frame = {
                        "observation.state": state,
                        "observation.images.camera1": obs["camera1"],
                        "observation.images.camera2": obs["camera2"],
                        "action": action_vec,
                        "task": episode_task,
                    }
                    dataset.add_frame(frame)

                    
                    # follow through action 
                    robot.send_action(action)
                    show_cameras(obs, "camera1")



                    # quit program handler logic
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_p]:
                        print_joint_angles(robot)
                        precise_sleep(0.3)
                    if keys[pygame.K_q]:
                        break


                    # time update logic
                    prev_time = current_time
                    precise_sleep(max(1.0 / FPS - (time.perf_counter() - loop_start), 0.0))


                pygame.display.quit()
                
                try:
                    print("Moving to rest pose between episodes... ")
                    go_to_rest(robot)
                except Exception as exc:
                    print(f"Rest pose failed: {exc}")

                print("episode_index:", episode_index)
                good_episode = input("good episode? [y/N] ").strip().lower()
                if good_episode == "y":
                    episode_index += 1
                    dataset.save_episode()
                else:

                    print("discarding episode")

                    
                    print("BEFORE CLEAR:")
                    print("pending frames:", dataset.has_pending_frames())
                    print("buffer size:", dataset.writer.episode_buffer["size"])
                    dataset.clear_episode_buffer(delete_images=True)
                    print("AFTER CLEAR:")
                    print("pending frames:", dataset.has_pending_frames())
                    print("buffer size:", dataset.writer.episode_buffer["size"])

         

            else:
                break

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
        pygame.quit()

        try:
            print("Encoding episode videos...")
            dataset.finalize()
        except Exception as exc:
            print(f"Dataset finalize failed: {exc}")

        answer = input("Push dataset to Hugging Face? [y/N] ").strip().lower()
        if answer == "y":
            dataset.push_to_hub(
                branch="main",
                tags=["robotics", "manipulation"],
                license="apache-2.0",
                push_videos=True,
            )


if __name__ == "__main__":
        main()
