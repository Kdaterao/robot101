import os

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import shutil
import time
import sys


import cv2
import numpy as np
import pygame
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from xboxController import MyTeleopConfig, xboxController

from utility import FPS, show_cameras, print_joint_angles, go_to_rest, ask_question, ease_to_position, NEUTRAL_POS, focus_pygame_window



from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME



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

        if whichCamera == 't':
            top_index = i

        if whichCamera == 'f':
            front_index = i


    if front_index == -1 or top_index == -1:
        print("both cameras not found")
        sys.exit()
    '''

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
        max_relative_target=20.0
    )

    robot = SO100Follower(robot_config)


    #----- CONTROLLER -------
    controller_config = MyTeleopConfig(id="xbox_controller")
    controller = xboxController(controller_config, robot)



    #---------------------------
    #        DATASET
    #---------------------------

    repo_id = "kdaterao/so101_data"
    dataset_root = HF_LEROBOT_HOME / repo_id
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": joint_names,
        },
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.camera2": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": joint_names,
        },
    }
    
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


        episode_index = 0

        #------ EPISODE RECORD LOOP ------
        while True:

            
            pygame.display.set_caption("") # reset caption


            #----- EPISODE??? ----------------
            answer = ask_question(pygame, screen, "type y to record new episode")
            
            if answer == 'y':

                #------ INPUT EPISODE TASK ------
                episode_task = ask_question(pygame, screen, "name for episode?")


                
                pygame.display.set_caption("teleop — press q to quit")
                focus_pygame_window()

                ease_to_position(robot, NEUTRAL_POS)
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
                    show_cameras(obs)



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


                #---- SAVE EPISDOE ------
                # Sequential encode avoids macOS leaking multiprocessing semaphores
                # after pygame has already initialized SDL.
                dataset.save_episode(parallel_encoding=False)

                episode_index += 1
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



        try:
            dataset.finalize()
        except Exception as exc:
            print(f"Dataset finalize failed: {exc}")

        try:
            answer = ask_question(pygame, screen, "y = save episodes")
        except Exception:
            answer = None

        if answer == "y":
            dataset.push_to_hub(
                branch="main",
                tags=["robotics", "manipulation"],
                license="apache-2.0",
                push_videos=True,
            )



        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
        main()
