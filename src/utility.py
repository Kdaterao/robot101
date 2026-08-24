import time

import cv2
import numpy as np
import pygame
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep






#-------------------------------
#         CONSTANTS 
#_------------------------------
FPS = 30




NEUTRAL_POS = {
    "shoulder_pan.pos": -12.92,
    "shoulder_lift.pos": -1.41,
    "elbow_flex.pos": 16.53,
    "wrist_flex.pos": 5.01,
    "wrist_roll.pos": -0.22,
    "gripper.pos": 3.46,
}


REST_POSE = {
    "shoulder_pan.pos": -13.54,
    "shoulder_lift.pos": -99.08,
    "elbow_flex.pos": 95.91,
    "wrist_flex.pos": 65.67,
    "wrist_roll.pos": -0.13,
    "gripper.pos": 98.00,
}


# random starting positions for recording 
RAND_POS1 = {
    "shoulder_pan.pos": -62.07,
    "shoulder_lift.pos": -20.84,
    "elbow_flex.pos": 76.57,
    "wrist_flex.pos": 6.68,
    "wrist_roll.pos": -0.48,
    "gripper.pos": 2.92,
}

RAND_POS2 = {
    "shoulder_pan.pos": -85.10,
    "shoulder_lift.pos": 11.69,
    "elbow_flex.pos": 46.07,
    "wrist_flex.pos": 6.68,
    "wrist_roll.pos": -0.57,
    "gripper.pos": 2.92,
}

RAND_POS3 = {
    "shoulder_pan.pos": 16.97,
    "shoulder_lift.pos": 21.27,
    "elbow_flex.pos": 31.56,
    "wrist_flex.pos": 7.03,
    "wrist_roll.pos": -0.40,
    "gripper.pos": 17.18,
}

RAND_POS4 = {
    "shoulder_pan.pos": -82.02,
    "shoulder_lift.pos": 50.64,
    "elbow_flex.pos": -15.03,
    "wrist_flex.pos": 7.03,
    "wrist_roll.pos": -0.57,
    "gripper.pos": 17.04,
}

RAND_POS5 = {
    "shoulder_pan.pos": -16.97,
    "shoulder_lift.pos": -9.76,
    "elbow_flex.pos": -50.64,
    "wrist_flex.pos": 6.95,
    "wrist_roll.pos": -0.40,
    "gripper.pos": 17.04,
}

RANDOM_START_POSES = (RAND_POS1, RAND_POS2, RAND_POS3, RAND_POS4, RAND_POS5)





#-------------------------------
#           STRUCTS 
#_-------------------------------

JOINT_POS_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


joint_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Arm joints stay capped for safety; gripper can jump its full 0-100 range.
MAX_RELATIVE_TARGET = {
    name: (100.0 if name == "gripper" else 20.0) for name in joint_names
}


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


#-------------------------------
#  PRINTING + DEBUG UTILITEIES
#-------------------------------

def preview_camera(index: int) -> None:
    """Preview a camera with the same capture pipeline the SO101 uses."""

    camera = OpenCVCamera(
        OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=FPS)
    )
    camera.connect()

    window = f"camera {index} | press q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = np.asarray(camera.read())
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()





# for visualization
def focus_pygame_window() -> None:
    """Bring the pygame/SDL window to the front so Xbox input is received on macOS."""
    pygame.event.pump()
    try:
        from pygame._sdl2.video import Window

        Window.from_display_module().focus()
    except Exception:
        pass
    pygame.display.flip()


_cv_windows = set()
_refocused_pygame = False


def show_cameras(obs: dict, camera_key = None) -> None:
    """Display robot camera frames in OpenCV windows."""
    global _refocused_pygame

    for key, value in obs.items():
        if key != camera_key and camera_key: 
            continue
            
        if not hasattr(value, "ndim") or value.ndim != 3:
            continue
        frame = np.asarray(value)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        name = str(key)
        if name not in _cv_windows:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            _cv_windows.add(name)
        cv2.imshow(name, frame)
    cv2.waitKey(1)
    # OpenCV's first imshow steals macOS focus; give it back to pygame once.
    if _cv_windows and not _refocused_pygame:
        focus_pygame_window()
        _refocused_pygame = True



def print_joint_angles(robot: SO100Follower) -> dict[str, float]:
    """Read the arm's current joint positions and print them."""
    obs = robot.get_observation()
    angles = {key: float(obs[key]) for key in JOINT_POS_KEYS if key in obs}
    print("Current joint angles:")
    for key, val in angles.items():
        print(f"  {key}: {val:.2f}")
    print("REST_POSE = {")
    for key, val in angles.items():
        print(f'    "{key}": {val:.2f},')
    print("}")
    return angles




#--------------------------------------
#       SHUTDOWN UTILITIES 
#--------------------------------------


# Folded rest so the arm can sit on the base when torque is disabled.
# Lift is negative because stick-up (raise) is positive in teleop.


def go_to_rest(robot: SO100Follower, duration: float = 2.5) -> None:
    obs = robot.get_observation()
    start = {key: float(obs[key]) for key in REST_POSE}
    t0 = time.perf_counter()
    while True:
        alpha = min(1.0, (time.perf_counter() - t0) / duration)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        action = {
            key: start[key] + (target - start[key]) * smooth
            for key, target in REST_POSE.items()
        }
        robot.send_action(action)
        if alpha >= 1.0:
            break
        precise_sleep(1.0 / FPS)
    for _ in range(int(0.3 * FPS)):
        robot.send_action(REST_POSE)
        precise_sleep(1.0 / FPS)



def ease_to_position(
    robot: SO100Follower,
    target_position: dict[str, float],
    duration: float = 2.5,
) -> None:
    """Smoothly move the robot from its current position to a target position."""

    obs = robot.get_observation()

    start_position = {
        key: float(obs[key])
        for key in target_position
        if key in obs
    }

    t0 = time.perf_counter()

    while True:
        alpha = min(1.0, (time.perf_counter() - t0) / duration)

        # Smoothstep easing: starts and ends slowly.
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)

        action = {
            key: start_position[key]
            + (target - start_position[key]) * smooth
            for key, target in target_position.items()
            if key in start_position
        }

        robot.send_action(action)

        if alpha >= 1.0:
            break

        precise_sleep(1.0 / FPS)

    # Hold the final position briefly.
    for _ in range(int(0.3 * FPS)):
        robot.send_action(target_position)
        precise_sleep(1.0 / FPS)
#----------------------------
#   INPUT
#----------------------------


def ask_question(pygame, screen, question):
    font = pygame.font.Font(None, 32)
    clock = pygame.time.Clock()

    text = ""
    pygame.event.clear()

    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_RETURN:
                    return text

                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]

                else:
                    text += event.unicode

        screen.fill((30, 30, 30))

        question_surface = font.render(
            question,
            True,
            (255, 255, 255)
        )

        text_surface = font.render(
            text,
            True,
            (255, 255, 255)
        )

        screen.blit(question_surface, (20, 10))
        screen.blit(text_surface, (20, 50))

        pygame.display.flip()
        clock.tick(30)
