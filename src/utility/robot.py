import time
import numpy as np

from lerobot.robots.so_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.datasets import LeRobotDataset


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


def print_joint_angles(robot: SO100Follower) -> dict[str, float]:
    """Read the arm's current joint positions and print them."""
    obs = robot.get_observation()
    angles = {key: float(obs[key]) for key in JOINT_POS_KEYS if key in obs}
    print("Current joint angles:")
    for key, val in angles.items():
        print(f"  {key}: {val:.2f}")
    print("}")
    return angles


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
    for _ in range(int(0.3 * FPS)):
        robot.send_action(target_position)
        precise_sleep(1.0 / FPS)




def record_ease_to_position(
    ctx,
    target_position: dict[str, float],
    duration: float = 2.5,
) -> None:
    """Smoothly move the robot from its current position to a target position."""

    #------ GET CTX OBJECTS ------
    robot = ctx["robot"]
    dataset= ctx["dataset"]
    episode_task = ctx["task"]


    #------ GET START AND END POS --------
    obs = robot.get_observation()
    start_position = {
        key: float(obs[key])
        for key in target_position
        if key in obs
    }
    t0 = time.perf_counter()


    #------ RUN MOVEMENT ----------
    while True:

        # GET NEXT ACTION STEP 
        alpha = min(1.0, (time.perf_counter() - t0) / duration)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        action = {
            key: start_position[key]
            + (target - start_position[key]) * smooth
            for key, target in target_position.items()
            if key in start_position
        }


        # RECORD STATE 
        obs = robot.get_observation()

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



        # FOLLOW THROUGH WITH ACTION STEP 
        robot.send_action(action)


        #[ LOOP HANDLING ]
        if alpha >= 1.0:
            break
        precise_sleep(1.0 / FPS)


def record_pause(ctx, dt: float = 2.0) -> None:
    """Record the robot holding its current position for dt seconds."""

    # ------ GET CTX OBJECTS ------
    robot = ctx["robot"]
    dataset = ctx["dataset"]
    episode_task = ctx["task"]

    # ------ GET CURRENT POSITION ------
    obs = robot.get_observation()

    action = {
        f"{name}.pos": float(obs[f"{name}.pos"])
        for name in joint_names
    }

    # ------ RECORD FOR dt SECONDS ------
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < dt:

        # Get latest observation
        obs = robot.get_observation()

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

        # Keep robot at its current position
        robot.send_action(action)

        precise_sleep(1.0 / FPS)


