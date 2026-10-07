"""Record wrist-cam demos for tapnetCreate.

Writes:
  data/demo_001/
    video.mp4
    robot_states.npy   # (T, 6) joint degrees; last col = gripper 0 closed..100 open
    timestamps.npy
    events.json        # stage boundaries: grasp (clamp) / release / manual
    segment.npy        # [0, last_frame] (legacy single-stage)

Usage:
  python src/tapnetRecord.py --out-dir data --camera 0 --port COM3 --leader-port COM4
  # y to start episode, q to stop recording, then y/N to keep

Always teleops with SO101LeaderController (default leader port COM4).
Stage boundaries come from gripper clamp detection: closing on an object
stalls the follower gripper -> [CLAMP] (grasp); opening again -> [RELEASE].
Space or e ends the current stage (manual boundary). q ends the episode.
No AprilTag / LilyTags. Raw frames are saved; tapnetCreate undistorts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import cv2
import numpy as np
import pygame
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so_leader import SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gripper_clamp import GripperClampDetector
from SO101LeaderController import SOLeaderController
from utility import (
    FPS,
    MAX_RELATIVE_TARGET,
    ease_to_position,
    go_to_rest,
    joint_names,
    print_joint_angles,
)

JOINT_KEYS = [f"{name}.pos" for name in joint_names]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record demos for tapnetCreate")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--camera", type=int, default=0, help="Wrist OpenCV camera index")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="SO-101 follower serial port",
    )
    p.add_argument(
        "--leader-port",
        type=str,
        default="COM4",
        help="SO-101 leader serial port",
    )
    p.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")
    p.add_argument("--leader-id", type=str, default="my_awesome_leader_arm")
    p.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Force next demo index (default: max existing demo_N + 1)",
    )
    return p.parse_args()


def _next_demo_index(out_dir: Path, start_index: int | None) -> int:
    if start_index is not None:
        return int(start_index)
    best = 0
    if out_dir.is_dir():
        for d in out_dir.iterdir():
            if d.is_dir() and d.name.startswith("demo_"):
                try:
                    best = max(best, int(d.name.split("_", 1)[1]))
                except ValueError:
                    pass
    return best + 1


def _obs_to_bgr(obs: dict, width: int, height: int) -> np.ndarray:
    img = obs.get("camera1")
    if img is None:
        raise RuntimeError("observation missing camera1")
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height))
    return bgr


def _show_bgr_pygame(
    screen: pygame.Surface, bgr: np.ndarray, caption: str, max_w: int = 960
) -> pygame.Surface:
    """Preview via pygame (scaled; opencv-python-headless has no GUI)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        rgb = cv2.resize(
            rgb, (max_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA
        )
    # frombuffer needs a live buffer; copy keeps the surface valid
    rgb = np.ascontiguousarray(rgb)
    surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
    if screen.get_size() != (rgb.shape[1], rgb.shape[0]):
        screen = pygame.display.set_mode((rgb.shape[1], rgb.shape[0]))
    screen.blit(surf, (0, 0))
    pygame.display.set_caption(caption)
    pygame.display.flip()
    return screen


def _save_demo(
    demo_dir: Path,
    frames_bgr: list[np.ndarray],
    states: list[np.ndarray],
    timestamps: list[float],
    width: int,
    height: int,
    events: list[dict],
) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    video_path = demo_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(FPS), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {video_path}")
    for fr in frames_bgr:
        writer.write(fr)
    writer.release()

    np.save(demo_dir / "robot_states.npy", np.stack(states, axis=0).astype(np.float32))
    np.save(
        demo_dir / "timestamps.npy",
        np.asarray(timestamps, dtype=np.float64),
    )
    # Full episode = TAPIR segment (ended by keypress in the recorder)
    np.save(
        demo_dir / "segment.npy",
        np.array([0, len(frames_bgr) - 1], dtype=np.int32),
    )
    with (demo_dir / "events.json").open("w", encoding="utf-8") as f:
        json.dump({"num_frames": len(frames_bgr), "events": events}, f, indent=2)
    summary = ", ".join(f"{e['type']}@{e['frame']}" for e in events) or "none"
    print(
        f"Saved {demo_dir}  frames={len(frames_bgr)}  "
        f"events=[{summary}]  "
        f"gripper end={float(states[-1][-1]):.1f}"
    )


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    demo_index = _next_demo_index(args.out_dir, args.start_index)

    camera_config = {
        "camera1": OpenCVCameraConfig(
            index_or_path=args.camera,
            width=args.width,
            height=args.height,
            fps=FPS,
        ),
    }
    robot = SO100Follower(
        SO100FollowerConfig(
            port=args.port,
            id=args.robot_id,
            cameras=camera_config,
            use_degrees=True,
            max_relative_target=MAX_RELATIVE_TARGET,
        )
    )
    controller = SOLeaderController(
        SO100LeaderConfig(
            port=args.leader_port,
            id=args.leader_id,
            use_degrees=True,
        )
    )

    controller.connect()
    robot.connect(calibrate=True)

    pygame.init()
    print(
        f"tapnetRecord -> {args.out_dir.resolve()}  "
        f"next=demo_{demo_index:03d}  cam={args.camera}  "
        f"leader={args.leader_port}  follower={args.port}"
    )
    print(
        "Prompt: y = start episode. During record: q = END episode, "
        "space/e = end current stage, p = print joints."
    )
    print(
        "Use space/e for a stage boundary when clamp detection is not enough "
        "(e.g. approach pose above the cube)."
    )

    try:
        while True:
            answer = input("type y to record new demo: ").strip().lower()
            if answer != "y":
                break

            demo_name = f"demo_{demo_index:03d}"
            demo_dir = args.out_dir / demo_name

            pygame.display.init()
            preview_w = min(960, args.width)
            preview_h = max(
                1, int(round(args.height * (preview_w / float(max(args.width, 1)))))
            )
            screen = pygame.display.set_mode((preview_w, preview_h))
            pygame.display.set_caption(f"tapnetRecord {demo_name} - q to stop")

            obs = robot.get_observation()
            start_pos = controller.get_action(0, obs)
            ease_to_position(robot, start_pos)
            obs = robot.get_observation()
            controller.sync_from_observation(obs)
            pygame.event.pump()
            prev_time = time.perf_counter()
            t0 = prev_time

            frames_bgr: list[np.ndarray] = []
            states: list[np.ndarray] = []
            timestamps: list[float] = []
            events: list[dict] = []
            clamp = GripperClampDetector(label=demo_name)
            clamp.reset(float(obs["gripper.pos"]))

            print(
                f"Recording {demo_name}... "
                "(q = end episode, space/e = end stage; "
                "[CLAMP]/[RELEASE] print on grasp/release)"
            )
            while True:
                loop_start = time.perf_counter()
                current_time = time.perf_counter()
                dt = current_time - prev_time

                obs = robot.get_observation()
                action = controller.get_action(dt, obs)
                robot.send_action(action)

                bgr = _obs_to_bgr(obs, args.width, args.height)
                state = np.array(
                    [float(obs[k]) for k in JOINT_KEYS],
                    dtype=np.float32,
                )
                frame_idx = len(frames_bgr)
                frames_bgr.append(bgr)
                states.append(state)
                timestamps.append(current_time - t0)

                g = float(state[-1])
                ev = clamp.update(controller.last_leader_gripper, g, frame=frame_idx)
                if ev is not None:
                    events.append(
                        {
                            "frame": frame_idx,
                            "type": "grasp" if ev == "clamp" else "release",
                            "gripper": g,
                        }
                    )

                preview = bgr.copy()
                cv2.putText(
                    preview,
                    f"{demo_name}  n={len(frames_bgr)}  grip={g:.1f}  "
                    f"{'CLAMPED' if clamp.clamped else ''}  stages={len(events) + 1}  "
                    f"[space/e=end stage]",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255) if clamp.clamped else (0, 255, 255),
                    2,
                )
                screen = _show_bgr_pygame(
                    screen,
                    preview,
                    f"tapnetRecord {demo_name} - q stop, space/e end stage",
                )

                quit_requested = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            quit_requested = True
                        elif event.key == pygame.K_p:
                            print_joint_angles(robot)
                        elif event.key in (pygame.K_SPACE, pygame.K_e):
                            events.append(
                                {"frame": frame_idx, "type": "manual", "gripper": g}
                            )
                            print(
                                f"{demo_name} [STAGE END] frame={frame_idx} grip={g:.1f} "
                                f"(manual)  -> stage {len(events) + 1}",
                                flush=True,
                            )
                if quit_requested:
                    break

                prev_time = current_time
                precise_sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - loop_start)))

            pygame.display.quit()

            try:
                go_to_rest(robot)
            except Exception as exc:  # noqa: BLE001
                print(f"Rest pose failed: {exc}")

            if len(frames_bgr) < 2:
                print("Too few frames; discarding.")
                continue

            keep = input(f"keep {demo_name}? [{len(frames_bgr)} frames] [y/N] ").strip().lower()
            if keep == "y":
                _save_demo(
                    demo_dir,
                    frames_bgr,
                    states,
                    timestamps,
                    args.width,
                    args.height,
                    events,
                )
                demo_index += 1
            else:
                print("discarded")

    finally:
        try:
            go_to_rest(robot)
        except Exception as exc:  # noqa: BLE001
            print(f"Rest pose failed: {exc}")
        try:
            controller.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"Controller disconnect failed: {exc}")
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"Robot disconnect failed: {exc}")
        pygame.quit()


if __name__ == "__main__":
    main()
