"""Runtime RoboTAP visual servoing for SO-101 (wrist camera).

Runs the multi-stage motion plan from tapnetCreate:
  - tracks the task's saved query features live (no re-init on the first frame),
  - per stage, targets a demo frame slightly ahead of the nearest demo frame,
    switching to the across-demo mean goal near the end of the stage,
  - 4-DoF servo (translation first, rotation, z from spread),
  - when the stage converges, runs its primitive: grasp closes the gripper
    until clamp detection fires, release opens to the demo's gripper value.

Modes:
  viz         — track + overlay + stage logic (robot does not move)
  servo_print — also print camera deltas each frame
  ik_print    — solve IK, print joint targets (no send_action)
  robot       — send joint targets and gripper commands; q = e-stop

Keys (pygame window): q quit, n force next stage.

Usage:
  python src/tapnetGrab.py --task tasks/pick_place_task.npz --mode servo_print
  python src/tapnetGrab.py --task tasks/pick_place_task.npz --mode robot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from utility.robot import go_to_rest
import pygame

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utility.robot import NEUTRAL_POS, ease_to_position

# ---------------------------------------------------------------------------
# Wrist camera intrinsics — edit these (same as test_lily_relative.py)
# Calibrated at 1280x720, scaled x1.5 for 1920x1080. Use the full 8-param
# rational model (5-param alone warps badly). Rebuild tasks after changing.
# ---------------------------------------------------------------------------
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAM_FX = 1262.480288175
CAM_FY = 1264.71879012
CAM_CX = 913.586772795
CAM_CY = 609.63948621
CAM_DIST = (
    3.83407503,  # k1
    -2.80771278,  # k2
    -0.00124016194,  # p1
    0.00296903720,  # p2
    -0.755297301,  # k3
    4.27744077,  # k4
    -1.18548628,  # k5
    -2.35292973,  # k6
)
# Real wrist camera vs MuJoCo wrist_cam, rotation about the optical axis.
# 180 = real image is upside down relative to the model (fitted from demo
# image motion vs FK camera motion).
CAM_MOUNT_ROLL_DEG = 180.0
# False = use raw camera frames. Must match how the task was built
# (UNDISTORT in tapnetCreate.py) since goals are stored in pixels.
UNDISTORT = True

from gripper_clamp import GripperClampDetector
from robotap import (
    choose_target,
    mean_pixel_error,
    remap_plan_first_primitive,
    select_servo_inliers,
    solve_robotap_servo,
    unpack_plan,
)
from tapnet_utils import (
    BootsTAPIR,
    DEFAULT_CHECKPOINT,
    apply_ee_delta,
    bgr_to_rgb,
    camera_delta_to_world,
    clamp_delta,
    default_wrist_calib,
    features_from_numpy,
    load_task_npz,
)

FPS = 30
# Camera (OpenCV) roll per degree of wrist_roll, measured in the MuJoCo model.
WRIST_ROLL_PER_CAM_ROLL = -0.88
JOINT_POS_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

try:
    from utility.robot import MAX_RELATIVE_TARGET
except Exception:  # noqa: BLE001
    MAX_RELATIVE_TARGET = {
        "shoulder_pan": 20.0,
        "shoulder_lift": 20.0,
        "elbow_flex": 20.0,
        "wrist_flex": 20.0,
        "wrist_roll": 20.0,
        "gripper": 100.0,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoboTAP visual-servoing controller")
    p.add_argument("--task", type=Path, default=Path("tasks/pick_place_task.npz"))
    p.add_argument(
        "--mode", choices=("viz", "servo_print", "ik_print", "robot"), default="viz"
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--camera", type=int, default=0, help="OpenCV wrist camera index")
    p.add_argument("--port", type=str, default="COM3", help="SO-101 follower port")
    p.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")
    p.add_argument(
        "--no-robot",
        action="store_true",
        help="Webcam only (viz / servo_print); primitives are dry-run",
    )
    p.add_argument(
        "--first-primitive",
        choices=("task", "grasp", "release"),
        default="task",
        help="First gripper action after a servo stage: close (grasp), open (release), "
        "or keep whatever is stored in the task (default)",
    )
    p.add_argument(
        "--freeze-after-first",
        action="store_true",
        help="Test: after stage 0 finishes (servo + its primitive), hold the current "
        "joint pose and do not run later stages",
    )
    p.add_argument(
        "--test-down-cm",
        type=float,
        default=0.0,
        help="Test: after stage 0 hits the pixel checkpoint, move the gripper "
        "straight down this many cm in the SO-101/MuJoCo world frame (-Z), "
        "then run the normal stage primitive. 0 = off",
    )
    p.add_argument(
        "--undistort",
        action=argparse.BooleanOptionalAction,
        default=UNDISTORT,
        help=f"Undistort frames with CAM_* (default from UNDISTORT={UNDISTORT})",
    )
    p.add_argument("--dt", type=float, default=1.0 / FPS)
    p.add_argument(
        "--gain-trans",
        type=float,
        default=0.15,
        help="m/s per unit of normalized image error (x, y)",
    )
    p.add_argument("--gain-z", type=float, default=0.10, help="m/s per unit log-scale error")
    p.add_argument("--gain-rot", type=float, default=1.0, help="rad/s per rad of rotation error")
    p.add_argument(
        "--servo-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Flip all translation if the arm moves away from the goal",
    )
    p.add_argument(
        "--rot-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Flip wrist rotation direction",
    )
    p.add_argument(
        "--mount-roll",
        type=float,
        default=CAM_MOUNT_ROLL_DEG,
        help="Real camera roll vs MuJoCo wrist_cam in degrees (default CAM_MOUNT_ROLL_DEG)",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.3,
        help="Velocity low-pass factor 0..1 (1 = no smoothing, lower = smoother)",
    )
    p.add_argument(
        "--deadband-px",
        type=float,
        default=8.0,
        help="Hold still when the mean pixel error is below this",
    )
    p.add_argument(
        "--ik-min-move",
        type=float,
        default=0.006,
        help="Re-solve IK only after the target moves this far (m); else resend last pose",
    )
    p.add_argument(
        "--roll-min-deg",
        type=float,
        default=1.5,
        help="Re-solve once accumulated wrist roll reaches this (deg)",
    )
    p.add_argument(
        "--max-joint-step",
        type=float,
        default=4.0,
        help="Max change per arm joint per tick (deg)",
    )
    p.add_argument("--max-trans", type=float, default=0.01, help="m per tick")
    p.add_argument(
        "--leash",
        type=float,
        default=0.03,
        help="Max distance (m) the integrated EE target may lead the real gripper",
    )
    p.add_argument("--max-rot", type=float, default=0.05, help="rad per tick")
    p.add_argument("--min-points", type=int, default=3)
    p.add_argument(
        "--inlier-frac",
        type=float,
        default=0.45,
        help="Fraction of consistent visible points used for servo / converge",
    )
    p.add_argument(
        "--inlier-min-cos",
        type=float,
        default=0.25,
        help="Min cosine vs median error pull; rejects opposite / irrelevant tracks",
    )
    p.add_argument("--lookahead", type=int, default=5, help="Demo frames ahead of nearest")
    p.add_argument(
        "--end-frac",
        type=float,
        default=0.85,
        help="Switch to the across-demo mean goal past this stage progress",
    )
    p.add_argument(
        "--converge-px",
        type=float,
        default=40.0,
        help="Mean pixel-error threshold: finish this servo step when under this "
        "for --converge-frames (same idea as tapnetGrabGoal)",
    )
    p.add_argument(
        "--converge-frames",
        type=int,
        default=1,
        help="Consecutive frames under --converge-px required before advancing "
        "(default 1 = advance on first frame under threshold)",
    )
    p.add_argument(
        "--advance-progress",
        type=float,
        default=0.0,
        help="Optional escape hatch: also finish if nearest-demo progress reaches "
        "this (0..1). 0 = disabled; stage ends on pixel error only",
    )
    p.add_argument("--grip-step", type=float, default=2.0, help="Gripper close step per tick")
    p.add_argument(
        "--release-open",
        type=float,
        default=95.0,
        help="Minimum gripper command for a release (0=closed, 100=open). "
        "Demo targets below this are raised so the jaw actually opens.",
    )
    p.add_argument(
        "--grip-timeout",
        type=int,
        default=90,
        help="Ticks before a grasp/release primitive gives up and advances",
    )
    return p.parse_args()


def _real_to_model_cam(delta: np.ndarray, roll_deg: float) -> np.ndarray:
    """Rotate a real-camera [dx, dy, dz, droll] into the MuJoCo wrist_cam frame."""
    a = np.deg2rad(roll_deg)
    c, s = np.cos(a), np.sin(a)
    out = np.asarray(delta, dtype=np.float64).copy()
    out[0], out[1] = c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]
    return out


def _draw_overlay(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    goals: np.ndarray | None,
    valid: np.ndarray,
    lines: list[str],
    inliers: np.ndarray | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    for i, p in enumerate(points):
        if inliers is not None and inliers[i]:
            color = (0, 255, 0)  # used for servo / converge
        elif valid[i]:
            color = (0, 140, 255)  # visible but rejected
        else:
            color = (80, 80, 80)
        cv2.circle(out, (int(p[0]), int(p[1])), 5, color, -1)
        if goals is not None:
            g = goals[i]
            cv2.circle(out, (int(g[0]), int(g[1])), 5, (0, 255, 255), 2)
            if inliers is not None and inliers[i]:
                cv2.line(
                    out, (int(p[0]), int(p[1])), (int(g[0]), int(g[1])), (255, 128, 0), 1
                )
    for k, text in enumerate(lines):
        cv2.putText(
            out, text, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
    return out


def _show_bgr_pygame(
    screen: pygame.Surface, bgr: np.ndarray, caption: str, max_w: int = 960
) -> pygame.Surface:
    """Preview window (scaled down so 1080p is not fullscreen)."""
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        rgb = cv2.resize(rgb, (max_w, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(rgb)
    surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
    if screen.get_size() != (rgb.shape[1], rgb.shape[0]):
        screen = pygame.display.set_mode((rgb.shape[1], rgb.shape[0]))
    screen.blit(surf, (0, 0))
    pygame.display.set_caption(caption)
    pygame.display.flip()
    return screen


def _open_webcam(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")
    return cap


def _read_bgr_robot(robot, cam_key: str = "camera1") -> tuple[np.ndarray, dict]:
    obs = robot.get_observation()
    if cam_key in obs:
        img = obs[cam_key]
    else:
        keys = [k for k, v in obs.items() if hasattr(v, "shape") and len(getattr(v, "shape", ())) == 3]
        if not keys:
            raise RuntimeError("no camera image in observation")
        img = obs[keys[0]]
    # LeRobot images are RGB
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), obs


class StageRunner:
    """Stage state machine: servo → primitive (grasp / release) → next stage."""

    def __init__(self, plan, args, live: bool):
        self.plan = plan
        self.args = args
        self.live = live  # gripper commands are actually sent
        self.stage = 0
        self.phase = "servo"  # servo | descend | grasp | release | freeze | done
        self.converged_count = 0
        self.max_progress = 0.0  # nearest-demo progress only increases within a stage
        self.prim_ticks = 0
        self.grip_cmd: float | None = None
        self.freeze_joints: dict[str, float] | None = None
        self.descend_remaining_m = 0.0
        self.descend_total_m = 0.0
        self.descend_done = False  # only once after first checkpoint
        self.clamp = GripperClampDetector(verbose=True, label="[grab]")
        self._announce()

    @property
    def n_stages(self) -> int:
        return len(self.plan.stages)

    @property
    def current(self):
        return self.plan.stages[self.stage]

    def _announce(self) -> None:
        st = self.current
        extra = "  (--freeze-after-first: will hold after this stage)" if (
            self.args.freeze_after_first and self.stage == 0
        ) else ""
        print(
            f"[STAGE {self.stage}/{self.n_stages - 1}] servo with {len(st.point_ids)} points, "
            f"then '{st.primitive}'{extra}",
            flush=True,
        )

    def next_stage(self, why: str) -> None:
        if self.args.freeze_after_first and self.stage == 0 and self.phase != "freeze":
            print(
                f"[FREEZE] first stage complete ({why}); holding current pose "
                "(later stages skipped — test mode)",
                flush=True,
            )
            self.phase = "freeze"
            self.converged_count = 0
            self.max_progress = 0.0
            return
        if self.stage + 1 >= self.n_stages:
            if self.phase != "done":
                print(f"[DONE] task complete ({why})", flush=True)
            self.phase = "done"
            return
        self.stage += 1
        self.phase = "servo"
        self.converged_count = 0
        self.max_progress = 0.0
        print(f"[ADVANCE] -> stage {self.stage} ({why})", flush=True)
        self._announce()

    def on_converged_frame(self, err_px: float, using_mean: bool, progress: float) -> bool:
        """Advance when mean pixel error stays under --converge-px for N frames.

        Demo progress is tracked for the overlay / --end-frac mean-goal switch
        only. With default ``--advance-progress 0``, progress alone never ends
        the stage (same stop rule as tapnetGrabGoal).
        """
        self.max_progress = max(self.max_progress, float(progress))
        thr = float(self.args.converge_px)
        at_goal = err_px <= thr
        prog_thr = float(self.args.advance_progress)
        far_enough = prog_thr > 0.0 and self.max_progress >= prog_thr
        if at_goal or far_enough:
            self.converged_count += 1
        elif self.converged_count > 0 and err_px <= 1.25 * thr:
            # hysteresis: brief noise does not reset a pixel-error streak
            pass
        else:
            self.converged_count = 0
        need = max(1, int(self.args.converge_frames))
        if self.converged_count > 0 and self.converged_count < need:
            print(
                f"[THRESHOLD] stage {self.stage} "
                f"{self.converged_count}/{need}  "
                f"err={err_px:.1f}px (need <= {thr:.1f}px)  "
                f"progress={self.max_progress * 100:.0f}% "
                f"(mean={'Y' if using_mean else 'N'})",
                flush=True,
            )
        if self.converged_count < need:
            return False
        prim = self.current.primitive
        why = (
            f"err {err_px:.1f}px <= {thr:.1f}px"
            if at_goal
            else (
                f"progress {self.max_progress * 100:.0f}% >= "
                f"{prog_thr * 100:.0f}% (escape)"
            )
        )
        print(
            f"[CONVERGED] stage {self.stage} ({why}) -> next step '{prim}'",
            flush=True,
        )
        # Test hook: after first checkpoint, drop straight down before primitive.
        if (
            self.stage == 0
            and not self.descend_done
            and float(self.args.test_down_cm) > 0
        ):
            self.descend_total_m = float(self.args.test_down_cm) / 100.0
            self.descend_remaining_m = self.descend_total_m
            self.phase = "descend"
            self.grip_cmd = None
            print(
                f"[TEST DOWN] stage 0 checkpoint hit — descending "
                f"{self.args.test_down_cm:.1f} cm (world -Z), then '{prim}'",
                flush=True,
            )
            return True
        self._start_primitive_or_advance(prim)
        return True

    def on_descend_done(self) -> None:
        self.descend_done = True
        self.descend_remaining_m = 0.0
        prim = self.current.primitive
        print(
            f"[TEST DOWN] done ({self.descend_total_m * 100:.1f} cm) -> '{prim}'",
            flush=True,
        )
        self._start_primitive_or_advance(prim)

    def _start_primitive_or_advance(self, prim: str) -> None:
        if prim == "none":
            self.next_stage("converged")
        elif not self.live:
            print(f"[{prim.upper()}] dry run (no gripper command in this mode)", flush=True)
            self.next_stage(f"dry-run {prim}")
        else:
            self.phase = prim
            self.prim_ticks = 0
            self.grip_cmd = None

    def gripper_command(self, measured: float, frame: int) -> float:
        """Gripper value to send this tick (holds the clamp after a grasp)."""
        if self.phase in ("freeze", "descend"):
            if self.grip_cmd is None:
                self.grip_cmd = measured
            return float(self.grip_cmd)
        if self.grip_cmd is None:
            self.grip_cmd = measured
            self.clamp.reset(measured)
        if self.phase == "grasp":
            self.prim_ticks += 1
            self.grip_cmd = max(0.0, self.grip_cmd - self.args.grip_step)
            # servo latency: the gripper has not started moving yet, which
            # would otherwise look like a stall
            if self.prim_ticks <= 10:
                self.clamp.reset(measured)
                ev = None
            else:
                ev = self.clamp.update(self.grip_cmd, measured, frame)
            if ev == "clamp":
                self.grip_cmd = max(0.0, measured - 3.0)
                print(
                    f"[CLAMP] confirmed at grip={measured:.1f}, advancing to stage "
                    f"{self.stage + 1}",
                    flush=True,
                )
                self.next_stage("clamp")
            elif measured <= 1.0 or self.prim_ticks >= self.args.grip_timeout:
                print(
                    f"[GRASP] WARNING: gripper closed to {measured:.1f} without clamp "
                    "(missed the object?); advancing anyway",
                    flush=True,
                )
                self.next_stage("grasp timeout")
        elif self.phase == "release":
            self.prim_ticks += 1
            # Demo settle values are often half-open; floor so the jaw opens fully.
            target = max(float(self.current.gripper_target), float(self.args.release_open))
            # Step open (same idea as grasp step-close) so the follower can keep up.
            self.grip_cmd = min(100.0, max(self.grip_cmd, measured) + self.args.grip_step)
            self.grip_cmd = min(self.grip_cmd, target)
            self.clamp.update(self.grip_cmd, measured, frame)
            if measured >= target - 3.0 or self.prim_ticks >= self.args.grip_timeout:
                self.grip_cmd = target
                print(f"[RELEASE] opened to {measured:.1f} (target {target:.1f})", flush=True)
                self.clamp.reset(measured)
                self.next_stage("release")
        return float(self.grip_cmd)


def main() -> None:
    args = _parse_args()
    if not args.task.is_file():
        raise SystemExit(
            f"Task file not found: {args.task}\n"
            "Build one with: python src/tapnetCreate.py --data-dir data "
            "--out tasks/pick_place_task.npz"
        )
    task = load_task_npz(args.task)
    try:
        plan = unpack_plan(task)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if args.no_robot and args.mode in ("ik_print", "robot"):
        raise SystemExit("--no-robot only works with --mode viz or servo_print")
    if args.first_primitive != "task":
        before = [st.primitive for st in plan.stages]
        remap_plan_first_primitive(plan, args.first_primitive)
        after = [st.primitive for st in plan.stages]
        print(
            f"first-primitive={args.first_primitive}: {before} -> {after}",
            flush=True,
        )

    calib = default_wrist_calib(
        fx=CAM_FX, fy=CAM_FY, cx=CAM_CX, cy=CAM_CY,
        dist=CAM_DIST, width=FRAME_WIDTH, height=FRAME_HEIGHT,
    )
    width, height = calib.width, calib.height
    task_w = int(np.asarray(task["camera_width"]).reshape(-1)[0])
    task_h = int(np.asarray(task["camera_height"]).reshape(-1)[0])
    n_track = int(np.asarray(task["query_points_tyx"]).shape[0])
    print(
        f"Task {args.task}: {len(plan.stages)} stage(s), {n_track} tracked points, "
        f"{len(plan.demo_names)} demos, built at {task_w}x{task_h}  mode={args.mode}"
    )
    for s, st in enumerate(plan.stages):
        print(
            f"  stage {s}: {len(st.point_ids)} points, primitive={st.primitive}, "
            f"gripper target={st.gripper_target:.1f}"
        )
    if args.undistort:
        print(
            f"Undistort ON from CAM_*  fx={CAM_FX} fy={CAM_FY} cx={CAM_CX} cy={CAM_CY}  "
            f"dist={CAM_DIST}"
        )
    else:
        print("Undistort OFF (raw frames)")
    if "undistorted" in task:
        task_und = bool(np.asarray(task["undistorted"]).reshape(-1)[0])
        if task_und != args.undistort:
            print(
                f"WARNING: task was built with undistort {'ON' if task_und else 'OFF'} but "
                f"runtime is {'ON' if args.undistort else 'OFF'}; goals will not line up. "
                f"Use --{'' if task_und else 'no-'}undistort or rebuild the task."
            )

    tapir = BootsTAPIR(checkpoint=args.checkpoint)
    features = features_from_numpy(task, tapir.device)
    causal = tapir.initial_causal_state(n_track, features)

    robot = None
    sim = None
    gripper_site_id = gripper_body_id = cam_id = -1
    gripper_pose = None
    precise_sleep = time.sleep

    if not args.no_robot:
        import mujoco
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
        from lerobot.utils.robot_utils import precise_sleep as _precise_sleep

        from Auto.simController import simController
        from utility.lily_relative_helpers import gripper_pose as _gripper_pose

        gripper_pose = _gripper_pose
        precise_sleep = _precise_sleep
        robot = SO100Follower(
            SO100FollowerConfig(
                port=args.port,
                id=args.robot_id,
                cameras={
                    "camera1": OpenCVCameraConfig(
                        index_or_path=args.camera, width=width, height=height, fps=FPS
                    )
                },
                use_degrees=True,
                max_relative_target=MAX_RELATIVE_TARGET,
            )
        )
        robot.connect(calibrate=True)
        print(f"Robot connected on {args.port}.")
        if args.mode in ("ik_print", "robot"):
            sim = simController()
            gripper_site_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
            gripper_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
            cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            if cam_id < 0:
                raise RuntimeError("wrist_cam not in MuJoCo model")
            print("MuJoCo IK ready.")

    cap = _open_webcam(args.camera, width, height) if robot is None else None
    runner = StageRunner(plan, args, live=args.mode == "robot")

    pygame.init()
    preview_w = min(960, width)
    preview_h = max(1, int(round(height * (preview_w / float(max(width, 1))))))
    screen = pygame.display.set_mode((preview_w, preview_h))
    pygame.display.set_caption("tapnetGrab - q quit, n next stage")
    print("Keys (pygame window): q quit, n force next stage.")

    t0 = time.time()
    frame_i = 0
    maps = None
    maps_ready = False
    try:
        if robot is not None:
            ease_to_position(robot, NEUTRAL_POS)
        prev_loop = time.time()
        cmd_pose = None  # integrated EE target (pos, quat wxyz)
        vel_f = None  # smoothed servo velocity
        held_action = None  # last IK solution, resent until the target moves enough
        solved_pos = None
        roll_acc = 0.0
        while True:
            lag_n = 0.0
            loop_start = time.time()
            # the loop runs well below FPS (TAPIR + 1080p capture), so scale
            # velocity by the real period
            step_dt = min(max(loop_start - prev_loop, args.dt), 0.5)
            prev_loop = loop_start
            if robot is not None:
                frame_bgr, obs = _read_bgr_robot(robot)
            else:
                ok, frame_bgr = cap.read()
                if not ok:
                    print("camera read failed")
                    break
                obs = None

            if not maps_ready:
                live_h, live_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
                print(f"Camera native {live_w}x{live_h}; K is for {width}x{height}.")
                if (live_w, live_h) != (task_w, task_h):
                    print(
                        f"WARNING: live {live_w}x{live_h} != task {task_w}x{task_h}; "
                        "demo pixel targets will not line up."
                    )
                if args.undistort:
                    _, maps = calib.maps_for_frame(live_w, live_h)
                maps_ready = True
            if maps is not None:
                frame_bgr = cv2.remap(frame_bgr, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
            rgb = bgr_to_rgb(frame_bgr)

            all_pts, all_vis, causal = tapir.predict(rgb, features, causal)
            in_bounds = (
                (all_pts[:, 0] >= 0) & (all_pts[:, 0] < live_w)
                & (all_pts[:, 1] >= 0) & (all_pts[:, 1] < live_h)
            )
            all_valid = all_vis & in_bounds

            st = runner.current
            ids = st.point_ids
            points = all_pts[ids]
            visible = all_valid[ids]

            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    runner.next_stage("key n")
            if quit_requested:
                print("E-stop / quit")
                break

            target = None
            err_px = float("inf")
            delta = None
            skip_reason = None
            inliers = np.zeros(len(points), dtype=bool)
            if runner.phase == "servo":
                target = choose_target(
                    st, points, visible,
                    lookahead=args.lookahead,
                    end_frac=args.end_frac,
                    min_points=args.min_points,
                )
                if target is None or target.valid.sum() < args.min_points:
                    skip_reason = f"need {args.min_points} matched points"
                else:
                    inliers = select_servo_inliers(
                        points,
                        target.goals,
                        target.valid,
                        keep_frac=args.inlier_frac,
                        min_points=args.min_points,
                        min_cos=args.inlier_min_cos,
                    )
                    if int(inliers.sum()) < args.min_points:
                        skip_reason = (
                            f"need {args.min_points} inliers, have {int(inliers.sum())}"
                        )
                    else:
                        # Servo + converge on the consistent subset only.
                        err_px = mean_pixel_error(points, target.goals, inliers)
                        runner.on_converged_frame(
                            err_px, target.using_mean, target.progress
                        )
                        if runner.phase == "servo":
                            v = solve_robotap_servo(
                                points[inliers],
                                target.goals[inliers],
                                live_w,
                                live_h,
                            )
                            vel = np.array(
                                [
                                    args.servo_sign * args.gain_trans * v[0],
                                    args.servo_sign * args.gain_trans * v[1],
                                    args.servo_sign * args.gain_z * v[2],
                                    args.rot_sign * args.gain_rot * v[3],
                                ]
                            )
                            vel_f = vel if vel_f is None else (
                                args.smooth * vel + (1.0 - args.smooth) * vel_f
                            )
                            if err_px < args.deadband_px:
                                skip_reason = (
                                    f"in deadband ({err_px:.1f}px < {args.deadband_px}px)"
                                )
                                vel_f = None
                            else:
                                delta = clamp_delta(
                                    vel_f * step_dt, args.max_trans, args.max_rot
                                )
            elif runner.phase == "descend":
                if args.mode not in ("ik_print", "robot"):
                    print(
                        f"[TEST DOWN] dry run ({args.test_down_cm:.1f} cm) — "
                        "no IK in this mode",
                        flush=True,
                    )
                    runner.on_descend_done()
                    skip_reason = f"test descend dry-run -> {runner.phase}"
                else:
                    done_cm = (
                        runner.descend_total_m - runner.descend_remaining_m
                    ) * 100.0
                    skip_reason = (
                        f"test descend {done_cm:.1f}/"
                        f"{runner.descend_total_m * 100:.1f} cm"
                    )
            elif runner.phase in ("grasp", "release"):
                skip_reason = f"primitive {runner.phase}"
            elif runner.phase == "freeze":
                skip_reason = "freeze after first stage"
            else:
                skip_reason = "done"

            if target is not None:
                where = "MEAN goal" if target.using_mean else f"demo {target.demo} f{target.frame}"
                prog = f"{target.progress * 100:.0f}%"
            else:
                where, prog = "-", "-"
            clamp_txt = "CLAMPED" if runner.clamp.clamped else "open"
            n_valid = int(target.valid.sum()) if target is not None else int(visible.sum())
            lines = [
                f"stage {runner.stage}/{runner.n_stages - 1} [{runner.phase}] next={st.primitive}",
                f"target {where} progress {prog} inlier err {err_px:.1f}px",
                f"inliers {int(inliers.sum())}/{n_valid} visible  "
                f"gripper {clamp_txt}  {args.mode}",
            ]
            overlay = _draw_overlay(
                frame_bgr,
                points,
                target.goals if target is not None else None,
                target.valid if target is not None else visible,
                lines,
                inliers=inliers,
            )
            screen = _show_bgr_pygame(screen, overlay, "tapnetGrab - q quit, n next stage")

            if args.mode == "servo_print" and delta is not None:
                print(
                    f"s{runner.stage} {where:>12}  err={err_px:6.1f}px  "
                    f"dcam X={delta[0]:+.4f} Y={delta[1]:+.4f} Z={delta[2]:+.4f} "
                    f"Rz={delta[3]:+.4f}  inliers={int(inliers.sum())}/{n_valid}"
                )

            if robot is not None and obs is not None and args.mode == "robot":
                grip_meas = float(obs["gripper.pos"])
                grip_cmd = runner.gripper_command(grip_meas, frame_i)
                if runner.phase == "freeze" and runner.freeze_joints is None:
                    runner.freeze_joints = {k: float(obs[k]) for k in JOINT_POS_KEYS}
                    runner.freeze_joints["gripper.pos"] = float(grip_cmd)
                    print(
                        "[FREEZE] latched joints: "
                        + " ".join(
                            f"{k.split('.')[0]}={runner.freeze_joints[k]:+.1f}"
                            for k in JOINT_POS_KEYS
                        ),
                        flush=True,
                    )
            else:
                grip_cmd = float(obs["gripper.pos"]) if obs is not None else 0.0

            if args.mode in ("ik_print", "robot") and obs is not None:
                assert robot is not None and sim is not None
                action = None
                if runner.phase == "freeze" and runner.freeze_joints is not None:
                    action = dict(runner.freeze_joints)
                    delta = None
                elif runner.phase == "descend":
                    # Straight down in MuJoCo/world frame (+Z up → move -Z).
                    vel_f = None
                    delta = None
                    sim.sync(obs)
                    pose = gripper_pose(sim.data, gripper_site_id, gripper_body_id)
                    if pose is None:
                        skip_reason = "no gripper pose (descend)"
                    elif runner.descend_remaining_m <= 1e-6:
                        runner.on_descend_done()
                    else:
                        ee_pos, ee_quat = pose
                        step = min(float(args.max_trans), runner.descend_remaining_m)
                        if cmd_pose is None:
                            cmd_pose = (
                                np.asarray(ee_pos, float).copy(),
                                np.asarray(ee_quat, float).copy(),
                            )
                        target_pos = np.asarray(cmd_pose[0], float).copy()
                        target_pos[2] -= step
                        target_quat = np.asarray(ee_quat, float)
                        lag = target_pos - ee_pos
                        lag_n = float(np.linalg.norm(lag))
                        if lag_n > args.leash:
                            target_pos = ee_pos + lag * (args.leash / lag_n)
                            step = max(0.0, float(cmd_pose[0][2] - target_pos[2]))
                        cmd_pose = (target_pos, target_quat)
                        runner.descend_remaining_m = max(
                            0.0, runner.descend_remaining_m - step
                        )
                        action = sim.solveIK(target_pos, target_quat)
                        if action is None:
                            action = sim.solveIK(target_pos)
                        if action is None:
                            skip_reason = "IK failed (descend)"
                            cmd_pose = held_action = solved_pos = None
                        else:
                            held_action = dict(action)
                            solved_pos = target_pos.copy()
                            for k in JOINT_POS_KEYS[:5]:
                                cur = float(obs[k])
                                action[k] = float(
                                    np.clip(
                                        action[k],
                                        cur - args.max_joint_step,
                                        cur + args.max_joint_step,
                                    )
                                )
                        if runner.descend_remaining_m <= 1e-6:
                            runner.on_descend_done()
                        if action is not None and args.mode == "ik_print":
                            print(
                                f"[TEST DOWN] z={target_pos[2]:+.3f} "
                                f"left={runner.descend_remaining_m * 100:.1f}cm",
                                flush=True,
                            )
                            action = None
                elif delta is None:
                    cmd_pose = None
                    held_action = solved_pos = None
                    roll_acc = 0.0
                    if runner.phase != "servo":
                        vel_f = None
                else:
                    sim.sync(obs)
                    pose = gripper_pose(sim.data, gripper_site_id, gripper_body_id)
                    if pose is None:
                        skip_reason = "no gripper pose"
                    else:
                        ee_pos, ee_quat = pose
                        # Integrate deltas into a persistent target: per-tick steps
                        # are smaller than solveIK's 4 mm / 0.25 rad tolerance, so
                        # re-solving from the current pose each tick never moves.
                        if cmd_pose is None:
                            cmd_pose = (np.asarray(ee_pos, float).copy(), np.asarray(ee_quat, float).copy())
                        cam_R = np.asarray(sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
                        d_world, d_rz = camera_delta_to_world(
                            _real_to_model_cam(delta, args.mount_roll), sim.data.cam_xpos[cam_id], cam_R
                        )
                        # Roll goes straight to wrist_roll below (IK's 0.25 rad
                        # orientation tolerance swallows small roll steps), so the
                        # IK target keeps the current orientation.
                        target_pos, _ = apply_ee_delta(cmd_pose[0], ee_quat, d_world, 0.0, cam_R)
                        target_quat = np.asarray(ee_quat, float)
                        lag = target_pos - ee_pos
                        lag_n = float(np.linalg.norm(lag))
                        if lag_n > args.leash:
                            target_pos = ee_pos + lag * (args.leash / lag_n)
                        cmd_pose = (target_pos, target_quat)
                        roll_acc += d_rz
                        moved = (
                            np.inf if solved_pos is None
                            else float(np.linalg.norm(target_pos - solved_pos))
                        )
                        roll_deg = float(np.degrees(roll_acc) / WRIST_ROLL_PER_CAM_ROLL)
                        if (
                            held_action is not None
                            and moved < args.ik_min_move
                            and abs(roll_deg) < args.roll_min_deg
                        ):
                            action = dict(held_action)
                        else:
                            action = sim.solveIK(target_pos, target_quat)
                            if action is None:
                                action = sim.solveIK(target_pos)
                            if action is None:
                                skip_reason = "IK failed"
                                cmd_pose = held_action = solved_pos = None
                            else:
                                action["wrist_roll.pos"] = float(obs["wrist_roll.pos"]) + roll_deg
                                roll_acc = 0.0
                                solved_pos = target_pos.copy()
                        if action is not None:
                            # hold the full IK pose; only the sent step is clipped
                            held_action = dict(action)
                            for k in JOINT_POS_KEYS[:5]:
                                cur = float(obs[k])
                                action[k] = float(np.clip(
                                    action[k], cur - args.max_joint_step, cur + args.max_joint_step
                                ))
                        if action is not None and args.mode == "ik_print":
                            joints = " ".join(
                                f"{k.split('.')[0]}={action[k]:+.1f}" for k in JOINT_POS_KEYS if k in action
                            )
                            print(
                                f"target xyz={target_pos[0]:+.3f} {target_pos[1]:+.3f} "
                                f"{target_pos[2]:+.3f}  |  {joints}"
                            )
                            action = None
                if args.mode == "robot":
                    if action is None:
                        # hold the arm, still drive the gripper
                        action = {k: float(obs[k]) for k in JOINT_POS_KEYS}
                    if runner.phase != "freeze":
                        action["gripper.pos"] = grip_cmd
                    robot.send_action(action)
                    if frame_i % 5 == 0 and delta is not None:
                        dj = max(
                            abs(float(action[k]) - float(obs[k])) for k in JOINT_POS_KEYS[:5]
                        )
                        prog = (
                            f"{target.progress * 100:.0f}%/{runner.max_progress * 100:.0f}%"
                            if target is not None
                            else "-"
                        )
                        print(
                            f"SENT s{runner.stage} err={err_px:.1f}px prog={prog} "
                            f"near={runner.converged_count}/{args.converge_frames} "
                            f"dcam=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f},"
                            f"{delta[3]:+.3f}) lag={lag_n * 1000:.0f}mm "
                            f"dj={dj:.1f}deg dt={step_dt:.2f}s grip={grip_cmd:.1f}"
                        )

            if skip_reason and frame_i % 30 == 0:
                print(f"hold: {skip_reason}")

            frame_i += 1
            if frame_i % 90 == 0:
                print(f"~{frame_i / max(time.time() - t0, 1e-6):.1f} fps")
            precise_sleep(max(0.0, args.dt - (time.time() - loop_start)))
    finally:
        if robot is not None:
            go_to_rest(robot)
        
        pygame.quit()
        if cap is not None:
            cap.release()
        if robot is not None and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
