"""Direct-goal IBVS + gripper orientation runtime for SO-101 (wrist camera).

Like tapnetGrabGoal: servo inliers toward each stage's goal_mean via an image
Jacobian. Additionally FK the stage's demo end joint pose (SO-101 base frame)
to a goal gripper quaternion and slerp the EE toward it. A stage converges only
when pixel error AND orientation error are both under threshold.

Rebuild the task with tapnetCreate so ``stage_*_goal_joints`` is stored.

Modes:
  viz         — track + overlay (robot does not move)
  servo_print — also print camera deltas each frame
  ik_print    — solve IK, print joint targets (no send_action)
  robot       — send joint targets and gripper commands; q = e-stop

Keys (pygame window): q quit, n force next stage.

Usage:
  python src/tapnetGrabPose.py --task tasks/pick_place_task_steps.npz --mode robot --undistort
  python src/tapnetGrabPose.py --task tasks/pick_place_task_steps.npz --mode robot --undistort --converge-orient-deg 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utility.robot import NEUTRAL_POS, ease_to_position, go_to_rest

# ---------------------------------------------------------------------------
# Wrist camera intrinsics — keep in sync with tapnetGrab.py / tapnetCreate.py
# ---------------------------------------------------------------------------
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAM_FX = 1262.480288175
CAM_FY = 1264.71879012
CAM_CX = 913.586772795
CAM_CY = 609.63948621
CAM_DIST = (
    3.83407503,
    -2.80771278,
    -0.00124016194,
    0.00296903720,
    -0.755297301,
    4.27744077,
    -1.18548628,
    -2.35292973,
)
CAM_MOUNT_ROLL_DEG = 180.0
UNDISTORT = True

from gripper_clamp import GripperClampDetector
from robotap import (
    joints_to_obs,
    mean_pixel_error,
    remap_plan_first_primitive,
    select_servo_inliers,
    solve_jacobian_servo,
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
from utility.transforms import quat_angular_distance_deg, quat_slerp_wxyz

FPS = 30
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
    p = argparse.ArgumentParser(
        description="GrabGoal IBVS + align gripper quat to demo end pose"
    )
    p.add_argument("--task", type=Path, default=Path("tasks/pick_place_task.npz"))
    p.add_argument(
        "--mode", choices=("viz", "servo_print", "ik_print", "robot"), default="viz"
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--port", type=str, default="COM3")
    p.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")
    p.add_argument("--no-robot", action="store_true")
    p.add_argument(
        "--first-primitive",
        choices=("task", "grasp", "release"),
        default="task",
        help="First gripper action: grasp / release / keep task",
    )
    p.add_argument(
        "--freeze-after-first",
        action="store_true",
        help="After stage 0 finishes, hold pose and skip later stages",
    )
    p.add_argument(
        "--undistort",
        action=argparse.BooleanOptionalAction,
        default=UNDISTORT,
    )
    p.add_argument("--dt", type=float, default=1.0 / FPS)
    p.add_argument(
        "--depth",
        type=float,
        default=0.20,
        help="Assumed point depth Z (m) for the image Jacobian",
    )
    p.add_argument("--gain-trans", type=float, default=0.15)
    p.add_argument("--gain-z", type=float, default=0.10)
    p.add_argument("--gain-rot", type=float, default=1.0)
    p.add_argument("--servo-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    p.add_argument("--rot-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    p.add_argument("--mount-roll", type=float, default=CAM_MOUNT_ROLL_DEG)
    p.add_argument("--smooth", type=float, default=0.3)
    p.add_argument("--deadband-px", type=float, default=8.0)
    p.add_argument("--ik-min-move", type=float, default=0.006)
    p.add_argument("--roll-min-deg", type=float, default=1.5)
    p.add_argument("--max-joint-step", type=float, default=4.0)
    p.add_argument("--max-trans", type=float, default=0.01)
    p.add_argument("--leash", type=float, default=0.03)
    p.add_argument("--max-rot", type=float, default=0.05)
    p.add_argument("--min-points", type=int, default=3)
    p.add_argument(
        "--inlier-frac",
        type=float,
        default=0.45,
        help="Fraction of consistent visible points used for Jacobian / converge "
        "(lowest error among those agreeing with the median pull)",
    )
    p.add_argument(
        "--inlier-min-cos",
        type=float,
        default=0.25,
        help="Min cosine between a point's error vector and the median error; "
        "rejects opposite / irrelevant pulls",
    )
    p.add_argument(
        "--converge-px",
        type=float,
        default=40.0,
        help="Mean inlier pixel-error threshold (with orient) to finish servo",
    )
    p.add_argument(
        "--converge-orient-deg",
        type=float,
        default=15.0,
        help="Max gripper quaternion error vs demo end pose (deg) to finish servo",
    )
    p.add_argument(
        "--orient-after-px",
        type=float,
        default=50.0,
        help="Only slerp toward goal quat once inlier error is at/below this; "
        "while farther, keep current gripper orient (pure IBVS like GrabGoal). "
        "Prevents demo-quat IK from yanking the wrist-cam away from inliers.",
    )
    p.add_argument(
        "--orient-gain",
        type=float,
        default=0.15,
        help="Per-tick slerp fraction toward goal gripper quat (0..1)",
    )
    p.add_argument(
        "--orient-min-deg",
        type=float,
        default=2.0,
        help="Re-solve IK when orientation error exceeds this (deg)",
    )
    p.add_argument(
        "--converge-frames",
        type=int,
        default=1,
        help="Consecutive frames under pixel+orient thresholds before advancing",
    )
    p.add_argument("--grip-step", type=float, default=2.0)
    p.add_argument("--release-open", type=float, default=95.0)
    p.add_argument("--grip-timeout", type=int, default=90)
    return p.parse_args()


def _real_to_model_cam(delta: np.ndarray, roll_deg: float) -> np.ndarray:
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
            color = (0, 255, 0)  # used for servo
        elif valid[i]:
            color = (0, 140, 255)  # visible but rejected as outlier
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
        keys = [
            k for k, v in obs.items()
            if hasattr(v, "shape") and len(getattr(v, "shape", ())) == 3
        ]
        if not keys:
            raise RuntimeError("no camera image in observation")
        img = obs[keys[0]]
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), obs


class StageRunner:
    """servo → primitive → next stage. Converge = pixels + gripper orient."""

    def __init__(self, plan, args, live: bool):
        self.plan = plan
        self.args = args
        self.live = live
        self.stage = 0
        self.phase = "servo"  # servo | grasp | release | freeze | done
        self.converged_count = 0
        self.prim_ticks = 0
        self.grip_cmd: float | None = None
        self.freeze_joints: dict[str, float] | None = None
        self.clamp = GripperClampDetector(verbose=True, label="[grabPose]")
        self._announce()

    @property
    def n_stages(self) -> int:
        return len(self.plan.stages)

    @property
    def current(self):
        return self.plan.stages[self.stage]

    def _announce(self) -> None:
        st = self.current
        extra = (
            "  (--freeze-after-first: will hold after this stage)"
            if self.args.freeze_after_first and self.stage == 0
            else ""
        )
        has_j = st.goal_joints is not None
        print(
            f"[STAGE {self.stage}/{self.n_stages - 1}] IBVS+orient to goal_mean "
            f"({len(st.point_ids)} pts"
            f"{', goal_joints OK' if has_j else ', NO goal_joints — rebuild task'}"
            f"), then '{st.primitive}'{extra}",
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
            return
        if self.stage + 1 >= self.n_stages:
            if self.phase != "done":
                print(f"[DONE] task complete ({why})", flush=True)
            self.phase = "done"
            return
        self.stage += 1
        self.phase = "servo"
        self.converged_count = 0
        print(f"[ADVANCE] -> stage {self.stage} ({why})", flush=True)
        self._announce()

    def on_converged_frame(self, err_px: float, orient_deg: float) -> bool:
        """Advance when pixel AND orientation stay under thresholds for N frames."""
        thr_px = float(self.args.converge_px)
        thr_o = float(self.args.converge_orient_deg)
        px_ok = err_px <= thr_px
        o_ok = orient_deg <= thr_o
        if px_ok and o_ok:
            self.converged_count += 1
        elif (
            self.converged_count > 0
            and err_px <= 1.25 * thr_px
            and orient_deg <= 1.25 * thr_o
        ):
            pass
        else:
            self.converged_count = 0
        need = max(1, int(self.args.converge_frames))
        if self.converged_count > 0 and self.converged_count < need:
            print(
                f"[THRESHOLD] stage {self.stage} "
                f"{self.converged_count}/{need}  err={err_px:.1f}px "
                f"orient={orient_deg:.1f}deg "
                f"(need <= {thr_px:.1f}px and <= {thr_o:.1f}deg)",
                flush=True,
            )
        if self.converged_count < need:
            return False
        prim = self.current.primitive
        print(
            f"[CONVERGED] stage {self.stage} "
            f"(err {err_px:.1f}px, orient {orient_deg:.1f}deg) -> next step '{prim}'",
            flush=True,
        )
        if prim == "none":
            self.next_stage("converged")
        elif not self.live:
            print(f"[{prim.upper()}] dry run (no gripper command in this mode)", flush=True)
            self.next_stage(f"dry-run {prim}")
        else:
            self.phase = prim
            self.prim_ticks = 0
            self.grip_cmd = None
        return True

    def gripper_command(self, measured: float, frame: int) -> float:
        if self.phase == "freeze":
            if self.grip_cmd is None:
                self.grip_cmd = measured
            return float(self.grip_cmd)
        if self.grip_cmd is None:
            self.grip_cmd = measured
            self.clamp.reset(measured)
        if self.phase == "grasp":
            self.prim_ticks += 1
            self.grip_cmd = max(0.0, self.grip_cmd - self.args.grip_step)
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
                    f"[GRASP] WARNING: gripper closed to {measured:.1f} without clamp; "
                    "advancing anyway",
                    flush=True,
                )
                self.next_stage("grasp timeout")
        elif self.phase == "release":
            self.prim_ticks += 1
            target = max(float(self.current.gripper_target), float(self.args.release_open))
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
    missing_j = [i for i, st in enumerate(plan.stages) if st.goal_joints is None]
    if missing_j:
        raise SystemExit(
            f"Task missing stage_*_goal_joints for stage(s) {missing_j}. "
            "Rebuild with: python src/tapnetCreate.py --data-dir data "
            "--out tasks/pick_place_task_steps.npz"
        )
    if args.no_robot and args.mode in ("ik_print", "robot"):
        raise SystemExit("--no-robot only works with --mode viz or servo_print")
    if args.first_primitive != "task":
        before = [st.primitive for st in plan.stages]
        remap_plan_first_primitive(plan, args.first_primitive)
        after = [st.primitive for st in plan.stages]
        print(f"first-primitive={args.first_primitive}: {before} -> {after}", flush=True)

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
        f"{len(plan.demo_names)} demos, built at {task_w}x{task_h}  mode={args.mode}  "
        f"control=IBVS+orient (Z={args.depth:.2f}m, "
        f"converge <= {args.converge_px:.0f}px & {args.converge_orient_deg:.0f}deg)"
    )
    for s, st in enumerate(plan.stages):
        print(
            f"  stage {s}: {len(st.point_ids)} points, primitive={st.primitive}, "
            f"gripper target={st.gripper_target:.1f}, "
            f"goal_joints={np.array2string(st.goal_joints, precision=1)}"
        )
    if args.undistort:
        print(
            f"Undistort ON from CAM_*  fx={CAM_FX} fy={CAM_FY} cx={CAM_CX} cy={CAM_CY}"
        )
    else:
        print("Undistort OFF (raw frames)")
    if "undistorted" in task:
        task_und = bool(np.asarray(task["undistorted"]).reshape(-1)[0])
        if task_und != args.undistort:
            print(
                f"WARNING: task was built with undistort {'ON' if task_und else 'OFF'} but "
                f"runtime is {'ON' if args.undistort else 'OFF'}; goals will not line up."
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

    # FK demo end joints → goal gripper quat in SO-101 / MuJoCo world (base) frame.
    goal_quats: list[np.ndarray | None] = [None] * len(plan.stages)
    if sim is not None and gripper_pose is not None:
        for s, st in enumerate(plan.stages):
            assert st.goal_joints is not None
            sim.sync(joints_to_obs(st.goal_joints))
            gp = gripper_pose(sim.data, gripper_site_id, gripper_body_id)
            if gp is None:
                raise RuntimeError(f"FK failed for stage {s} goal_joints")
            goal_quats[s] = np.asarray(gp[1], dtype=float).copy()
            print(
                f"  stage {s} goal_quat (wxyz)="
                f"{np.array2string(goal_quats[s], precision=3)}",
                flush=True,
            )

    cap = _open_webcam(args.camera, width, height) if robot is None else None
    runner = StageRunner(plan, args, live=args.mode == "robot")

    pygame.init()
    preview_w = min(960, width)
    preview_h = max(1, int(round(height * (preview_w / float(max(width, 1))))))
    screen = pygame.display.set_mode((preview_w, preview_h))
    pygame.display.set_caption("tapnetGrabPose - q quit, n next stage")
    print("Keys (pygame window): q quit, n force next stage.")

    t0 = time.time()
    frame_i = 0
    maps = None
    maps_ready = False
    try:
        if robot is not None:
            ease_to_position(robot, NEUTRAL_POS)
        prev_loop = time.time()
        cmd_pose = None
        vel_f = None
        held_action = None
        solved_pos = None
        roll_acc = 0.0
        while True:
            lag_n = 0.0
            need_orient = False
            loop_start = time.time()
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
                        "goals will not line up."
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
            goals = st.goal_mean
            valid = visible & st.goal_valid

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

            err_px = float("inf")
            orient_deg = float("inf")
            goal_q = goal_quats[runner.stage]
            ee_quat_now: np.ndarray | None = None
            if (
                sim is not None
                and obs is not None
                and gripper_pose is not None
                and goal_q is not None
            ):
                sim.sync(obs)
                pose_now = gripper_pose(sim.data, gripper_site_id, gripper_body_id)
                if pose_now is not None:
                    ee_quat_now = np.asarray(pose_now[1], dtype=float)
                    orient_deg = quat_angular_distance_deg(ee_quat_now, goal_q)
            elif goal_q is None:
                # viz / no MuJoCo: don't block on orientation
                orient_deg = 0.0

            delta = None
            skip_reason = None
            inliers = np.zeros(len(points), dtype=bool)
            if runner.phase == "servo":
                n_valid = int(valid.sum())
                if n_valid < args.min_points:
                    skip_reason = f"need {args.min_points} visible goal points, have {n_valid}"
                else:
                    inliers = select_servo_inliers(
                        points,
                        goals,
                        valid,
                        keep_frac=args.inlier_frac,
                        min_points=args.min_points,
                        min_cos=args.inlier_min_cos,
                    )
                    err_px = mean_pixel_error(points, goals, inliers)
                    # Orient only after pixels are close — otherwise quat IK
                    # rotates the wrist-cam and drives inliers away from goals.
                    allow_orient = err_px <= float(args.orient_after_px)
                    need_orient = (
                        orient_deg > float(args.converge_orient_deg) and allow_orient
                    )
                    advanced = runner.on_converged_frame(err_px, orient_deg)
                    if advanced or runner.phase != "servo":
                        vel_f = None
                        skip_reason = f"threshold met -> {runner.phase}"
                    else:
                        v = solve_jacobian_servo(
                            points[inliers],
                            goals[inliers],
                            live_w,
                            live_h,
                            z=args.depth,
                            robust_iters=2,
                        )
                        vel = np.array(
                            [
                                args.servo_sign * args.gain_trans * v[0],
                                args.servo_sign * args.gain_trans * v[1],
                                args.servo_sign * args.gain_z * v[2],
                                args.rot_sign * args.gain_rot * v[3],
                            ]
                        )
                        vel_f = (
                            vel if vel_f is None
                            else args.smooth * vel + (1.0 - args.smooth) * vel_f
                        )
                        if err_px < args.deadband_px:
                            vel_f = None
                            if not need_orient:
                                skip_reason = (
                                    f"in deadband ({err_px:.1f}px) and "
                                    f"orient ok ({orient_deg:.1f}deg)"
                                )
                            else:
                                skip_reason = (
                                    f"pixels ok; aligning orient "
                                    f"{orient_deg:.1f}->{args.converge_orient_deg:.0f}deg"
                                )
                        else:
                            delta = clamp_delta(
                                vel_f * step_dt, args.max_trans, args.max_rot
                            )
            elif runner.phase in ("grasp", "release"):
                skip_reason = f"primitive {runner.phase}"
            elif runner.phase == "freeze":
                skip_reason = "freeze after first stage"
            else:
                skip_reason = "done"

            clamp_txt = "CLAMPED" if runner.clamp.clamped else "open"
            lines = [
                f"stage {runner.stage}/{runner.n_stages - 1} [{runner.phase}] "
                f"next={st.primitive}  IBVS+R",
                f"inlier err {err_px:.1f}px  orient {orient_deg:.1f}deg  "
                f"near {runner.converged_count}/{args.converge_frames}",
                f"inliers {int(inliers.sum())}/{int(valid.sum())} visible  "
                f"gripper {clamp_txt}  {args.mode}",
            ]
            overlay = _draw_overlay(
                frame_bgr, points, goals, valid, lines, inliers=inliers
            )
            screen = _show_bgr_pygame(
                screen, overlay, "tapnetGrabPose - q quit, n next stage"
            )

            if args.mode == "servo_print" and delta is not None:
                print(
                    f"s{runner.stage} goal_mean  err={err_px:6.1f}px  "
                    f"dcam X={delta[0]:+.4f} Y={delta[1]:+.4f} Z={delta[2]:+.4f} "
                    f"Rz={delta[3]:+.4f}  valid={int(valid.sum())}"
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
                want_servo = (
                    runner.phase == "servo"
                    and (delta is not None or need_orient)
                    and goal_q is not None
                )
                if runner.phase == "freeze" and runner.freeze_joints is not None:
                    action = dict(runner.freeze_joints)
                    delta = None
                elif not want_servo:
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
                        if cmd_pose is None:
                            cmd_pose = (
                                np.asarray(ee_pos, float).copy(),
                                np.asarray(ee_quat, float).copy(),
                            )
                        cam_R = np.asarray(
                            sim.data.cam_xmat[cam_id], dtype=float
                        ).reshape(3, 3)
                        if delta is not None:
                            d_world, d_rz = camera_delta_to_world(
                                _real_to_model_cam(delta, args.mount_roll),
                                sim.data.cam_xpos[cam_id],
                                cam_R,
                            )
                            target_pos, _ = apply_ee_delta(
                                cmd_pose[0], ee_quat, d_world, 0.0, cam_R
                            )
                            roll_acc += d_rz
                        else:
                            # Hold XY/Z from last cmd; only rotate toward goal quat.
                            target_pos = np.asarray(cmd_pose[0], float).copy()
                            d_rz = 0.0
                        # Far from goals: keep current quat (GrabGoal). Near: slerp.
                        do_orient = (
                            err_px <= float(args.orient_after_px) and goal_q is not None
                        )
                        if do_orient:
                            alpha = float(np.clip(args.orient_gain, 0.0, 1.0))
                            target_quat = quat_slerp_wxyz(ee_quat, goal_q, alpha)
                        else:
                            target_quat = np.asarray(ee_quat, float)
                        lag = target_pos - ee_pos
                        lag_n = float(np.linalg.norm(lag))
                        if lag_n > args.leash:
                            target_pos = ee_pos + lag * (args.leash / lag_n)
                        cmd_pose = (target_pos, target_quat)
                        moved = (
                            np.inf if solved_pos is None
                            else float(np.linalg.norm(target_pos - solved_pos))
                        )
                        roll_deg = float(np.degrees(roll_acc) / WRIST_ROLL_PER_CAM_ROLL)
                        orient_hold = (not do_orient) or (
                            orient_deg < args.orient_min_deg
                        )
                        if (
                            held_action is not None
                            and moved < args.ik_min_move
                            and orient_hold
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
                                if do_orient:
                                    if abs(roll_deg) >= args.roll_min_deg:
                                        action["wrist_roll.pos"] = (
                                            float(
                                                action.get(
                                                    "wrist_roll.pos",
                                                    obs["wrist_roll.pos"],
                                                )
                                            )
                                            + roll_deg
                                        )
                                else:
                                    # Pure IBVS like GrabGoal: image-roll via wrist_roll.
                                    action["wrist_roll.pos"] = (
                                        float(obs["wrist_roll.pos"]) + roll_deg
                                    )
                                roll_acc = 0.0
                                solved_pos = target_pos.copy()
                        if action is not None:
                            held_action = dict(action)
                            for k in JOINT_POS_KEYS[:5]:
                                cur = float(obs[k])
                                action[k] = float(
                                    np.clip(
                                        action[k],
                                        cur - args.max_joint_step,
                                        cur + args.max_joint_step,
                                    )
                                )
                        if action is not None and args.mode == "ik_print":
                            joints = " ".join(
                                f"{k.split('.')[0]}={action[k]:+.1f}"
                                for k in JOINT_POS_KEYS
                                if k in action
                            )
                            print(
                                f"target xyz={target_pos[0]:+.3f} {target_pos[1]:+.3f} "
                                f"{target_pos[2]:+.3f}  orient={orient_deg:.1f}deg  "
                                f"|  {joints}"
                            )
                            action = None
                if args.mode == "robot":
                    if action is None:
                        action = {k: float(obs[k]) for k in JOINT_POS_KEYS}
                    if runner.phase != "freeze":
                        action["gripper.pos"] = grip_cmd
                    robot.send_action(action)
                    if frame_i % 5 == 0 and (delta is not None or need_orient):
                        dj = max(
                            abs(float(action[k]) - float(obs[k]))
                            for k in JOINT_POS_KEYS[:5]
                        )
                        dcam = (
                            f"({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f},"
                            f"{delta[3]:+.3f})"
                            if delta is not None
                            else "(hold)"
                        )
                        print(
                            f"SENT s{runner.stage} err={err_px:.1f}px "
                            f"orient={orient_deg:.1f}deg "
                            f"near={runner.converged_count}/{args.converge_frames} "
                            f"dcam={dcam} lag={lag_n * 1000:.0f}mm "
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
            try:
                go_to_rest(robot)
            except Exception as exc:  # noqa: BLE001
                print(f"Rest pose failed: {exc}")
        pygame.quit()
        if cap is not None:
            cap.release()
        if robot is not None and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
