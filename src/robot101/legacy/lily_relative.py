


"""Xbox-teleop the SO101, teach a gripper pose relative to a tracked block.

Same AprilTag + teleop loop as test_lily_external.py, plus a relative-pose
teach/execute path:

    1. Press 1 to track the block, then 1 again to HOLD the pose.
    2. Teleop the gripper to the pose you want relative to the block.
    3. Press G. Uses the frozen (or last tracked) block pose; the camera
       does not need to see the block. Saves T_block_gripper and the
       gripper clamp (0 closed … 100 open).
    4. Move the physical block. Press 1 to track, then HOLD.
    5. Press E. Reconstructs T_world_goal_gripper = T_world_block @
       T_block_gripper, then solveIK(goal_pos, goal_quat) and applies
       the recorded clamp.
    5. Prints block pose, target gripper pose, and the IK solution.
       The robot only moves IRL if you type y at the prompt.

    uv run python -m robot101.legacy.lily_relative
    uv run python -m robot101.legacy.lily_relative --objects cube --camera 1
    uv run python -m robot101.legacy.lily_relative --displacement 0.50 0 0.40 --rpy 180 0 0

Press q to quit. Press F to follow. Press P for gripper pose from the origin.
Press G to record the current gripper pose relative to the block.
Press E to reconstruct that relative pose on the current block and (after
[y/N]) send the SO101 through IK, then optionally clamp the jaw shut.
Press 1 to toggle object tracking (off until you turn it on).
Press B to lock/unlock object orientation (tags keep driving XYZ).
Nudge the camera pose live from the MuJoCo / pygame / OpenCV windows:
  i/k x  j/l y  u/o z   t/; roll  y/h pitch  n/m yaw
  [ ] slower/faster   shift 5x   c print CLI   0 reset pose
"""

from __future__ import annotations

'''


ok so it seems that we just need to make a utitliy py file for all the utitlies 

we also should make a class for ui/update our ui class to handle overlay text in a more abstracted way 

uhh we should perhaps make a more abstracted functions that handle different sectors such as a function
to handle capturing a pose, a function that handles moving the robot to a position, so.. basically more high level ones 

this way we can make it more like scripting rather than something very hard to repeat over and over again...

i dont wanna do sum xml bull shi tho, lets just keep it as functions so it simpler and easier to trace n stuff 


'''

import os

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import argparse
import sys
import time
from pathlib import Path

import cv2
import glfw
import mujoco
import numpy as np
import pygame
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.robot_utils import precise_sleep


from robot101.legacy.simulation.aprilRosTracker import AprilTagRosTracker
from robot101.legacy.simulation.joints import real_to_sim
from robot101.legacy.simulation.lilyObject import LilyObjectManager
from robot101.legacy.simulation.simController import simController
from robot101.legacy.simulation.viewer import MainThreadViewer
from robot101.robot.controllers import ControllerType, make_controller
from robot101.robot.common import (
    FPS,
    MAX_RELATIVE_TARGET,
    REST_POSE,
    ease_to_position,
    go_to_rest,
)
from robot101.legacy.simulation.lily_camera_ui import (
    cam_hold_nudge as _cam_hold_nudge,
    cam_tap_nudge as _cam_tap_nudge,
    drain_ui_events as _drain_ui_events,
    format_cam_cli as _format_cam_cli,
    mocap_id as _mocap_id,
)
from robot101.legacy.simulation.lily_relative_helpers import (
    clamp_gripper_hard as _clamp_gripper_hard,
    fmt_action as _fmt_action,
    fmt_clamp as _fmt_clamp,
    fmt_quat as _fmt_quat,
    fmt_xyz as _fmt_xyz,
    gripper_pose as _gripper_pose,
    obs_gripper_clamp as _obs_gripper_clamp,
)
from robot101.legacy.simulation.lily_shared import parse_cam_quat, select_objects as _select_objects
from robot101.calibration.transforms import (
    T_from_pose as _pose_T,
    T_relative as _relative_T,
    T_to_pose as _T_to_pose,
    camera_matrix as _camera_matrix,
    intrinsics_from_K as _params_from_K,
    quat_angular_distance_deg as _quat_ang_deg,
    rpy_deg_to_quat_wxyz as _rpy_deg_to_quat_wxyz,
)


'''
This is obvious but we need our real life situation to be perfect 

'''



'''

CALIBRATION VALUES

'''
TEST_POSE = {
    "shoulder_pan.pos": -15.00,
    "shoulder_lift.pos": -90.00,
    "elbow_flex.pos": 90.00,
    "wrist_flex.pos": 0.00,
    "wrist_roll.pos": -0.00,
    "gripper.pos": 90.00,
}


# From tags 7/8/9: geometry was correct, world Z was +13.88 cm high.
CAM_XYZ = (0.28105 + 0.0012, 0.0237,0.3901 + 0.0004) 
CAM_RPY_DEG = (180.0, 0.0, 90.0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Calibrated detector K for FRAME_WIDTH x FRAME_HEIGHT. Do not resize frames
# to another size, and do not replace this with getOptimalNewCameraMatrix.
CAM_FX = 822.317
CAM_FY = 822.317
CAM_CX = 319.495
CAM_CY = 242.502
CAM_DIST = (
    -0.0449369,
    1.17277,
    0,
    0,
    -3.63244,
    0,
    0,
    0,
)

UNDISTORT = True
LOCK_ORI = False
OBJ_RPY_DEG = (90.0, 0.0, 0.0)
CAM_NUDGE_MPS = 0.05
CAM_NUDGE_DPS = 10.0
CAM_NUDGE_TAP_M = 0.001
CAM_NUDGE_TAP_DEG = 1.0


def _parse_cam_quat(rpy, quat) -> np.ndarray:
    return parse_cam_quat(rpy, quat, CAM_RPY_DEG)


#-------------------------
#    MAIN LOOPS 
#-------------------------



def main() -> None:


    #------ ARGS + VARIABLES ----------


    parser = argparse.ArgumentParser(
        description="Xbox teleop + teach gripper pose relative to an AprilTag block"
    )
    parser.add_argument(
        "--objects",
        default="",
        help="Comma-separated Lily object names (default: all)",
    )
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=0.024)
    parser.add_argument("--fx", type=float, default=CAM_FX)
    parser.add_argument("--fy", type=float, default=CAM_FY)
    parser.add_argument("--cx", type=float, default=CAM_CX)
    parser.add_argument("--cy", type=float, default=CAM_CY)
    parser.add_argument(
        "--undistort",
        action=argparse.BooleanOptionalAction,
        default=UNDISTORT,
        help="Undistort each frame with OpenCV before AprilTag tracking "
        "(uses CAM_DIST / --dist). Keeps fx/fy/cx/cy; does not compute a new_K.",
    )
    parser.add_argument(
        "--dist",
        nargs="+",
        type=float,
        default=None,
        metavar="K",
        help="OpenCV distortion coefficients (default: CAM_DIST at top of file).",
    )
    parser.add_argument(
        "--displacement",
        nargs=3,
        type=float,
        default=list(CAM_XYZ),
        metavar=("X", "Y", "Z"),
        help="External camera XYZ in meters from the SO101 / world origin "
        "(default: 0.50 0 0.40).",
    )
    parser.add_argument(
        "--rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Camera orientation in degrees (ZYX yaw-pitch-roll) from the "
        "SO101 / world origin. Default: 180 0 0 (facing straight down).",
    )
    parser.add_argument(
        "--quat",
        nargs=4,
        type=float,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Camera orientation as a MuJoCo wxyz quaternion (instead of --rpy).",
    )
    parser.add_argument(
        "--lock-ori",
        action=argparse.BooleanOptionalAction,
        default=LOCK_ORI,
        help="Use AprilTags for object XYZ only; keep a hardcoded world "
        "orientation (default: off). Toggle live with B.",
    )
    parser.add_argument(
        "--obj-rpy",
        nargs=3,
        type=float,
        default=list(OBJ_RPY_DEG),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Hardcoded object world RPY in degrees when --lock-ori is on "
        f"(default: {OBJ_RPY_DEG[0]:g} {OBJ_RPY_DEG[1]:g} {OBJ_RPY_DEG[2]:g}).",
    )
    args = parser.parse_args()

    cam_xyz = np.array([float(v) for v in args.displacement], dtype=float)
    cam_rpy = (
        None
        if args.quat is not None
        else np.array(
            [float(v) for v in (args.rpy or CAM_RPY_DEG)], dtype=float
        )
    )
    cam_quat = _parse_cam_quat(args.rpy, args.quat)
    start_xyz = cam_xyz.copy()
    start_rpy = None if cam_rpy is None else cam_rpy.copy()
    start_quat = cam_quat.copy()
    T_world_cam = _pose_T(cam_xyz, cam_quat)
    nudge_scale = 1.0
    dist_coeffs = np.asarray(
        args.dist if args.dist is not None else CAM_DIST,
        dtype=np.float64,
    ).reshape(-1, 1)
    K_distorted = _camera_matrix(
        float(args.fx), float(args.fy), float(args.cx), float(args.cy)
    )
    want_undistort = (
        bool(args.undistort) and float(np.linalg.norm(dist_coeffs)) > 1e-12
    )
    lock_ori = bool(args.lock_ori)
    obj_rpy = np.array([float(v) for v in args.obj_rpy], dtype=float)
    obj_lock_quat = _rpy_deg_to_quat_wxyz(*obj_rpy)

    names = [n.strip() for n in args.objects.split(",") if n.strip()] or None
    mgr = LilyObjectManager()
    objects = _select_objects(mgr, names)

    sim = simController(objects=objects)
    model = sim.model
    data = sim.data
    floor_origin_id = _mocap_id(model, "floor_origin")
    floor_cam_id = _mocap_id(model, "floor_tag_11")

    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5B610348321",
        id="my_awesome_follower_arm",
        cameras={},
        use_degrees=True,
        max_relative_target=MAX_RELATIVE_TARGET,
    )
    robot = SO100Follower(robot_config)
    controller = make_controller(ControllerType.XBOX, robot)

    tracker = AprilTagRosTracker(
        family=args.family,
        tag_size=args.tag_size,
        camera_index=args.camera,
        camera_params=_params_from_K(K_distorted),
    )
    tracker.obj_tags(objects)

    cap = cv2.VideoCapture(tracker.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Could not open camera {tracker.camera_index}")
        sys.exit(1)

    probe = None
    for _ in range(8):
        ok, probe = cap.read()
        if ok and probe is not None and probe.size:
            break
    if probe is None or not probe.size:
        print("Could not read a camera frame to check resolution.")
        sys.exit(1)
    native_h, native_w = int(probe.shape[0]), int(probe.shape[1])
    native_size = (native_w, native_h)
    calib_size = (int(FRAME_WIDTH), int(FRAME_HEIGHT))
    print(
        f"Camera native frame.shape={probe.shape} "
        f"({native_w}x{native_h}). "
        f"Requested {args.width}x{args.height}. "
        f"K is for {FRAME_WIDTH}x{FRAME_HEIGHT}."
    )
    k_for_frame = True
    if native_size != calib_size:
        sx = native_w / float(FRAME_WIDTH)
        sy = native_h / float(FRAME_HEIGHT)
        if abs(sx - sy) < 0.02:
            print(
                f"Same aspect as calibration; scaling K by "
                f"sx={sx:.4f} sy={sy:.4f}. Not stretching the image."
            )
            K_distorted = K_distorted.copy()
            K_distorted[0, 0] *= sx
            K_distorted[0, 2] *= sx
            K_distorted[1, 1] *= sy
            K_distorted[1, 2] *= sy
        else:
            k_for_frame = False
            print(
                f"WARNING: camera is {native_w}x{native_h} "
                f"({native_w / max(native_h, 1):.3f}:1), calibration is "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT} "
                f"({FRAME_WIDTH / max(FRAME_HEIGHT, 1):.3f}:1). "
                "Not resizing (that stretches pixels and invalidates K). "
                "Set the camera to the calibration resolution."
            )
    frame_size = native_size
    undistort = want_undistort
    if undistort and not k_for_frame:
        print(
            "WARNING: undistort skipped; native size does not match K."
        )
        undistort = False

    undistort_maps: tuple[np.ndarray, np.ndarray] | None = None
    if undistort:
        # Remap with P=K so fx/fy/cx/cy stay the calibrated values.
        undistort_maps = cv2.initUndistortRectifyMap(
            K_distorted,
            dist_coeffs,
            None,
            K_distorted,
            frame_size,
            cv2.CV_16SC2,
        )
        track_K = K_distorted
        axes_dist = np.zeros((5, 1), dtype=np.float64)
        print(
            "Undistort ON: remap keeps the calibrated K "
            "(no getOptimalNewCameraMatrix / new_K)."
        )
    else:
        track_K = K_distorted
        axes_dist = dist_coeffs if k_for_frame else np.zeros((5, 1), dtype=np.float64)
        print("Undistort off: tracker uses the calibrated K on native frames.")

    cam_k_params = _params_from_K(track_K)
    tracker.camera_params = cam_k_params
    cam_k = track_K

    for obj in objects:
        ids = sorted(obj.get("tags") or {})
        print(f"{obj['name']} LilyTag IDs: {ids or 'none'}")
    print(
        f"SO101 is origin. Camera marker at "
        f"{cam_xyz[0]:+.3f} {cam_xyz[1]:+.3f} {cam_xyz[2]:+.3f} m "
        f"quat {cam_quat[0]:+.3f} {cam_quat[1]:+.3f} "
        f"{cam_quat[2]:+.3f} {cam_quat[3]:+.3f}. "
        "T_world_obj = T_world_cam @ T_cam_obj."
    )
    print(
        f"External camera {tracker.camera_index}. "
        f"tracker K "
        f"fx={cam_k_params[0]:.1f} fy={cam_k_params[1]:.1f} "
        f"cx={cam_k_params[2]:.1f} cy={cam_k_params[3]:.1f}. "
        f"undistort={'ON' if undistort else 'off'}. "
        "apriltag_ros bundle tracking. Xbox teleop. "
        "Press q quit, F follow, P gripper pose, G record block→gripper, "
        "E reconstruct + IK (y/N IRL), "
        "1 toggle object tracking (starts off), B lock object orientation."
    )
    print(
        f"Object ori lock={'ON' if lock_ori else 'off'} "
        f"rpy {obj_rpy[0]:g} {obj_rpy[1]:g} {obj_rpy[2]:g} "
        "(tags still drive XYZ)."
    )
    print(
        "Nudge camera (focus MuJoCo, pygame, or OpenCV): "
        "i/k x, j/l y, u/o z, arrows/pgup/pgdn; "
        "t/; roll, y/h pitch, n/m yaw; "
        "[ ] speed, shift 5x, c print, 0 reset."
    )

    pygame.init()
    pygame.display.set_mode((320, 80))
    pygame.display.set_caption("teleop — G record  E execute  i/k x  j/l y  u/o z")

    pending: list[str] = []

    def on_key(keycode: int) -> None:
        if keycode == glfw.KEY_F:
            pending.append("follow")
        if keycode == glfw.KEY_P:
            pending.append("print-gripper")
        if keycode == glfw.KEY_C:
            pending.append("print-cam")
        if keycode == glfw.KEY_0:
            pending.append("reset-cam")
        if keycode == glfw.KEY_LEFT_BRACKET:
            pending.append("nudge-slower")
        if keycode == glfw.KEY_RIGHT_BRACKET:
            pending.append("nudge-faster")
        if keycode == glfw.KEY_E:
            pending.append("go-to-relative")
        if keycode == glfw.KEY_G:
            pending.append("record-goal")
        if keycode == glfw.KEY_B:
            pending.append("lock-ori")
        if keycode in (glfw.KEY_1, glfw.KEY_KP_1):
            pending.append("toggle-track")

    viewer = MainThreadViewer(model, data, key_callback=on_key)
    viewer.cam.lookat[:] = model.stat.center
    viewer.cam.distance = max(0.6, 1.5 * float(model.stat.extent))
    viewer.cam.azimuth = 160
    viewer.cam.elevation = -20

    main_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, objects[0]["name"]
    )
    gripper_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
    )
    gripper_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
    )
    follow_cam = False
    track_objects = False
    connected = False
    previous_quat: dict[str, np.ndarray] = {}
    gripper_origin_text: str | None = None
    last_world_pose: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    GOAL_T_BLOCK_GRIPPER: np.ndarray | None = None
    GOAL_GRIPPER_CLAMP: float | None = None
    cube_name = next(
        (obj["name"] for obj in objects if obj["name"] == "cube"),
        objects[0]["name"],
    )

    try:



        #------ CONNECTIONS ----
        controller.connect() # LEADER 
        robot.connect(calibrate=True) #FOLLOWER
        connected = True
        time.sleep(0.3)

        #------ intilaize both for position ... -------
        ease_to_position(robot, TEST_POSE) 
        obs = robot.get_observation()
        controller.sync_from_observation(obs)
        sim.sync(obs)

        
        cv2.namedWindow("LilyTag camera (apriltag_ros)", cv2.WINDOW_NORMAL)
        pygame.event.pump()
        prev_time = time.perf_counter()

        while viewer.is_running():
            
           
            loop_start = time.perf_counter() # for dt 
            pygame.event.pump() # pump 

            obs = robot.get_observation()
            current_time = time.perf_counter()
            dt = current_time - prev_time


             #---- update state  ----
            action = controller.get_action(dt, obs)
            robot.send_action(action)
            sim.sync(obs)


            pygame_keys = pygame.key.get_pressed()
            boost = (
                5.0
                if (
                    pygame_keys[pygame.K_LSHIFT]
                    or pygame_keys[pygame.K_RSHIFT]
                    or glfw.get_key(viewer.window, glfw.KEY_LEFT_SHIFT)
                    == glfw.PRESS
                    or glfw.get_key(viewer.window, glfw.KEY_RIGHT_SHIFT)
                    == glfw.PRESS
                )
                else 1.0
            )



            #----- CAMERA POSITION UPDATE -------
            dxyz, drpy = _cam_hold_nudge(
                viewer.window,
                pygame_keys,
                dt,
                nudge_scale * boost,
                nudge_mps=CAM_NUDGE_MPS,
                nudge_dps=CAM_NUDGE_DPS,
            )
            cam_xyz += dxyz
            if cam_rpy is not None:
                cam_rpy += drpy
                cam_quat = _rpy_deg_to_quat_wxyz(*cam_rpy)
            T_world_cam = _pose_T(cam_xyz, cam_quat)

            ok, frame = cap.read()
            if not ok:
                print("Camera frame failed.")
                break

            

            #----- HANDLE APRIL TAGS OBJECTS _-----

            # distort camera frame 
            if undistort_maps is not None:
                frame = cv2.remap(
                    frame,
                    undistort_maps[0],
                    undistort_maps[1],
                    interpolation=cv2.INTER_LINEAR,
                )

            # detect poses of objects 
            tracker.camera_params = list(cam_k_params)
            detections = tracker.detect(frame)
            localized = tracker.poses_from_bundles(detections)




            #------ UI LINJES ----------
            overlay_lines = [
                f"seen: {sorted(detections) or 'none'}",
                f"cam {cam_xyz[0]:+.3f} {cam_xyz[1]:+.3f} {cam_xyz[2]:+.3f}m"
                + (
                    f"  rpy {cam_rpy[0]:+.1f} {cam_rpy[1]:+.1f} {cam_rpy[2]:+.1f}"
                    if cam_rpy is not None
                    else f"  quat {cam_quat[0]:+.2f} {cam_quat[1]:+.2f} "
                    f"{cam_quat[2]:+.2f} {cam_quat[3]:+.2f}"
                )
                + f"  nudge x{nudge_scale:g}",
                "i/k x  j/l y  u/o z  t/; r  y/h p  n/m yaw  c print  0 reset",
                "G record block→gripper   E reconstruct + IK [y/N]",
                f"follow [{objects[0]['name']}]: {'ON (F)' if follow_cam else 'off (F)'}",
                f"track objects: {'ON (1)' if track_objects else 'HOLD (1)'}",
                f"undistort: {'ON' if undistort else 'off'}",
                (
                    f"ori: LOCKED rpy {obj_rpy[0]:g} {obj_rpy[1]:g} "
                    f"{obj_rpy[2]:g} (B)"
                    if lock_ori
                    else "ori: tracked (B lock)"
                ),
                "pose: T_world_cam @ T_cam_obj",
            ]
            if gripper_origin_text:
                overlay_lines.append(gripper_origin_text)
            if GOAL_T_BLOCK_GRIPPER is None:
                overlay_lines.append("goal: none (G to record)")
            else:
                rel_t = GOAL_T_BLOCK_GRIPPER[:3, 3]
                clamp_s = (
                    f"  clamp {GOAL_GRIPPER_CLAMP:.1f}"
                    if GOAL_GRIPPER_CLAMP is not None
                    else ""
                )
                overlay_lines.append(
                    f"goal: recorded  t_block_grip "
                    f"{rel_t[0]:+.3f} {rel_t[1]:+.3f} {rel_t[2]:+.3f}"
                    f"{clamp_s}"
                )




            data.mocap_pos[floor_origin_id] = 0.0
            data.mocap_quat[floor_origin_id] = np.array([1.0, 0.0, 0.0, 0.0])
            data.mocap_pos[floor_cam_id] = cam_xyz
            data.mocap_quat[floor_cam_id] = cam_quat



            #------ LOOP THROGH OBJECTS AND UPDATE THEIR POSOTION
            for obj in objects:

                #---- check if detected ------
                hit = localized.get(obj["name"])
                if hit is None:
                    overlay_lines.append(
                        f"{obj['name']}: waiting (1 tag is enough)"
                    )
                    continue


                #------ set pos via the simulator object ------
                pos, quat, primary_tag = hit
                wpos, wquat = _T_to_pose(T_world_cam @ _pose_T(pos, quat))
                if lock_ori:
                    wquat = obj_lock_quat.copy()
                else:
                    prev = previous_quat.get(obj["name"])
                    if prev is not None and float(np.dot(prev, wquat)) < 0.0:
                        wquat = -wquat
                    previous_quat[obj["name"]] = wquat
                if track_objects:
                    sim.set_object_pose(obj["name"], wpos, wquat)
                    last_world_pose[obj["name"]] = (
                        np.asarray(wpos, dtype=float).reshape(3).copy(),
                        np.asarray(wquat, dtype=float).reshape(4).copy(),
                    )

                n_tags = tracker.last_bundle_counts.get(obj["name"], 1)
                held = "" if track_objects else " HOLD"
                overlay_lines.append(
                    f"{obj['name']} via {n_tags} tag(s) "
                    f"(primary {primary_tag}): "
                    f"world {wpos[0]:+.3f} {wpos[1]:+.3f} {wpos[2]:+.3f}"
                    f"{held}"
                )
                overlay_lines.append(
                    f"{obj['name']} from cam: "
                    f"dx={pos[0]:+.3f} dy={pos[1]:+.3f} "
                    f"dz={pos[2]:+.3f}  "
                    f"dist={float(np.linalg.norm(pos)):.3f}m"
                )

            if GOAL_T_BLOCK_GRIPPER is not None and cube_name in last_world_pose:
                bpos, bquat = last_world_pose[cube_name]
                gpos, _gquat = _T_to_pose(
                    _pose_T(bpos, bquat) @ GOAL_T_BLOCK_GRIPPER
                )
                overlay_lines.append(
                    f"goal gripper world {_fmt_xyz(gpos)}"
                )

            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_forward(model, data)

            if follow_cam and main_body_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = int(main_body_id)
            else:
                if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    viewer.cam.lookat[:] = data.xpos[main_body_id]
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            viewer.overlay = "\n".join(overlay_lines)
            viewer.sync()

            for seen_id, (R, t) in tracker.last_tag_poses.items():
                rvec, _ = cv2.Rodrigues(R)
                cv2.drawFrameAxes(
                    frame, cam_k, axes_dist, rvec, t.reshape(3, 1), args.tag_size
                )
                corners = tracker.last_corners.get(seen_id)
                color = (0, 255, 0)
                if corners is not None:
                    pts = corners.astype(np.int32)
                    cv2.polylines(frame, [pts], True, color, 2)
                cx, cy = tracker.last_centers[seen_id]
                cv2.putText(
                    frame,
                    str(seen_id),
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            y = 24
            opencv_lines = [
                line
                for line in overlay_lines
                if " from cam:" not in line
            ]
            for tag_id, (_R, t) in sorted(tracker.last_tag_poses.items()):
                t = np.asarray(t, dtype=float).reshape(3)
                opencv_lines.append(
                    f"tag {tag_id} from cam: "
                    f"dx={t[0]:+.3f} dy={t[1]:+.3f} "
                    f"dz={t[2]:+.3f}  "
                    f"dist={float(np.linalg.norm(t)):.3f}m"
                )
            for line in opencv_lines:
                waiting = "waiting" in line
                cv2.putText(
                    frame,
                    line,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255) if waiting else (0, 255, 0),
                    2,
                )
                y += 22
            cv2.imshow("LilyTag camera (apriltag_ros)", frame)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                break
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pending.append("print-gripper")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    pending.append("follow")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    pending.append("print-cam")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_0:
                    pending.append("reset-cam")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFTBRACKET:
                    pending.append("nudge-slower")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHTBRACKET:
                    pending.append("nudge-faster")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                    pending.append("go-to-relative")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                    pending.append("record-goal")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                    pending.append("lock-ori")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                    pending.append("toggle-track")
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_q,
                    pygame.K_ESCAPE,
                ):
                    pending.append("quit")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("f"), ord("F")):
                pending.append("follow")
            if key in (ord("p"), ord("P")):
                pending.append("print-gripper")
            if key in (ord("e"), ord("E")):
                pending.append("go-to-relative")
            if key in (ord("g"), ord("G")):
                pending.append("record-goal")
            if key in (ord("b"), ord("B")):
                pending.append("lock-ori")
            if key == ord("1"):
                pending.append("toggle-track")
            if key in (ord("c"), ord("C")):
                pending.append("print-cam")
            if key == ord("0"):
                pending.append("reset-cam")
            if key == ord("["):
                pending.append("nudge-slower")
            if key == ord("]"):
                pending.append("nudge-faster")
            if key not in (0xFF, 0, 255):
                tap_xyz, tap_rpy = _cam_tap_nudge(
                    key,
                    nudge_scale,
                    tap_m=CAM_NUDGE_TAP_M,
                    tap_deg=CAM_NUDGE_TAP_DEG,
                )
                cam_xyz += tap_xyz
                if cam_rpy is not None:
                    cam_rpy += tap_rpy
                    cam_quat = _rpy_deg_to_quat_wxyz(*cam_rpy)
            if key == ord("q"):
                break
            if glfw.get_key(viewer.window, glfw.KEY_Q) == glfw.PRESS:
                break
            if "quit" in pending:
                break
            if "follow" in pending:
                follow_cam = not follow_cam
            if "nudge-slower" in pending:
                nudge_scale = max(0.125, nudge_scale * 0.5)
                print(f"camera nudge scale x{nudge_scale:g}")
            if "nudge-faster" in pending:
                nudge_scale = min(8.0, nudge_scale * 2.0)
                print(f"camera nudge scale x{nudge_scale:g}")
            if "reset-cam" in pending:
                cam_xyz[:] = start_xyz
                if start_rpy is not None and cam_rpy is not None:
                    cam_rpy[:] = start_rpy
                cam_quat = start_quat.copy()
                print("camera pose reset to launch values.")
            if "print-cam" in pending:
                line = _format_cam_cli(cam_xyz, cam_rpy, cam_quat)
                print(line)
            if "print-gripper" in pending:
                pose = _gripper_pose(data, gripper_site_id, gripper_body_id)
                if pose is None:
                    print("Could not read gripper pose.")
                else:
                    gpos, gquat = pose
                    dist_m = float(np.linalg.norm(gpos))
                    gripper_origin_text = (
                        f"gripper from origin: "
                        f"dx={gpos[0]:+.4f} dy={gpos[1]:+.4f} "
                        f"dz={gpos[2]:+.4f}  dist={dist_m:.4f}m  "
                        f"quat {_fmt_quat(gquat)}"
                    )
                    print(gripper_origin_text)
            if "lock-ori" in pending:
                lock_ori = not lock_ori
                print(
                    f"object ori lock={'ON' if lock_ori else 'off'} "
                    f"rpy {obj_rpy[0]:g} {obj_rpy[1]:g} {obj_rpy[2]:g}"
                )
            if "toggle-track" in pending:
                track_objects = not track_objects
                print(
                    f"object tracking={'ON' if track_objects else 'HOLD'} (1)"
                )
            if "record-goal" in pending:
                gripper = _gripper_pose(
                    data, gripper_site_id, gripper_body_id
                )
                if cube_name not in last_world_pose:
                    print(
                        f"G: no known pose for '{cube_name}'. "
                        "Press 1 to track until it is seen, then HOLD."
                    )
                elif gripper is None:
                    print("G: could not read gripper pose.")
                else:
                    block_pos, block_quat = last_world_pose[cube_name]
                    grip_pos, grip_quat = gripper
                    T_world_block = _pose_T(block_pos, block_quat)
                    T_world_gripper = _pose_T(grip_pos, grip_quat)
                    GOAL_T_BLOCK_GRIPPER = _relative_T(
                        T_world_block, T_world_gripper
                    )
                    GOAL_GRIPPER_CLAMP = _obs_gripper_clamp(obs)
                    rel_t, rel_q = _T_to_pose(GOAL_T_BLOCK_GRIPPER)
                    src = "live" if track_objects else "HOLD"
                    print(
                        f"G: recorded block→gripper relative pose "
                        f"(not a world coordinate) from {src} block pose.\n"
                        f"  block    pos {_fmt_xyz(block_pos)}  "
                        f"quat {_fmt_quat(block_quat)}\n"
                        f"  gripper  pos {_fmt_xyz(grip_pos)}  "
                        f"quat {_fmt_quat(grip_quat)}\n"
                        f"  clamp    {_fmt_clamp(GOAL_GRIPPER_CLAMP)}\n"
                        f"  T_block_gripper  t {_fmt_xyz(rel_t)}  "
                        f"quat {_fmt_quat(rel_q)}"
                    )
            if "go-to-relative" in pending:
                if GOAL_T_BLOCK_GRIPPER is None:
                    print("E: no recorded goal. Press G first.")
                elif cube_name not in last_world_pose:
                    print(
                        f"E: no known pose for '{cube_name}'. "
                        "Press 1 to track until it is seen, then HOLD."
                    )
                else:
                    block_pos, block_quat = last_world_pose[cube_name]
                    T_world_block = _pose_T(block_pos, block_quat)
                    T_world_goal = T_world_block @ GOAL_T_BLOCK_GRIPPER
                    goal_pos, goal_quat = _T_to_pose(T_world_goal)
                    print(
                        f"\nE: reconstruct gripper pose from current "
                        f"'{cube_name}' pose.\n"
                        f"  Block pose\n"
                        f"    pos  {_fmt_xyz(block_pos)}\n"
                        f"    quat {_fmt_quat(block_quat)}\n"
                        f"  Target gripper pose\n"
                        f"    pos  {_fmt_xyz(goal_pos)}\n"
                        f"    quat {_fmt_quat(goal_quat)}\n"
                        f"    clamp {_fmt_clamp(GOAL_GRIPPER_CLAMP if GOAL_GRIPPER_CLAMP is not None else _obs_gripper_clamp(obs))}"
                    )
                    ik_action = sim.solveIK(goal_pos, goal_quat)
                    if ik_action is None:
                        print(
                            "  IK with orientation failed (5-DOF arm). "
                            "Retrying position-only..."
                        )
                        ik_action = sim.solveIK(goal_pos)
                    if ik_action is not None and GOAL_GRIPPER_CLAMP is not None:
                        ik_action["gripper.pos"] = float(GOAL_GRIPPER_CLAMP)
                    if ik_action is None:
                        print(
                            "  IK solution: failed. Target XYZ may be out of "
                            "reach from the current arm pose; teleop closer "
                            "and press E again."
                        )
                    else:
                        q_ik = real_to_sim(ik_action)
                        for name, q in q_ik.items():
                            data.qpos[sim._joint_qposadr[name]] = q
                        data.qvel[:] = 0.0
                        mujoco.mj_forward(model, data)
                        viewer.sync()
                        achieved = _gripper_pose(
                            data, gripper_site_id, gripper_body_id
                        )
                        print(f"  IK solution\n    {_fmt_action(ik_action)}")
                        if achieved is not None:
                            apos, aquat = achieved
                            print(
                                f"  IK gripper (sim preview, MuJoCo window)\n"
                                f"    pos  {_fmt_xyz(apos)}  "
                                f"err {float(np.linalg.norm(apos - goal_pos))*1000:.1f} mm\n"
                                f"    quat {_fmt_quat(aquat)}  "
                                f"err {_quat_ang_deg(aquat, goal_quat):.1f} deg"
                            )
                        try:
                            answer = input(
                                "\nMove robot IRL to this pose? [y/N]: "
                            )
                        except EOFError:
                            answer = ""
                        _drain_ui_events()
                        sim.sync(obs)
                        viewer.sync()
                        if answer.strip().lower() == "y":
                            ease_to_position(robot, ik_action)
                            obs = robot.get_observation()
                            controller.sync_from_observation(obs)
                            sim.sync(obs)
                            print("E: reached IK pose.")
                            try:
                                squeeze = input(
                                    "\nClamp gripper hard on the block? [y/N]: "
                                )
                            except EOFError:
                                squeeze = ""
                            _drain_ui_events()
                            if squeeze.strip().lower() == "y":
                                _clamp_gripper_hard(robot)
                                obs = robot.get_observation()
                                controller.sync_from_observation(obs)
                                sim.sync(obs)
                                print("E: clamp finished (Xbox teleop resumed).")
                            else:
                                print("E: skipped clamp (Xbox teleop resumed).")
                        else:
                            print("E: cancelled. Robot not moved.")
            pending.clear()

            prev_time = current_time
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - loop_start), 0.0))
    finally:
        cap.release()
        try:
            viewer.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if connected:
            try:
                print("Moving to rest pose...")
                go_to_rest(robot)
            except Exception as exc:
                print(f"Rest pose failed: {exc}")
            try:
                controller.disconnect()
            except Exception as exc:
                print(f"Controller disconnect failed: {exc}")
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"Robot disconnect failed (motors may need a power cycle): {exc}")
        pygame.quit()


if __name__ == "__main__":
    main()
