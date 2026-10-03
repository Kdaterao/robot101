import numpy as np
import mujoco
from lerobot.types import RobotAction, RobotObservation

from utility.transforms import (
    R_to_quat_wxyz,
    T_CV_TO_MJ,
    T_from_R_t,
    quat_wxyz_to_R,
    rot_error,
)

from .joints import ARM_JOINTS, MOTOR_NAMES, real_to_sim, sim_to_real
from .sceneLoader import SceneLoader


class simController:
    """
    
    Holds the MuJoCo SO101 and copies real LeRobot 
    joint readings into qpos.
    
    """

    motor_names = list(MOTOR_NAMES)

    gripper_error_threshold = 3.0
    _stall_frames_needed = 3
    _stall_motion_eps = 0.5

    def __init__(self, model=None, data=None, scene_path=None, objects=None):

        #LOAD MODEL AND DATA IF NOT PROVIDED
        if model is None or data is None:
            model, data = SceneLoader(scene_path).load(objects=objects)


        #SET VARIABLES
        self.model = model # specification of our sim
        self.data = data # state
        self._synced = False
        self.solver = None
        self.IK_positions = None

        self.targets = {name: 0.0 for name in self.motor_names}
        self._prev_gripper = 0.0
        self._stall_count = 0


        #INITIALIZE JOINT QPOS ADDRESS DICTIONARY (SO ITS ALWAYS CORRECT AND VALID)
        self._joint_qposadr = {}
        for name in self.motor_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            self._joint_qposadr[name] = int(self.model.jnt_qposadr[jid])

        self._index_mocap_objects(objects)



    def _index_mocap_objects(self, objects=None) -> None:
        self._mocap_ids: dict[str, int] = {}
        names: list[str] = []
        if objects:
            names = [
                obj["name"] if isinstance(obj, dict) else str(obj)
                for obj in objects
            ]
        else:
            for i in range(self.model.nbody):
                if int(self.model.body_mocapid[i]) >= 0:
                    name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, i
                    )
                    if name:
                        names.append(name)
        for name in names:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id < 0:
                raise ValueError(f"Body '{name}' not found in MuJoCo model")
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(f"Body '{name}' is not a mocap body")
            self._mocap_ids[name] = mocap_id



    def sync(self, obs: RobotObservation) -> None:
        """
        Match the simulated arm to a real observation 
        or named pose dict.
        
        """
        
        qpos = real_to_sim(obs) 
        for name, q in qpos.items():
            self.data.qpos[self._joint_qposadr[name]] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for name in self.motor_names:
            self.targets[name] = float(obs[f"{name}.pos"])
        self._prev_gripper = float(obs["gripper.pos"])
        self._stall_count = 0
        self._synced = True



    def sim_qpos(self) -> dict[str, float]:
        '''
        Returns the current position of the mujuco 
        model as a dictionary.
        '''
        return {
            name: float(self.data.qpos[adr])
            for name, adr in self._joint_qposadr.items()
        }



    def set_object_pose(self, name: str, pos, quat) -> None:
        '''Set a named mocap object's world pose (quat is wxyz).'''
        mid = self._mocap_ids.get(name)
        if mid is None:
            raise ValueError(f"Unknown mocap object '{name}'")
        self.data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64).reshape(3)
        self.data.mocap_quat[mid] = np.asarray(quat, dtype=np.float64).reshape(4)



    def camera_intrinsics(
        self, camera: str, width: int, height: int
    ) -> list[float]:
        '''fx, fy, cx, cy from MuJoCo vertical fovy and image size.'''
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        if cam_id < 0:
            raise ValueError(f"Camera '{camera}' not found in MuJoCo model")
        fovy = float(np.deg2rad(self.model.cam_fovy[cam_id]))
        fy = (0.5 * float(height)) / np.tan(0.5 * fovy)
        fx = fy * (float(width) / float(height))
        return [fx, fy, 0.5 * float(width), 0.5 * float(height)]

    def pose_from_known_tags(
        self,
        obj_pos,
        obj_quat,
        detections: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]],
        tag_in_base_quat=(1.0, 0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Camera-frame object pose -> world using known tags at the origin.

        Landmark XYZ is measured from the world origin, not the STL base
        mesh (that body is only a mount offset).
        """
        if not detections:
            return None

        cam_R = []
        cam_t = []
        for tag_R, tag_t, tag_in_origin_pos in detections:
            t_world_tag = T_from_R_t(
                quat_wxyz_to_R(tag_in_base_quat),
                tag_in_origin_pos,
            )
            t_cam_tag = T_from_R_t(tag_R, tag_t)
            t_world_cam = t_world_tag @ np.linalg.inv(t_cam_tag)
            cam_R.append(t_world_cam[:3, :3])
            cam_t.append(t_world_cam[:3, 3])

        mean_t = np.mean(np.stack(cam_t, axis=0), axis=0)
        mean_R = np.mean(np.stack(cam_R, axis=0), axis=0)
        U, _, Vt = np.linalg.svd(mean_R)
        R_cam = U @ Vt
        if float(np.linalg.det(R_cam)) < 0.0:
            U[:, -1] *= -1.0
            R_cam = U @ Vt
        t_world_cam = T_from_R_t(R_cam, mean_t)
        t_cam_obj = T_from_R_t(quat_wxyz_to_R(obj_quat), obj_pos)
        t_world_obj = t_world_cam @ t_cam_obj
        delta = t_world_obj[:3, 3] - t_world_cam[:3, 3]
        return (
            t_world_obj[:3, 3].copy(),
            R_to_quat_wxyz(t_world_obj[:3, :3]),
            np.asarray(delta, dtype=float).reshape(3).copy(),
            t_world_cam[:3, 3].copy(),
        )

    def known_tag_world_pose(
        self,
        tag_in_base_pos=(0.20, 0.0, 0.0),
        tag_in_base_quat=(1.0, 0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fixed landmark pose in world origin (not the STL base mesh)."""
        t_world_tag = T_from_R_t(
            quat_wxyz_to_R(tag_in_base_quat),
            tag_in_base_pos,
        )
        return (
            t_world_tag[:3, 3].copy(),
            R_to_quat_wxyz(t_world_tag[:3, :3]),
        )

    def pose_from_cam_pose(
        self, pos, quat, cam_pos, cam_R
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """OpenCV camera-frame object pose -> world from a known camera pose.

        cam_pos is the camera in the SO101 / world origin. cam_R is the
        MuJoCo camera rotation (columns: +X right, +Y up, +Z back).
        """
        t_world_cam = T_from_R_t(cam_R, cam_pos)
        t_cam_obj = T_from_R_t(quat_wxyz_to_R(quat), pos)
        t_world_obj = t_world_cam @ T_CV_TO_MJ @ t_cam_obj
        delta = t_world_obj[:3, 3] - t_world_cam[:3, 3]
        return (
            t_world_obj[:3, 3].copy(),
            R_to_quat_wxyz(t_world_obj[:3, :3]),
            np.asarray(delta, dtype=float).reshape(3).copy(),
            t_world_cam[:3, 3].copy(),
        )

    def pose_from_camera(
        self, pos, quat, camera: str = "wrist_cam"
    ) -> tuple[np.ndarray, np.ndarray]:
        '''Camera-frame pose -> world pose, via the SO101 base.'''
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
        )
        if cam_id < 0:
            raise ValueError(f"Camera '{camera}' not found in MuJoCo model")
        base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        if base_id < 0:
            raise ValueError("Body 'base' not found in MuJoCo model")

        t_world_base = T_from_R_t(
            self.data.xmat[base_id].reshape(3, 3),
            self.data.xpos[base_id],
        )
        t_world_cam = T_from_R_t(
            self.data.cam_xmat[cam_id].reshape(3, 3),
            self.data.cam_xpos[cam_id],
        )
        t_base_cam = np.linalg.inv(t_world_base) @ t_world_cam
        t_cam_obj = T_from_R_t(quat_wxyz_to_R(quat), pos)
        t_base_obj = t_base_cam @ T_CV_TO_MJ @ t_cam_obj
        t_world_obj = t_world_base @ t_base_obj
        return t_world_obj[:3, 3].copy(), R_to_quat_wxyz(t_world_obj[:3, :3])



    def set_object_pose_from_camera(
        self, name: str, pos, quat, camera: str = "wrist_cam"
    ) -> None:
        wpos, wquat = self.pose_from_camera(pos, quat, camera)
        self.set_object_pose(name, wpos, wquat)



    def get_action(self, dt: float, obs: RobotObservation) -> RobotAction:
        """Turns the current sim pose into a follower action, with gripper stall guard."""
        if not self._synced:
            self.sync(obs)

        action = sim_to_real(self.sim_qpos())

        actual_gripper = float(obs["gripper.pos"])
        leader_gripper = float(action["gripper.pos"])
        closing = leader_gripper < actual_gripper - 1.0
        moved_closed = (self._prev_gripper - actual_gripper) > self._stall_motion_eps

        if closing and not moved_closed:
            self._stall_count += 1
        else:
            self._stall_count = 0

        if self._stall_count >= self._stall_frames_needed:
            action["gripper.pos"] = max(
                leader_gripper,
                actual_gripper - self.gripper_error_threshold,
            )

        self._prev_gripper = actual_gripper
        return action

    def solveIK(self, end_pos, end_quat=None, site: str = "gripperframe"):
        """Damped Jacobian IK. Position first; orientation is best-effort.

        The SO101 arm is 5-DOF, so a 6-DOF target cannot always be met.
        Position is solved first, then orientation is applied with a low
        weight so it cannot pull the gripper off the target.
        Returns a LeRobot action dict or None.
        """
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if site_id < 0:
            raise ValueError(f"Site '{site}' not found in MuJoCo model")
        target = np.asarray(end_pos, dtype=float).reshape(3)
        use_ori = end_quat is not None
        R_des = quat_wxyz_to_R(end_quat) if use_ori else None

        qadr: list[int] = []
        dofadr: list[int] = []
        jids: list[int] = []
        for name in ARM_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")
            jids.append(jid)
            qadr.append(int(self.model.jnt_qposadr[jid]))
            dofadr.append(int(self.model.jnt_dofadr[jid]))

        q0 = np.array(self.data.qpos, copy=True)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        pos_tol = 0.004
        ori_tol = 0.25
        accept_pos = 0.03

        def _step(ori_weight: float, n_iter: int, damping: float) -> None:
            n_err = 6 if ori_weight > 0.0 else 3
            step = 0.6
            for _ in range(n_iter):
                mujoco.mj_forward(self.model, self.data)
                pos_err = target - np.asarray(
                    self.data.site_xpos[site_id], dtype=float
                )
                pos_n = float(np.linalg.norm(pos_err))
                if ori_weight > 0.0:
                    R_cur = np.asarray(
                        self.data.site_xmat[site_id], dtype=float
                    ).reshape(3, 3)
                    ori_err = rot_error(R_cur, R_des)
                    ori_n = float(np.linalg.norm(ori_err))
                    err = np.concatenate([pos_err, ori_weight * ori_err])
                    mujoco.mj_jacSite(
                        self.model, self.data, jacp, jacr, site_id
                    )
                    J = np.vstack(
                        [jacp[:, dofadr], ori_weight * jacr[:, dofadr]]
                    )
                else:
                    ori_n = 0.0
                    err = pos_err
                    mujoco.mj_jacSite(
                        self.model, self.data, jacp, None, site_id
                    )
                    J = jacp[:, dofadr]
                if pos_n < pos_tol and (ori_weight <= 0.0 or ori_n < ori_tol):
                    return
                dq = J.T @ np.linalg.solve(
                    J @ J.T + damping * np.eye(n_err), err
                )
                for i, (jid, adr) in enumerate(zip(jids, qadr)):
                    q = float(self.data.qpos[adr] + step * dq[i])
                    if int(self.model.jnt_limited[jid]):
                        lo, hi = self.model.jnt_range[jid]
                        q = float(np.clip(q, lo, hi))
                    self.data.qpos[adr] = q
                self.data.qvel[:] = 0.0

        try:
            _step(0.0, 250, 1e-3)
            q_pos = np.array(self.data.qpos, copy=True)
            mujoco.mj_forward(self.model, self.data)
            pos_1 = float(
                np.linalg.norm(
                    target
                    - np.asarray(self.data.site_xpos[site_id], dtype=float)
                )
            )
            if use_ori:
                _step(0.12, 200, 1e-2)
                mujoco.mj_forward(self.model, self.data)
                pos_2 = float(
                    np.linalg.norm(
                        target
                        - np.asarray(self.data.site_xpos[site_id], dtype=float)
                    )
                )
                if pos_2 > pos_1 + 0.005:
                    self.data.qpos[:] = q_pos
                    self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            pos_n = float(
                np.linalg.norm(
                    target
                    - np.asarray(self.data.site_xpos[site_id], dtype=float)
                )
            )
            if pos_n < accept_pos:
                return sim_to_real(self.sim_qpos())
            return None
        finally:
            self.data.qpos[:] = q0
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
