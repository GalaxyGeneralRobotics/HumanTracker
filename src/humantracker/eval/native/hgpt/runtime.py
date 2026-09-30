# Derived from GalaxyGeneralRobotics/Humanoid-GPT, de8498f (Apache-2.0).
# Evaluation-only extraction; see ../SOURCES.md for modifications.
from types import SimpleNamespace
from typing import Any
import numpy as np
import mujoco
import onnxruntime as rt
from scipy.spatial.transform import Rotation as R
from humantracker.eval.core.mj_sim import MJSim, State
from . import constants as consts
from .constants import ACTION_JOINT_NAMES, OBS_JOINT_NAMES, KPT_NAMES
from .transforms import batch_base2navi, quat2mat, quat2yaw, WarpPi

NUM_ACTION = 29

def g1_infer_env_config(ctrl_dt=0.02):
    return SimpleNamespace(ctrl_dt=ctrl_dt, action_scale=0.25,
                           soft_joint_pos_limit_factor=0.90)

def get_sensor_data(mj_model, mj_data, sensor_name: str) -> np.ndarray:
    """Gets sensor data given sensor name."""
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim]

def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get the distance and normal of the collision between two geoms."""
    mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = np.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, :3]
    return dist, normal

def geoms_colliding(state: mujoco.MjData, geom1: int, geom2: int):
    """Return True if the two geoms are colliding."""
    if len(state.contact) == 0:
        return 0
    return get_collision_info(state.contact, geom1, geom2)[0] < 0

mj_coll = geoms_colliding
mj_sensor = get_sensor_data

class G1TrackMjSim(MJSim):
    def __init__(
        self,
        init_qpos,
        headless=False,
        ctrl_dt=0.02,
        sim_dt=0.001,
        xml_path=None,
    ):
        if xml_path is None:
            from humantracker.eval.paths import required_file
            xml_path = required_file("storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml")
        super().__init__(xml_path=xml_path, ctrl_dt=ctrl_dt, sim_dt=sim_dt, headless=headless)
        self.kps = np.float32(consts.KPs)
        self.kds = np.float32(consts.KDs)
        self.torque_limit = np.float32(consts.TORQUE_LIMIT)
        self.init_qpos = np.float32(init_qpos)



class G1TrackInferFn:
    def __init__(
        self,
        env_config,
        mj_model: mujoco.MjModel,
        nn_policy: rt.InferenceSession,
        num_envs: int = 1,
        privileged: bool = False,
    ):
        self.env_config = env_config
        self.mj_model = mj_model
        self.nn_policy = nn_policy
        self.num_envs = num_envs
        self.privileged = privileged

        # init
        self.ctrl_id_act = np.int32(
            [self.mj_model.actuator(n).id for n in ACTION_JOINT_NAMES]
        )
        self.ctrl_id_obs = np.int32(
            [self.mj_model.actuator(n).id for n in OBS_JOINT_NAMES]
        )
        self.dof_ids_g1 = np.int32(
            np.hstack([self.mj_model.joint(n).dofadr for n in consts.MotorName.FULL])
        )
        self.qpos_ids_g1 = np.int32(
            np.hstack([self.mj_model.joint(n).qposadr for n in consts.MotorName.FULL])
        )

        self._nom_jnt_qpos = np.full(
            (self.num_envs, consts.NUM_JOINT), consts.DEFAULT_QPOS[7:]
        )
        self.act_scale = np.array(consts.ACTION_SCALE)
        self.dt = env_config.ctrl_dt

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self.env_config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self.env_config.soft_joint_pos_limit_factor

        self.site_id_chest = self.mj_model.site("chest").id
        self.site_ids_feet = np.array(
            [self.mj_model.site(name).id for name in consts.FEET_SITES]
        )
        self.geom_id_floor = self.mj_model.geom("floor").id
        self.site_id_torso_imu = self.mj_model.site("imu_in_torso").id
        self.site_id_pelvis_imu = self.mj_model.site("imu_in_pelvis").id
        self.geom_ids_left_feet = np.array(
            [self.mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS]
        )
        self.geom_ids_right_feet = np.array(
            [self.mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS]
        )
        self.body_ids_kpt_full = np.array([self.mj_model.body(n).id for n in KPT_NAMES])
        self.kpt_id_pelvis = KPT_NAMES.index("pelvis")
        self.kpt_id_torso = KPT_NAMES.index("torso_link")
        self.kpt_id_peltor = np.hstack([self.kpt_id_pelvis, self.kpt_id_torso])

        self.num_jnt = len(consts.DEFAULT_QPOS[7:])
        self.num_kpt = len(KPT_NAMES)

        self.info = {
            "step": 0,
            "nn_action": np.zeros((self.num_envs, NUM_ACTION), dtype=np.float32),
            "gyro_pelvis": np.zeros((self.num_envs, 3)),
            "gvec_pelvis": np.zeros((self.num_envs, 3)),
            "linvel_pelvis": np.zeros((self.num_envs, 3)),
            "qpos": np.zeros((self.num_envs, self.mj_model.nq)),
            "qvel": np.zeros((self.num_envs, self.mj_model.nv)),
            "last_action": np.zeros((self.num_envs, len(self.ctrl_id_act))),
            "motor_targets": self._nom_jnt_qpos.copy(),
            # hint state
            "navi_pelvis_rpy": np.zeros((self.num_envs, 3)),
            "navi_torso_rpy": np.zeros((self.num_envs, 3)),
            "feet_contact": np.zeros((self.num_envs, 2)),
            # kpt actual state
            "acu_kpt2gv_pose": np.full((self.num_envs, self.num_kpt, 4, 4), np.eye(4)),
            "acu_kpt_cvel_in_gv": np.zeros((self.num_envs, self.num_kpt, 6)),
            # kpt reference state
            # "ref_next_navi_vel": np.zeros((self.num_envs, 3)),
            # "next_jnt_qpos_res": np.zeros((self.num_envs, self.num_jnt)),
            # "next_jnt_qvel_res": np.zeros((self.num_envs, self.num_jnt)),
            # "ref_next_kpt_npose": np.full(
            #     (self.num_envs, self.num_kpt, 4, 4), np.eye(4)
            # ),
            # "ref_next_kpt_cvel": np.zeros((self.num_envs, self.num_kpt, 6)),
        }

    def infer_onnx(self, state: State, ref_state: np.ndarray | dict) -> np.ndarray:
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])
        self.update_state(state, ref_state)
        if self.privileged:
            obs = self.get_nn_priv_state(self.info, ref_next, self.info["last_action"])
        else:
            obs = self.get_nn_state(self.info, ref_next, self.info["last_action"])

        nn_action = self.nn_policy.infer(obs)

        motor_targets = self.nn2motor_action(nn_action)
        self.info["motor_targets"] = motor_targets.copy()

        self.info["step"] += 1
        self.info["nn_action"] = nn_action
        self.info["last_action"] = nn_action.copy()
        return motor_targets

    def get_nn_state(self, info, ref_state: np.ndarray, last_action: np.ndarray):
        ref_qpos = ref_state["qpos"][:, 7:]
        ref_qvel = ref_state["qvel"][:, 6:]

        gyro_pelvis = info["gyro_pelvis"]
        gvec_pelvis = info["gvec_pelvis"]
        joint_qpos = info["qpos"][:, self.qpos_ids_g1]
        joint_qvel = info["qvel"][:, self.dof_ids_g1]

        ref_root_gv_pose = ref_state["kpt2gv_pose"][:, 0]
        ref_root_cvel_in_gv = ref_state["kpt_cvel_in_gv"][:, 0]

        # Encode yaw as [cos, sin] to match training logic
        yaw_cmd = np.stack([np.cos(info["yaw_d"]), np.sin(info["yaw_d"])], axis=-1)  # (num_envs, 2)
        # xy_d is already in navi frame (R(-yaw_curr) @ (ref-curr)); training uses same
        xy_cmd = info["xy_d"]

        state = np.hstack(
            [
                # pose state
                gyro_pelvis,  # 3
                gvec_pelvis,  # 3
                # joint state
                (joint_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 29
                joint_qvel[:, self.ctrl_id_obs],  # 29  * scale_vel
                last_action,
                # commands
                (ref_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],
                # ref_qvel[:, self.ctrl_id_obs],
                ref_root_gv_pose[:, 2, 3][:, None],  # height (num_envs, 1)
                -ref_root_gv_pose[:, :3, :3].transpose((0, 2, 1))[..., 2],
                ref_root_cvel_in_gv,
                # global
                yaw_cmd,  # (num_envs, 2)
                xy_cmd    # (num_envs, 2)
            ]
        ).astype(np.float32)
        return state

    def get_nn_priv_state(self, info, ref_state: np.ndarray, last_action: np.ndarray):
        ref_qpos = ref_state["qpos"][:, 7:]
        ref_qvel = ref_state["qvel"][:, 6:]

        gyro_pelvis = info["gyro_pelvis"]
        gvec_pelvis = info["gvec_pelvis"]
        linvel_pelvis = info["linvel_pelvis"]
        joint_qpos = info["qpos"][:, self.qpos_ids_g1]
        joint_qvel = info["qvel"][:, self.dof_ids_g1]

        ref_root_gv_pose = ref_state["kpt2gv_pose"][:, 0]
        ref_root_cvel_in_gv = ref_state["kpt_cvel_in_gv"][:, 0]

        # Encode yaw as [cos, sin] to match training's priv_yaw_cmd
        priv_yaw_cmd = np.stack([np.cos(info["yaw_d"]), np.sin(info["yaw_d"])], axis=-1)  # (num_envs, 2)

        privileged_state = np.hstack(
            [
                # pose state
                gyro_pelvis,  # 3
                gvec_pelvis,  # 3
                # joint state
                (joint_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],  # 23
                joint_qvel[:, self.ctrl_id_obs],  # 23
                last_action,  # num_actions
                # reference motion
                (ref_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],
                # ref_qvel[:, self.ctrl_id_obs],
                # hint state
                linvel_pelvis,  # 3
                info["acu_root2gv_lin_vel"],
                info["acu_root2gv_ang_vel"],
                info["acu_kpt2gv_pose"][..., :3, :2].reshape(self.num_envs, -1),
                info["acu_kpt2gv_pose"][..., :3, 3].reshape(self.num_envs, -1),
                info["acu_kpt_cvel_in_gv"].reshape(self.num_envs, -1),
                # res
                info["next_ref2acu_gv_vel"],
                info["next_ref2acu_kpt_pose"][..., :3, :2].reshape(self.num_envs, -1),
                info["next_ref2acu_kpt_pose"][..., :3, 3].reshape(self.num_envs, -1),
                info["next_ref2acu_kpt_cvel"].reshape(self.num_envs, -1),
                # root command
                ref_root_gv_pose[:, 2, 3][:, None],  # height (num_envs, 1)
                -ref_root_gv_pose[:, :3, :3].transpose((0, 2, 1))[..., 2],
                ref_root_cvel_in_gv,
                # global
                priv_yaw_cmd,  # (num_envs, 2)
                info["xy_d"],  # (num_envs, 2)
            ]
        ).astype(np.float32)
        return privileged_state

    def update_state(self, state: State, ref_state):
        mj_data = state.mj_data
        gyro_pelvis = mj_sensor(self.mj_model, mj_data, "gyro_pelvis")
        pelvis2world_rot = quat2mat(mj_data.qpos[3:7][None])
        pelvis2world_pos = mj_data.qpos[:3][None]
        qpos = mj_data.qpos.copy()[None]
        qvel = mj_data.qvel.copy()[None]

        # privileged state
        linvel_pelvis = mj_sensor(self.mj_model, mj_data, "local_linvel_pelvis")
        torso2world_rot = mj_data.site_xmat[self.site_id_torso_imu]
        torso2world_rot = torso2world_rot.reshape(1, 3, 3)
        feet_contact = self.get_mj_feet_contact(mj_data)[None]
        kpt2wrd_rot = mj_data.xmat[self.body_ids_kpt_full].reshape(-1, 3, 3)
        kpt2wrd_pos = mj_data.xpos[self.body_ids_kpt_full][None]
        acu_kpt_cvel_in_wrd = mj_data.cvel[self.body_ids_kpt_full][None]

        # state
        gvec_pelvis = -pelvis2world_rot.transpose(0, 2, 1)[..., 2]
        self.info["gyro_pelvis"][:] = gyro_pelvis.copy()
        self.info["linvel_pelvis"][:] = linvel_pelvis.copy()
        self.info["gvec_pelvis"][:] = gvec_pelvis.copy()
        self.info["qpos"][:] = qpos.copy()
        self.info["qvel"][:] = qvel.copy()

        # privileged state
        navi2world_rot = batch_base2navi(pelvis2world_rot)
        pelvis2navi_rot = navi2world_rot.transpose(0, 2, 1) @ pelvis2world_rot
        torso2navi_rot = navi2world_rot.transpose(0, 2, 1) @ torso2world_rot
        self.info["navi2world_rot"] = navi2world_rot.copy()
        self.info["navi_pelvis_rpy"][:] = (
            R.from_matrix(pelvis2navi_rot).as_euler("xyz").copy()
        )
        self.info["navi_torso_rpy"][:] = (
            R.from_matrix(torso2navi_rot).as_euler("xyz").copy()
        )
        self.info["feet_contact"][:] = feet_contact.copy()

        # gravity view frame
        acu_gv2wrd_pose = np.full((self.num_envs, 4, 4), np.eye(4))
        acu_gv2wrd_pose[:, :3, :3] = navi2world_rot
        acu_gv2wrd_pose[:, :2, 3] = pelvis2world_pos[:, :2]
        acu_kpt2wrd_pose = np.full((self.num_envs, self.num_kpt, 4, 4), np.eye(4))
        acu_kpt2wrd_pose[:, :, :3, :3] = kpt2wrd_rot
        acu_kpt2wrd_pose[:, :, :3, 3] = kpt2wrd_pos
        acu_kpt2gv_pose = np.linalg.inv(acu_gv2wrd_pose[:, None]) @ acu_kpt2wrd_pose
        self.info["acu_kpt2gv_pose"] = acu_kpt2gv_pose

        acu_kpt_cvel_in_gv = np.zeros_like(acu_kpt_cvel_in_wrd)
        R_wrd2gv = np.swapaxes(acu_gv2wrd_pose[..., :3, :3], -1, -2)  # R_gv2wrd^T = R_wrd2gv
        acu_kpt_cvel_in_gv[..., :3] = np.einsum("...ij,...kj->...ki", R_wrd2gv, acu_kpt_cvel_in_wrd[..., :3])
        acu_kpt_cvel_in_gv[..., 3:] = np.einsum("...ij,...kj->...ki", R_wrd2gv, acu_kpt_cvel_in_wrd[..., 3:])
        self.info["acu_kpt_cvel_in_gv"][:] = acu_kpt_cvel_in_gv

        self.update_coord_cmd(ref_state)

        if "ref_next" in ref_state:
            ref_next = ref_state["ref_next"]

            next_ref_gv2wrd_pose = ref_next["gv2wrd_pose"]
            self.info["next_ref2acu_gv_pose"] = (
                np.linalg.inv(acu_gv2wrd_pose) @ next_ref_gv2wrd_pose
            )
            next_ref2acu_kpt_pose = (
                np.linalg.inv(acu_kpt2gv_pose) @ ref_next["kpt2gv_pose"]
            )
            self.info["next_ref2acu_kpt_pose"] = next_ref2acu_kpt_pose

            ref_next_navi_vel = ref_next["gv_vel"]
            acu_root2gv_lin_vel = pelvis2navi_rot @ linvel_pelvis
            acu_root2gv_ang_vel = pelvis2navi_rot @ gyro_pelvis
            self.info["acu_root2gv_lin_vel"] = acu_root2gv_lin_vel
            self.info["acu_root2gv_ang_vel"] = acu_root2gv_ang_vel
            # self.info["acu_navi_vel"] = np.hstack(
            #     [navi_pelvis_lin_vel[:, :2], navi_pelvis_ang_vel[:, 2:3]]
            # )
            next_vel_lin_res = ref_next_navi_vel[:, :2] - acu_root2gv_lin_vel[:, :2]
            next_vel_ang_res = ref_next_navi_vel[:, 2] - acu_root2gv_ang_vel[:, 2]
            self.info["next_ref2acu_gv_vel"] = np.hstack(
                [next_vel_lin_res, next_vel_ang_res[:, None]]
            )
            # self.info["ref_next_navi_vel"][:] = ref_next["navi_vel"]
            # self.info["ref_next_kpt_npose"] = ref_next["kpt_cvel_in_gv"]
            # self.info["ref_next_kpt_cvel"] = ref_next["kpt_cvel"]
            # self.info["next_jnt_qpos_res"] = ref_next["qpos"][:, 7:] - qpos[:, 7:]
            # self.info["next_jnt_qvel_res"] = ref_next["qvel"][:, 6:] - qvel[:, 6:]
            self.info["next_ref2acu_kpt_cvel"] = (
                ref_next["kpt_cvel_in_gv"] - acu_kpt_cvel_in_gv
            )

    def update_coord_cmd(self, ref_state):
        curr_ref_state = ref_state["ref_curr"]
        q_ref = curr_ref_state["qpos"][:, 3:7]
        q_curr = self.info["qpos"][:, 3:7]
        yaw_ref = quat2yaw(q_ref)
        yaw_curr = quat2yaw(q_curr)
        yaw_cmd = curr_ref_state.get("yaw_cmd", np.zeros_like(yaw_curr))
        yaw_target = yaw_cmd + yaw_ref
        yaw_d = WarpPi(yaw_target - yaw_curr)

        p_ref = curr_ref_state["qpos"][:, :2]
        p_curr = self.info["qpos"][:, :2]

        c, s = np.cos(yaw_cmd), np.sin(yaw_cmd)
        R_m = np.stack([
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ], axis=-2)
        xy_ref = p_ref
        xy_curr = p_curr
        xy_cmd = curr_ref_state.get("xy_cmd", np.zeros_like(xy_curr))
        xy_target = np.einsum("...ij,...j->...i", R_m, (xy_cmd + xy_ref))
        xy_d = xy_target - xy_curr

        c, s = np.cos(-yaw_curr), np.sin(-yaw_curr)
        R_m = np.stack([
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ], axis=-2)
        xy_d = np.einsum("...ij,...j->...i", R_m, xy_d)
        # xy_d = np.clip(xy_d, self.coord_cfg["xy_move_range"][0], self.coord_cfg["xy_move_range"][1])

        self.info["yaw_d"] = yaw_d
        self.info["xy_d"] = xy_d

    def get_mj_feet_contact(self, mj_data: mujoco.MjData) -> np.ndarray:
        """
        Returns the contact state of the left and right feet.

        Contact state encoding:
                -1: fully in air (no contact)
                 1: partial contact (some or all foot geoms in contact)
        """
        left_contacts = np.array(
            [
                mj_coll(mj_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_left_feet
            ]
        )
        right_contacts = np.array(
            [
                mj_coll(mj_data, geom_id, self.geom_id_floor)
                for geom_id in self.geom_ids_right_feet
            ]
        )

        left_state = np.where(left_contacts.any(), 1, -1)
        right_state = np.where(right_contacts.any(), 1, -1)

        return np.array([left_state, right_state])

    def nn2motor_action(self, nn_action):
        motor_targets = self._nom_jnt_qpos.copy()
        motor_targets[:, self.ctrl_id_act] = (
            self._nom_jnt_qpos[:, self.ctrl_id_act]
            + nn_action
            * self.env_config.action_scale
            * self.act_scale[self.ctrl_id_act]
        )
        return motor_targets
