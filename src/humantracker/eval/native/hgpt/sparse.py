"""HGPT 5-point sparse observation, with no upstream source dependency.

Derived from projects/sparse_tracking/infer_utils.py in Humanoid-GPT
de8498f4aaa6f1a0ce05ca69531c344e9f4b0c40 (Apache-2.0; see LICENSE).
Only the five-point MuJoCo evaluation path is included.
"""
import numpy as np

from .constants import KPT_NAMES
from .runtime import G1TrackInferFn

SPARSE_KPT_NAMES = (
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
NUM_OBSERVATIONS = 93 + 5 * 15 + 14


class G1TrackSparseInferFn(G1TrackInferFn):
    """Preserve upstream bookkeeping and PD action conversion; replace observations."""

    def __init__(self, env_config, mj_model, nn_policy, num_envs=1, privileged=False):
        if privileged:
            raise ValueError("HGPT 5-point sparse tracking is non-privileged")
        super().__init__(env_config, mj_model, nn_policy,
                         num_envs=num_envs, privileged=False)
        self.sparse_mode = "5point"
        self.sparse_kpt_ids = np.array(
            [KPT_NAMES.index(name) for name in SPARSE_KPT_NAMES], dtype=np.int32
        )
        self.sparse_kpt_local_offsets = np.zeros((5, 3), dtype=np.float32)
        self.num_sparse_kpt = 5

    def get_nn_state(
        self,
        info: dict,
        ref_state: dict,
        last_action: np.ndarray,
    ) -> np.ndarray:
        """Construct the sparse-keypoint observation vector.

        Layout (matches ``G1TrackSparse._get_obs`` from training)::

            proprio_part     93   gyro(3) gvec(3) jnt_pos(29) jnt_vel(29) act(29)
            sparse_ref       K*15 pos(K*3) rot6d(K*6) vel6d(K*6)
            root_cmd_part    14   height(1) gvec(3) cvel(6) yaw(2) xy(2)
        """
        B = self.num_envs

        # ---- proprioception (identical to parent) ----
        gyro_pelvis = info["gyro_pelvis"]                          # (B, 3)
        gvec_pelvis = info["gvec_pelvis"]                          # (B, 3)
        joint_qpos = info["qpos"][:, self.qpos_ids_g1]            # (B, 29)
        joint_qvel = info["qvel"][:, self.dof_ids_g1]             # (B, 29)

        # ---- sparse keypoint reference ----
        sparse_pose = ref_state["kpt2gv_pose"][:, self.sparse_kpt_ids]      # (B, K, 4, 4)
        sparse_cvel = ref_state["kpt_cvel_in_gv"][:, self.sparse_kpt_ids]   # (B, K, 6)

        # Preserve upstream local-frame offset handling (all zero for five points).
        # pos' = pos + R @ local_offset ; rotation is unchanged.
        offset_in_gv = np.einsum(
            "bkij,kj->bki",
            sparse_pose[:, :, :3, :3],
            self.sparse_kpt_local_offsets,
        )                                                           # (B, K, 3)
        sparse_pos = sparse_pose[:, :, :3, 3] + offset_in_gv       # (B, K, 3)
        sparse_rot = sparse_pose[:, :, :3, :2]                    # (B, K, 3, 2)
        sparse_vel = sparse_cvel                                   # (B, K, 6)

        # ---- root commands (same as parent) ----
        ref_root_gv_pose = ref_state["kpt2gv_pose"][:, 0]         # (B, 4, 4)
        ref_root_cvel = ref_state["kpt_cvel_in_gv"][:, 0]         # (B, 6)
        yaw_cmd = np.stack(
            [np.cos(info["yaw_d"]), np.sin(info["yaw_d"])], axis=-1
        )                                                          # (B, 2)
        xy_cmd = info["xy_d"]                                      # (B, 2)

        state = np.hstack([
            # proprio
            gyro_pelvis,                                               # 3
            gvec_pelvis,                                               # 3
            (joint_qpos - self._nom_jnt_qpos)[:, self.ctrl_id_obs],   # 29
            joint_qvel[:, self.ctrl_id_obs],                           # 29
            last_action,                                               # 29
            # sparse reference
            sparse_pos.reshape(B, -1),                                 # K*3
            sparse_rot.reshape(B, -1),                                 # K*6
            sparse_vel.reshape(B, -1),                                 # K*6
            # root commands
            ref_root_gv_pose[:, 2, 3][:, None],                       # 1
            -ref_root_gv_pose[:, :3, :3].transpose((0, 2, 1))[..., 2],  # 3
            ref_root_cvel,                                             # 6
            yaw_cmd,                                                   # 2
            xy_cmd,                                                    # 2
        ]).astype(np.float32)

        return state
