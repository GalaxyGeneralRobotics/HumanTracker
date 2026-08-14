import numpy as np
import pickle
from pathlib import Path

# -----------------------------------
# 配置
# -----------------------------------
npz_path = "data/mocap/walk.npz"     # 输入 npz
pkl_path = "data/parc/walking/walk_000.pkl"    # 输出 pkl

SRC_FPS = 50.0   # npz 采样率
TGT_FPS = 30.0   # pkl 目标采样率
LOOP_MODE = "CLAMP"  # 或 "WRAP"，按你任务需求改


# ===============================
# 工具：quat(wxyz) → exp map(so3)
# ===============================
def quat_to_expmap(q):
    """
    q: (...,4)  wxyz
    return: (...,3)  exponential-map representation
    """
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))

    axis = np.zeros(q.shape[:-1] + (3,), dtype=q.dtype)
    small = sin_half < 1e-8
    # 正常情况
    axis[~small] = np.stack([x, y, z], axis=-1)[~small] / sin_half[~small][..., None]
    # 非常小角度时，axis 不重要，直接给 0
    axis[small] = 0.0

    return axis * angle[..., None]


# ===============================
# 1) 加载 npz
# ===============================
npz = np.load(npz_path, allow_pickle=True)
qpos_raw = npz["qpos"]              # (T, 7 + dof_src)
T, D = qpos_raw.shape
dof_src = D - 7

print(f"Loaded npz: {npz_path}")
print(f"qpos shape: {qpos_raw.shape}, dof_src = {dof_src}")

if "foot_contact" in npz:
    foot_contact_raw = npz["foot_contact"]  # (T, 2) [left, right]
    print(f"foot_contact shape: {foot_contact_raw.shape}")
else:
    print("[WARN] npz 中没有 foot_contact，默认全 0")
    foot_contact_raw = np.zeros((T, 2), dtype=np.float32)


# ===============================
# 2) 从 50Hz 重采样到 20Hz (最近邻)
# ===============================
src_dt = 1.0 / SRC_FPS
t_max = (T - 1) * src_dt

# 目标时间轴 [0, t_max] 按 1/20s 采样
tgt_times = np.arange(0.0, t_max + 1e-8, 1.0 / TGT_FPS)
K = tgt_times.shape[0]
print(f"Target frames: {K} at {TGT_FPS}Hz")

# 每个目标时刻，映射到原 50Hz 的最近帧索引
src_indices = np.round(tgt_times * SRC_FPS).astype(int)
src_indices = np.clip(src_indices, 0, T - 1)

qpos = qpos_raw[src_indices]              # (K, 7 + dof_src)
foot_contact = foot_contact_raw[src_indices]  # (K, 2)


# ===============================
# 3) 处理关节 DoF 维度：23 → 29 或直接 29
#    (参考 npz_check 里的 mapping)
# ===============================
if dof_src == 29:
    # 已经是 29 DoF
    joint_dof_29 = qpos[:, 7:]  # (K, 29)
elif dof_src == 23:
    # 23 → 29 的映射，参考你的 npz_check 代码
    # 23 维 joint_dof 的顺序与 npz_check.joint_names['66155'] 一致
    joint_23 = qpos[:, 7:]      # (K, 23)
    joint_dof_29 = np.zeros((K, 29), dtype=joint_23.dtype)

    # 映射关系同 npz_check:
    # （这里 index 是“29 DoF 空间中的下标”）
    idx_29_for_23 = [
        0, 1, 2, 3, 4, 5,           # 左腿 6
        6, 7, 8, 9, 10, 11,         # 右腿 6
        12,                         # 腰 yaw
        15, 16, 17, 18, 19,         # 左臂 5
        22, 23, 24, 25, 26          # 右臂 5
    ]
    assert len(idx_29_for_23) == 23

    joint_dof_29[:, idx_29_for_23] = joint_23
else:
    raise ValueError(f"Unsupported dof_src={dof_src}, expected 23 or 29")

num_dof = joint_dof_29.shape[1]
print(f"joint_dof_29 shape: {joint_dof_29.shape}")


# ===============================
# 4) root_pos, root_rot(expmap)
# ===============================
root_pos = qpos[:, :3]        # (K, 3)
root_quat = qpos[:, 3:7]      # (K, 4), wxyz
root_rot_exp = quat_to_expmap(root_quat)  # (K, 3)

frames = np.concatenate([root_pos, root_rot_exp, joint_dof_29], axis=-1).astype(np.float32)
print(f"frames shape: {frames.shape}  # = (K, 3+3+{num_dof})")


# ===============================
# 5) contacts 扩展为全身 29 维
#    foot_contact: (K, 2) [left, right]
#    这里选择: 左脚接触 → 左腿三个 DOF（knee, ankle_pitch, ankle_roll）
#             右脚接触 → 右腿三个 DOF
#    索引依据 29DoF joint_names['66377'] 的顺序：
#    0~5 左腿, 6~11 右腿, 12~14 腰, 后面是手臂与手腕
# ===============================
contacts = np.zeros((K, num_dof), dtype=np.float32)

left_fc = foot_contact[:, 0:1]   # (K,1)
right_fc = foot_contact[:, 1:2]  # (K,1)

# 左腿：3,4,5 = left_knee, left_ankle_pitch, left_ankle_roll
contacts[:, 3:6] = left_fc

# 右腿：9,10,11 = right_knee, right_ankle_pitch, right_ankle_roll
contacts[:, 9:12] = right_fc


# ===============================
# 6) 组装成 MotionLib 所需 pkl dict
# ===============================
motion = {
    "fps": TGT_FPS,           # 目标 20Hz
    "loop_mode": LOOP_MODE,   # "CLAMP" 或 "WRAP"
    "frames": frames,         # (K, 3+3+29)
    "contacts": contacts,     # (K, 29)；全身标志位（腿随脚接触）
    "hf_mask_inds": None,     # 平地
    "terrain": None,          # 平地
}

# ===============================
# 7) 保存为 pkl
# ===============================
with open(pkl_path, "wb") as f:
    pickle.dump(motion, f)

print(f"Saved MotionLib pkl to: {pkl_path}")
