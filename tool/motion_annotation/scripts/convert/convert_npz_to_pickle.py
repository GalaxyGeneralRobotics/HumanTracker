import numpy as np
import pickle
import sys
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ==============================
# 配置 link 层级和 link_body_list
# ==============================
link_body_list = [
    'pelvis',
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'left_toe_link',
    'pelvis_contour_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'right_toe_link',
    'waist_yaw_link', 'waist_roll_link', 'torso_link', 'head_link', 'head_mocap', 'imu_in_torso',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_rubber_hand',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand'
]

# 关节父子层级定义
parent_map = {
    0: None,  # pelvis root
    1: 0, 2: 1, 3: 2, 4:3, 5:4, 6:5, 7:6,
    8:0,
    9:0, 10:9, 11:10, 12:11, 13:12, 14:13, 15:14,
    16:0, 17:16, 18:17, 19:18, 20:19, 21:18,
    22:18, 23:22, 24:23, 25:24, 26:25, 27:26, 28:27, 29:28,
    30:18, 31:30, 32:31, 33:32, 34:33, 35:34, 36:35, 37:36
}

def convert_npz_to_pickle(npz_file, output_dir):
    """
    将单个 npz 文件转换为 pkl 文件

    Args:
        npz_file: npz 文件路径
        output_dir: 输出 pkl 的目录

    Returns:
        输出 pickle 文件路径
    """
    # 加载 npz 文件
    data = np.load(npz_file, allow_pickle=True)
    qpos_all = data['qpos']        # shape (frame, 36)

    # fps 从 frequency 获得
    fps = float(data['frequency'])

    # ==============================
    # 提取 root 和 dof
    # ==============================
    root_pos = qpos_all[:, 0:3]      # 3维
    # 原始 npz 中 qpos 存的是 w,x,y,z 顺序（wxyz），先读取为 wxyz，然后重排为 xyzw（scipy 需要 xyzw）
    root_rot_wxyz = qpos_all[:, 3:7]      # wxyz
    root_rot = root_rot_wxyz[:, [1, 2, 3, 0]]  # 转为 xyzw (x, y, z, w)
    dof_pos = qpos_all[:, 7:]        # 29维

    num_frames = root_pos.shape[0]
    num_links = len(link_body_list)

    # ==============================
    # 初始化 local_body_pos
    # ==============================
    local_body_pos = np.zeros((num_frames, num_links, 3), dtype=np.float32)

    # ==============================
    # 假设初始偏移为第一帧的 world pos
    # ==============================
    # 先构建 world pos 数组
    world_pos = np.zeros((num_frames, num_links, 3), dtype=np.float32)

    # 对于 pelvis(root)
    world_pos[:, 0, :] = root_pos

    # 对其他 link
    for i in range(1, num_links):
        parent = parent_map[i]
        if parent is None:
            continue
        # 简单假设每帧相对父节点的偏移固定，用第一帧计算
        offset = np.zeros(3, dtype=np.float32)
        if i < dof_pos.shape[1]:  # 如果有对应的关节自由度
            offset = dof_pos[0, i-1:i+2] if (i-1+3)<=dof_pos.shape[1] else np.zeros(3)
        # 第一帧 world pos
        world_pos[0, i, :] = world_pos[0, parent, :] + offset

    # 后续帧沿用上一帧偏移（不考虑 joint 旋转）
    for f in range(1, num_frames):
        for i in range(1, num_links):
            parent = parent_map[i]
            if parent is None:
                continue
            offset = world_pos[0, i, :] - world_pos[0, parent, :]
            world_pos[f, i, :] = world_pos[f, parent, :] + offset

    # ==============================
    # 将 world pos 转换为 root 局部坐标
    # ==============================
    for f in range(num_frames):
        r_rot = R.from_quat(root_rot[f])  # xyzw
        local_body_pos[f] = r_rot.inv().apply(world_pos[f] - root_pos[f])

    # ==============================
    # 保存为 pkl
    # ==============================
    pkl_data = {
        'fps': fps,
        'root_pos': root_pos.astype(np.float32),
        'root_rot': root_rot.astype(np.float32),
        'dof_pos': dof_pos.astype(np.float32),
        'local_body_pos': local_body_pos,
        'link_body_list': link_body_list
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名（保持原 npz 的名称，但扩展名改为 .pkl）
    npz_filename = os.path.basename(npz_file)
    pkl_filename = os.path.splitext(npz_filename)[0] + '.pkl'
    output_path = os.path.join(output_dir, pkl_filename)

    with open(output_path, "wb") as f:
        pickle.dump(pkl_data, f)

    return output_path


def main():
    """主函数：接受单个文件或目录路径"""
    if len(sys.argv) < 2:
        print("Usage: python convert_npz_to_pickle.py <npz_file_or_directory>")
        print("Example (single file): python convert_npz_to_pickle.py /path/to/file.npz")
        print("Example (directory):   python convert_npz_to_pickle.py /path/to/npz_folder")
        sys.exit(1)

    input_path = sys.argv[1]
    input_path = os.path.abspath(input_path)

    # 如果是单个文件
    if os.path.isfile(input_path):
        if not input_path.lower().endswith('.npz'):
            print(f"错误：指定的文件不是 .npz 文件：{input_path}")
            sys.exit(1)
        npz_files = [Path(input_path)]
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        parent_dir = os.path.dirname(input_path)
        output_dir = parent_dir

    # 如果是目录
    elif os.path.isdir(input_path):
        npz_files = list(Path(input_path).glob('*.npz'))
        if not npz_files:
            print(f"错误：文件夹 '{input_path}' 中没有找到 .npz 文件")
            sys.exit(1)
        input_dir_name = os.path.basename(input_path.rstrip('/').rstrip('\\'))
        parent_dir = os.path.dirname(input_path)
        output_dir = os.path.join(parent_dir, input_dir_name + '_converted_pkl')

    else:
        print(f"错误：路径 '{input_path}' 不存在")
        sys.exit(1)

    print(f"找到 {len(npz_files)} 个 npz 文件:")
    for npz_file in npz_files:
        print(f"  - {npz_file.name}")
    print()

    # 批量转换（单文件时也是这个流程）
    successful = []

    for npz_file in npz_files:
        print(f"正在转换: {npz_file.name} ...", end=" " )
        result = convert_npz_to_pickle(str(npz_file), output_dir)
        print("✓")
        successful.append((npz_file.name, result))

    # 输出转换结果
    print()
    print("=" * 60)
    print("转换完成")
    print("=" * 60)
    print(f"\n成功转换 ({len(successful)} 个):")
    for npz_name, pkl_path in successful:
        pkl_name = os.path.basename(pkl_path)
        print(f"  {npz_name} → {pkl_name}")


    print(f"\n输出目录: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
