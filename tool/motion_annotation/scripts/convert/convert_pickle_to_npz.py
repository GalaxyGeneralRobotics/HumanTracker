import numpy as np
import pickle
import sys
import os
from pathlib import Path


def quaternion_xyzw_to_wxyz(quat_xyzw):
    """
    将四元数从XYZW格式转换为WXYZ格式
    Args:
        quat_xyzw: 形状为(4,)或(n,4)的数组，XYZW格式的四元数
    Returns:
        quat_wxyz: WXYZ格式的四元数
    """
    if len(quat_xyzw.shape) == 1:
        # 单个四元数 [x, y, z, w] -> [w, x, y, z]
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    else:
        # 多个四元数 (n, 4)
        quat_wxyz = np.zeros_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]  # w
        quat_wxyz[:, 1] = quat_xyzw[:, 0]  # x
        quat_wxyz[:, 2] = quat_xyzw[:, 1]  # y
        quat_wxyz[:, 3] = quat_xyzw[:, 2]  # z
        return quat_wxyz


def convert_pickle_to_npz(pkl_file, output_dir=None):
    """
    将单个 pkl 文件转换为 npz 文件
    
    Args:
        pkl_file: pkl 文件路径
        output_dir: 输出 npz 的目录，如果为None则使用脚本同目录
    
    Returns:
        输出 NPZ 文件路径
    """
    # 加载 pkl 文件
    with open(pkl_file, 'rb') as f:
        pkl_data = pickle.load(f)

    # 读取数据
    root_pos = pkl_data['root_pos']
    root_rot_xyzw = pkl_data['root_rot']  # XYZW 格式
    dof_pos = pkl_data['dof_pos']

    # 将四元数从 XYZW 格式转换为 WXYZ 格式
    root_rot_wxyz = quaternion_xyzw_to_wxyz(root_rot_xyzw)

    # 拼接 qpos: root_pos (3维) + root_rot (4维 WXYZ) + dof_pos (23或29维)
    qpos = np.hstack((root_pos, root_rot_wxyz, dof_pos))

    # 准备导出数据
    export_data = {
        'qpos': qpos,
        'frequency': float(pkl_data['fps'])
    }

    # 确保输出目录存在
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件名（保持原 pkl 的名称，但扩展名改为 .npz）
    pkl_filename = os.path.basename(pkl_file)
    npz_filename = os.path.splitext(pkl_filename)[0] + '.npz'
    output_path = os.path.join(output_dir, npz_filename)

    # 导出
    np.savez(output_path, **export_data)

    return output_path



def main():
    """主函数：接受单个文件或目录路径"""
    if len(sys.argv) < 2:
        print("Usage: python convert_npz_to_pickle.py <pkl_file_or_directory>")
        print("Example (single file): python convert_pickle_to_npz.py /path/to/file.pkl")
        print("Example (directory):   python convert_pickle_to_npz.py /path/to/pkl_folder")
        sys.exit(1)

    input_path = sys.argv[1]
    input_path = os.path.abspath(input_path)

    # 如果是单个文件
    if os.path.isfile(input_path):
        if not input_path.lower().endswith('.pkl'):
            print(f"错误：指定的文件不是 .pkl 文件：{input_path}")
            sys.exit(1)
        pkl_files = [Path(input_path)]
        parent_dir = os.path.dirname(input_path)
        output_dir = parent_dir

    # 如果是目录
    elif os.path.isdir(input_path):
        pkl_files = list(Path(input_path).glob('*.pkl'))
        if not pkl_files:
            print(f"错误：文件夹 '{input_path}' 中没有找到 .pkl 文件")
            sys.exit(1)
        input_dir_name = os.path.basename(input_path.rstrip('/').rstrip('\\'))
        parent_dir = os.path.dirname(input_path)
        output_dir = os.path.join(parent_dir, input_dir_name + '_converted_npz')

    else:
        print(f"错误：路径 '{input_path}' 不存在")
        sys.exit(1)

    print(f"找到 {len(pkl_files)} 个 pkl 文件:")
    for pkl_file in pkl_files:
        print(f"  - {pkl_file.name}")
    print()

    # 批量转换
    successful = []

    for pkl_file in pkl_files:
        print(f"正在转换: {pkl_file.name} ...", end=" " )
        result = convert_pickle_to_npz(str(pkl_file), output_dir)
        print("✓")
        successful.append((pkl_file.name, result))

    # 输出转换结果
    print()
    print("=" * 60)
    print("转换完成")
    print("=" * 60)
    print(f"\n成功转换 ({len(successful)} 个):")
    for pkl_name, npz_path in successful:
        npz_name = os.path.basename(npz_path)
        print(f"  {pkl_name} → {npz_name}")


    print(f"\n输出目录: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()