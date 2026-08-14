#!/usr/bin/env python3
"""
检查start_folder和end_folder之间的所有文件夹，
如果文件夹中traj_csv_path下的NPZ数量未达到标准，则标记该文件夹，
在程序最后请求是否删除对应文件夹
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tyro


@dataclass
class Args:
    """清理NPZ文件数量不足的文件夹的配置"""
    # ==================== 实验范围 ====================
    start_folder: str = "20260121_111436_01201112_G1-TrackV5_v0_bvh1219PrivOri_['jump']"
    end_folder: str = "20260121_122926_01205757_G1-TrackV5_v0_bvh1219Ori_['jump']"
    
    # ==================== 路径配置 ====================
    output_dir: str = "../play_logs"  # play_logs输出目录
    traj_csv_path: str = "traj_csv"  # NPZ文件所在子目录名称
    
    # ==================== 操作控制 ====================
    auto_delete: bool = False  # 是否自动删除（跳过交互确认）
    dry_run: bool = False  # 仅显示将要删除的文件夹，不实际删除


def get_npz_count(folder: Path, traj_csv_name: str) -> int:
    """
    计算指定文件夹下traj_csv目录中的NPZ文件数量
    
    Args:
        folder: 要检查的文件夹路径
        traj_csv_name: traj_csv目录名称
    
    Returns:
        NPZ文件数量
    """
    traj_csv_path = folder / traj_csv_name
    if not traj_csv_path.exists():
        return 0
    
    npz_files = list(traj_csv_path.glob("*.npz"))
    return len(npz_files)


def check_folders(start_folder: Path, end_folder: Path, traj_csv_name: str) -> Tuple[List[Tuple[Path, int]], int]:
    """
    检查start_folder到end_folder之间的所有文件夹
    
    Args:
        start_folder: 起始文件夹
        end_folder: 结束文件夹
        traj_csv_name: traj_csv目录名称
    
    Returns:
        (未达标的文件夹列表, 最大NPZ数量)
        每个元素为(文件夹路径, NPZ数量)
    """
    parent = start_folder.parent
    folders = sorted([f for f in parent.iterdir() if f.is_dir()])
    
    # 获取start_folder和end_folder在列表中的索引
    try:
        start_idx = folders.index(start_folder)
        end_idx = folders.index(end_folder)
    except ValueError as e:
        raise ValueError(f"无法找到指定的文件夹: {e}")
    
    # 获取范围内的文件夹
    target_folders = folders[start_idx : end_idx + 1]
    
    bad_folders = []
    max_npz = 0
    for folder in target_folders:
        npz_count = get_npz_count(folder, traj_csv_name)
        max_npz = max(max_npz, npz_count)
        print(f"检查文件夹: {folder.name}")
        print(f"  - NPZ文件数量: {npz_count}")
        
        if npz_count < max_npz:
            status = "未达标"
            bad_folders.append((folder, npz_count))
        else:
            status = "达标"
        
        print(f"  - 状态: {status}\n")
    
    return bad_folders, max_npz


def main():
    args: Args = tyro.cli(Args)
    
    # 构建完整路径
    output_root = Path(args.output_dir)
    start_folder = output_root / args.start_folder
    end_folder = output_root / args.end_folder
    
    # 验证文件夹存在
    if not start_folder.exists():
        raise FileNotFoundError(f"起始文件夹不存在: {start_folder}")
    if not end_folder.exists():
        raise FileNotFoundError(f"结束文件夹不存在: {end_folder}")
    
    # 验证文件夹在同一目录下
    if start_folder.parent != end_folder.parent:
        raise ValueError("起始文件夹和结束文件夹必须在同一目录下")
    
    print(f"检查范围: {start_folder.name} 到 {end_folder.name}")
    print(f"父目录: {start_folder.parent}")
    print(f"traj_csv目录名称: {args.traj_csv_path}")
    if args.dry_run:
        print("模式: 仅显示（不实际删除）")
    print("=" * 80)
    print()
    
    # 检查文件夹，自动获取最大NPZ数量作为标准
    bad_folders, max_npz = check_folders(start_folder, end_folder, args.traj_csv_path)
    
    # 输出结果
    print("=" * 80)
    print(f"检查完成！共检查了 {len([f for f in (start_folder.parent).iterdir() if f.is_dir()])} 个文件夹")
    print(f"自动设置的最小NPZ数量标准（最大值）: {max_npz}")
    print(f"未达标的文件夹数量: {len(bad_folders)}")
    print()
    
    if bad_folders:
        print("未达标的文件夹列表:")
        for folder, count in bad_folders:
            print(f"  - {folder.name} (NPZ数量: {count})")
        print()
        
        # 如果是dry_run模式，只显示不删除
        if args.dry_run:
            print("DRY RUN模式：以上文件夹将被删除，但实际未执行删除操作。")
            return
        
        # 请求确认删除
        if not args.auto_delete:
            response = input("是否删除这些文件夹？(yes/no): ").strip().lower()
            if response not in ["yes", "y"]:
                print("取消删除操作。")
                return
        
        # 删除文件夹
        print("开始删除文件夹...")
        for folder, count in bad_folders:
            print(f"  删除: {folder.name}")
            try:
                shutil.rmtree(folder)
                print(f"    ✓ 已删除")
            except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError) as e:
                print(f"    ✗ 删除失败: {e}")
        
        print("\n删除操作完成！")
    else:
        print("所有文件夹都达到了NPZ数量标准，无需删除。")


if __name__ == "__main__":
    main()
