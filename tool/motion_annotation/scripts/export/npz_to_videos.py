"""
python npz_export_videos.py \
  --name_dir "../play_logs/example/c1" \
  --npz_dir  "data/mocap/bvh_1219" \
  --out_dir  "data/video_source/sim_c1" \
  --recursive_npz


"""

import re

import os
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco
from mujoco import Renderer

# play_track.py 用的是你工程里的 images_to_video
# 如果你想完全独立运行，可把它替换成 imageio 或 ffmpeg 管道
from galbot_mj.common.video import images_to_video


# ====== 可配置项 ======
DEFAULT_SCENE_XML = "assets/unitree_g1/scene_mjx.xml"


_RE_NAME = re.compile(r"^\d{8}_\d{6}_(.+?)\.npz(?:\.mp4)?$")
# 更具体的解析，提取关键信息
# _RE_NAME = re.compile(r'''
#     ^                           # 开始
#     \d{8}_\d{6}_                # 日期时间前缀
#     (                           # 捕获组开始：名字部分
#         Take\ \d{4}-\d{2}-\d{2}\ \d{2}\.\d{2}\.\d{2}\ [AP]M_  # Take部分
#         [^_]+_                  # 用户名（下划线前）
#         \d+Hz_                  # 频率
#         \d+dof                  # 自由度
#         -\d+to\d+               # 帧范围
#     )                           # 捕获组结束
#     \.npz(?:\.mp4)?$           # 后缀
# ''', re.VERBOSE)

def key_from_name_file(p: Path) -> str:
    """
    20251219_104616_1028_walk_omni-0-0to6480-0to6480.npz.mp4
    -> 1028_walk_omni-0-0to6480-0to6480
    """
    match = _RE_NAME.fullmatch(p.name)
    if match is None:
        raise ValueError(f"invalid timestamped trajectory filename: {p.name}")
    return match.group(1)

def strip_known_suffixes(p: Path, known=(".mp4", ".npz", ".npy", ".csv")) -> str:
    """
    对于 xxx.npz.mp4 / xxx.mp4 / xxx.npz 这种，剥离多重后缀，得到稳定的 base name。
    """
    name = p.name
    changed = True
    while changed:
        changed = False
        for s in known:
            if name.endswith(s):
                name = name[: -len(s)]
                changed = True
    return name


def build_npz_index(npz_dir: Path, recursive: bool = True) -> dict[str, Path]:
    """
    在 npz_dir 里建立 {stem: path} 索引，stem 用 strip_known_suffixes 规则。
    """
    files = npz_dir.rglob("*.npz") if recursive else npz_dir.glob("*.npz")
    mp = {}
    for f in files:
        k = strip_known_suffixes(f, known=(".npz",))
        if k in mp:
            raise ValueError(f"duplicate trajectory key {k!r}: {mp[k]} and {f}")
        mp[k] = f
    return mp


def qpos_to_sim_qpos(qpos_frame: np.ndarray) -> np.ndarray:
    """
    兼容 npz_check.py 的两种关节模式：
    - (7+23)=30 维：需要映射成 (7+29)=36 的模型 qpos（根+29）
    - (7+29)=36 维：直接用
    """
    qpos_frame = np.asarray(qpos_frame).reshape(-1)
    if qpos_frame.shape[0] == 7 + 29:
        return qpos_frame

    if qpos_frame.shape[0] == 7 + 23:
        # 与 npz_check.py 的 66155 映射一致 :contentReference[oaicite:2]{index=2}
        qpos_23 = qpos_frame
        qpos_29 = np.zeros((29,), dtype=qpos_23.dtype)
        qpos_29[:7] = qpos_23[:7]
        # 23d -> 29d indices（和你 npz_check 保持一致）
        idx29 = [
            0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11,
            12,
            15, 16, 17, 18, 19,
            22, 23, 24, 25, 26
        ]
        qpos_29[idx29] = qpos_23[7:]
        return np.concatenate([qpos_23[:7], qpos_29], axis=0)

    raise ValueError(f"Unsupported qpos dim: {qpos_frame.shape[0]} (expect 36 or 30)")


def update_camera_front(viewer_cam, pelvis_pos: np.ndarray, pelvis_rot: np.ndarray):
    """
    参考 play_track 的“正对机器人正面”相机逻辑 :contentReference[oaicite:3]{index=3}
    """
    distance = 2.0
    height_offset = 1.

    forward_vector = pelvis_rot[:, 0]
    cam_pos = pelvis_pos - forward_vector * distance
    cam_pos[2] += height_offset

    lookat = pelvis_pos.copy()
    lookat[2] += 0.

    # Renderer 的 camera 是 mujoco.MjvCamera
    viewer_cam.lookat[:] = lookat
    viewer_cam.distance = distance
    viewer_cam.elevation = -20.0
    viewer_cam.azimuth = 90.0


def render_npz_to_video(
    npz_path: Path,
    out_mp4: Path,
    scene_xml: str = DEFAULT_SCENE_XML,
    fps: int = 50,
    width: int = 640,
    height: int = 480,
    max_frames: int | None = None,
):
    data_npz = np.load(npz_path, allow_pickle=True)
    qpos_seq = data_npz["qpos"]
    if max_frames is not None:
        qpos_seq = qpos_seq[:max_frames]

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    renderer = Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()

    pelvis_id = model.body("pelvis").id

    buf = []
    for i in range(qpos_seq.shape[0]):
        sim_qpos = qpos_to_sim_qpos(qpos_seq[i])
        data.qpos[:] = sim_qpos
        mujoco.mj_forward(model, data)

        pelvis_pos = data.xpos[pelvis_id].copy()
        pelvis_rot = data.xmat[pelvis_id].reshape(3, 3).copy()
        update_camera_front(cam, pelvis_pos, pelvis_rot)

        renderer.update_scene(data, cam)
        buf.append(renderer.render())

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    images_to_video(buf, str(out_mp4), fps=fps, color_format="RGB")


@dataclass
class Args:
    # A：读“名字”的文件夹（可以是 mp4 / npz.mp4 / 任意文件，只用来取 base name）
    name_dir: str
    # B：真正存放 npz 的文件夹
    npz_dir: str
    # 输出视频目录
    out_dir: str

    scene_xml: str = DEFAULT_SCENE_XML
    recursive_npz: bool = True

    fps: int = 50
    width: int = 640
    height: int = 480
    max_frames: int | None = None  # 调试用：限制渲染帧数


def main(args: Args):
    name_dir = Path(args.name_dir)
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)

    npz_index = build_npz_index(npz_dir, recursive=args.recursive_npz)

    name_files = sorted([p for p in name_dir.iterdir() if p.is_file()])
    if not name_files:
        raise RuntimeError(f"name_dir is empty: {name_dir}")

    missing = []
    ok = 0

    for f in name_files:
        key = key_from_name_file(f)

        npz_path = npz_index.get(key, None)
        if npz_path is None:
            missing.append(key)
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_mp4 = out_dir / f"{ts}_{key}.npz.mp4"
        print(f"[OK] {key}\n  name_file: {f}\n  npz      : {npz_path}\n  out      : {out_mp4}")
        render_npz_to_video(
            npz_path=npz_path,
            out_mp4=out_mp4,
            scene_xml=args.scene_xml,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
        )
        ok += 1

    print(f"\nDone. rendered={ok}, missing={len(missing)}")
    if missing:
        print("Missing keys (show first 50):")
        for k in missing[:50]:
            print("  -", k)


if __name__ == "__main__":
    # 不强依赖 tyro，避免你环境里没有；你也可以改回 tyro.cli
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--name_dir", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scene_xml", default=DEFAULT_SCENE_XML)
    ap.add_argument("--recursive_npz", action="store_true")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max_frames", type=int, default=0)
    ns = ap.parse_args()

    args = Args(
        name_dir=ns.name_dir,
        npz_dir=ns.npz_dir,
        out_dir=ns.out_dir,
        scene_xml=ns.scene_xml,
        recursive_npz=ns.recursive_npz,
        fps=ns.fps,
        width=ns.width,
        height=ns.height,
        max_frames=(None if ns.max_frames <= 0 else ns.max_frames),
    )
    main(args)
