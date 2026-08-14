r"""
python npz_export_videos.py \
  --csv_path data/names.csv \  # or --name_dir data/name_dir \
  --npz_dir  data/mocap/qzk_test \
  --out_dir  data/video_source/f1 \
  --recursive_npz

python npz_export_videos_from_csv.py \
  --csv_path "../play_logs/example/metrics_csv/cluster_min_values.csv" \
  --npz_dir  "data/mocap/bvh_1219" \
  --out_dir "data/video_source" \
  --recursive_npz

python npz_export_videos_from_csv.py \
  --csv_path "../play_logs/example/metrics_csv/failed_trajectories_comparison.csv" \
  --npz_dir  "data/mocap/train_c" \
  --out_dir  "data/video_source/failed" \
  --recursive_npz
"""

r"""
python npz_export_videos.py \
  --csv_path data/names.csv \  # or --name_dir data/name_dir \
  --npz_dir  data/mocap/qzk_test \
  --out_dir  data/video_source/f1 \
  --recursive_npz


"""

import re
import csv

import os
import time
from pathlib import Path
import gc
import subprocess
import shutil
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

def keys_from_csv(csv_path: Path, col: str | None = None) -> list[str]:
    """Read trajectory/npz names from a CSV file.

    - If `col` is None, will try common column names in order:
      file_name, npz_name, name, names.
    - Values may include suffixes like .npz/.mp4; we normalize with strip_known_suffixes.
    """
    import csv as _csv

    with csv_path.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        cols = [c.strip() for c in reader.fieldnames if c is not None]
        if col is None:
            for cand in ("file_name", "npz_name", "name", "names"):
                if cand in cols:
                    col = cand
                    break
        if col is None or col not in cols:
            raise RuntimeError(
                f"Cannot find name column in CSV. available={cols}, requested={col}"
            )

        out: list[str] = []
        for row in reader:
            v = (row.get(col) or "").strip()
            if not v:
                continue
            out.append(strip_known_suffixes(Path(v)))
        return out


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


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _encode_with_ffmpeg(frames_iter, out_mp4: Path, *, fps: int, width: int, height: int, crf: int = 23, preset: str = "veryfast"):
    """Stream RGB frames to ffmpeg (no frame buffering)."""
    _ensure_parent(out_mp4)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH. Install it or use --encoder cv2 / images_to_video.")
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames_iter:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError(f"Frame shape mismatch: got {frame.shape}, expect {(height, width, 3)}")
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}")


def _encode_with_cv2(frames_iter, out_mp4: Path, *, fps: int, width: int, height: int):
    """Stream frames with cv2.VideoWriter."""
    import cv2
    _ensure_parent(out_mp4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (width, height))
    try:
        for frame in frames_iter:
            # cv2 expects BGR
            bgr = frame[..., ::-1]
            vw.write(bgr)
    finally:
        vw.release()


def render_npz_to_video(
    npz_path: Path,
    out_mp4: Path,
    scene_xml: str = DEFAULT_SCENE_XML,
    fps: int = 50,
    width: int = 640,
    height: int = 480,
    max_frames: int | None = None,
    *,
    encoder: str = "ffmpeg",
    ffmpeg_crf: int = 23,
    ffmpeg_preset: str = "veryfast",
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

    def frames():
        for i in range(qpos_seq.shape[0]):
            sim_qpos = qpos_to_sim_qpos(qpos_seq[i])
            data.qpos[:] = sim_qpos
            mujoco.mj_forward(model, data)

            pelvis_pos = data.xpos[pelvis_id].copy()
            pelvis_rot = data.xmat[pelvis_id].reshape(3, 3).copy()
            update_camera_front(cam, pelvis_pos, pelvis_rot)

            renderer.update_scene(data, cam)
            img = renderer.render()
            yield img

    try:
        enc = encoder.lower().strip()
        if enc == "ffmpeg":
            _encode_with_ffmpeg(frames(), out_mp4, fps=fps, width=width, height=height, crf=ffmpeg_crf, preset=ffmpeg_preset)
        elif enc == "cv2":
            _encode_with_cv2(frames(), out_mp4, fps=fps, width=width, height=height)
        elif enc == "images_to_video":
            # 兼容旧逻辑：会缓存所有帧，容易 OOM
            buf = list(frames())
            _ensure_parent(out_mp4)
            images_to_video(buf, str(out_mp4), fps=fps, color_format="RGB")
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
    finally:
        try:
            renderer.close()
        except Exception:
            pass
        del renderer, data, model
        gc.collect()


@dataclass
class Args:
    # A1：读“名字”的文件夹（可以是 mp4 / npz.mp4 / 任意文件，只用来取 base name）
    # 也可以不传，改用 csv_path
    name_dir: str = ""

    # A2：从 CSV 读取 name 列（优先于 name_dir）
    csv_path: str = ""
    csv_col: str = ""

    # B：真正存放 npz 的文件夹
    npz_dir: str = ""
    # 输出视频目录
    out_dir: str = ""

    scene_xml: str = DEFAULT_SCENE_XML
    recursive_npz: bool = True

    fps: int = 50
    width: int = 640
    height: int = 480
    max_frames: int | None = None  # 调试用：限制渲染帧数

    # 导出控制
    max_videos: int = 0          # <=0 表示不限制
    skip_existing: bool = True  # out_mp4 已存在则跳过
    sleep_ms: int = 0           # 每导出一个视频后 sleep，缓解系统卡顿

    # 编码方式：ffmpeg（推荐，流式写入，低内存）/ cv2 / images_to_video（会占用大量内存）
    encoder: str = "ffmpeg"
    ffmpeg_crf: int = 23        # ffmpeg 编码质量（更小更清晰更大）
    ffmpeg_preset: str = "veryfast"

def main(args: Args):
    name_dir = Path(args.name_dir)
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)

    npz_index = build_npz_index(npz_dir, recursive=args.recursive_npz)

    # 读取 keys：优先 csv_path，其次 name_dir
    keys: list[str] = []
    if args.csv_path:
        keys = keys_from_csv(Path(args.csv_path), col=(args.csv_col or None))
    else:
        name_files = sorted([p for p in name_dir.iterdir() if p.is_file()])
        if not name_files:
            raise RuntimeError(f"name_dir is empty: {name_dir}")
        keys = [key_from_name_file(f) for f in name_files]

    missing = []
    ok = 0

    for key in keys:
        if args.max_videos > 0 and ok >= args.max_videos:
            break

        npz_path = npz_index.get(key, None)
        if npz_path is None:
            missing.append(key)
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_mp4 = out_dir / f"{ts}_{key}.npz.mp4"
        if args.skip_existing and out_mp4.exists():
            print(f"[SKIP] exists: {out_mp4}")
            continue
        print(f"[OK] {key}\n  npz      : {npz_path}\n  out      : {out_mp4}")
        render_npz_to_video(
            npz_path=npz_path,
            out_mp4=out_mp4,
            scene_xml=args.scene_xml,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            encoder=args.encoder,
            ffmpeg_crf=args.ffmpeg_crf,
            ffmpeg_preset=args.ffmpeg_preset,
        )
        ok += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"\nDone. rendered={ok}, missing={len(missing)}")
    if missing:
        print("Missing keys (show first 50):")
        for k in missing[:50]:
            print("  -", k)


if __name__ == "__main__":
    # 不强依赖 tyro，避免你环境里没有；你也可以改回 tyro.cli
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--name_dir", default="")
    ap.add_argument("--csv_path", default="")
    ap.add_argument("--csv_col", default="")
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scene_xml", default=DEFAULT_SCENE_XML)
    ap.add_argument("--recursive_npz", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--skip_existing", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--sleep_ms", type=int, default=0)
    ap.add_argument("--encoder", default="ffmpeg", choices=["ffmpeg","cv2","images_to_video"])
    ap.add_argument("--ffmpeg_crf", type=int, default=23)
    ap.add_argument("--ffmpeg_preset", default="veryfast")
    ns = ap.parse_args()

    if not ns.csv_path and not ns.name_dir:
        raise SystemExit("Need --csv_path or --name_dir")

    args = Args(
        name_dir=ns.name_dir,
        csv_path=ns.csv_path,
        csv_col=ns.csv_col,
        npz_dir=ns.npz_dir,
        out_dir=ns.out_dir,
        scene_xml=ns.scene_xml,
        recursive_npz=ns.recursive_npz,
        fps=ns.fps,
        width=ns.width,
        height=ns.height,
        max_frames=(None if ns.max_frames <= 0 else ns.max_frames),
        max_videos=ns.max_videos,
        skip_existing=ns.skip_existing,
        sleep_ms=ns.sleep_ms,
        encoder=ns.encoder,
        ffmpeg_crf=ns.ffmpeg_crf,
        ffmpeg_preset=ns.ffmpeg_preset,
    )
    main(args)
