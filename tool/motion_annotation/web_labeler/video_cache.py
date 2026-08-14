"""NPZ -> MP4 rendering with disk caching.

Renders tracker motion (normal color) + ref motion (red, semi-transparent)
in the same video using a dual-robot MuJoCo scene.
Ref motion is loaded from the original mocap dataset (source_motion_path).
"""

from __future__ import annotations

import hashlib
import fcntl
import os
import subprocess
import shutil
import threading
from pathlib import Path

# MuJoCo chooses and imports its GL backend during module import.  Keep this
# before ``import mujoco`` so direct module users also work without DISPLAY.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from mujoco import Renderer

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1"
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_XML = str(
    _REPO_ROOT
    / "storage"
    / "assets"
    / "unitree_g1"
    / "scene_mjx_track_papergray.xml"
)
DEFAULT_G1_XML = str(_ASSETS_DIR / "g1.xml")
DEFAULT_FPS = 50
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

REF_RGBA = [0.8, 0.1, 0.1, 0.4]  # red, semi-transparent

# ---------------------------------------------------------------------------
# Shared render context cache.
#
# Building the dual-robot model (MjSpec compile) costs ~1.7s and creating a
# Renderer costs ~0.7s. Re-doing this on every clip dominates render time, so
# we build them once per (scene, size) and reuse them. MuJoCo's Renderer and
# its EGL context are NOT thread-safe and are bound to the thread that created
# them, so all rendering is serialized through _RENDER_LOCK and is expected to
# run on a single dedicated worker thread.
# ---------------------------------------------------------------------------
_RENDER_LOCK = threading.Lock()
_CTX_CACHE: dict = {}


class _RenderCtx:
    __slots__ = ("model", "data", "renderer", "cam", "pelvis_id", "ref_pelvis_id")

    def __init__(self, scene_xml: str, width: int, height: int):
        self.model = _build_dual_model(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.pelvis_id = self.model.body("pelvis").id
        self.ref_pelvis_id = self.model.body("ref_pelvis").id


def _get_ctx(scene_xml: str, width: int, height: int) -> "_RenderCtx":
    key = (scene_xml, width, height)
    ctx = _CTX_CACHE.get(key)
    if ctx is None:
        ctx = _RenderCtx(scene_xml, width, height)
        _CTX_CACHE[key] = ctx
    return ctx


def cache_filename(npz_path: Path | str, start: int, end: int) -> str:
    """Public helper: the cache MP4 filename for a clip (used by the prefetch manager)."""
    return _cache_key(Path(npz_path), int(start), int(end))


def qpos_to_sim_qpos(qpos_frame: np.ndarray) -> np.ndarray:
    """Map 30-dim or 36-dim qpos to the 36-dim model qpos."""
    qpos_frame = np.asarray(qpos_frame).reshape(-1)
    if qpos_frame.shape[0] == 7 + 29:
        return qpos_frame
    if qpos_frame.shape[0] == 7 + 23:
        qpos_23 = qpos_frame
        qpos_29 = np.zeros((29,), dtype=qpos_23.dtype)
        qpos_29[:7] = qpos_23[:7]
        idx29 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
        qpos_29[idx29] = qpos_23[7:]
        return np.concatenate([qpos_23[:7], qpos_29], axis=0)
    raise ValueError(f"Unsupported qpos dim: {qpos_frame.shape[0]} (expect 36 or 30)")


def update_camera_front(viewer_cam, pelvis_pos: np.ndarray, pelvis_rot: np.ndarray):
    """Position camera facing the robot's front."""
    viewer_cam.lookat[:] = pelvis_pos.copy()
    viewer_cam.distance = 2.0
    viewer_cam.elevation = -20.0
    viewer_cam.azimuth = 90.0


def update_camera_dual(viewer_cam, trk_pos: np.ndarray, ref_pos: np.ndarray):
    """Frame both the tracker and ref robots, centering on their midpoint and
    zooming out enough that both stay visible even when they diverge."""
    mid = (trk_pos + ref_pos) / 2.0
    viewer_cam.lookat[:] = mid
    sep = float(np.linalg.norm(trk_pos[:2] - ref_pos[:2]))
    # 2.0 keeps the single-robot framing when overlapping; grow with separation.
    viewer_cam.distance = max(2.0, sep * 0.85 + 1.6)
    viewer_cam.elevation = -20.0
    viewer_cam.azimuth = 90.0


def _build_dual_model(scene_xml: str) -> mujoco.MjModel:
    """Build a scene with two robots: tracker (original) + ref (red)."""
    spec = mujoco.MjSpec.from_file(scene_xml)
    spec.copy_during_attach = True
    ref_spec = mujoco.MjSpec.from_file(DEFAULT_G1_XML)
    ref_spec.copy_during_attach = True

    frame = spec.worldbody.add_frame()
    frame.name = "ref_attach"
    # Attaching the complete child spec in MuJoCo 3.3.7 also merges its global
    # model configuration and causes file-backed floor textures in the parent
    # scene to render as plain white. The reference robot only needs its body,
    # joints, geoms and referenced assets, so attach that body tree directly.
    frame.attach_body(ref_spec.worldbody.first_body(), prefix="ref_")

    model = spec.compile()

    # Find all body IDs belonging to the ref robot
    ref_body_ids = set()
    for i in range(model.nbody):
        if model.body(i).name.startswith("ref_"):
            ref_body_ids.add(i)

    # Track which materials we've already recolored
    recolored_mats = set()

    # Color ref geoms red + disable collision
    for i in range(model.ngeom):
        if model.geom_bodyid[i] in ref_body_ids:
            model.geom_contype[i] = 0
            model.geom_conaffinity[i] = 0
            matid = model.geom_matid[i]
            if matid >= 0:
                if matid not in recolored_mats:
                    model.mat_rgba[matid] = REF_RGBA
                    recolored_mats.add(matid)
            else:
                model.geom_rgba[i] = REF_RGBA

    return model


def _load_ref_qpos(side_info: dict, start_frame: int, end_frame: int) -> np.ndarray | None:
    """Load ref motion from the original mocap dataset.

    Returns (N, 36) array of ref qpos, or None if unavailable.
    """
    source_path = side_info.get("source_motion_path")
    if not source_path:
        return None
    source_path = Path(source_path)
    if not source_path.exists():
        return None

    # Slice the original mocap by its own frame numbers, not the clip's: the clip NPZ
    # was already cut out of this motion, so start_frame here counts from the clip.
    # task_manager requires both fields on every candidate, so a missing one is a
    # malformed task rather than something to guess at.
    src_start = side_info["source_start_frame"]
    src_end = side_info["source_end_frame"]

    data = np.load(source_path, allow_pickle=True)
    if "qpos" not in data:
        return None

    ref_qpos = data["qpos"][src_start:src_end]  # (N, 36)
    return ref_qpos


def _cache_key(npz_path: Path, start: int, end: int) -> str:
    # Different trackers/policies produce identically-named clip NPZs (the
    # tracker name only appears in the parent directory), so the stem alone is
    # NOT unique. Hash the full resolved path to keep left/right clips distinct.
    identity = str(npz_path.resolve())
    h = hashlib.md5(identity.encode("utf-8")).hexdigest()[:8]
    return f"{npz_path.stem}_{start}_{end}_{h}.mp4"


def get_or_render(
    npz_path: Path | str,
    start_frame: int,
    end_frame: int,
    cache_dir: Path | str,
    scene_xml: str | None = None,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    side_info: dict | None = None,
) -> Path:
    """Return path to cached MP4, rendering tracker + ref (red) if needed."""
    npz_path = Path(npz_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = scene_xml or DEFAULT_SCENE_XML

    mp4_name = _cache_key(npz_path, start_frame, end_frame)
    mp4_path = cache_dir / mp4_name

    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        return mp4_path

    # Serialize: the shared renderer / EGL context is single-threaded.
    with _RENDER_LOCK:
        # The web service and the offline pre-renderer can share the same kpfs2
        # cache.  An advisory file lock prevents two hosts/processes from
        # encoding the same clip at once.  The lock is released automatically
        # if a worker exits; the small .lock file can safely remain on disk.
        lock_path = mp4_path.with_suffix(mp4_path.suffix + ".lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Re-check after taking both locks: another process may have
                # completed the file while this worker was waiting.
                if mp4_path.exists() and mp4_path.stat().st_size > 0:
                    return mp4_path
                return _render_to_mp4(
                    npz_path, start_frame, end_frame, mp4_path,
                    scene_xml, fps, width, height, side_info,
                )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _render_to_mp4(
    npz_path: Path,
    start_frame: int,
    end_frame: int,
    mp4_path: Path,
    scene_xml: str,
    fps: int,
    width: int,
    height: int,
    side_info: dict | None,
) -> Path:
    # Load tracker qpos
    data_npz = np.load(npz_path, allow_pickle=True)

    # Use imu_pose + joint_pos as the actual tracker output (not qpos which may equal ref)
    if "imu_pose" in data_npz and "joint_pos" in data_npz:
        imu_pose = data_npz["imu_pose"][start_frame:end_frame]  # (N, 7)
        joint_pos = data_npz["joint_pos"][start_frame:end_frame]  # (N, 29)
        qpos_seq = np.concatenate([imu_pose, joint_pos], axis=1)  # (N, 36)
    else:
        qpos_seq = data_npz["qpos"][start_frame:end_frame]

    if qpos_seq.shape[0] == 0:
        raise ValueError(f"Empty qpos slice [{start_frame}:{end_frame}] from {npz_path}")

    num_frames = qpos_seq.shape[0]

    # Load ref motion: prefer from same NPZ (ref_pose + ref_joint_pos), else from source_motion_path
    ref_qpos_seq = None
    if "ref_pose" in data_npz and "ref_joint_pos" in data_npz:
        ref_pose = data_npz["ref_pose"][start_frame:end_frame]  # (N, 7)
        ref_joint = data_npz["ref_joint_pos"][start_frame:end_frame]  # (N, 29)
        ref_qpos_seq = np.concatenate([ref_pose, ref_joint], axis=1)  # (N, 36)
    elif side_info:
        ref_qpos_seq = _load_ref_qpos(side_info, start_frame, end_frame)

    # The tracker rollout (imu_pose, absolute sim-world frame) and the reference
    # often do NOT share a world origin: a clip taken mid-episode has the robot
    # wherever it wandered, while the reference uses its own origin. Left as-is,
    # the ref can sit several metres away and fall off-screen. Remove only the
    # constant XY offset at frame 0 so the two overlay at the start; genuine
    # tracking error (relative drift over the clip) is preserved.
    if ref_qpos_seq is not None and ref_qpos_seq.shape[0] > 0:
        ref_qpos_seq = np.array(ref_qpos_seq, dtype=float, copy=True)
        ref_qpos_seq[:, :2] += qpos_seq[0, :2] - ref_qpos_seq[0, :2]

    # Reuse the cached dual-robot model + renderer (built once, ~2.5s saved per clip).
    ctx = _get_ctx(scene_xml, width, height)
    model, data, renderer, cam, pelvis_id, ref_pelvis_id = (
        ctx.model, ctx.data, ctx.renderer, ctx.cam, ctx.pelvis_id, ctx.ref_pelvis_id,
    )

    # Encode with ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    # Render to a temp file then atomically rename, so readers (the web server's
    # static file mount) never see a half-written MP4.
    # A process-specific temp name makes the atomic publish step safe even
    # when multiple pre-render workers share the same cache directory.
    tmp_path = mp4_path.with_name(
        f".{mp4_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
    )
    tmp_path.unlink(missing_ok=True)
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
        "-preset", "veryfast",
        "-crf", "23",
        # Short keyframe interval -> instant seek-to-start on loop / scrubbing.
        "-g", str(max(1, fps // 2)),
        # Put the moov atom at the front so HTML5 <video> can start playing
        # immediately without downloading the whole file.
        "-movflags", "+faststart",
        str(tmp_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for i in range(num_frames):
            # Tracker qpos (first 36 dims)
            sim_qpos = qpos_to_sim_qpos(qpos_seq[i])
            data.qpos[:36] = sim_qpos

            # Ref qpos (next 36 dims) from original mocap
            has_ref = ref_qpos_seq is not None and i < ref_qpos_seq.shape[0]
            if has_ref:
                data.qpos[36:] = qpos_to_sim_qpos(ref_qpos_seq[i])
            else:
                # Hide ref robot underground if no ref data
                data.qpos[36:] = sim_qpos
                data.qpos[36 + 2] = -10.0  # z far below ground

            mujoco.mj_forward(model, data)

            pelvis_pos = data.xpos[pelvis_id].copy()
            if has_ref:
                # Frame both robots so the red ref stays visible even when the
                # tracker drifts far from it.
                update_camera_dual(cam, pelvis_pos, data.xpos[ref_pelvis_id].copy())
            else:
                pelvis_rot = data.xmat[pelvis_id].reshape(3, 3).copy()
                update_camera_front(cam, pelvis_pos, pelvis_rot)

            renderer.update_scene(data, cam)
            frame = renderer.render()
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            proc.stdin.write(frame.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr[:500]}")

    tmp_path.replace(mp4_path)
    return mp4_path
