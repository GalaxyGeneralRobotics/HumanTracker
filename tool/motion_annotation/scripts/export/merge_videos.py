"""
python merge_videos.py \
  --source_root data/video_source \
  --folders f1,f2,f3 \
  --folder_labels "f1=retarget,f2=track_v5,f3=track_v6" \
  --output_dir data/video_source/merged


"""


import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def ffprobe_json(path: str) -> dict:
    code, out, err = run([
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-print_format", "json", path
    ])
    if code != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{err}")
    return json.loads(out)


def get_wh(path: str) -> Tuple[int, int]:
    info = ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s["width"]), int(s["height"])
    raise RuntimeError(f"No video stream in {path}")


def has_audio(path: str) -> bool:
    info = ffprobe_json(path)
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def get_duration_seconds(path: str) -> float:
    info = ffprobe_json(path)
    duration = info.get("format", {}).get("duration")
    if duration is not None:
        return float(duration)
    stream_durations = [
        float(stream["duration"])
        for stream in info["streams"]
        if stream.get("duration") is not None
    ]
    if not stream_durations:
        raise ValueError(f"video duration is missing: {path}")
    return max(stream_durations)


def natural_key(s: str) -> List:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_videos(folder: Path, recursive: bool = False) -> List[Path]:
    if not folder.is_dir():
        return []
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files, key=lambda p: natural_key(p.name))


def discover_folders(source_root: Path, prefix: str = "f") -> List[str]:
    if not source_root.is_dir():
        return []
    dirs = [p.name for p in source_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(dirs, key=natural_key)


def parse_folder_labels(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    s = (s or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v if v else k
    return out


def task_key_from_filename(p: Path) -> Optional[str]:
    name = p.name

    # 去掉视频后缀
    for ext in VIDEO_EXTS:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    # 必须包含 .npz
    i = name.lower().rfind(".npz")
    if i < 0:
        return None
    base = name[:i]  # 去掉 .npz 及后面

    # 必须以 YYYYMMDD_HHMMSS_ 开头
    m = re.match(r"^(\d{8})_(\d{6})_(.+)$", base)
    if not m:
        return None

    return m.group(3)  # “时间后、npz前”



def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-. ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:200] if len(s) > 200 else s


def norm_name(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def build_xstack_layout(n: int) -> str:
    if n <= 1:
        return "0_0"
    parts = ["0_0", "w0_0"]
    expr = "w0"
    for i in range(2, n):
        expr = expr + f"+w{i-1}"
        parts.append(f"{expr}_0")
    return "|".join(parts)


def merge_side_by_side(
    inputs: List[str],
    column_titles: List[str],
    out: str,
    height: Optional[int],
    audio: str,          # first / mix / none
    vcodec: str,
    crf: int,
    preset: str,
    shortest: bool,
):
    assert len(inputs) == len(column_titles) and len(inputs) >= 1

    if height is None:
        hs = []
        for p in inputs:
            _, h = get_wh(p)
            hs.append(h)
        height = min(hs)

    durs = [get_duration_seconds(p) for p in inputs]
    d_min = min([d for d in durs if d > 0] or [0.0])
    target_duration = d_min if d_min > 0 else None

    seeks: List[float] = []
    if target_duration is None:
        seeks = [0.0] * len(inputs)
    else:
        for d in durs:
            if d > target_duration + 1e-3:
                seeks.append(max(0.0, (d - target_duration) / 2.0))
            else:
                seeks.append(0.0)

    v_filters = []
    v_tags = []
    for i, title in enumerate(column_titles):
        tag = f"s{i}"
        safe_title = title.replace(":", "\\:").replace("'", "\\'")
        vf = (
            f"[{i}:v]"
            f"scale=-2:{height}:flags=lanczos,setsar=1,"
            f"drawtext=text='{safe_title}':x=10:y=10:fontsize=28:"
            f"fontcolor=white:box=1:boxcolor=black@0.5"
            f"[{tag}]"
        )
        v_filters.append(vf)
        v_tags.append(f"[{tag}]")

    if len(inputs) == 1:
        stack = f"{v_tags[0]}copy[vout]"
    else:
        layout = build_xstack_layout(len(inputs))
        stack = f"{''.join(v_tags)}xstack=inputs={len(inputs)}:layout={layout}"
        if shortest:
            stack += ":shortest=1"
        stack += "[vout]"

    map_args = ["-map", "[vout]"]
    a_present = [has_audio(p) for p in inputs]
    a_filters = []
    map_audio = False

    if audio == "none":
        map_audio = False
    elif audio == "first":
        for i, ok in enumerate(a_present):
            if ok:
                map_args += ["-map", f"{i}:a"]
                map_audio = True
                break
    elif audio == "mix":
        idxs = [i for i, ok in enumerate(a_present) if ok]
        if len(idxs) == 1:
            map_args += ["-map", f"{idxs[0]}:a"]
            map_audio = True
        elif len(idxs) >= 2:
            for k, i in enumerate(idxs):
                a_filters.append(
                    f"[{i}:a]aformat=sample_fmts=fltp:channel_layouts=stereo,aresample=48000[a{k}]"
                )
            amix_in = "".join([f"[a{k}]" for k in range(len(idxs))])
            a_filters.append(
                f"{amix_in}amix=inputs={len(idxs)}:duration={'shortest' if shortest else 'longest'}:dropout_transition=0[aout]"
            )
            map_args += ["-map", "[aout]"]
            map_audio = True

    filter_complex = ";".join(v_filters + [stack] + a_filters)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for i, p in enumerate(inputs):
        if seeks[i] > 0:
            cmd += ["-ss", f"{seeks[i]:.3f}"]
        if target_duration is not None:
            cmd += ["-t", f"{target_duration:.3f}"]
        cmd += ["-i", p]

    cmd += ["-filter_complex", filter_complex]
    cmd += map_args
    cmd += ["-c:v", vcodec, "-crf", str(crf), "-preset", preset, "-movflags", "+faststart"]
    if map_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    if shortest:
        cmd += ["-shortest"]
    cmd += [out]

    code, _, err = run(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err}\nCMD:\n{' '.join(cmd)}")


def pick_best_candidate(ref_path: Path, candidates: List[Path]) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    refn = norm_name(ref_path.stem)
    best = candidates[0]
    best_sc = -1.0
    for c in candidates:
        sc = name_similarity(refn, norm_name(c.stem))
        if sc > best_sc:
            best_sc = sc
            best = c
    return best


def main():
    ap = argparse.ArgumentParser(
        description="Align videos across sibling folders by the task key parsed from each "
                    "filename, and output one side-by-side merged video per task key."
    )
    ap.add_argument("--source_root", required=True,
                    help="Root dir containing folders like f1,f2,f3...")
    ap.add_argument("--folders", default="",
                    help="Folder order, comma-separated. Example: f1,f3,f2 . If empty, auto-discover and sort.")
    ap.add_argument("--recursive", action="store_true",
                    help="Search videos recursively inside each folder.")

    ap.add_argument("--task_key_regex", default=r"^\d{8}_\d{6}_\d+_(.+)$",
                    help="Regex to extract task key from filename stem. Must have one capture group.")
    ap.add_argument("--prefer_folder", default="",
                    help="Use this folder as reference for iterating keys and resolving duplicates (default: first).")

    ap.add_argument("--folder_labels", default="",
                    help='Overlay column titles. Example: "f1=retarget,f2=track_v5,f3=v6"')

    ap.add_argument("--output_dir", default="./output",
                    help="Directory to write outputs.")
    ap.add_argument("--height", type=int, default=None,
                    help="Target output height for stacking. Default=min height among inputs for each task.")
    ap.add_argument("--audio", choices=["first", "mix", "none"], default="first",
                    help="Audio strategy: first available / mix all / none.")
    ap.add_argument("--vcodec", default="libx264",
                    help="Video encoder (e.g., libx264).")
    ap.add_argument("--crf", type=int, default=20,
                    help="CRF for H.264 (lower=better quality).")
    ap.add_argument("--preset", default="veryfast",
                    help="x264 preset.")
    ap.add_argument("--shortest", action="store_true",
                    help="End stacked clip when the shortest input ends.")

    args = ap.parse_args()

    source_root = Path(args.source_root)
    if not source_root.is_dir():
        print(f"Not found source_root: {source_root}", file=sys.stderr)
        sys.exit(1)

    if args.folders.strip():
        folder_order = [s.strip() for s in args.folders.split(",") if s.strip()]
    else:
        folder_order = discover_folders(source_root, prefix="f")

    if not folder_order:
        print(f"No folders found under: {source_root}", file=sys.stderr)
        sys.exit(1)

    prefer = args.prefer_folder.strip() or folder_order[0]
    if prefer not in folder_order:
        prefer = folder_order[0]

    folder_labels = parse_folder_labels(args.folder_labels)

    # per folder: task_key -> [paths]
    per_folder: Dict[str, Dict[str, List[Path]]] = {}
    for f in folder_order:
        vids = list_videos(source_root / f, recursive=args.recursive)
        m: Dict[str, List[Path]] = {}
        for p in vids:
            key = task_key_from_filename(p)
            if key is None:
                continue
            if key not in m:
                m[key] = p
        per_folder[f] = m

    prefer = args.prefer_folder.strip() or folder_order[0]
    if prefer not in folder_order:
        prefer = folder_order[0]

    ref_keys = sorted(per_folder.get(prefer, {}).keys(), key=natural_key)
    if not ref_keys:
        print(f"No task keys found in prefer_folder={prefer}. Check filenames or --task_key_regex.", file=sys.stderr)
        sys.exit(2)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir + f"/{ts}_merge_export")
    out_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    skipped = 0

    for key in ref_keys:
        # 只有当每个 folder 都有同 key 才输出
        miss = [f for f in folder_order if key not in per_folder.get(f, {})]
        if miss:
            skipped += 1
            continue

        group_paths = {f: per_folder[f][key] for f in folder_order}

        inputs = [str(group_paths[f]) for f in folder_order]
        titles = [folder_labels.get(f, f) for f in folder_order]

        out_name = safe_filename(key) + ".mp4"
        out_path = out_dir / out_name

        merge_side_by_side(
            inputs=inputs,
            column_titles=titles,
            out=str(out_path),
            height=args.height,
            audio=args.audio,
            vcodec=args.vcodec,
            crf=args.crf,
            preset=args.preset,
            shortest=args.shortest,
        )

        produced += 1
        print(f"[OK] {key} -> {out_path}")

    print(f"Done. produced={produced}, skipped_missing_pairs={skipped}")
    print(f"Folders order: {folder_order}")
    print(f"Prefer folder: {prefer}")


if __name__ == "__main__":
    main()

