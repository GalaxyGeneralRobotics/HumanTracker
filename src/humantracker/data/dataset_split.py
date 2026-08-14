import os
import glob
import json
import argparse
import random
import zipfile
import math
import re
import numpy as np
import concurrent.futures
from pathlib import Path
from collections import Counter

FPS = None
SEED = 42
TEST_RATIO = 0.1
SMALL_TRAIN_RATIO = 0.1
OUTPUT_MULTIPLIER = 2


def _has_npz_files(directory):
    return any(child.is_file() and child.suffix == ".npz" for child in directory.iterdir())


def discover_categories(dataset_root):
    root = Path(dataset_root)
    first_level_dirs = sorted([child for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")])

    # Flat layout: npz files are placed directly under dataset root.
    if _has_npz_files(root):
        return {root.name: sorted(root.glob("*.npz"))}

    # Nested layout: use leaf directories that actually contain npz files.
    leaf_categories = {}
    has_second_level = any(
        any(grandchild.is_dir() for grandchild in child.iterdir())
        for child in first_level_dirs
    )

    if has_second_level:
        for first_dir in first_level_dirs:
            for current_dir, dirs, _ in os.walk(first_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                current_path = Path(current_dir)
                npz_files = sorted(current_path.glob("*.npz"))
                child_dirs = [child for child in current_path.iterdir() if child.is_dir()]
                if npz_files and not child_dirs:
                    rel_category = current_path.relative_to(root).as_posix()
                    leaf_categories[rel_category] = npz_files
        if leaf_categories:
            return leaf_categories

    # Single-level category layout: first-level subdirectories are categories.
    first_level_categories = {}
    for child in first_level_dirs:
        npz_files = sorted(child.glob("*.npz"))
        if npz_files:
            first_level_categories[child.relative_to(root).as_posix()] = npz_files
    return first_level_categories


def split_category_files(files, test_ratio):
    total = len(files)
    if total == 0:
        return [], []

    if test_ratio <= 0:
        return list(files), []

    test_count = math.floor(total * test_ratio)
    if total > 0 and test_ratio > 0 and test_count == 0:
        test_count = 1
    if test_count >= total and total > 1:
        test_count = total - 1
    if total == 1:
        test_count = 1 if test_ratio >= 1 else min(test_count, 1)

    train_count = total - test_count
    return list(files[:train_count]), list(files[train_count:])


def case_key_from_path(file_path):
    name = file_path.stem
    if name.endswith("_flipped"):
        return name[:-8]
    return name


def group_files_by_case(files):
    groups = {}
    for fpath in files:
        key = case_key_from_path(fpath)
        if key not in groups:
            groups[key] = []
        groups[key].append(fpath)

    # Keep deterministic order inside each case: original first, then flipped/others.
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda p: ("_flipped" in p.stem, p.stem))
    return groups


def extract_fps_from_name(file_path):
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hz|fps)", file_path.stem, re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))


def build_fps_resolver(category_to_files, default_fps=50.0):
    if FPS is not None:
        manual_fps = float(FPS)

        def resolve_fps(_file_path, _category):
            return manual_fps, "manual"

        return resolve_fps, {
            "default_fps": manual_fps,
            "default_source": "manual",
            "global_counter": Counter({manual_fps: 1}),
            "category_defaults": {},
        }

    global_counter = Counter()
    category_defaults = {}

    for category, files in category_to_files.items():
        category_counter = Counter()
        for fpath in files:
            fps = extract_fps_from_name(fpath)
            if fps is None:
                continue
            category_counter[fps] += 1
            global_counter[fps] += 1
        if category_counter:
            category_defaults[category] = category_counter.most_common(1)[0][0]

    if global_counter:
        global_default_fps = global_counter.most_common(1)[0][0]
        default_source = "inferred"
    else:
        global_default_fps = float(default_fps)
        default_source = "default"

    def resolve_fps(file_path, category):
        parsed_fps = extract_fps_from_name(file_path)
        if parsed_fps is not None:
            return parsed_fps, "filename"

        category_fps = category_defaults.get(category)
        if category_fps is not None:
            return category_fps, "category_default"

        return global_default_fps, default_source

    return resolve_fps, {
        "default_fps": global_default_fps,
        "default_source": default_source,
        "global_counter": global_counter,
        "category_defaults": category_defaults,
    }


def build_item(rel_path, category, frames):
    return {
        "path": rel_path,
        "category": category,
        "frames": int(frames)
    }


def update_stats(stats, split_name, category, frames, fps):
    hrs = frames / fps / 3600.0

    stats["overall"]["clips"] += 1
    stats["overall"]["frames"] += frames
    stats["overall"]["duration_hrs"] += hrs

    stats[split_name]["clips"] += 1
    stats[split_name]["frames"] += frames
    stats[split_name]["duration_hrs"] += hrs

    stats["categories"][category]["clips"] += 1
    stats["categories"][category]["frames"] += frames
    stats["categories"][category]["duration_hrs"] += hrs


def scale_output_value(value):
    return value * OUTPUT_MULTIPLIER


def build_case_json_items(case_id, category, real_files, frame_map, dataset_root):
    real_items = []
    for fpath in real_files:
        # Indexed, not defaulted: the caller has already refused to continue if any
        # file scanned to zero frames, so a missing path is a bug rather than a
        # motion to leave out of the split without saying so.
        frames = frame_map[fpath]
        rel_path = os.path.relpath(str(fpath), dataset_root)
        real_items.append(build_item(rel_path, category, frames))

    if not real_items:
        return [], 0

    stem_to_item = {
        Path(item["path"]).stem: item
        for item in real_items
    }

    ordered_items = []
    original_stem = case_id
    flipped_stem = f"{case_id}_flipped"

    reference_item = stem_to_item.get(original_stem)
    if reference_item is None:
        reference_item = next(iter(real_items))

    if original_stem in stem_to_item:
        ordered_items.append(stem_to_item[original_stem])
    else:
        original_path = str(Path(reference_item["path"]).with_name(f"{original_stem}.npz"))
        ordered_items.append(build_item(original_path, category, reference_item["frames"]))

    if flipped_stem in stem_to_item:
        ordered_items.append(stem_to_item[flipped_stem])
    else:
        flipped_path = str(Path(reference_item["path"]).with_name(f"{flipped_stem}.npz"))
        ordered_items.append(build_item(flipped_path, category, reference_item["frames"]))

    extra_items = [
        item for stem, item in sorted(stem_to_item.items())
        if stem not in {original_stem, flipped_stem}
    ]
    ordered_items.extend(extra_items)

    virtual_count = sum(1 for item in ordered_items if Path(item["path"]).stem not in stem_to_item)
    return ordered_items, virtual_count

def get_frames(fpath):
    """Frame count of a motion npz, read from the qpos header without inflating it."""
    with zipfile.ZipFile(fpath) as archive:
        if "qpos.npy" not in archive.namelist():
            raise KeyError(f"{fpath}: npz contains no qpos.npy")
        with archive.open("qpos.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"{fpath}: unsupported npy format {version}")
    return shape[0]


def process_file(fpath):
    frames = get_frames(fpath)
    return fpath, frames

def allocate_test_counts(category_case_counts, test_ratio):
    """Use largest-remainder apportionment for the closest stratified split."""
    total_cases = sum(category_case_counts.values())
    target_total = round(total_cases * test_ratio)
    raw_counts = {
        category: count * test_ratio
        for category, count in category_case_counts.items()
    }
    test_counts = {
        category: math.floor(raw_count)
        for category, raw_count in raw_counts.items()
    }

    remaining = target_total - sum(test_counts.values())
    remainder_order = sorted(
        category_case_counts,
        key=lambda category: (
            -(raw_counts[category] - test_counts[category]),
            category,
        ),
    )
    for category in remainder_order[:remaining]:
        test_counts[category] += 1

    return test_counts


def write_json_atomic(output_path, data):
    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w") as f:
        json.dump(data, f, indent=4)
    os.replace(temporary_path, output_path)


def validate_split(train_data, test_data, all_paths, expected_test_counts):
    train_paths = {item["path"] for item in train_data}
    test_paths = {item["path"] for item in test_data}

    if len(train_paths) != len(train_data) or len(test_paths) != len(test_data):
        raise RuntimeError("Duplicate paths detected in a split")
    if train_paths & test_paths:
        raise RuntimeError("Train/test path leakage detected")
    if train_paths | test_paths != all_paths:
        missing = all_paths - (train_paths | test_paths)
        extra = (train_paths | test_paths) - all_paths
        raise RuntimeError(
            f"Split coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    actual_test_counts = Counter(item["category"] for item in test_data)
    expected_test_file_counts = {
        category: count * 2
        for category, count in expected_test_counts.items()
    }
    if dict(actual_test_counts) != expected_test_file_counts:
        raise RuntimeError(
            "Category counts do not match allocation: "
            f"actual={dict(actual_test_counts)}, "
            f"expected={expected_test_file_counts}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a motion dataset into train/test/10pct manifests."
    )
    parser.add_argument("dataset_root", type=Path,
                        help="dataset root to scan; manifests are written here")
    return parser.parse_args()


def main():
    dataset_root = parse_args().dataset_root
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    dataset_root = str(dataset_root)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    category_to_files = discover_categories(dataset_root)
    categories = sorted(category_to_files)
    if not categories:
        raise RuntimeError(f"No .npz categories found under {dataset_root}")

    category_case_groups = {}
    shuffled_case_ids = {}
    all_files = []
    for category in categories:
        files = list(category_to_files[category])
        case_groups = group_files_by_case(files)
        category_case_groups[category] = case_groups
        case_ids = sorted(case_groups)
        rng.shuffle(case_ids)
        shuffled_case_ids[category] = case_ids
        all_files.extend(files)

    print(f"Dataset root: {dataset_root}")
    print(f"Seed: {SEED}")
    print(f"Categories: {len(categories)} ({', '.join(categories)})")
    print(f"Cases: {sum(len(groups) for groups in category_case_groups.values())}")
    print(f"Files: {len(all_files)}")
    print("Reading frame counts...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        frame_results = list(executor.map(process_file, all_files))
    frame_map = dict(frame_results)
    unreadable_files = [str(path) for path, frames in frame_results if frames <= 0]
    if unreadable_files:
        examples = "\n".join(unreadable_files[:20])
        raise RuntimeError(
            f"Failed to read frames from {len(unreadable_files)} files:\n{examples}"
        )

    case_json_items = {}
    invalid_case_ids = {category: set() for category in categories}
    virtual_flipped_count = 0
    for category in categories:
        for case_id, real_files in category_case_groups[category].items():
            items, virtual_count = build_case_json_items(
                case_id,
                category,
                real_files,
                frame_map,
                dataset_root,
            )
            if virtual_count:
                # Keep incomplete cases with their real files only; never emit a virtual path.
                real_items = [
                    build_item(
                        os.path.relpath(str(fpath), dataset_root),
                        category,
                        frame_map[fpath],
                    )
                    for fpath in real_files
                ]
                case_json_items[(category, case_id)] = real_items
                virtual_flipped_count += virtual_count
                print(f"Including incomplete case with existing files only: {category}/{case_id}")
                continue
            case_json_items[(category, case_id)] = items
            virtual_flipped_count += virtual_count

    for category in categories:
        for case_id in invalid_case_ids[category]:
            category_case_groups[category].pop(case_id, None)
        shuffled_case_ids[category] = [
            case_id for case_id in shuffled_case_ids[category]
            if case_id not in invalid_case_ids[category]
        ]

    all_paths = {
        item["path"]
        for items in case_json_items.values()
        for item in items
    }
    category_case_counts = {
        category: len(category_case_groups[category])
        for category in categories
    }

    # Split complete cases, so original/flipped files never leak across splits.
    test_case_counts = allocate_test_counts(category_case_counts, TEST_RATIO)
    train_case_ids = {}
    test_case_ids = {}
    train_data = []
    test_data = []

    for category in categories:
        category_case_ids = shuffled_case_ids[category]
        test_count = test_case_counts[category]
        test_case_ids[category] = category_case_ids[:test_count]
        train_case_ids[category] = category_case_ids[test_count:]

        for case_id in train_case_ids[category]:
            train_data.extend(case_json_items[(category, case_id)])
        for case_id in test_case_ids[category]:
            test_data.extend(case_json_items[(category, case_id)])

    validate_split(train_data, test_data, all_paths, test_case_counts)

    # Small set: approximately 10% of all cases, sampled only from train cases.
    train_case_count_by_category = {
        category: len(train_case_ids[category])
        for category in categories
    }
    small_case_counts = allocate_test_counts(
        train_case_count_by_category,
        SMALL_TRAIN_RATIO / (1.0 - TEST_RATIO),
    )
    small_data = []
    for category in categories:
        small_count = small_case_counts[category]
        for case_id in train_case_ids[category][:small_count]:
            small_data.extend(case_json_items[(category, case_id)])

    small_paths = {item["path"] for item in small_data}
    train_paths = {item["path"] for item in train_data}
    if not small_paths <= train_paths:
        raise RuntimeError("10pct set contains items outside train.json")
    if not small_paths:
        raise RuntimeError("10pct set is empty")

    write_json_atomic(Path(dataset_root) / "train.json", train_data)
    write_json_atomic(Path(dataset_root) / "test.json", test_data)
    write_json_atomic(Path(dataset_root) / "10pct.json", small_data)

    print(f"\n[9/1] train={len(train_data)}, test={len(test_data)}")
    for category in categories:
        total = category_case_counts[category]
        test_count = test_case_counts[category]
        print(
            f"  {category:<20} cases: train={total - test_count:>5}, "
            f"test={test_count:>5}, test_ratio={test_count / total:.6%}"
        )
    print(f"[10pct] cases={sum(small_case_counts.values())}, files={len(small_data)}")
    print(f"Saved: {Path(dataset_root) / 'train.json'}")
    print(f"Saved: {Path(dataset_root) / 'test.json'}")
    print(f"Saved: {Path(dataset_root) / '10pct.json'}")
    print("Splitting and validation complete.")


if __name__ == "__main__":
    main()
