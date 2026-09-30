"""Make an any4hdmi view of a HumanTracker test manifest without copying motions.

sim2real reads reference motions as an any4hdmi dataset: a ``manifest.json`` naming
the robot model the motions' forward kinematics use, beside a ``motions/`` tree. The
view hard-links every manifest motion into that tree at its dataset-relative path,
so the dataset itself is neither modified nor copied; the view must therefore be on
the same filesystem. Re-running verifies existing links instead of recreating them.

The FK model is sim2real's G1 MJCF. It is a different file from HumanTracker's scene
-- it has the toe links some released observations read -- and is used only to
build policy references; rollouts and metrics still use the HumanTracker scene.

Run from sim2real's environment; the model is fetched into the Hugging Face cache
(``HF_ENDPOINT`` is honored), after which evaluation can run with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
from any4hdmi.utils.mjcf import resolve_mjcf_path
from sim2real.config.robots import get_robot_cfg

QPOS_ROOT_NAMES = ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz"]


def load_fk_model(mjcf: str) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(resolve_mjcf_path(mjcf)))
    if model.nq != 36 or model.njnt != 30 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Expected a free-root, 29-joint G1 model: {mjcf}")
    body_names = {model.body(i).name for i in range(model.nbody)}
    if not {"left_toe_link", "right_toe_link"} <= body_names:
        raise ValueError(f"Motion FK model must include both G1 toe links: {mjcf}")
    return model


def link_motions(mocap: Path, entries: list, output: Path) -> int:
    paths = [Path(entry["path"]) for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate motion paths in test manifest")
    for relative in paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid motion path: {relative}")
        source = mocap / relative
        target = output / "motions" / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)
        elif not os.path.samefile(source, target):
            raise FileExistsError(f"View entry is not a hard link to its source: {target}")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocap_path", type=Path, required=True, help="motion dataset root")
    parser.add_argument("--test_json", type=Path, required=True, help="manifest of {path, category} entries")
    parser.add_argument("--output", type=Path, required=True, help="view directory, on the dataset's filesystem")
    args = parser.parse_args()

    mjcf = get_robot_cfg("g1").mjcf_path
    model = load_fk_model(mjcf)
    test_json = args.test_json.resolve()
    output = args.output.resolve()
    count = link_motions(args.mocap_path.resolve(), json.loads(test_json.read_text()), output)

    manifest = {
        "format_version": 2,
        "dataset_name": "HumanTracker_test_view",
        "mjcf": mjcf,
        "motions_subdir": "motions",
        "timestep": 0.02,
        "qpos_dim": model.nq,
        "qpos_names": QPOS_ROOT_NAMES + [model.joint(i).name for i in range(1, model.njnt)],
        "num_motions": count,
        "source": {"test_json": str(test_json)},
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    print(f"Linked {count} motions into {output}")


if __name__ == "__main__":
    main()
