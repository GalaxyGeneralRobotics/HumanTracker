"""Make an any4hdmi view of a HumanTracker test manifest without copying motions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mocap_path", type=Path, required=True)
    parser.add_argument("--test_json", type=Path, required=True)
    parser.add_argument("--robot_xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mocap = args.mocap_path.resolve()
    test_json = args.test_json.resolve()
    robot_xml = args.robot_xml.expanduser().absolute()
    output = args.output.resolve()
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    if model.nq != 36 or model.njnt != 30 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected a free-root, 29-joint G1 model")
    body_names = {model.body(i).name for i in range(model.nbody)}
    if not {"left_toe_link", "right_toe_link"} <= body_names:
        raise ValueError("Motion FK model must include both G1 toe links")
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]

    entries = json.loads(test_json.read_text())
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
        if target.exists():
            if not os.path.samefile(source, target):
                raise FileExistsError(f"View entry is not a hard link to source: {target}")
        else:
            os.link(source, target)

    manifest = {
        "format_version": 2,
        "dataset_name": "HumanTracker_test_view",
        "mjcf": "hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml",
        "motions_subdir": "motions",
        "timestep": 0.02,
        "qpos_dim": model.nq,
        "qpos_names": ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz", *joint_names],
        "num_motions": len(paths),
        "source": {"test_json": str(test_json)},
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    print(f"Linked {len(paths)} motions into {output}")


if __name__ == "__main__":
    main()
