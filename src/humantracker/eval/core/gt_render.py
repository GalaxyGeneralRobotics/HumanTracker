"""Dual-robot renderer: tracker in native colors, reference as a light-red ghost.

Matches the preference-labeler overlay (``REF_RGBA`` in
``tool/motion_annotation/web_labeler/video_cache.py``). Physics stays on the
single-robot sim; this model is display-only.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from humantracker.eval.paths import ROOT

# Same tint as the annotation overlay: light red, semi-transparent.
REF_RGBA = np.array([0.8, 0.1, 0.1, 0.4], dtype=np.float64)
DEFAULT_G1_XML = ROOT / "storage" / "assets" / "unitree_g1_5010" / "g1_mjx_track_papergray.xml"


class DualRobotRenderer:
    """Offscreen renderer with a collision-free red copy of the G1 for GT qpos."""

    def __init__(
        self,
        scene_xml: str,
        width: int,
        height: int,
        robot_xml: str | None = None,
        cam_distance: float = 3.0,
        cam_azimuth: float = 90.0,
        cam_elevation: float = -20.0,
    ):
        robot_xml = str(robot_xml or DEFAULT_G1_XML)
        if not Path(robot_xml).is_file():
            raise FileNotFoundError(f"GT robot XML not found: {robot_xml}")
        self.model = _build_dual_model(scene_xml, robot_xml)
        self.data = mujoco.MjData(self.model)
        self.nq_robot = self.model.nq // 2
        if self.model.nq != 2 * self.nq_robot:
            raise ValueError(f"Dual-robot nq={self.model.nq} is not two equal robots")
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = cam_distance
        self.cam.azimuth = cam_azimuth
        self.cam.elevation = cam_elevation
        self.pelvis_id = self.model.body("pelvis").id
        self.ref_pelvis_id = self.model.body("ref_pelvis").id

    def render(self, robot_qpos: np.ndarray, ref_qpos: np.ndarray) -> np.ndarray:
        robot_qpos = np.asarray(robot_qpos, dtype=np.float64).reshape(-1)
        ref_qpos = np.asarray(ref_qpos, dtype=np.float64).reshape(-1)
        if robot_qpos.shape[0] != self.nq_robot or ref_qpos.shape[0] != self.nq_robot:
            raise ValueError(
                f"Expected qpos dim {self.nq_robot}, got "
                f"robot={robot_qpos.shape[0]} ref={ref_qpos.shape[0]}"
            )
        self.data.qpos[: self.nq_robot] = robot_qpos
        self.data.qpos[self.nq_robot :] = ref_qpos
        mujoco.mj_forward(self.model, self.data)
        trk = self.data.xpos[self.pelvis_id]
        ref = self.data.xpos[self.ref_pelvis_id]
        self.cam.lookat[:] = 0.5 * (trk + ref)
        sep = float(np.linalg.norm(trk[:2] - ref[:2]))
        self.cam.distance = max(3.0, sep * 0.85 + 2.4)
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def _build_dual_model(scene_xml: str, robot_xml: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(scene_xml)
    spec.copy_during_attach = True
    ref_spec = mujoco.MjSpec.from_file(robot_xml)
    ref_spec.copy_during_attach = True

    frame = spec.worldbody.add_frame()
    frame.name = "ref_attach"
    # Attach the body tree only. Merging the child spec's global config makes
    # the parent floor texture render as plain white (same issue as the labeler).
    frame.attach_body(ref_spec.worldbody.first_body(), prefix="ref_")
    model = spec.compile()

    ref_body_ids = {
        i for i in range(model.nbody) if model.body(i).name.startswith("ref_")
    }
    recolored_mats: set[int] = set()
    for i in range(model.ngeom):
        if model.geom_bodyid[i] not in ref_body_ids:
            continue
        model.geom_contype[i] = 0
        model.geom_conaffinity[i] = 0
        matid = int(model.geom_matid[i])
        if matid >= 0:
            if matid not in recolored_mats:
                model.mat_rgba[matid] = REF_RGBA
                recolored_mats.add(matid)
        else:
            model.geom_rgba[i] = REF_RGBA
    return model
