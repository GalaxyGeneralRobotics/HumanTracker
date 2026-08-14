# Changelog – Unitree G1 Description (mode_15 / 5010 wrist)

All notable changes to this model will be documented in this file.

## [2026-04-17] — Upgrade to mode_machine 15 (black-pelvis G1)

Forked from `unitree_g1/` (mode_machine 5, `g1_29dof_rev_1_0`) and upgraded to
the new **`g1_29dof_mode_15`** variant (the "black-pelvis" hardware refresh,
per Unitree `unitree_ros/robots/g1_description` README).

Changes applied to both `g1_mjx.xml` and `g1_mjx_track.xml`:

- **Wrist upgraded from 4010 to 5010 motors**
  - Meshes: `*_wrist_{roll,pitch,yaw}_link_5010.STL` (new files under `assets/`).
  - Inertial (both sides): `wrist_pitch_link` mass `0.48405 → 0.684` kg,
    updated `pos/quat/diaginertia` to match `g1_29dof_mode_15` URDF.
  - `wrist_yaw_link` attachment: `pos="0.046 0 0" → "0.051 0 0"` (outboard +5 mm).
  - `wrist_collision` capsule extended: `fromto=... 0.06 0 0 → 0.065 0 0`.
  - Actuator: `wrist_pitch/yaw` effort `5 → 13.4 N·m`, armature `0.00425 → 0.01`.
- **Hip-pitch upgraded to 139 N·m** (was 88 N·m). Matches knee/hip_roll spec
  (22.5:22.5 gear ratio per mode_15 URDF). Armature `0.01017752004 → 0.025101925`.
- **Ankle / waist_pitch / waist_roll effort reduced** from 50 to 35 N·m
  (armature unchanged; same motor, updated firmware limits).
- `mujoco model` name: `g1_29dof_rev_1_0 mjx → g1_29dof_mode_15 mjx`.

**Deploy note:** when using this model, set `mode_machine = 15` in the
low-level control stream (was 5 for the old hardware).

## [2025-05-30]

- Add MJX variant of [g1.xml](g1.xml), with manually designed collision geoms and contact pairs.

<p float="left">
  <img src="g1_mjx_colliders.png" width="400">
</p>

## [2024-12-10]

- Use updated models from Unitree's official [repo](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description) (sha: c20ca8f1fe5e519474c6c8d10b1ce5c719dd7a65).
  - Model without hands: [g1_29dof_rev_1_0](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.xml)
  - Model with hands: [g1_29dof_with_hand_rev_1_0](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml)

## [2024-05-20]

- Initial release.
