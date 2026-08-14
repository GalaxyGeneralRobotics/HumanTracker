# Unitree G1 Description — mode_15 / 5010 wrist (MJCF)

> [!IMPORTANT]
> Requires MuJoCo 2.3.4 or later.
> **This package targets the new "black-pelvis" G1 (`mode_machine = 15`)**,
> which uses 5010 wrist motors and a symmetric 22.5/22.5 hip gear ratio.
> This is the only G1 description this repository ships.

## Changelog

- **17/04/2026: Upgraded from `g1_29dof_rev_1_0` to `g1_29dof_mode_15`.**
  See `CHANGELOG.md` for the full diff.
- 10/12/2024: Updated base model from Unitree's official [repo](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description)
  (sha: c20ca8f1fe5e519474c6c8d10b1ce5c719dd7a65).
- 20/05/2024: Initial release (Menagerie).

## Overview

This package contains the MJCF description of the [G1 Humanoid
Robot](https://www.unitree.com/g1/) developed by [Unitree
Robotics](https://www.unitree.com/), specifically the `g1_29dof_mode_15`
variant — 29 DoF, fully-actuated waist, rubber hands, **new 5010 wrist motors
with 13.4 N·m peak torque** (vs. 5 N·m on the old 4010 wrist). Derived from
[`g1_29dof_mode_15.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_mode_15.urdf)
by re-using the Menagerie MJX-style structure (contact pairs, actuators,
sensors) from `unitree_g1/`.

<p float="left">
  <img src="g1.png" width="400">
  <img src="g1_with_hands.png" width="400">
</p>

## MJCF derivation steps

1. Copied the MJCF description from [g1_description](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.xml).
2. Manually edited the MJCF to extract common properties into the `<default>` section.
3. Added stand keyframe.
4. Added joint position actuators (needs tuning).
5. Applied similar edits to `g1_with_hands.xml`.

## License

This model is released under a [BSD-3-Clause License](LICENSE).
