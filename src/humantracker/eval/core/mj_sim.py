"""MuJoCo PD-control wrapper shared by the GMT, SONIC and TWIST2 backends.

Each backend subclasses :class:`MJSim` to supply its own PD gains, torque limits
and initial pose, and passes its own control/physics rates in -- those differ per
policy. The Humanoid-GPT backend drives its upstream ``G1TrackMjSim`` instead, so it
does not use this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = ["State", "MJSim"]


@dataclass
class State:
    mj_data: mujoco.MjData = None


class MJSim:
    """PD position controller: ``num_sim_substeps`` physics steps per action.

    Subclasses must set ``kps``, ``kds``, ``torque_limit`` and ``init_qpos``
    before :meth:`reset` or :meth:`step` is called.
    """

    kps: np.ndarray
    kds: np.ndarray
    torque_limit: np.ndarray
    init_qpos: np.ndarray

    def __init__(self, xml_path: str, ctrl_dt=0.02, sim_dt=0.001, headless=True):
        self.ctrl_dt = ctrl_dt
        self.sim_dt = sim_dt
        self.num_sim_substeps = int(self.ctrl_dt / self.sim_dt)
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.timestep = sim_dt
        self.headless = headless

    def init_state(self) -> State:
        return State(mj_data=mujoco.MjData(self.mj_model))

    def reset(self, state: State) -> State:
        mj_data = state.mj_data
        mj_data.qpos[:] = self.init_qpos
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, mj_data)
        return State(mj_data=mj_data)

    def step(self, state: State, action: np.ndarray) -> State:
        mj_data = state.mj_data
        for _ in range(self.num_sim_substeps):
            torques = self.kps * (action - mj_data.qpos[7:]) + self.kds * (-mj_data.qvel[6:])
            mj_data.ctrl[:] = np.clip(torques, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.mj_model, mj_data)
        return State(mj_data=mj_data)
