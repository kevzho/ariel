"""Command interface for the humanoid's native position servos.

The servos themselves (PD gains, torque limits and back-EMF torque-speed behaviour)
are MuJoCo actuators defined in ``humanoid.py``, so physics steps run in C without a
Python callback. This module maps policy actions to joint targets and rate-limits
the telescoping targets so a commanded length change takes visible time.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .humanoid import RANGES, RobotSpec, joint_kind, standing_pose


@dataclass
class ActuatorMap:
    names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    default: np.ndarray
    is_slide: np.ndarray
    qpos_adr: np.ndarray
    qvel_adr: np.ndarray


def actuator_map(model: mujoco.MjModel, actuators: list[str], spec: RobotSpec) -> ActuatorMap:
    n = len(actuators)
    lo, hi, default = np.zeros(n), np.zeros(n), np.zeros(n)
    slide = np.zeros(n, bool)
    qadr, vadr = np.zeros(n, int), np.zeros(n, int)
    pose = standing_pose(0.1)
    for i, name in enumerate(actuators):
        j = model.joint(name)
        lo[i], hi[i] = j.range
        qadr[i], vadr[i] = j.qposadr[0], j.dofadr[0]
        slide[i] = name.endswith("_ext")
        default[i] = 0.0 if slide[i] else pose.get(name, 0.0)
    return ActuatorMap(actuators, lo, hi, default, slide, qadr, vadr)


class Commander:
    """Policy action in [-1, 1]^n -> servo targets (ctrl), with telescoping rate limits."""

    def __init__(self, amap: ActuatorMap, spec: RobotSpec, control_dt: float, smoothing: float = 0.4):
        self.m = amap
        # first-order filter on rotary targets (~5 Hz command bandwidth at 50 Hz control):
        # real servos cannot follow arbitrarily fast command changes, and it removes twitching
        self.alpha = smoothing
        self.max_step = spec.slide_speed * control_dt
        span_up = amap.hi - amap.default
        span_dn = amap.default - amap.lo
        self.scale = np.maximum(span_up, span_dn)
        self.target = amap.default.copy()

    def reset(self, data: mujoco.MjData) -> None:
        self.target = np.clip(data.qpos[self.m.qpos_adr].copy(), self.m.lo, self.m.hi)
        data.ctrl[:] = self.target

    def command(self, action: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        a = np.clip(action, -1.0, 1.0)
        desired = np.clip(self.m.default + a * self.scale, self.m.lo, self.m.hi)
        step = np.clip(desired - self.target, -self.max_step, self.max_step)
        smoothed = self.target + self.alpha * (desired - self.target)
        self.target = np.where(self.m.is_slide, self.target + step, smoothed)
        data.ctrl[:] = self.target
        return self.target

    def set_targets(self, targets: np.ndarray, data: mujoco.MjData) -> np.ndarray:
        """Scripted use: absolute targets, still rate-limited on the slides."""
        desired = np.clip(targets, self.m.lo, self.m.hi)
        step = np.clip(desired - self.target, -self.max_step, self.max_step)
        self.target = np.where(self.m.is_slide, self.target + step, desired)
        data.ctrl[:] = self.target
        return self.target


__all__ = ["ActuatorMap", "actuator_map", "Commander", "RANGES", "joint_kind"]
