"""Named whole-body postures and helpers to place the robot on the ground in them.

Used by the crawlspace fit check and the recording verification traversal; not by
the learned policy.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .humanoid import RobotSpec, standing_pose

ROBOT_NQ = 39


def _joints(name: str) -> dict[str, float]:
    if name == "stand":
        return standing_pose(0.1)
    if name == "tall":
        p = standing_pose(0.05)
        for s in ("left", "right"):
            p[f"{s}_shoulder_pitch"] = 0.3
        return p
    if name == "crawl":
        # army crawl: thighs down, shins along the ground, upper arms down, forearms forward
        p = {}
        for s in ("left", "right"):
            p.update({f"{s}_hip_pitch": 1.57, f"{s}_knee": 1.57, f"{s}_ankle_pitch": -0.6,
                      f"{s}_shoulder_pitch": 1.57, f"{s}_elbow": 1.57, f"{s}_shoulder_roll": 0.15,
                      f"{s}_hip_roll": 0.1})
        return p
    raise KeyError(name)


SLIDES = {"stand": 1.0, "tall": 1.4, "crawl": 0.8}
ROOT_PITCH = {"stand": 0.0, "tall": 0.0, "crawl": math.pi / 2}


def pose_vector(name: str, actuators: list[str], spec: RobotSpec) -> np.ndarray:
    """Joint/segment values in actuator order for a named posture."""
    j = _joints(name)
    out = []
    for a in actuators:
        if a.endswith("_ext"):
            key = a.replace("left_", "").replace("right_", "").replace("_ext", "")
            tel = getattr(spec, key)
            scale = SLIDES[name] if key != "torso" else min(max(SLIDES[name], 0.85), 1.25)
            out.append(tel.nominal * (scale - 1.0))
        else:
            out.append(j.get(a, 0.0))
    return np.array(out)


def robot_top_and_bottom(model: mujoco.MjModel, data: mujoco.MjData, robot_geoms: np.ndarray) -> tuple[float, float]:
    """Highest and lowest points of the robot's collision shapes (exact for spheres, boxes, capsules)."""
    top, bot = -1e9, 1e9
    for g in np.nonzero(robot_geoms)[0]:
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        c = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        sz = model.geom_size[g]
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = sz[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ext = abs(R[2, 2]) * sz[1] + sz[0]
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ext = abs(R[2, 2]) * sz[1] + sz[0] * math.hypot(R[2, 0], R[2, 1])
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            ext = float(np.sum(np.abs(R[2, :]) * sz[:3]))
        else:
            continue
        top = max(top, c[2] + ext)
        bot = min(bot, c[2] - ext)
    return float(top), float(bot)


def place(model, data, amap, robot_geoms, name: str, x: float, y: float, ground_fn, spec: RobotSpec,
          yaw: float = 0.0) -> np.ndarray:
    """Set the robot into a posture at (x, y) with its lowest point resting on the ground."""
    vec = pose_vector(name, amap.names, spec)
    pitch = ROOT_PITCH[name]
    qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qz, qy)
    data.qpos[0:3] = (x, y, 3.0)
    data.qpos[3:7] = q
    data.qpos[7:ROBOT_NQ] = 0.0
    data.qpos[amap.qpos_adr] = vec
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)
    _, bot = robot_top_and_bottom(model, data, robot_geoms)
    data.qpos[2] += ground_fn(x, y) - bot + 0.003
    mujoco.mj_kinematics(model, data)
    return vec
