"""Export the telescoping CMU walker and a recorded episode for the Blender renderer.

The Blender side (``ascent.art``) receives the same two things as for the first robot: a rig
(every body with its collision shapes in the body frame) and one world pose per body per frame,
computed here with forward kinematics from recorded ``qpos``. Each telescoping segment adds a
helper frame ``tele_<label>`` that sits at the moving joint but keeps the parent bone's
orientation, so the polished inner tube can be drawn sliding out of its housing.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .walker import DEFAULT_RANGE, TELESCOPES


def _vec(v, default):
    return [float(x) for x in (default if v is None else v)]


def export_rig(env) -> dict:
    walker = env.walker
    root = walker.mjcf_model
    ranges = {**DEFAULT_RANGE, **env.cfg.ranges}
    bodies = []
    for body in root.find_all("body"):
        geoms = []
        for g in body.geom:
            name = g.name or ""
            if name.startswith(("tube_", "housing_")):
                continue
            geoms.append({"name": name, "type": g.type or "capsule", "size": _vec(g.size, (0.03, 0.0, 0.0)),
                          "pos": _vec(g.pos, (0, 0, 0)), "quat": _vec(g.quat, (1, 0, 0, 0))})
        parent = body.parent
        bodies.append({"name": body.name, "parent": parent.name if getattr(parent, "tag", "") == "body" else "",
                       "pos": _vec(body.pos, (0, 0, 0)), "geoms": geoms})
    telescopes = []
    for child, parent_geom, label, group in TELESCOPES:
        b = root.find("body", child)
        offset = np.asarray(b.pos, dtype=float)
        length = float(np.linalg.norm(offset))
        lo, hi = ranges[group]
        radius = float(root.find("geom", parent_geom).size[0]) if parent_geom else 0.06
        telescopes.append({"label": label, "group": group, "child": child, "parent": b.parent.name,
                           "parent_geom": parent_geom, "lo": float(lo), "hi": float(hi), "length": length,
                           "axis": (offset / length).tolist(), "radius": radius})
        bodies.append({"name": f"tele_{label}", "parent": b.parent.name, "pos": [0, 0, 0], "geoms": []})
    return {"kind": "cmu", "bodies": bodies, "telescopes": telescopes}


def export_poses(env, qpos: np.ndarray, fps: float) -> dict:
    """World pose of every walker body (and telescope helper frame) for each recorded frame."""
    physics = env.env.physics
    root = env.walker.mjcf_model
    bodies = root.find_all("body")
    names = [b.name for b in bodies]
    bound = physics.bind(bodies)
    tele = [(names.index(child), names.index(root.find("body", child).parent.name))
            for child, _, _, _ in TELESCOPES]
    n = len(names) + len(tele)
    pos = np.zeros((len(qpos), n, 3), np.float32)
    quat = np.zeros((len(qpos), n, 4), np.float32)
    saved = physics.data.qpos.copy()
    for i, q in enumerate(qpos):
        physics.data.qpos[:] = q
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
        xp, xq = np.asarray(bound.xpos), np.asarray(bound.xquat)
        pos[i, :len(names)], quat[i, :len(names)] = xp, xq
        for k, (c, p) in enumerate(tele):
            pos[i, len(names) + k], quat[i, len(names) + k] = xp[c], xq[p]
    physics.data.qpos[:] = saved
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    return {"names": np.array(names + [f"tele_{t[2]}" for t in TELESCOPES]), "pos": pos, "quat": quat,
            "fps": float(fps)}


def pack_terrain(course, vista_res: float = 0.2, near_width: float = 9.0, near_dy: float = 0.05) -> bytes:
    """Course grid, near-field strips on both sides and a coarse vista (same layout as
    ``ascent.world.settle.pack_terrain``, without importing the MuJoCo 3 stack)."""
    import io

    from ..world.course import HFIELD_RES, terrain_function, vista_grid

    xs, ys, vz = vista_grid(course.params, res=vista_res)
    f = terrain_function(course.params)
    x0, y0, x1, y1 = course.extent
    cx = x0 + HFIELD_RES * np.arange(course.heights.shape[1])
    n = int(round(near_width / near_dy))
    strips = {}
    for side, ylim in (("near_lo", y0), ("near_hi", y1)):
        sgn = -1 if side == "near_lo" else 1
        ny = ylim + sgn * near_dy * np.arange(n + 1)
        if sgn < 0:
            ny = ny[::-1]
        X, Y = np.meshgrid(cx, ny)
        z = f(X, Y).astype(np.float32)
        if sgn < 0:
            z[-1] = course.heights[0]
        else:
            z[0] = course.heights[-1]
        strips[side + "_ys"] = ny.astype(np.float32)
        strips[side + "_z"] = z
    buf = io.BytesIO()
    np.savez_compressed(buf, heights=course.heights, vista_xs=xs, vista_ys=ys, vista_z=vz,
                        near_xs=cx.astype(np.float32), **strips)
    return buf.getvalue()
