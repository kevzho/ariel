"""Export a MuJoCo model's bodies and geoms, and recorded body poses, for Blender.

The Blender side (``ariel.art``) never imports MuJoCo. It receives:

* ``rig``: every body with its parent and its geoms in the body frame;
* ``poses``: world position and quaternion of every body for each frame,
  computed here with forward kinematics from recorded ``qpos``.
"""

from __future__ import annotations

import mujoco
import numpy as np

GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane", mujoco.mjtGeom.mjGEOM_HFIELD: "hfield",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}


def export_rig(model: mujoco.MjModel) -> dict:
    bodies = []
    for b in range(1, model.nbody):
        body = model.body(b)
        geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b:
                continue
            geom = model.geom(g)
            mat = model.mat(model.geom_matid[g]).name if model.geom_matid[g] >= 0 else ""
            geoms.append({
                "name": geom.name,
                "type": GEOM_TYPES.get(int(model.geom_type[g]), "other"),
                "size": [float(x) for x in model.geom_size[g]],
                "pos": [float(x) for x in model.geom_pos[g]],
                "quat": [float(x) for x in model.geom_quat[g]],
                "material": mat,
            })
        joints = []
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
            joints.append({
                "name": model.joint(j).name,
                "type": int(model.jnt_type[j]),
                "axis": [float(x) for x in model.jnt_axis[j]],
                "range": [float(x) for x in model.jnt_range[j]],
                "qposadr": int(model.jnt_qposadr[j]),
            })
        bodies.append({
            "name": body.name,
            "parent": model.body(int(model.body_parentid[b])).name if model.body_parentid[b] > 0 else "",
            "geoms": geoms,
            "joints": joints,
        })
    return {"bodies": bodies}


def body_poses(model: mujoco.MjModel, qpos: np.ndarray) -> dict:
    """World pose of every body (excluding the world body) for each frame."""
    data = mujoco.MjData(model)
    n = model.nbody - 1
    pos = np.zeros((len(qpos), n, 3), np.float32)
    quat = np.zeros((len(qpos), n, 4), np.float32)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        pos[i] = data.xpos[1:]
        quat[i] = data.xquat[1:]
    return {"names": [model.body(b).name for b in range(1, model.nbody)], "pos": pos, "quat": quat,
            "qpos": np.asarray(qpos, np.float32)}
