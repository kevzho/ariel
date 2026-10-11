"""Settle a course under physics: the robot stands on the pad while loose debris comes to rest.

Produces a short trace (labelled as a settle run, not learned behaviour) used for
course look-development and as the cached reset state for training.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict

import mujoco
import numpy as np

from ..robot.control import Commander, actuator_map
from ..robot.humanoid import RobotSpec
from .course import CourseParams, generate_course, vista_grid
from .scene import HFIELD_FILE, build_world_xml, hfield_bytes, model_assets


def settle_course(params: CourseParams = CourseParams(), seconds: float = 6.0, fps: float = 50.0) -> dict:
    spec = RobotSpec()
    course = generate_course(params)
    xml, meta = build_world_xml(spec, course=course)
    model = mujoco.MjModel.from_xml_string(xml, model_assets(course))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    amap = actuator_map(model, meta["actuators"], spec)
    cmd = Commander(amap, spec, control_dt=1.0 / fps)
    cmd.reset(data)
    steps = int(round(1.0 / (fps * model.opt.timestep)))
    frames = int(seconds * fps)
    qpos = np.zeros((frames, model.nq), np.float32)
    for f in range(frames):
        cmd.set_targets(amap.default, data)
        mujoco.mj_step(model, data, nstep=steps)
        qpos[f] = data.qpos
    loose_speed = 0.0
    moving = {}
    for name in meta["loose"]:
        j = model.joint(name + "_free")
        sp = float(np.linalg.norm(data.qvel[j.dofadr[0]:j.dofadr[0] + 3]))
        loose_speed = max(loose_speed, sp)
        p_end = data.qpos[j.qposadr[0]:j.qposadr[0] + 3]
        moved = float(np.linalg.norm(p_end[:2] - qpos[0, j.qposadr[0]:j.qposadr[0] + 2]))
        if sp > 0.02 or moved > 0.15:
            moving[name] = {"speed": round(sp, 3), "moved_m": round(moved, 3),
                            "pos": [round(float(v), 3) for v in p_end]}
    w, x, y, _ = data.qpos[3:7]
    return {
        "xml": xml, "qpos": qpos, "fps": fps, "course": course, "meta": meta,
        "metrics": {
            "label": "course settle run (robot holds a standing pose; not learned)",
            "course_id": course.course_id, "params": asdict(params),
            "seconds": seconds, "nq": int(model.nq), "nbody": int(model.nbody),
            "loose_bodies": len(meta["loose"]), "max_loose_speed_end_mps": round(loose_speed, 4),
            "still_moving": moving,
            "robot_upright_end": round(float(1 - 2 * (x * x + y * y)), 4),
            "pelvis_end": [round(float(v), 3) for v in data.qpos[:3]],
            "contacts_end": int(data.ncon),
        },
    }


def pack_trace(res: dict) -> bytes:
    """A self-contained trace: model XML, heightfield asset, scene description and states."""
    course = res["course"]
    buf = io.BytesIO()
    np.savez_compressed(buf, qpos=res["qpos"], fps=res["fps"], xml=np.array(res["xml"]),
                        hfield=np.frombuffer(hfield_bytes(course), dtype=np.uint8),
                        scene=np.array(json.dumps(course.scene_json())))
    return buf.getvalue()


def pack_terrain(course, vista_res: float = 0.2, near_width: float = 9.0, near_dy: float = 0.05) -> bytes:
    """Course grid, near-field strips on both sides (same x samples, so edges match exactly)
    and a coarse vista for the horizon."""
    from .course import HFIELD_RES, terrain_function
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
        # the shared row is copied from the physics grid so the seam is exact
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


def load_model(trace: np.lib.npyio.NpzFile) -> mujoco.MjModel:
    assets = {HFIELD_FILE: trace["hfield"].tobytes()} if "hfield" in trace.files else {}
    return mujoco.MjModel.from_xml_string(str(trace["xml"]), assets)
