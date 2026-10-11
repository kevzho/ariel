"""Scripted body check (NOT learned behaviour).

Drives the humanoid through a fixed sequence of poses and segment lengths to
prove that the body stands, telescopes and crouches under physics during one
continuous episode. Used for A1 evidence and visual review only.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from ..world.scene import build_world_xml
from .control import Commander, actuator_map
from .humanoid import RobotSpec, standing_pose


def _targets(meta: dict, pose: dict, scales: dict, spec: RobotSpec) -> np.ndarray:
    out = []
    for name in meta["actuators"]:
        if name.endswith("_ext"):
            key = name.replace("left_", "").replace("right_", "").replace("_ext", "")
            tel = getattr(spec, key)
            out.append(tel.nominal * (scales.get(key, 1.0) - 1.0))
        else:
            out.append(pose.get(name, 0.0))
    return np.array(out)


def _pose(crouch=0.1, arms_pitch=0.1, arms_roll=0.12, elbow=0.35, waist_pitch=0.0):
    p = standing_pose(crouch)
    for s in ("left", "right"):
        p[f"{s}_shoulder_pitch"] = arms_pitch
        p[f"{s}_shoulder_roll"] = arms_roll
        p[f"{s}_elbow"] = elbow
    p["waist_pitch"] = waist_pitch
    return p


ALL = lambda v: {"torso": v, "thigh": v, "shin": v, "upper_arm": v, "forearm": v}  # noqa: E731

# (time, pose, segment scales)
def _program(crouch: float):
    return [
        (0.0, _pose(), ALL(1.0)),
        (1.5, _pose(), ALL(1.0)),
        (3.5, _pose(arms_pitch=2.6, arms_roll=0.25, elbow=0.1), {**ALL(1.4), "torso": 1.25}),
        (5.5, _pose(arms_pitch=2.6, arms_roll=0.25, elbow=0.1), {**ALL(1.4), "torso": 1.25}),
        (8.5, _pose(crouch=crouch, arms_pitch=0.7, elbow=0.8, waist_pitch=0.1), {**ALL(0.8), "torso": 0.85}),
        (10.0, _pose(crouch=crouch, arms_pitch=0.7, elbow=0.8, waist_pitch=0.1), {**ALL(0.8), "torso": 0.85}),
        (12.5, _pose(), ALL(1.0)),
        (14.0, _pose(), ALL(1.0)),
    ]


PROGRAM = _program(0.7)


def _interp(t: float, keys: list[np.ndarray]) -> np.ndarray:
    times = [k[0] for k in PROGRAM]
    if t >= times[-1]:
        return keys[-1]
    i = max(0, np.searchsorted(times, t, side="right") - 1)
    t0, t1 = times[i], times[i + 1]
    a = (t - t0) / max(t1 - t0, 1e-9)
    a = a * a * (3 - 2 * a)            # smoothstep
    return keys[i] * (1 - a) + keys[i + 1] * a


BALANCE = {"ankle_kp": 2.0, "ankle_kd": 0.8, "hip_k": 0.0, "crouch": 0.7}


def run_body_check(spec: RobotSpec = RobotSpec(), fps: float = 50.0, course=None, balance: dict | None = None) -> dict:
    bal = {**BALANCE, **(balance or {})}
    global PROGRAM
    PROGRAM = _program(bal["crouch"])
    from ..world.scene import model_assets
    xml, meta = build_world_xml(spec, course=course)
    model = mujoco.MjModel.from_xml_string(xml, model_assets(course))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    amap = actuator_map(model, meta["actuators"], spec)
    cmd = Commander(amap, spec, control_dt=1.0 / fps)
    cmd.reset(data)
    keys = [_targets(meta, p, s, spec) for _, p, s in PROGRAM]

    duration = PROGRAM[-1][0]
    steps_per_frame = int(round(1.0 / (fps * model.opt.timestep)))
    n_frames = int(duration * fps)
    qpos = np.zeros((n_frames, model.nq), np.float32)
    ctrl = np.zeros((n_frames, model.nu), np.float32)
    pelvis = model.body("pelvis").id
    head_geom = model.geom("head").id
    slide_dofs = amap.qvel_adr[amap.is_slide]
    rot_dofs = amap.qvel_adr[~amap.is_slide]

    max_pen = 0.0
    max_slide_speed = 0.0
    max_rot_speed = 0.0
    energy = 0.0
    ankle_ids = [i for i, a in enumerate(meta["actuators"]) if a.endswith("ankle_pitch")]
    hip_ids = [i for i, a in enumerate(meta["actuators"]) if a.endswith("hip_pitch")]
    feet = [model.site("left_foot_site").id, model.site("right_foot_site").id]
    prev_err = 0.0
    wall = time.perf_counter()
    for f in range(n_frames):
        target = _interp(f / fps, keys).copy()
        # Ankle strategy: keep the centre of mass over the feet (scripted check only).
        com_x = data.subtree_com[pelvis][0]
        support_x = float(np.mean([data.site_xpos[i][0] for i in feet]))
        err = com_x - support_x
        derr = (err - prev_err) * fps
        prev_err = err
        target[ankle_ids] -= bal["ankle_kp"] * err + bal["ankle_kd"] * derr
        target[hip_ids] += bal["hip_k"] * err
        cmd.set_targets(target, data)
        mujoco.mj_step(model, data, nstep=steps_per_frame)
        energy += float(np.sum(np.abs(data.actuator_force * data.qvel[amap.qvel_adr]))) / fps
        if data.ncon:
            max_pen = max(max_pen, float(-min(c.dist for c in data.contact[:data.ncon])))
        max_slide_speed = max(max_slide_speed, float(np.max(np.abs(data.qvel[slide_dofs]))))
        max_rot_speed = max(max_rot_speed, float(np.max(np.abs(data.qvel[rot_dofs]))))
        qpos[f] = data.qpos
        ctrl[f] = data.ctrl
    wall = time.perf_counter() - wall

    heights = qpos[:, 2]
    head_z = []
    d2 = mujoco.MjData(model)
    for q in qpos[:: int(fps / 5)]:
        d2.qpos[:] = q
        mujoco.mj_kinematics(model, d2)
        head_z.append(float(d2.geom_xpos[head_geom][2]))
    upright = []
    for q in qpos:
        w, x, y, z = q[3:7]
        upright.append(1 - 2 * (x * x + y * y))   # z component of the body z axis
    metrics = {
        "label": "scripted body check (not learned)",
        "duration_s": duration,
        "physics_steps": n_frames * steps_per_frame,
        "wall_seconds": round(wall, 3),
        "realtime_factor": round(duration / wall, 2),
        "pelvis_height_min_max": [round(float(heights.min()), 3), round(float(heights.max()), 3)],
        "head_height_min_max": [round(min(head_z), 3), round(max(head_z), 3)],
        "min_uprightness": round(float(min(upright)), 3),
        "pelvis_xy_drift_m": round(float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])), 3),
        "course_id": course.course_id if course is not None else "flat-test-floor",
        "max_penetration_m": round(max_pen, 4),
        "max_slide_speed_mps": round(max_slide_speed, 3),
        "max_rotary_speed_radps": round(max_rot_speed, 2),
        "actuator_energy_J": round(energy, 1),
    }
    return {"xml": xml, "qpos": qpos, "ctrl": ctrl, "fps": fps, "metrics": metrics,
            "actuators": meta["actuators"], "course": course, "meta": meta}
