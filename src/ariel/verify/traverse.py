"""Verification traversal: a kinematic "puppet" pass over the whole showcase course.

This is NOT physics or learned behaviour. It exists to exercise the recording path
(recorder -> chunk on disk -> reload -> render) and every camera and lighting
situation (under the module, over the fissure, up the scarp) before training.
"""

from __future__ import annotations

import io
import json
import math

import mujoco
import numpy as np

from ..env.lunar import LunarRescueEnv
from ..robot.poses import ROBOT_NQ, ROOT_PITCH, pose_vector, robot_top_and_bottom
from ..world.course import CourseParams
from ..world.scene import hfield_bytes


def _keyframes(P: CourseParams) -> list[tuple[float, str]]:
    R = P.hab_radius
    return [
        (0.0, "stand"), (P.hab_x - R - 0.9, "stand"), (P.hab_x - R - 0.25, "crawl"),
        (P.hab_x + R + 0.25, "crawl"), (P.hab_x + R + 0.9, "stand"), (P.fissure_x - 0.45, "stand"),
        (P.fissure_x - 0.2, "tall"), (P.fissure_x + P.fissure_width + 0.3, "tall"), (P.scarp_x - 0.45, "tall"),
        (P.scarp_x + 0.55, "tall"), (P.mast_x - 0.45, "stand"), (P.mast_x + 0.5, "tall"),
        (P.goal_x - 1.1, "stand"),
    ]


def traverse(params: CourseParams = CourseParams(), speed: float = 0.6, hz: float = 50.0):
    env = LunarRescueEnv(seed=0, pool_size=1, fixed_params=params)
    env.reset(spawn=0)
    s = env.slot
    m, d = s.model, s.data
    course = s.course
    P = params
    keys = _keyframes(P)
    poses = {n: pose_vector(n, s.amap.names, env.spec) for n in ("stand", "tall", "crawl")}

    # pick the lane with the most room between settled debris
    loose_xy = []
    for b in course.loose:
        j = m.joint(b["name"] + "_free")
        loose_xy.append(d.qpos[j.qposadr[0]:j.qposadr[0] + 2].copy())
    lanes = np.linspace(-1.6, 1.6, 33)
    def room(y):
        return min([math.hypot(0, y - p[1]) if P.pad_end < p[0] < P.debris_end + 0.5 else 9.0 for p in loose_xy] + [9.0])
    lane = float(max(lanes, key=room))

    def ground(x: float) -> float:
        hs = [course.height_at(x + dx, lane) for dx in (-0.3, -0.12, 0.12, 0.3)]
        if P.fissure_x - 0.1 < x < P.fissure_x + P.fissure_width + 0.1:
            hs = [max(course.height_at(P.fissure_x - 0.05, lane), course.height_at(P.fissure_x + P.fissure_width + 0.05, lane))]
        return max(hs)

    total = keys[-1][0] / speed
    n = int(total * hz)
    xs = np.linspace(0.0, keys[-1][0], n)
    kx = np.array([k[0] for k in keys])
    frames = []
    for x in xs:
        i = int(np.clip(np.searchsorted(kx, x, side="right") - 1, 0, len(keys) - 2))
        u = (x - kx[i]) / max(kx[i + 1] - kx[i], 1e-6)
        u = u * u * (3 - 2 * u)
        a, b = keys[i][1], keys[i + 1][1]
        vec = poses[a] * (1 - u) + poses[b] * u
        pitch = ROOT_PITCH[a] * (1 - u) + ROOT_PITCH[b] * u
        d.qpos[0:3] = (x, lane, 3.0)
        d.qpos[3:7] = (math.cos(pitch / 2), 0, math.sin(pitch / 2), 0)
        d.qpos[7:ROBOT_NQ] = 0.0
        d.qpos[s.amap.qpos_adr] = vec
        mujoco.mj_kinematics(m, d)
        _, bot = robot_top_and_bottom(m, d, s.robot_geom)
        d.qpos[2] += ground(x) - bot + 0.004
        frames.append(d.qpos.copy())
    return env, np.array(frames, np.float32), lane


def record_and_reload(root: str, run_id: str, params: CourseParams = CourseParams()) -> tuple[bytes, dict]:
    """Push the traversal through the real recorder, read it back from disk, and build a render trace."""
    from ..record.recorder import WorkerRecorder, load_episode

    env, frames, lane = traverse(params)
    rec = WorkerRecorder(root, run_id, worker=0)
    meta = {**env.episode_meta(), "origin": "verification_puppet (kinematic, not physics or learned)"}
    rec.start(0, meta, 0, frames[0])
    for t in range(1, len(frames)):
        rec.step(0, np.zeros(env.slot.model.nu), 0.0, {}, -1, float(frames[t][0]), frames[t])
    row = rec.finish(0, "complete", "verification", len(frames), {})
    rec.flush()
    ep = load_episode(f"{root}/{run_id}", row)
    course = env.slot.course
    buf = io.BytesIO()
    np.savez_compressed(buf, qpos=ep["qpos"], fps=row["playback_hz"], xml=np.array(env.slot.xml),
                        hfield=np.frombuffer(hfield_bytes(course), dtype=np.uint8),
                        scene=np.array(json.dumps(course.scene_json())))
    info = {"frames_written": len(frames), "frames_reloaded": int(len(ep["qpos"])),
            "playback_hz": row["playback_hz"], "lane_y": round(lane, 2), "episode_id": row["episode_id"],
            "chunk": row["chunk"]}
    return buf.getvalue(), info
