"""Run frozen checkpoints on fixed courses (showcase and held-out layouts).

Evaluation episodes are labelled ``origin = "checkpoint_evaluation"`` and kept apart
from training recordings. The policy acts deterministically (mean action) unless
``stochastic`` is set; every evaluation episode is recorded the same way.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict

import numpy as np


def load_policy(path: str):
    import torch

    from .ppo import build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    actor_dim = sd["actor.0.weight"].shape[1]
    critic_dim = sd["critic.0.weight"].shape[1]
    act_dim = sd["log_std"].shape[0]
    model = build_model(actor_dim, critic_dim, act_dim, tuple(ck["config"]["hidden"]), 0.3)
    model.load_state_dict(sd)
    model.eval()
    mean, var = ck["norm_actor"]["mean"], ck["norm_actor"]["var"]
    norm = lambda x: np.clip((x - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)  # noqa: E731
    return model, norm, ck


def evaluate(checkpoint: str, params_list: list[dict], episodes_per_course: int = 2, stochastic: bool = False,
             lock_telescoping: bool = False, seed: int = 0) -> list[dict]:
    import torch

    from ..env.lunar import LunarRescueEnv
    from ..world.course import CourseParams
    from ..world.scene import hfield_bytes

    model, norm, ck = load_policy(checkpoint)
    results = []
    for pi, p in enumerate(params_list):
        env = LunarRescueEnv(seed=seed + pi, pool_size=1, fixed_params=CourseParams(**p))
        slides = env.slot.amap.is_slide
        for k in range(episodes_per_course):
            obs = env.reset(spawn=0)
            qpos = [env.slot.data.qpos.astype(np.float32).copy()]
            ret, info, steps = 0.0, {}, 0
            while True:
                with torch.no_grad():
                    d = model.dist(torch.from_numpy(norm(obs["actor"][None])))
                    a = (d.sample() if stochastic else d.mean)[0].numpy()
                if lock_telescoping:
                    a[slides] = 0.0
                obs, r, term, trunc, info = env.step(a)
                ret += r
                steps += 1
                if steps % 2 == 0:
                    qpos.append(env.slot.data.qpos.astype(np.float32).copy())
                if term or trunc:
                    break
            course = env.slot.course
            buf = io.BytesIO()
            np.savez_compressed(buf, qpos=np.stack(qpos), fps=25.0, xml=np.array(env.slot.xml),
                                hfield=np.frombuffer(hfield_bytes(course), dtype=np.uint8),
                                scene=np.array(json.dumps(course.scene_json())))
            results.append({
                "origin": "checkpoint_evaluation", "checkpoint": checkpoint.split("/")[-1],
                "policy_version": int(ck["version"]), "global_step": int(ck["global_step"]),
                "course_id": course.course_id, "course_params": asdict(course.params), "episode": k,
                "deterministic": not stochastic, "lock_telescoping": lock_telescoping,
                "reason": info.get("reason", ""), "steps": steps, "return": round(ret, 2),
                "max_x": round(float(info.get("frontier", 0.0)), 3), "passed": info.get("passed", {}),
                "success": info.get("reason") == "success", "trace": buf.getvalue(),
            })
    return results
