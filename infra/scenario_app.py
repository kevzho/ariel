"""Re-ordered showcase courses: the same four sections (crates, deck, fissure, slope) in other orders,
run with an existing checkpoint. Evaluation only; nothing here trains.

    uv run modal run infra/scenario_app.py::scenarios --orders DCFS,CFDS --seeds 24
"""


import io
import json
from pathlib import Path

import modal

HF = "https://huggingface.co/microsoft/mocapact-models/resolve/main"

mocapact_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libosmesa6", "libgl1", "libglfw3", "libegl1", "ffmpeg")
    .pip_install("pip==23.3.1", "setuptools==65.5.0", "wheel==0.38.4")
    .pip_install("numpy==1.23.5")
    .pip_install("torch==1.13.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "mujoco==2.2.2", "dm_control==1.0.7", "protobuf==3.20.1", "gym==0.21.0",
        "stable-baselines3==1.5.0", "pytorch-lightning==1.6.4", "torchmetrics==0.9.3",
        "cloudpickle==2.1.0", "h5py==3.8.0", "ml_collections==0.1.1", "imageio==2.31.1",
        "imageio-ffmpeg==0.4.8",
    )
    .run_commands(
        "pip install --no-deps git+https://github.com/microsoft/MoCapAct@main",
        # Modal's in-container runtime needs protobuf 4; Lightning's <=3.20.1 pin only matters for
        # tensorboard, which is not installed
        "pip install --no-deps protobuf==4.25.3",
    )
    .env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    .run_commands(
        # reference clips (CMU 2020 conversion used by MoCapAct) and the model zoo
        "python -c \"from dm_control.locomotion.mocap import cmu_mocap_data as c; print(c.get_path_for_cmu(version='2020'))\"",
        "mkdir -p /models && cd /models"
        f" && wget -q {HF}/multiclip_policy.tar.gz && tar -xzf multiclip_policy.tar.gz && rm multiclip_policy.tar.gz"
        f" && wget -q {HF}/transfer.tar.gz && tar -xzf transfer.tar.gz && rm transfer.tar.gz",
    )
)
image = mocapact_image.add_local_python_source("ascent")

app = modal.App("ascent-scenarios")
runs = modal.Volume.from_name("ascent-runs")
RUNS = Path("/runs")

ELEMENTS = {"C": "crates", "D": "deck", "F": "fissure", "S": "slope"}


def _jsonable(o):
    return o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o)


def ordered_params(cfg, level: dict, order: str):
    """The showcase course with its four sections placed in ``order`` (letters of ELEMENTS).

    Sizes, counts and the terrain seed are the showcase course's; only the positions change. The gaps
    are the ones the original order has: 4.6 m of lead before the first crate, 4.4 m from the end of
    the previous section to the deck centre, 6.8 m of run-up before the fissure, 2.7 m from the far
    edge of the previous section to the foot of the slope, and ``goal_gap`` to the astronaut. A section
    placed after the slope gets 1.5 m more, so the prior's height correction has finished blending in.
    ``order="CDFS"`` reproduces ``showcase_params`` exactly.
    """
    import dataclasses

    from ascent.mocap.env import showcase_params

    base = showcase_params(cfg, level)
    pad = base.pad_end
    cur, extra, kw = pad, 0.0, {}
    for el in order:
        add, extra = extra, 0.0
        if el == "C":
            b0 = cur + cfg.block_start + add
            kw["block_start"] = b0 - pad
            cur = b0 + max(cfg.lane_blocks - 1, 0) * cfg.block_spacing
            kw["debris_end"] = cur + 1.2
        elif el == "D":
            kw["hab_x"] = cur + 4.4 + add
            cur = kw["hab_x"] + base.roof_len / 2
        elif el == "F":
            kw["fissure_x"] = cur + 6.8 + add
            cur = kw["fissure_x"] + base.fissure_width
        elif el == "S":
            kw["scarp_x"] = cur + 2.7 + add
            cur = kw["scarp_x"] + cfg.ramp_run
            extra = 1.5
        else:
            raise ValueError(f"unknown section {el!r}")
    goal = cur + cfg.goal_gap
    kw.update(goal_x=goal, mast_x=cur + 0.8, broken_end=cur + max(2.4, cfg.goal_gap - 1.5))
    return dataclasses.replace(base, **kw)


@app.function(image=image, cpu=4.0, memory=8192, timeout=3600, volumes={RUNS: runs})
def scenario_job(out_id: str, job: dict) -> dict:
    """One evaluation episode of a checkpoint on the showcase course with its sections re-ordered.

    job: order, seed, level, source {"run_id", "checkpoint"}, size (h, w), every (render each n-th step).
    """
    import io

    import imageio
    import numpy as np
    import torch
    from dm_control.mujoco import engine

    from ascent.mocap.env import ReshapeEnv
    from ascent.mocap.export import export_poses, export_rig, pack_terrain
    from ascent.mocap.train import load_policy, make_env_cfg

    torch.set_num_threads(1)
    runs.reload()
    src, order, seed = job["source"], job["order"], int(job["seed"])
    policy, ck = load_policy(str(RUNS / src["run_id"] / "checkpoints" / src["checkpoint"]))
    cfg = make_env_cfg(ck["config"]["env"])
    level = job["level"]
    params = ordered_params(cfg, level, order)
    env = ReshapeEnv(cfg, seed=seed, fixed_params=params)
    env.teach = 0.0
    obs = env.reset(spawn=0)
    physics = env.env.physics
    h, w = job.get("size", (540, 960))
    every = int(job.get("every", 2))
    cam = engine.MovableCamera(physics, height=h, width=w)
    zc = env.course.height_at(env.x, 0.0)
    qpos, frames, track, info = [env.qpos.astype(np.float32).copy()], [], [], {"reason": ""}
    for k in range(cfg.episode_steps):
        if k % every == 0:
            # the camera follows the robot along the course; its height follows the ground, smoothed,
            # and ignores the fissure floor
            g = max(env.course.height_at(env.x - 1.0, 0.0), env.course.height_at(env.x + 1.0, 0.0))
            zc += 0.15 * (g - zc)
            cam.set_pose(lookat=[env.x, 0.0, zc + 1.0], distance=6.0, azimuth=90.0, elevation=-8.0)
            frames.append(cam.render().copy())
            track.append([env.x, env.y, env.head_top - zc, env.expected_head, float(env.stand_count)])
        obs, r, term, trunc, info = env.step(policy(obs["actor"]))
        qpos.append(env.qpos.astype(np.float32).copy())
        if term or trunc:
            break
    track = np.array(track, np.float32)
    P = env.course.params
    passed = {k: bool(v) for k, v in info.get("passed", {}).items()}
    max_x = float(track[:, 0].max())
    usable = all(passed.values()) and max_x >= env.stop_x - 2.0
    summary = {"name": f"{order}_s{seed}", "order": order, "sections": [ELEMENTS[c] for c in order], "seed": seed,
               "origin": "checkpoint_evaluation", "controller": "learned layer + frozen MoCapAct prior",
               "run_id": src["run_id"], "checkpoint": src["checkpoint"], "policy_version": int(ck["version"]),
               "level": level, "gravity": 9.81, "reason": info["reason"], "steps": len(qpos) - 1,
               "seconds": round((len(qpos) - 1) * 0.03, 1), "max_x": round(max_x, 2), "stop_x": round(env.stop_x, 2),
               "goal_x": round(P.goal_x, 2), "passed": passed, "usable": usable,
               "stand_steps_max": int(track[:, 4].max()), "fps": round(1.0 / (0.03 * every), 3),
               "layout": {"first_crate_x": round(P.pad_end + P.block_start, 2), "deck_x": round(P.hab_x, 2),
                          "fissure_x": round(P.fissure_x, 2), "slope_x": round(P.scarp_x, 2)},
               "course_params": {k: getattr(P, k) for k in ("hab_clearance", "fissure_width", "scarp_height",
                                                             "block_offset", "ramp_run")},
               "export": f"{out_id}/{order}_s{seed}"}
    out = RUNS / out_id / f"{order}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    imageio.mimwrite(buf, frames, format="mp4", fps=summary["fps"], quality=8, macro_block_size=1)
    mp4 = buf.getvalue()
    (out / "preview.mp4").write_bytes(mp4)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    np.save(out / "track.npy", track)
    if usable:
        q = np.stack(qpos)
        pz = io.BytesIO()
        np.savez_compressed(pz, **export_poses(env, q, fps=1.0 / 0.03), qpos=q)
        (out / "poses.npz").write_bytes(pz.getvalue())
        (out / "rig.json").write_text(json.dumps(export_rig(env)))
        (out / "scene.json").write_text(json.dumps(env.course.scene_json(), default=_jsonable))
        (out / "terrain.npz").write_bytes(pack_terrain(env.course))
    runs.commit()
    keep = usable or job.get("fetch_all", False)
    return {"summary": summary, "track": track.tolist() if keep else [], "mp4": mp4 if keep else b""}


@app.local_entrypoint()
def scenarios(orders: str = "DCFS,CFDS,FCDS,FDCS,DFCS,CFSD,DFSC,FSCD,CDSF", seeds: int = 24, first_seed: int = 200,
              run_id: str = "athletic-s0-20261011-031906", checkpoint: str = "policy_v001300.pt",
              levels: str = '{"blocks":0.8,"low":1.0,"fissure":0.8,"scarp":0.9}', fetch_all: bool = False, tag: str = ""):
    """Sample evaluation episodes of one checkpoint on re-ordered courses. Usable runs (all four sections
    passed and within 2 m of the stop point) are saved locally with their per-frame track; every episode
    stays on the volume. Prints k of n per order."""
    import time

    out_id = time.strftime("scenarios-%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    lv = json.loads(levels)
    jobs = [{"order": o, "seed": s, "level": dict(lv), "source": {"run_id": run_id, "checkpoint": checkpoint},
             "fetch_all": fetch_all}
            for o in orders.split(",") for s in range(first_seed, first_seed + seeds)]
    local = Path("runs") / out_id
    local.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in scenario_job.map([out_id] * len(jobs), jobs, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:700], flush=True)
            continue
        s = r["summary"]
        rows.append(s)
        print(json.dumps({k: s[k] for k in ("name", "reason", "seconds", "max_x", "stop_x", "passed", "usable",
                                            "stand_steps_max")}, default=_jsonable), flush=True)
        if r["mp4"]:
            (local / f"{s['name']}.mp4").write_bytes(r["mp4"])
            (local / f"{s['name']}.json").write_text(json.dumps({**s, "track": r["track"]}, default=_jsonable))
        (local / "scenarios.json").write_text(json.dumps(rows, indent=1, default=_jsonable))
    for o in orders.split(","):
        grp = [s for s in rows if s["order"] == o]
        ok = [s for s in grp if s["usable"]]
        keys = ("blocks", "low", "fissure", "scarp")
        print(f"SUMMARY {o} ({'-'.join(ELEMENTS[c] for c in o)}): {len(ok)} of {len(grp)} usable; passed "
              + ", ".join(f"{k} {sum(bool(s['passed'].get(k)) for s in grp)}" for k in keys)
              + f"; seeds {[s['seed'] for s in ok]}", flush=True)
    print("->", str(local))
