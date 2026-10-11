"""Modal jobs for the MoCapAct-based controller (pretrained human-motion prior + our course).

MoCapAct (Wagener et al., NeurIPS 2022, MIT) ships a multi-clip "neural probabilistic motor
primitive" policy for the dm_control CMU humanoid. Its low-level decoder maps
(proprioception, 60-d motor intention) -> 56 joint targets. We keep that decoder frozen and
learn the layer above it on the lunar course.

The pinned stack (dm_control 1.0.7, MuJoCo 2.2.2, SB3 1.5, Lightning 1.6) is what the
checkpoints were saved with, so it lives in its own image.

    uv run modal run infra/mocapact_app.py::probe
"""

from __future__ import annotations

import json
import time
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

app = modal.App("ascent-mocapact")
runs = modal.Volume.from_name("ariel-runs", create_if_missing=True)
RUNS = Path("/runs")


def _jsonable(o):
    return o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o)


def _tree(root: str, depth: int = 5) -> list[str]:
    import os
    out = []
    base = root.rstrip("/").count("/")
    for d, dirs, files in os.walk(root):
        if d.count("/") - base >= depth:
            dirs[:] = []
        for f in files:
            p = os.path.join(d, f)
            out.append(f"{p} {os.path.getsize(p)}")
    return out


@app.function(image=mocapact_image, cpu=8.0, memory=16384, timeout=3600, volumes={RUNS: runs})
def probe_job(run_id: str, speed: float = 1.2, steps: int = 400) -> dict:
    """Does the pretrained hierarchy walk? Earth vs Moon gravity, with video."""
    import glob
    import os

    import imageio
    import numpy as np
    import torch
    from dm_control.locomotion.tasks.reference_pose import tracking
    from mocapact.distillation import model as mmodel
    from mocapact.envs import dm_control_wrapper
    from mocapact.sb3 import utils
    from mocapact.tasks import velocity_control
    from stable_baselines3.common.utils import obs_as_tensor

    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"files": _tree("/models")}

    roots = sorted({os.path.dirname(p) for p in glob.glob("/models/**/best_model.zip", recursive=True)})
    report["transfer_roots"] = roots
    root = next(r for r in roots if "velocity_control" in r and "general_low_level" in r)

    class FixedVelocity(velocity_control.VelocityControl):
        def _sample_move_speed(self, random_state):
            self._move_speed, self._move_angle, self._move_speed_counter = speed, 0.0, 0

    ctor = dm_control_wrapper.DmControlWrapper.make_env_constructor(FixedVelocity)
    env = ctor(task_kwargs=dict(physics_timestep=tracking.DEFAULT_PHYSICS_TIMESTEP, control_timestep=0.03,
                                max_speed=4.5, reward_margin=0.75, direction_exponent=7.0,
                                steps_before_changing_velocity=10**9),
               environment_kwargs=dict(time_limit=0.03 * steps, random_state=0), arena_size=(200.0, 200.0))
    report["obs_keys"] = {k: list(v.shape) for k, v in env.observation_space.spaces.items()}
    report["action_dim"] = int(env.action_space.shape[0])

    high = utils.load_policy(root, list(env.observation_space.keys()), device="cpu")
    ll_path = os.path.join(root, "low_level_policy.ckpt")
    distilled = mmodel.NpmpPolicy.load_from_checkpoint(ll_path, map_location="cpu")
    report["embed_size"] = int(distilled.embed_size)
    report["decoder_keys"] = list(distilled.features_extractor._extractors["decoder"]._keys) \
        if hasattr(distilled.features_extractor, "_extractors") else None
    report["decoder"] = str(distilled.action_decoder)[:2000]
    report["state_dict"] = {k: list(v.shape) for k, v in distilled.state_dict().items()}

    physics = env.dm_env.physics
    report["cameras"] = [physics.model.id2name(i, "camera") for i in range(physics.model.ncam)]
    cam = next((i for i, n in enumerate(report["cameras"]) if "side" in n), 1)
    (out / "walker.xml").write_text(env.dm_env.task._walker.mjcf_model.to_xml_string())

    results = {}
    for label, g in (("earth", 9.81), ("moon", 1.62)):
        obs = env.reset()
        physics = env.dm_env.physics
        physics.model.opt.gravity[:] = (0.0, 0.0, -g)
        frames, xs, zs, t0 = [], [], [], time.time()
        done, k, ret = False, 0, 0.0
        while not done and k < steps:
            embed, _ = high.predict(obs, deterministic=True)
            embed = np.clip(embed, -3.0, 3.0)
            with torch.no_grad():
                t_obs = obs_as_tensor({kk: v.astype(np.float32)[None] for kk, v in obs.items()}, "cpu")
                act = distilled.low_level_policy(t_obs, torch.tensor(embed[None], dtype=torch.float32))[0].numpy()
            obs, r, done, _ = env.step(np.clip(act, -1.0, 1.0).astype(np.float32))
            ret += r
            root_pos = physics.named.data.xpos["walker/root"]
            xs.append(float(root_pos[0])); zs.append(float(root_pos[2]))
            if k % 2 == 0:
                frames.append(physics.render(height=360, width=640, camera_id=cam))
            k += 1
        dt = time.time() - t0
        imageio.mimwrite(out / f"probe_{label}.mp4", frames, fps=17, quality=8)
        results[label] = {"steps": k, "terminated_early": bool(done and k < steps), "return": float(ret),
                          "distance_x": xs[-1] - xs[0], "mean_speed": (xs[-1] - xs[0]) / (k * 0.03),
                          "min_root_z": min(zs), "mean_root_z": float(np.mean(zs)), "steps_per_s": k / dt}
    report["results"] = results
    (out / "probe.json").write_text(json.dumps(report, indent=1, default=_jsonable))
    runs.commit()
    return report


@app.local_entrypoint()
def probe(speed: float = 1.2, steps: int = 400):
    run_id = time.strftime("mocapact-probe-%Y%m%d-%H%M%S")
    rep = probe_job.remote(run_id, speed, steps)
    local = Path("runs") / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "probe.json").write_text(json.dumps(rep, indent=1, default=_jsonable))
    for name in ("probe_earth.mp4", "probe_moon.mp4", "walker.xml"):
        data = b"".join(runs.read_file(f"{run_id}/{name}"))
        (local / name).write_bytes(data)
    short = {k: v for k, v in rep.items() if k not in ("files", "state_dict", "decoder")}
    print(json.dumps(short, indent=1, default=_jsonable))
    print("files:", len(rep["files"]), "->", str(local))


# the ASCENT package (course generator + ascent.mocap) on top of the pinned stack
course_image = mocapact_image.add_local_python_source("ascent")


def _transfer_root() -> str:
    import glob
    import os
    roots = sorted({os.path.dirname(p) for p in glob.glob("/models/**/best_model.zip", recursive=True)})
    return next(r for r in roots if "velocity_control" in r and "general_low_level" in r)


@app.function(image=course_image, cpu=8.0, memory=16384, timeout=3600, volumes={RUNS: runs})
def course_smoke_job(run_id: str, steps: int = 500, telescoping: bool = True, speed: float = 1.0) -> dict:
    """Telescoping CMU humanoid on the showcase lunar course, driven by the frozen MoCapAct
    walking hierarchy (no learning). Checks the walker/arena/task modules and records a video."""
    import os

    import imageio
    import numpy as np
    import torch
    from dm_control import composer
    from mocapact.distillation import model as mmodel
    from mocapact.sb3 import utils
    from stable_baselines3.common.utils import obs_as_tensor

    from ascent.mocap.arena import LunarCourse
    from ascent.mocap.task import WALK_POLICY_KEYS, CourseTask, StandInitializer
    from ascent.mocap.walker import TelescopingCMUHumanoid
    from ascent.world.course import CourseParams, generate_course

    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    course = generate_course(CourseParams())
    init = StandInitializer()
    init.spawn = (course.start[0], course.start[1], course.height_at(*course.start))
    walker = TelescopingCMUHumanoid(initializer=init, telescoping=telescoping)
    arena = LunarCourse(course)
    task = CourseTask(walker, arena)
    env = composer.Environment(task, random_state=0, strip_singleton_obs_buffer_dim=True)
    ts = env.reset()
    build_s = time.time() - t0
    obs_shapes = {k: list(np.asarray(v).shape) for k, v in ts.observation.items()}
    walk_keys = list(WALK_POLICY_KEYS)

    root = _transfer_root()
    high = utils.load_policy(root, walk_keys, device="cpu")
    distilled = mmodel.NpmpPolicy.load_from_checkpoint(os.path.join(root, "low_level_policy.ckpt"), map_location="cpu")
    physics = env.physics
    cams = [physics.model.id2name(i, "camera") for i in range(physics.model.ncam)]
    cam = next((i for i, n in enumerate(cams) if "side" in n), 1)
    n_act = len(walker.actuators)
    slide_lo = np.array([j.range[0] for j in walker.slide_joints]) if walker.n_slides else np.zeros(0)
    slide_hi = np.array([j.range[1] for j in walker.slide_joints]) if walker.n_slides else np.zeros(0)
    tele = np.zeros(walker.n_slides)
    frames, xs, zs, tel_log = [], [], [], []
    physics_error = None
    t1 = time.time()
    for k in range(steps):
        task.command = np.array([speed, 0.0, 1.0, (k % 333) / 333.0])
        o = {kk: np.asarray(ts.observation[kk], np.float32).ravel() for kk in walk_keys if kk != "target_obs"}
        o["target_obs"] = task.command.astype(np.float32)
        t_obs = obs_as_tensor({kk: v[None] for kk, v in o.items()}, "cpu")
        with torch.no_grad():
            z = high.policy._predict(t_obs, deterministic=True).clamp(-3, 3)
            q = distilled.low_level_policy(t_obs, z)[0].numpy()
        if walker.n_slides:      # slow scripted reshaping sweep (rate-limited like the learned version)
            goal = np.where(np.sin(k * 0.03 * 0.6) > 0, slide_hi, slide_lo) * 0.6
            tele = tele + np.clip(goal - tele, -0.25 * 0.03, 0.25 * 0.03)
        try:
            ts = env.step(np.concatenate([np.clip(q, -1, 1), tele]))
        except Exception as exc:  # noqa: BLE001  (keep the video and report up to the failure)
            physics_error = f"step {k}: {exc!r}"[:300]
            break
        p = physics.named.data.xpos["walker/root"]
        xs.append(float(p[0])); zs.append(float(p[2])); tel_log.append(tele.tolist())
        if k % 2 == 0:
            frames.append(physics.render(height=360, width=640, camera_id=cam))
    step_s = (time.time() - t1) / max(1, len(xs))
    imageio.mimwrite(out / "course_smoke.mp4", frames, fps=17, quality=8)
    rep = {"build_s": build_s, "obs_shapes": obs_shapes, "walk_keys": walk_keys, "n_actuators": n_act,
           "n_slides": walker.n_slides, "slide_range": [slide_lo.tolist(), slide_hi.tolist()],
           "bone_lengths": walker.bone_lengths.tolist(), "cameras": cams, "nq": int(physics.model.nq),
           "ngeom": int(physics.model.ngeom), "gravity": physics.model.opt.gravity.tolist(),
           "x_start": xs[0], "x_end": xs[-1], "min_root_z": min(zs), "mean_root_z": float(np.mean(zs)),
           "s_per_control_step_incl_render": step_s, "hab_x": course.params.hab_x,
           "steps_run": len(xs), "physics_error": physics_error}
    (out / "course_smoke.json").write_text(json.dumps(rep, indent=1, default=_jsonable))
    runs.commit()
    return rep


@app.local_entrypoint()
def course_smoke(steps: int = 500, telescoping: bool = True, speed: float = 1.0):
    run_id = time.strftime("mocapact-course-%Y%m%d-%H%M%S")
    rep = course_smoke_job.remote(run_id, steps, telescoping, speed)
    local = Path("runs") / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "course_smoke.json").write_text(json.dumps(rep, indent=1, default=_jsonable))
    (local / "course_smoke.mp4").write_bytes(b"".join(runs.read_file(f"{run_id}/course_smoke.mp4")))
    print(json.dumps({k: v for k, v in rep.items() if k != "obs_shapes"}, indent=1, default=_jsonable))
    print("->", str(local))


# ---------------------------------------------------------------------- body tolerance sweep
def _load_walk():
    """The frozen MoCapAct walking hierarchy: (high-level SB3 policy, low-level decoder)."""
    import os

    from mocapact.distillation import model as mmodel
    from mocapact.sb3 import utils

    from ascent.mocap.task import WALK_POLICY_KEYS

    root = _transfer_root()
    high = utils.load_policy(root, list(WALK_POLICY_KEYS), device="cpu")
    low = mmodel.NpmpPolicy.load_from_checkpoint(os.path.join(root, "low_level_policy.ckpt"), map_location="cpu")
    return high, low


@app.function(image=course_image, cpu=2.0, memory=4096, timeout=1800, volumes={RUNS: runs})
def tolerance_job(run_id: str, cfg: dict) -> dict:
    """Frozen walking prior on a flat floor while the body reshapes (no learning).

    cfg: name, speed, a (group command per GROUPS in [-1, 1]), optional b + period (alternate
    between a and b), telescoping (False = the unmodified body), seconds, video.
    """
    import imageio
    import numpy as np
    import torch
    from dm_control import composer
    from dm_control.locomotion.arenas import floors
    from stable_baselines3.common.utils import obs_as_tensor

    from ascent.mocap.task import WALK_POLICY_KEYS, CourseTask, StandInitializer
    from ascent.mocap.walker import GROUPS, SLIDE_SPEED, TelescopingCMUHumanoid

    torch.set_num_threads(1)
    name, speed = cfg["name"], float(cfg.get("speed", 1.0))
    steps = int(float(cfg.get("seconds", 14.0)) / 0.03)
    walker = TelescopingCMUHumanoid(initializer=StandInitializer(), telescoping=bool(cfg.get("telescoping", True)),
                                    ranges=cfg.get("ranges"))
    task = CourseTask(walker, floors.Floor(size=(60, 12)))
    env = composer.Environment(task, random_state=0, strip_singleton_obs_buffer_dim=True)
    ts = env.reset()
    high, low = _load_walk()
    physics = env.physics
    cams = [physics.model.id2name(i, "camera") for i in range(physics.model.ncam)]
    cam = next((i for i, n in enumerate(cams) if n.endswith("side")), 1)
    keys = list(WALK_POLICY_KEYS)
    def command(key):
        # "a"/"b" are group commands in [-1, 1]; "ar"/"br" are length ratios (1.0 = nominal)
        if cfg.get(key + "r"):
            from ascent.mocap.walker import DEFAULT_RANGE
            rng = {**DEFAULT_RANGE, **(cfg.get("ranges") or {})}
            out = []
            for g in GROUPS:
                r, (lo, hi) = float(cfg[key + "r"].get(g, 1.0)), rng[g]
                out.append((r - 1.0) / (hi - 1.0) if r >= 1.0 else (r - 1.0) / (1.0 - lo))
            return np.array(out)
        return np.array([cfg.get(key, {}).get(g, 0.0) for g in GROUPS], dtype=float)

    a = command("a")
    b = command("b") if (cfg.get("b") or cfg.get("br")) else None
    feet = [walker.mjcf_model.find("body", n) for n in ("ltoes", "rtoes")]
    foot_prev, plant, plants = None, [[], []], [[], []]
    period = int(float(cfg.get("period", 6.0)) / 0.03)
    tele = np.zeros(walker.n_slides)
    frames, xs, head, pelvis, err, ratios = [], [], [], [], [], []
    t_pol = t_env = 0.0
    fell_at, physics_error = None, None
    for k in range(steps):
        task.command = np.array([speed, 0.0, 1.0, (k % 333) / 333.0])
        o = {kk: np.asarray(ts.observation[kk], np.float32).ravel() for kk in keys if kk != "target_obs"}
        o["target_obs"] = task.command.astype(np.float32)
        t0 = time.perf_counter()
        t_obs = obs_as_tensor({kk: v[None] for kk, v in o.items()}, "cpu")
        with torch.no_grad():
            z = high.policy._predict(t_obs, deterministic=True).clamp(-3, 3)
            q = low.low_level_policy(t_obs, z)[0].numpy()
        t1 = time.perf_counter()
        if walker.n_slides:
            want = np.zeros(len(GROUPS)) if k < 67 else a            # walk normally for 2 s first
            if b is not None and k >= 67 and ((k - 67) // (period // 2)) % 2 == 1:
                want = b
            goal = walker.slide_targets(want)
            tele = tele + np.clip(goal - tele, -SLIDE_SPEED * 0.03, SLIDE_SPEED * 0.03)
        try:
            ts = env.step(np.concatenate([np.clip(q, -1, 1), tele]))
        except Exception as exc:  # noqa: BLE001
            physics_error = f"step {k}: {exc!r}"[:300]
            break
        t_pol += t1 - t0
        t_env += time.perf_counter() - t1
        xs.append(float(physics.named.data.xpos["walker/root"][0]))
        pelvis.append(float(physics.named.data.xpos["walker/root"][2]))
        head.append(float(physics.named.data.geom_xpos["walker/head"][2]) + 0.143)
        if walker.n_slides:
            sq = np.asarray(physics.bind(walker.slide_joints).qpos)
            err.append(float(np.max(np.abs(sq - tele))))
            ratios.append(walker.group_ratios(sq).tolist())
        fx = np.asarray(physics.bind(feet).xpos)[:, 0].copy()
        if foot_prev is not None and k > 200:                 # after the shape change has settled
            for f in range(2):
                if abs(fx[f] - foot_prev[f]) < 0.006:         # foot planted (slower than 0.2 m/s)
                    plant[f].append(fx[f])
                elif len(plant[f]) >= 3:
                    plants[f].append(float(np.mean(plant[f])))
                    plant[f] = []
                else:
                    plant[f] = []
        foot_prev = fx
        if fell_at is None and head[-1] < 0.95:
            fell_at = k
        if cfg.get("video", True) and k % 3 == 0:
            frames.append(physics.render(height=270, width=480, camera_id=cam))
        if fell_at is not None and k > fell_at + 40:
            break
    n = len(xs)
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimwrite(out / f"{name}.mp4", frames, fps=11, quality=7)
        runs.commit()
    start = max(0, n - 167)
    strides = [d for f in range(2) for d in np.diff(plants[f]).tolist() if d > 0.2]
    return {"stride_m": round(float(np.median(strides)), 3) if strides else None, "n_strides": len(strides),
            "name": name, "cfg": cfg, "steps": n, "fell_at_s": None if fell_at is None else round(fell_at * 0.03, 2),
            "physics_error": physics_error, "distance": round(xs[-1] - xs[0], 2),
            "speed_last5s": round((xs[-1] - xs[start]) / max(1e-6, (n - start) * 0.03), 2),
            "head_top_last": round(float(np.mean(head[-100:])), 3), "head_top_min": round(min(head), 3),
            "head_top_max": round(max(head), 3), "pelvis_last": round(float(np.mean(pelvis[-100:])), 3),
            "ratios_last": [round(v, 3) for v in ratios[-1]] if ratios else None,
            "slide_err_max": round(max(err), 4) if err else None,
            "policy_ms": round(1000 * t_pol / n, 2), "env_step_ms": round(1000 * t_env / n, 2)}


def _legs(v):
    return {"thigh": v, "shin": v}


def _arms(v):
    return {"upper_arm": v, "forearm": v}


@app.local_entrypoint()
def tolerance(video: bool = True, sweep: int = 2):
    """Which body shapes does the frozen walking prior tolerate? One small container per shape."""
    run_id = time.strftime("mocapact-tolerance-%Y%m%d-%H%M%S")
    wide = {"thigh": (0.45, 1.4), "shin": (0.7, 1.8)}
    T = lambda t=1.0, s=1.0, o=1.0, arm=1.0: {"thigh": t, "shin": s, "torso": o, "upper_arm": arm, "forearm": arm}  # noqa: E731
    shapes = {
        "nominal": T(), "T60": T(0.6), "T50": T(0.5), "T45": T(0.45),
        "T60_o75": T(0.6, o=0.75), "T50_o75": T(0.5, o=0.75),
        "T70_S90_o75": T(0.7, 0.9, 0.75), "T60_S90_o75": T(0.6, 0.9, 0.75), "T50_S90_o75": T(0.5, 0.9, 0.75),
        "T50_S85_o75": T(0.5, 0.85, 0.75), "T70_S85_o75": T(0.7, 0.85, 0.75),
        "T50_S90_o75_arm70": T(0.5, 0.9, 0.75, 0.7),
        "S140": T(s=1.4), "S160": T(s=1.6), "S180": T(s=1.8), "S140_T120": T(1.2, 1.4), "S160_T120": T(1.2, 1.6),
        "S160_T110": T(1.1, 1.6), "S140_o145": T(s=1.4, o=1.45), "S160_o145_arm140": T(s=1.6, o=1.45, arm=1.4),
        "S140_T125_o145": T(1.25, 1.4, 1.45),
    }
    cfgs = [{"name": "locked_body", "telescoping": False}]
    for n, r in shapes.items():
        for sp in (1.0, 0.7):
            cfgs.append({"name": f"{n}_v{int(sp * 10):02d}", "ar": r, "speed": sp, "ranges": wide})
    for n, (ra, rb) in {"cyc_T50o75_S160": (shapes["T50_o75"], shapes["S160"]),
                        "cyc_T60S90o75_S140": (shapes["T60_S90_o75"], shapes["S140"]),
                        "cyc_T70S90o75_S140T120": (shapes["T70_S90_o75"], shapes["S140_T120"])}.items():
        cfgs.append({"name": n, "ar": ra, "br": rb, "period": 10.0, "seconds": 24.0, "ranges": wide})
    cfgs = [{**c, "video": video} for c in cfgs]
    rows = []
    for r in tolerance_job.map([run_id] * len(cfgs), cfgs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:400])
            continue
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "cfg"}, default=_jsonable))
    local = Path("runs") / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "tolerance.json").write_text(json.dumps(rows, indent=1, default=_jsonable))
    if video:
        for r in rows:
            try:
                (local / f"{r['name']}.mp4").write_bytes(b"".join(runs.read_file(f"{run_id}/{r['name']}.mp4")))
            except Exception as exc:  # noqa: BLE001
                print("no video for", r["name"], repr(exc)[:120])
    print("->", str(local))


# ---------------------------------------------------------------------- learned layer: checks, training, evaluation
def _side_camera(physics, x: float, z: float = 1.0, distance: float = 6.0, size=(360, 640)):
    from dm_control.mujoco import engine
    cam = engine.MovableCamera(physics, height=size[0], width=size[1])
    cam.set_pose(lookat=[x, 0.0, z], distance=distance, azimuth=90.0, elevation=-8.0)
    return cam


@app.function(image=course_image, cpu=4.0, memory=8192, timeout=3600, volumes={RUNS: runs})
def env_check_job(run_id: str, variant: dict) -> dict:
    """One rollout on the showcase course without any learning, with video.

    variant: name, spawn, shape ("none" = frozen prior alone, "rule" = the hand-written shape rule drives
    the telescoping, scripted), noise (std of random action noise, as PPO exploration would add), lock.
    """
    import imageio
    import numpy as np

    from ascent.mocap.env import EnvCfg, ReshapeEnv, showcase_params
    from ascent.mocap.train import make_env_cfg

    import torch
    torch.set_num_threads(1)
    cfg = make_env_cfg(variant.get("env", {}))
    name = variant["name"]
    t0 = time.perf_counter()
    env = ReshapeEnv(cfg, seed=int(variant.get("seed", 0)), fixed_params=showcase_params(cfg) if cfg.mode != "flat" else None,
                     level={"shape": float(variant.get("shape_level", 1.0))})
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    obs = env.reset(spawn=int(variant.get("spawn", 0)))
    reset_s = time.perf_counter() - t0
    physics = env.env.physics
    rng = np.random.default_rng(1)
    noise = float(variant.get("noise", 0.0))
    frames, xs, heads, ratios, reasons = [], [], [], [], ""
    cam = None
    info = {}
    steps = int(variant.get("steps", 1500))
    t_sim = 0.0
    for k in range(steps):
        a = np.zeros(env.act_dim)
        if variant.get("shape", "none") == "rule":
            a[env.tele_slice] = env.teacher()
            a[1] = env.steer_teacher()
        if variant.get("force"):
            from ascent.mocap.env import _ratios_to_command
            a[env.tele_slice] = _ratios_to_command(variant["force"], cfg.ranges)
        if noise:
            a = a + rng.normal(0.0, noise, env.act_dim)
        t1 = time.perf_counter()
        obs, r, term, trunc, info = env.step(a)
        t_sim += time.perf_counter() - t1
        xs.append(info["x"]); heads.append(info["head_top"]); ratios.append(info["ratios"])
        if variant.get("video", True) and k % 2 == 0:
            cam = _side_camera(physics, info["x"], z=env.ground + 1.0)
            frames.append(cam.render())
        if term or trunc:
            break
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimwrite(out / f"{name}.mp4", frames, fps=17, quality=8)
        runs.commit()
    warn = {str(i): int(w.number) for i, w in enumerate(physics.data.warning) if int(w.number)}
    r = np.array(ratios)
    return {"name": name, "variant": variant, "reason": info.get("reason"), "steps": len(xs), "x_start": env.start_x,
            "x_end": round(xs[-1], 2), "max_x": round(max(xs), 2), "passed": info.get("passed"),
            "head_min": round(min(heads), 3), "ratio_min": r.min(0).round(3).tolist(), "ratio_max": r.max(0).round(3).tolist(),
            "build_s": round(build_s, 2), "reset_s": round(reset_s, 3), "recompiles_each_reset": env.recompiles,
            "ms_per_step": round(1000 * t_sim / len(xs), 2), "obs_dims": [len(obs["actor"]), len(obs["critic"])],
            "act_dim": env.act_dim, "nq": int(physics.model.nq), "data_mb": round(physics.data.nbuffer / 1e6, 1),
            "warnings": warn, "zones": getattr(env, "zones", None), "roof": getattr(env.course, "roof", None)}


@app.local_entrypoint()
def env_check(low: float = 1.82, fissure: float = 0.5, video: bool = True):
    """Look before training: the frozen prior alone, and the scripted shape rule, on the showcase course."""
    run_id = time.strftime("mocapact-check-%Y%m%d-%H%M%S")
    env = {"low_clearance": low, "fissure_width": [fissure - 0.1, fissure]}
    variants = []
    for spawn, label in ((0, "start"), (1, "deck"), (2, "fissure"), (3, "scarp")):
        variants.append({"name": f"prior_only_{label}", "spawn": spawn, "shape": "none", "env": env})
        variants.append({"name": f"shape_rule_{label}", "spawn": spawn, "shape": "rule", "env": env})
    for seed in (1, 2, 3):
        variants.append({"name": f"shape_rule_start_s{seed}", "spawn": 0, "shape": "rule", "seed": seed, "env": env})
        variants.append({"name": f"prior_only_start_s{seed}", "spawn": 0, "shape": "none", "seed": seed, "env": env})
    variants.append({"name": "shape_rule_noise_start", "spawn": 0, "shape": "rule", "noise": 0.2, "env": env})
    variants.append({"name": "shape_rule_noise_deck", "spawn": 1, "shape": "rule", "noise": 0.2, "env": env})
    variants.append({"name": "flat_random_shapes", "spawn": 0, "env": {"mode": "flat"}, "shape_level": 0.3, "steps": 700})
    variants.append({"name": "flat_random_shapes_wide", "spawn": 0, "shape_level": 1.0, "steps": 700,
                     "env": {"mode": "flat", "ranges": {"thigh": [0.5, 1.4], "shin": [0.7, 1.6]}}})
    variants = [{**v, "video": video} for v in variants]
    rows = []
    for r in env_check_job.map([run_id] * len(variants), variants, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:1500])
            continue
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "variant"}, default=_jsonable))
    local = Path("runs") / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "check.json").write_text(json.dumps(rows, indent=1, default=_jsonable))
    if video:
        for r in rows:
            try:
                (local / f"{r['name']}.mp4").write_bytes(b"".join(runs.read_file(f"{run_id}/{r['name']}.mp4")))
            except Exception as exc:  # noqa: BLE001
                print("no video for", r["name"], repr(exc)[:120])
    print("->", str(local))


def _reshape_train(run_id: str, overrides: dict | None = None) -> dict:
    import threading

    from ascent.mocap.train import TrainConfig, train

    runs.reload()                                   # see checkpoints written before a restart
    cfg = TrainConfig(run_id=run_id, out_root=str(RUNS), **(overrides or {}))
    stop = threading.Event()

    def committer():
        while not stop.wait(60):
            for k in range(3):
                try:
                    runs.commit()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(5 * (k + 1))

    threading.Thread(target=committer, daemon=True).start()
    try:
        res = train(cfg, lambda row: print(json.dumps(row), flush=True))
    finally:
        stop.set()
        runs.commit()
    return res


@app.function(image=course_image, cpu=32.0, memory=49152, timeout=12 * 3600, volumes={RUNS: runs},
              retries=modal.Retries(max_retries=3, initial_delay=15.0, backoff_coefficient=1.0))
def reshape_train_job(run_id: str, overrides: dict | None = None) -> dict:
    """PPO on the learned layer. Every episode from every worker is recorded to /runs/<run_id>/episodes."""
    return _reshape_train(run_id, overrides)


@app.local_entrypoint()
def reshape_smoke(minutes: float = 6.0, mode: str = "course", workers: int = 30):
    """Short end-to-end training check: throughput, learning signal, recordings, checkpoints."""
    run_id = time.strftime(f"rsmoke-{mode}-%Y%m%d-%H%M%S")
    res = reshape_train_job.remote(run_id, {"workers": workers, "envs_per_worker": 6, "hours": minutes / 60,
                                            "checkpoint_every": 10, "log_every": 2, "env": {"mode": mode}})
    print(json.dumps(res, indent=1, default=str))
    print("run id:", run_id)


@app.function(image=course_image, cpu=2.0, memory=8192, timeout=1800)
def heading_debug_job(variant: dict) -> dict:
    import numpy as np
    import torch

    from ascent.mocap.env import ReshapeEnv, showcase_params
    from ascent.mocap.train import make_env_cfg

    torch.set_num_threads(1)
    cfg = make_env_cfg(variant.get("env", {}))
    env = ReshapeEnv(cfg, seed=0, fixed_params=showcase_params(cfg) if cfg.mode != "flat" else None)
    env.reset(spawn=int(variant.get("spawn", 0)))
    rows = []
    t0 = time.perf_counter()
    for k in range(int(variant.get("steps", 200))):
        _, _, term, trunc, info = env.step(np.zeros(env.act_dim))
        if k == 0:
            rows.append(["natural_yaw", round(env.init.natural_yaw, 3)])
        if k % 10 == 0:
            o = env.ts.observation
            rows.append([k, round(env.x, 2), round(env.y, 2), round(env.yaw, 2), round(float(o["walker/torso_xvel"]), 2),
                         round(float(o["walker/torso_yvel"]), 2), [round(float(v), 2) for v in env.task.command[:3]]])
        if term or trunc:
            break
    prof = {kk: round(1000 * v / max(1, env.prof["steps"]), 2) for kk, v in env.prof.items() if kk not in ("steps", "resets", "reset")}
    t1 = time.perf_counter()
    for _ in range(3):
        env.reset(spawn=0)
    return {"name": variant["name"], "reason": info["reason"], "ms_per_step": round(1000 * (time.perf_counter() - t0) / (k + 1), 2),
            "rows": rows, "prof_ms": prof, "reset_s": round((time.perf_counter() - t1) / 3, 3), "recompiles": env.recompiles,
            "data_mb": round(env.env.physics.data.nbuffer / 1e6, 1)}


@app.local_entrypoint()
def heading_debug():
    vs = [{"name": f"flat_yaw{y}_gain{g}", "env": {"mode": "flat", "spawn_yaw": y, "heading_gain": g}, "steps": 300}
          for y in (0.0, -1.5708, 1.5708, 3.1416, None) for g in (0.0, 0.3)]
    vs.append({"name": "course_start_gain0.6", "env": {}, "steps": 120})
    for r in heading_debug_job.map(vs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:800]); continue
        print(r["name"], r["reason"], r["ms_per_step"], "ms/step", r["prof_ms"], "reset_s", r["reset_s"], r["recompiles"], r["data_mb"])
        for row in r["rows"]:
            print("   ", row)


@app.function(image=course_image, cpu=4.0, memory=8192, timeout=3600, volumes={RUNS: runs})
def reshape_eval_job(run_id: str, checkpoint: str, job: dict) -> dict:
    """Run one saved policy for one episode and save video, trace and summary under <run>/evaluations.

    job: name, spawn, seed, video, policy ("learned", or "prior" for the frozen prior alone as a reference),
    course ("showcase" or a held-out seed), env (overrides on top of the training configuration).
    """
    import io

    import imageio
    import numpy as np
    import torch

    from ascent.mocap.env import ReshapeEnv, sample_params, showcase_params
    from ascent.mocap.train import load_policy, make_env_cfg

    torch.set_num_threads(1)
    runs.reload()
    ckpt_dir = RUNS / run_id / "checkpoints"
    name = checkpoint if checkpoint != "latest" else sorted(p.name for p in ckpt_dir.glob("policy_v*.pt"))[-1]
    policy, ck = load_policy(str(ckpt_dir / name))
    cfg = make_env_cfg({**ck["config"]["env"], **job.get("env", {})})
    if cfg.mode == "flat":
        params = None
    elif job.get("course", "showcase") == "showcase":
        params = showcase_params(cfg, job.get("level", ck["level"]))
    else:
        params = sample_params(np.random.default_rng(int(job["course"])), job.get("level", ck["level"]), cfg)
        params.seed = int(job["course"])
    env = ReshapeEnv(cfg, seed=int(job.get("seed", 0)), fixed_params=params,
                     level={"shape": float(job.get("shape_level", ck["level"].get("shape", 1.0)))})
    env.teach = 0.0
    obs = env.reset(spawn=int(job.get("spawn", 0)))
    physics = env.env.physics
    use_policy = job.get("policy", "learned") == "learned"
    frames, qpos, rows = [], [env.qpos.astype(np.float32).copy()], []
    info, ret = {"reason": ""}, 0.0
    for k in range(int(job.get("steps", cfg.episode_steps))):
        a = policy(obs["actor"]) if use_policy else np.zeros(env.act_dim)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        qpos.append(env.qpos.astype(np.float32).copy())
        rows.append([info["x"], env.y, info["head_top"], *info["ratios"], *info["teacher"], info["bump"],
                     float(np.abs(np.clip(a, -1, 1)[7:67]).mean()), float(np.abs(np.clip(a, -1, 1)[67:]).mean())])
        if job.get("video", True) and k % 2 == 0:
            frames.append(_side_camera(physics, info["x"], z=env.ground + 1.0).render())
        if term or trunc:
            break
    out = RUNS / run_id / "evaluations" / name.replace(".pt", "")
    out.mkdir(parents=True, exist_ok=True)
    tag = job["name"]
    rows = np.array(rows, np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, qpos=np.stack(qpos), rows=rows, fps=1.0 / 0.03)
    (out / f"{tag}_trace.npz").write_bytes(buf.getvalue())
    if frames:
        imageio.mimwrite(out / f"{tag}.mp4", frames, fps=17, quality=8, macro_block_size=1)
    summary = {"name": tag, "run_id": run_id, "checkpoint": name, "policy_version": int(ck["version"]),
               "global_step": int(ck["global_step"]), "origin": "checkpoint_evaluation",
               "controller": "learned layer + frozen MoCapAct prior" if use_policy else "frozen MoCapAct prior only",
               "course": job.get("course", "showcase") if cfg.mode != "flat" else "flat", "spawn_x": env.start_x,
               "reason": info["reason"], "steps": len(rows), "seconds": round(len(rows) * 0.03, 2), "return": round(ret, 2),
               "max_x": round(float(rows[:, 0].max()), 2), "passed": info.get("passed"), "attempted": info.get("attempted"),
               "ratio_min": rows[:, 3:8].min(0).round(3).tolist(), "ratio_max": rows[:, 3:8].max(0).round(3).tolist(),
               "head_top_min": round(float(rows[:, 2].min()), 3), "head_top_max": round(float(rows[:, 2].max()), 3),
               "bump_steps": int(rows[:, 13].sum()), "mean_abs_dz_action": round(float(rows[:, 14].mean()), 4),
               "mean_abs_dq_action": round(float(rows[:, 15].mean()), 4), "gravity": 9.81,
               "lock_telescoping": cfg.lock_telescoping, "job": job,
               "course_level": job.get("level", ck["level"]) if cfg.mode != "flat" else None,
               "course_params": {k: getattr(params, k) for k in ("hab_clearance", "fissure_width", "scarp_height",
                                                                  "block_offset")} if params is not None else None}
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    runs.commit()
    return summary


@app.local_entrypoint()
def reshape_eval(run_id: str, checkpoint: str = "latest", spawns: str = "0,1,2", prior: bool = True, video: bool = True,
                 episodes: int = 1):
    """Evaluate a checkpoint on the showcase course (and the frozen prior alone for reference); fetch videos."""
    jobs = []
    for sp in [int(s) for s in spawns.split(",")]:
        for e in range(episodes):
            jobs.append({"name": f"learned_spawn{sp}_e{e}", "spawn": sp, "seed": e, "video": video and e == 0})
        if prior:
            jobs.append({"name": f"prior_spawn{sp}", "spawn": sp, "policy": "prior", "video": video})
    local = Path("runs") / f"eval-{run_id}"
    local.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in reshape_eval_job.map([run_id] * len(jobs), [checkpoint] * len(jobs), jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:1500])
            continue
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("job", "origin", "run_id")}, default=_jsonable))
        if r["job"].get("video", True):
            src = f"{run_id}/evaluations/{r['checkpoint'].replace('.pt', '')}/{r['name']}.mp4"
            try:
                (local / f"{r['checkpoint'].replace('.pt', '')}_{r['name']}.mp4").write_bytes(b"".join(runs.read_file(src)))
            except Exception as exc:  # noqa: BLE001
                print("no video", src, repr(exc)[:100])
    (local / f"summary_{checkpoint.replace('.pt', '')}.json").write_text(json.dumps(rows, indent=1, default=_jsonable))
    print("->", str(local))


@app.local_entrypoint()
def reshape_resume(spec: str, hours: float = 3.5):
    """Relaunch runs into their existing ids (spec keys are full run ids); each resumes from its
    latest checkpoint. The time limit still counts from the run's first start."""
    specs = json.loads(Path(spec).read_text())
    calls = {}
    for run_id, over in specs.items():
        fn = reshape_train_job_xl if over.pop("xl", False) else reshape_train_job
        calls[run_id] = fn.spawn(run_id, {"hours": hours, **over}).object_id
        print(f"{run_id}: function call {calls[run_id]}")
    (Path("runs") / time.strftime("resume-%Y%m%d-%H%M%S.json")).write_text(json.dumps(calls, indent=1))


@app.local_entrypoint()
def reshape_launch(spec: str, hours: float = 3.0):
    """Launch training runs detached. ``spec`` is a JSON file: {run_name: TrainConfig overrides}."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    specs = json.loads(Path(spec).read_text())
    calls = {}
    for name, over in specs.items():
        run_id = f"{name}-{stamp}"
        fn = reshape_train_job_xl if over.pop("xl", False) else reshape_train_job
        calls[run_id] = fn.spawn(run_id, {"hours": hours, **over}).object_id
        print(f"{run_id}: function call {calls[run_id]}")
    Path("runs").mkdir(exist_ok=True)
    (Path("runs") / f"launch-{stamp}.json").write_text(json.dumps(calls, indent=1))


@app.function(image=course_image, cpu=64.0, memory=98304, timeout=12 * 3600, volumes={RUNS: runs},
              retries=modal.Retries(max_retries=3, initial_delay=15.0, backoff_coefficient=1.0))
def reshape_train_job_xl(run_id: str, overrides: dict | None = None) -> dict:
    """Same as reshape_train_job on a 64-core container (use workers=60)."""
    return _reshape_train(run_id, overrides)


@app.function(image=course_image, cpu=2.0, memory=8192, timeout=1800)
def spawn_check_job(level: float, seed: int) -> list:
    import numpy as np
    import torch

    from ascent.mocap.env import EnvCfg, ReshapeEnv, showcase_params

    torch.set_num_threads(1)
    cfg = EnvCfg()
    lv = {"low": level, "fissure": level, "scarp": level}
    env = ReshapeEnv(cfg, seed=seed, level=lv, fixed_params=showcase_params(cfg) if seed == 0 else None)
    out = []
    for sp in range(4):
        for rep in range(3):
            env.reset(spawn=sp)
            d = env.env.physics.data
            ncon0 = int(d.ncon)
            warn = [int(w.number) for w in d.warning]
            for _ in range(40):
                _, _, term, trunc, info = env.step(np.zeros(env.act_dim))
                if term or trunc:
                    break
            out.append({"level": level, "seed": seed, "spawn": sp, "x0": round(env.start_x, 2), "ncon0": ncon0,
                        "warn": [i for i, w in enumerate(warn) if w], "after40": info["reason"] or "ok",
                        "head": round(info["head_top"] - env.ground, 2)})
    return out


@app.local_entrypoint()
def spawn_check():
    args = [(1.0, 0), (0.0, 1), (0.5, 2), (1.0, 3), (1.0, 4), (0.2, 5)]
    for rows in spawn_check_job.starmap(args):
        for r in rows:
            print(json.dumps(r))


@app.local_entrypoint()
def v2_check(video: bool = True):
    """Course v2 sizing: slope steepness by body shape, block spacing, and full scripted runs."""
    run_id = time.strftime("mocapact-v2check-%Y%m%d-%H%M%S")
    shapes = {"nominal": None, "thigh070": {"thigh": 0.7}, "shin140": {"shin": 1.4}, "legs090": {"thigh": 0.9, "shin": 0.9},
              "legs115": {"thigh": 1.15, "shin": 1.15}}
    variants = []
    for h in (0.8, 1.2, 1.6, 2.0, 2.6):
        for sname, force in shapes.items():
            v = {"name": f"slope_h{int(h * 10):02d}_{sname}", "spawn": 3, "env": {"scarp_height": [h, h]},
                 "video": video and sname == "nominal", "steps": 500}
            if force:
                v["force"] = force
            variants.append(v)
        variants.append({"name": f"slope_h{int(h * 10):02d}_norocks", "spawn": 3, "steps": 500, "video": False,
                         "env": {"scarp_height": [h, h], "ramp_rocks": 0}})
    for sp in (2.8, 3.3, 4.0):
        for seed in (1, 2):
            variants.append({"name": f"slalom_sp{int(sp * 10)}_rule_s{seed}", "spawn": 0, "shape": "rule", "seed": seed,
                             "steps": 520, "env": {"block_spacing": sp}, "video": video and seed == 1})
            variants.append({"name": f"slalom_sp{int(sp * 10)}_none_s{seed}", "spawn": 0, "shape": "none", "seed": seed,
                             "steps": 520, "env": {"block_spacing": sp}, "video": False})
    for seed in (1, 2, 3, 4):
        variants.append({"name": f"full_rule_s{seed}", "spawn": 0, "shape": "rule", "seed": seed, "video": video})
    rows = []
    for r in env_check_job.map([run_id] * len(variants), variants, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:1500])
            continue
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "variant"}, default=_jsonable))
    local = Path("runs") / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "check.json").write_text(json.dumps(rows, indent=1, default=_jsonable))
    for r in rows:
        if r["variant"].get("video"):
            try:
                (local / f"{r['name']}.mp4").write_bytes(b"".join(runs.read_file(f"{run_id}/{r['name']}.mp4")))
            except Exception as exc:  # noqa: BLE001
                print("no video for", r["name"], repr(exc)[:120])
    print("->", str(local))


@app.function(image=course_image, cpu=4.0, memory=8192, timeout=3600, volumes={RUNS: runs})
def showcase_export_job(out_id: str, job: dict) -> dict:
    """One episode on the showcase course, exported for the Blender renderer (rig, poses, scene, terrain).

    job: name, seed, level (course difficulty), source ("rule" = scripted shape and steering rules,
    "prior" = frozen prior alone, or {"run_id":..., "checkpoint":...} for a learned policy), spawn.
    """
    import io

    import numpy as np
    import torch

    from ascent.mocap.env import EnvCfg, ReshapeEnv, showcase_params
    from ascent.mocap.export import export_poses, export_rig, pack_terrain
    from ascent.mocap.train import load_policy, make_env_cfg

    torch.set_num_threads(1)
    source = job.get("source", "rule")
    policy, cfg, origin = None, make_env_cfg(job.get("env", {})), {"controller": source}
    if isinstance(source, dict):
        runs.reload()
        ckpt_dir = RUNS / source["run_id"] / "checkpoints"
        name = source.get("checkpoint", "latest")
        name = name if name != "latest" else sorted(p.name for p in ckpt_dir.glob("policy_v*.pt"))[-1]
        policy, ck = load_policy(str(ckpt_dir / name))
        cfg = make_env_cfg({**ck["config"]["env"], **job.get("env", {})})
        origin = {"controller": "learned layer + frozen MoCapAct prior", "run_id": source["run_id"], "checkpoint": name,
                  "policy_version": int(ck["version"]), "global_step": int(ck["global_step"])}
        job.setdefault("level", ck["level"])
    level = job.get("level") or {}
    env = ReshapeEnv(cfg, seed=int(job.get("seed", 0)), fixed_params=showcase_params(cfg, level))
    env.teach = 0.0
    obs = env.reset(spawn=int(job.get("spawn", 0)))
    qpos, rows, info = [env.qpos.astype(np.float32).copy()], [], {"reason": ""}
    frames, physics = [], env.env.physics
    for k in range(int(job.get("steps", cfg.episode_steps))):
        if job.get("preview") and k % 3 == 0:
            frames.append(_side_camera(physics, env.x, z=env.ground + 1.0, size=(270, 480)).render())
        if policy is not None:
            a = policy(obs["actor"])
        else:
            a = np.zeros(env.act_dim)
            if source == "rule":
                a[env.tele_slice] = env.teacher()
                a[1] = env.steer_teacher()
        obs, r, term, trunc, info = env.step(a)
        qpos.append(env.qpos.astype(np.float32).copy())
        rows.append([info["x"], env.y, info["head_top"], *info["ratios"], info["head_top"] - env.ground, info["bump"]])
        if term or trunc:
            break
    qpos = np.stack(qpos)
    rows = np.array(rows, np.float32)
    poses = export_poses(env, qpos, fps=1.0 / 0.03)
    buf = io.BytesIO()
    np.savez_compressed(buf, **poses, qpos=qpos, rows=rows)
    summary = {"name": job["name"], **origin, "origin": "scripted_rules" if source == "rule" else
               ("frozen_prior" if source == "prior" else "checkpoint_evaluation"), "seed": int(job.get("seed", 0)),
               "reason": info["reason"], "steps": len(rows), "max_x": round(float(rows[:, 0].max()), 2),
               "passed": info.get("passed"), "level": level, "gravity": 9.81,
               "ratio_min": rows[:, 3:8].min(0).round(3).tolist(), "ratio_max": rows[:, 3:8].max(0).round(3).tolist(),
               "seconds": round(len(rows) * 0.03, 1), "min_head_height": round(float(rows[20:, 8].min()), 2) if len(rows) > 25 else None,
               "max_abs_y": round(float(np.abs(rows[:, 1]).max()), 2), "bump_steps": int(rows[:, 9].sum()),
               "course_params": {k: getattr(env.course.params, k) for k in ("hab_clearance", "fissure_width", "scarp_height",
                                                                            "block_offset", "ramp_run")},
               "export": f"{out_id}/{job['name']}"}
    files = {"rig.json": json.dumps(export_rig(env)).encode(), "poses.npz": buf.getvalue(),
             "scene.json": json.dumps(env.course.scene_json()).encode(), "terrain.npz": pack_terrain(env.course),
             "summary.json": json.dumps(summary, indent=1, default=_jsonable).encode()}
    out = RUNS / out_id / job["name"]
    out.mkdir(parents=True, exist_ok=True)
    for n, blob in files.items():
        (out / n).write_bytes(blob)
    if frames:
        import imageio
        imageio.mimwrite(out / "preview.mp4", frames, fps=11, quality=7, macro_block_size=1)
    runs.commit()
    return {"summary": summary, "files": files if job.get("fetch", True) else {}}


@app.local_entrypoint()
def showcase_export(source: str = "rule", seeds: str = "1,2,3,4,5,6", level: float = 1.0, run_id: str = "",
                    checkpoint: str = "latest", env: str = "", levels: str = ""):
    """Export showcase episodes for Blender. With --run-id the learned policy drives; else the scripted rules."""
    out_id = time.strftime("export-%Y%m%d-%H%M%S")
    src = {"run_id": run_id, "checkpoint": checkpoint} if run_id else source
    jobs = [{"name": f"s{s}", "seed": int(s), "source": src} for s in seeds.split(",")]
    if not run_id:
        for j in jobs:
            j["level"] = {**{k: level for k in ("blocks", "low", "fissure", "scarp")}, **(json.loads(levels) if levels else {})}
            j["env"] = json.loads(Path(env).read_text()) if env else {}
    elif levels:
        for j in jobs:
            j["level"] = json.loads(levels)
    best = None
    for r in showcase_export_job.map([out_id] * len(jobs), jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:1500])
            continue
        s = r["summary"]
        print(json.dumps(s, default=_jsonable))
        local = Path("runs") / out_id / s["name"]
        local.mkdir(parents=True, exist_ok=True)
        for n, blob in r["files"].items():
            (local / n).write_bytes(blob)
        if best is None or s["max_x"] > best[0]:
            best = (s["max_x"], str(local))
    print("best ->", best)


@app.local_entrypoint()
def run_check(video: bool = True):
    """Does the frozen prior run, and can a running stride clear a wider fissure?"""
    run_id = time.strftime("mocapact-runcheck-%Y%m%d-%H%M%S")
    variants = []
    for v in (1.6, 2.2, 2.8, 3.4):
        variants.append({"name": f"flat_run_v{int(v * 10)}", "steps": 400, "video": video,
                         "env": {"mode": "flat", "speed_base": v, "flat_shape_seconds": [99.0, 100.0]}, "shape_level": 0.0})
        for w in (0.6, 0.9, 1.2):
            for seed in (1, 2):
                variants.append({"name": f"fissure_w{int(w * 10):02d}_v{int(v * 10)}_s{seed}", "spawn": 2, "seed": seed,
                                 "steps": 260, "video": video and seed == 1 and w == 0.9,
                                 "env": {"speed_base": v, "fissure_width": [w, w]}})
    rows = []
    for r in env_check_job.map([run_id] * len(variants), variants, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:1500])
            continue
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "variant"}, default=_jsonable))
    local = Path("runs") / run_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "check.json").write_text(json.dumps(rows, indent=1, default=_jsonable))
    for r in rows:
        if r["variant"].get("video"):
            try:
                (local / f"{r['name']}.mp4").write_bytes(b"".join(runs.read_file(f"{run_id}/{r['name']}.mp4")))
            except Exception as exc:  # noqa: BLE001
                print("no video for", r["name"], repr(exc)[:120])
    print("->", str(local))


# ---------------------------------------------------------------------- training recordings: replay and progression
@app.function(image=course_image, cpu=4.0, memory=16384, timeout=3600, volumes={RUNS: runs})
def episode_index_job(run_id: str, buckets: int = 5, spawn: int = 0) -> dict:
    """Summarise a run's recorded training episodes and pick a typical one from each stage of training."""
    import numpy as np

    runs.reload()
    run_dir = RUNS / run_id
    rows = []
    for idx in sorted((run_dir / "episodes").glob("w*/index.jsonl")):
        rows += [json.loads(line) for line in idx.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["global_step_start"])
    steps = np.array([r["global_step_start"] for r in rows])
    edges = np.linspace(0, steps.max() + 1, buckets + 1)
    picks, stages = [], []
    for b in range(buckets):
        grp = [r for r in rows if edges[b] <= r["global_step_start"] < edges[b + 1]]
        sel = [r for r in grp if r.get("spawn_index") == spawn and r["steps"] > 20]
        stages.append({"stage": b, "step_range": [int(edges[b]), int(edges[b + 1])], "episodes": len(grp),
                       "from_start": len(sel),
                       "median_max_x": float(np.median([r["max_x"] for r in sel])) if sel else None,
                       "success_rate": float(np.mean([r["success"] for r in sel])) if sel else None})
        if sel:
            sel.sort(key=lambda r: r["max_x"])
            picks.append(sel[len(sel) // 2])                 # the median attempt of this stage
    return {"run_id": run_id, "episodes": len(rows), "steps_recorded": int(sum(r["steps"] for r in rows)),
            "status": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
            "stages": stages, "picks": picks}


@app.function(image=course_image, cpu=4.0, memory=8192, timeout=3600, volumes={RUNS: runs})
def replay_episode_job(run_id: str, row: dict, name: str) -> dict:
    """Render one recorded training episode from its saved states (no physics re-run)."""
    import imageio
    import numpy as np
    import torch

    from ascent.mocap.env import ReshapeEnv
    from ascent.mocap.train import make_env_cfg
    from ascent.record.recorder import load_episode
    from ascent.world.course import CourseParams

    torch.set_num_threads(1)
    runs.reload()
    run_dir = RUNS / run_id
    cfg = make_env_cfg(json.loads((run_dir / "config.json").read_text())["env"])
    ep = load_episode(run_dir, row)
    known = CourseParams.__dataclass_fields__
    params = CourseParams(**{k: v for k, v in row["course_params"].items() if k in known}) if row.get("course_params") else None
    env = ReshapeEnv(cfg, seed=0, fixed_params=params)
    env.reset(spawn=int(row.get("spawn_index", 0)))
    physics = env.env.physics
    frames = []
    for q in ep["qpos"]:
        physics.data.qpos[:] = q
        physics.forward()
        env.walker.sync_geometry(physics)
        x = float(q[0])
        frames.append(_side_camera(physics, x, z=env.course.height_at(x, 0.0) + 1.0).render())
    out = run_dir / "replays"
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out / f"{name}.mp4", frames, fps=float(row.get("playback_hz", 16.7)), quality=8, macro_block_size=1)
    runs.commit()
    return {"name": name, "episode_id": row["episode_id"], "origin": "training_recording", "video_source": "rerendered_from_trace",
            "global_step_start": row["global_step_start"], "policy_version_min": row["policy_version_min"],
            "policy_version_max": row["policy_version_max"], "reason": row["reason"], "max_x": row["max_x"],
            "steps": row["steps"], "frames": len(frames)}


@app.local_entrypoint()
def progression(run_id: str, buckets: int = 5):
    """Typical training attempts from early to late training, replayed from the recordings."""
    info = episode_index_job.remote(run_id, buckets)
    print(json.dumps({k: v for k, v in info.items() if k != "picks"}, indent=1))
    picks = info["picks"]
    names = [f"stage{i}_step{p['global_step_start']}" for i, p in enumerate(picks)]
    local = Path("runs") / f"progression-{run_id}"
    local.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in replay_episode_job.map([run_id] * len(picks), picks, names, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:1200])
            continue
        rows.append(r)
        print(json.dumps(r))
        (local / f"{r['name']}.mp4").write_bytes(b"".join(runs.read_file(f"{run_id}/replays/{r['name']}.mp4")))
    (local / "progression.json").write_text(json.dumps({"summary": {k: v for k, v in info.items() if k != "picks"},
                                                        "clips": rows}, indent=1))
    print("->", str(local))


@app.local_entrypoint()
def climb_check():
    """Does a body shape help on the slope, and how close to the centre line can the blocks sit? (scripted rules)"""
    run_id = time.strftime("mocapact-climbcheck-%Y%m%d-%H%M%S")
    shapes = {"nominal": {}, "legs090": {"thigh": 0.9, "shin": 0.9}, "thigh070": {"thigh": 0.7},
              "legs115": {"thigh": 1.15, "shin": 1.15}, "shin140": {"shin": 1.4}, "torso075": {"torso": 0.75},
              "thigh070_arms140": {"thigh": 0.7, "upper_arm": 1.4, "forearm": 1.4}}
    variants = []
    for h in (0.6, 0.9, 1.2):
        for sname, shape in shapes.items():
            for seed in (1, 2):
                variants.append({"name": f"climb_h{int(h * 10):02d}_{sname}_s{seed}", "spawn": 3, "shape": "rule", "seed": seed,
                                 "steps": 420, "video": False, "env": {"scarp_height": [h, h], "climb_shape": shape}})
    for off in (0.1, 0.25, 0.45):
        for sp in (4.0, 5.0):
            for seed in (1, 2, 3):
                variants.append({"name": f"blocks_off{int(off * 100):02d}_sp{int(sp * 10)}_s{seed}", "spawn": 0, "shape": "rule",
                                 "seed": seed, "steps": 520, "video": False,
                                 "env": {"block_offset": off, "block_spacing": sp}})
    rows = []
    for r in env_check_job.map([run_id] * len(variants), variants, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:800])
            continue
        print(json.dumps({k: r[k] for k in ("name", "reason", "steps", "x_start", "max_x", "passed", "zones")}, default=_jsonable))


HELDOUT_SEEDS = (90001, 90002, 90003)       # layouts never drawn in training (training seeds are random 30-bit draws)


@app.local_entrypoint()
def reshape_compare(run_ids: str, episodes: int = 10, levels: str = "", checkpoint: str = "latest", heldout: bool = True):
    """Evaluate several runs under the same course difficulty: N episodes on the showcase course and N on
    held-out layouts each. ``levels`` (JSON) fixes the difficulty for every run; default: each run's own."""
    import numpy as np

    level = json.loads(levels) if levels else None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    table = {}
    for run_id in run_ids.split(","):
        jobs = [{"name": f"cmp{stamp}_showcase_e{e}", "spawn": 0, "seed": 1000 + e, "video": False} for e in range(episodes)]
        if heldout:
            jobs += [{"name": f"cmp{stamp}_held{s}_e{e}", "spawn": 0, "seed": 2000 + e, "course": s, "video": False}
                     for s in HELDOUT_SEEDS for e in range(max(1, episodes // len(HELDOUT_SEEDS)))]
        if level:
            for j in jobs:
                j["level"] = level
        rows = [r for r in reshape_eval_job.map([run_id] * len(jobs), [checkpoint] * len(jobs), jobs, return_exceptions=True)
                if not isinstance(r, Exception)]
        out = {}
        for role in ("showcase", "heldout"):
            grp = [r for r in rows if (r["course"] == "showcase") == (role == "showcase")]
            if not grp:
                continue
            keys = list(grp[0]["passed"].keys())
            out[role] = {"n": len(grp), "success": sum(r["reason"] == "success" for r in grp),
                         "mean_max_x": round(float(np.mean([r["max_x"] for r in grp])), 2),
                         "passed": {k: sum(bool(r["passed"][k]) for r in grp) for k in keys},
                         "reasons": {x: sum(r["reason"] == x for r in grp) for x in {r["reason"] for r in grp}},
                         "ratio_min": np.min([r["ratio_min"] for r in grp], axis=0).round(2).tolist(),
                         "ratio_max": np.max([r["ratio_max"] for r in grp], axis=0).round(2).tolist()}
        meta = {k: rows[0][k] for k in ("checkpoint", "policy_version", "global_step", "course_level", "course_params",
                                        "lock_telescoping")} if rows else {}
        table[run_id] = {**meta, **out}
        print(run_id, json.dumps(table[run_id], default=_jsonable))
    local = Path("runs") / f"compare-{stamp}"
    local.mkdir(parents=True, exist_ok=True)
    (local / "compare.json").write_text(json.dumps(table, indent=1, default=_jsonable))
    print("->", str(local))


@app.local_entrypoint()
def sample_runs(run_id: str, levels: str, seeds: int = 24, first_seed: int = 100, checkpoint: str = "latest", tag: str = "",
                env: str = ""):
    """Run many showcase episodes of one checkpoint at one difficulty; everything stays on the volume
    (export files and a preview video per episode). Prints one summary row per episode."""
    out_id = time.strftime("sample-%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    lv = json.loads(levels)
    jobs = [{"name": f"s{s}", "seed": s, "source": {"run_id": run_id, "checkpoint": checkpoint}, "level": dict(lv),
             "preview": True, "fetch": False, "env": json.loads(env) if env else {}}
            for s in range(first_seed, first_seed + seeds)]
    rows = []
    for r in showcase_export_job.map([out_id] * len(jobs), jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:500])
            continue
        rows.append(r["summary"])
        print(json.dumps(r["summary"], default=_jsonable))
    local = Path("runs") / out_id
    local.mkdir(parents=True, exist_ok=True)
    (local / "samples.json").write_text(json.dumps(rows, indent=1, default=_jsonable))
    ok = [r for r in rows if r["reason"] == "success"]
    print(f"SUMMARY {out_id}: {len(ok)} of {len(rows)} complete; levels {lv}")


@app.local_entrypoint()
def fetch_export(export: str):
    """Download one sampled episode (``<sample id>/<name>``) for rendering."""
    local = Path("runs") / export
    local.mkdir(parents=True, exist_ok=True)
    for n in ("rig.json", "poses.npz", "scene.json", "terrain.npz", "summary.json", "preview.mp4"):
        try:
            (local / n).write_bytes(b"".join(runs.read_file(f"{export}/{n}")))
        except Exception as exc:  # noqa: BLE001
            print("missing", n, repr(exc)[:100])
    print("->", str(local))


@app.local_entrypoint()
def check_variants(spec: str):
    """Run env_check_job for every variant in a JSON list file; print one row each and fetch videos."""
    variants = json.loads(Path(spec).read_text())
    run_id = time.strftime("mocapact-variants-%Y%m%d-%H%M%S")
    local = Path("runs") / run_id
    local.mkdir(parents=True, exist_ok=True)
    for r in env_check_job.map([run_id] * len(variants), variants, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED:", repr(r)[:800])
            continue
        print(json.dumps({k: r[k] for k in ("name", "reason", "steps", "x_start", "x_end", "max_x", "head_min")}, default=_jsonable))
        if r["variant"].get("video", True):
            try:
                (local / f"{r['name']}.mp4").write_bytes(b"".join(runs.read_file(f"{run_id}/{r['name']}.mp4")))
            except Exception:  # noqa: BLE001
                pass
    print("->", str(local))


@app.local_entrypoint()
def heldout_eval(run_id: str, checkpoint: str = "latest", levels: str = "", n: int = 24, first_seed: int = 91001,
                 prior: bool = True, tag: str = ""):
    """One episode on each of ``n`` layouts never drawn in training (course seeds first_seed ..), for the
    learned policy and, as a reference, the frozen prior alone."""
    import numpy as np

    level = json.loads(levels) if levels else None
    jobs = []
    for s in range(first_seed, first_seed + n):
        for pol in (("learned", "prior") if prior else ("learned",)):
            j = {"name": f"held{tag}_{pol}_{s}", "spawn": 0, "seed": s, "course": s, "video": False, "policy": pol}
            if level:
                j["level"] = level
            jobs.append(j)
    rows = [r for r in reshape_eval_job.map([run_id] * len(jobs), [checkpoint] * len(jobs), jobs, return_exceptions=True)
            if not isinstance(r, Exception)]
    out = {"run_id": run_id, "checkpoint": rows[0]["checkpoint"] if rows else checkpoint, "level": level,
           "layout_seeds": [first_seed, first_seed + n - 1]}
    for pol in ("learned", "prior"):
        grp = [r for r in rows if r["job"]["policy"] == pol]
        if not grp:
            continue
        keys = list(grp[0]["passed"].keys())
        out[pol] = {"n": len(grp), "reached_astronaut": sum(r["reason"] in ("success", "overshot") for r in grp),
                    "passed": {k: sum(bool(r["passed"][k]) for r in grp) for k in keys},
                    "mean_max_x": round(float(np.mean([r["max_x"] for r in grp])), 2),
                    "reasons": {x: sum(r["reason"] == x for r in grp) for x in {r["reason"] for r in grp}}}
    print(json.dumps(out, default=_jsonable))
    local = Path("runs") / time.strftime(f"heldout-%Y%m%d-%H%M%S{tag}")
    local.mkdir(parents=True, exist_ok=True)
    (local / "heldout.json").write_text(json.dumps({"summary": out, "episodes": rows}, indent=1, default=_jsonable))
    print("->", str(local))


@app.local_entrypoint()
def rough_test(run_id: str, checkpoint: str = "latest", n: int = 12, stretch: float = 22.0):
    """Untrained-for terrain: after an easy start, a long stretch of cratered, rippled ground at several
    roughness settings (0.5 is what training used). Learned policy and the frozen prior, unseen layouts."""
    import numpy as np

    level = {"blocks": 0.0, "low": 0.0, "fissure": 0.0, "scarp": 0.0}
    settings = [(0.5, 9), (1.0, 9), (1.0, 30), (2.0, 30), (3.0, 30)]
    jobs = []
    for rough, craters in settings:
        for s in range(92001, 92001 + n):
            for pol in ("learned", "prior"):
                jobs.append({"name": f"rough{int(rough * 10):02d}_c{craters}_{pol}_{s}", "spawn": 3, "seed": s, "course": s,
                             "level": level, "policy": pol, "video": s == 92001, "steps": 1500,
                             "env": {"goal_gap": stretch, "rough": rough, "craters": craters, "stop_task": False}})
    rows = [r for r in reshape_eval_job.map([run_id] * len(jobs), [checkpoint] * len(jobs), jobs, return_exceptions=True)
            if not isinstance(r, Exception)]
    local = Path("runs") / time.strftime("roughtest-%Y%m%d-%H%M%S")
    local.mkdir(parents=True, exist_ok=True)
    table = []
    for rough, craters in settings:
        for pol in ("learned", "prior"):
            grp = [r for r in rows if r["job"]["env"]["rough"] == rough and r["job"]["env"]["craters"] == craters
                   and r["job"]["policy"] == pol]
            if not grp:
                continue
            row = {"rough": rough, "craters": craters, "controller": pol, "n": len(grp),
                   "crossed": sum(r["reason"] == "success" for r in grp),
                   "mean_distance_m": round(float(np.mean([r["max_x"] - r["spawn_x"] for r in grp])), 1),
                   "reasons": {x: sum(r["reason"] == x for r in grp) for x in {r["reason"] for r in grp}}}
            table.append(row)
            print(json.dumps(row))
    for r in rows:
        if r["job"].get("video"):
            src = f"{run_id}/evaluations/{r['checkpoint'].replace('.pt', '')}/{r['name']}.mp4"
            try:
                (local / f"{r['name']}.mp4").write_bytes(b"".join(runs.read_file(src)))
            except Exception:  # noqa: BLE001
                pass
    (local / "rough_test.json").write_text(json.dumps({"run_id": run_id, "checkpoint": rows[0]["checkpoint"] if rows else "",
                                                       "stretch_m": stretch, "table": table, "episodes": rows},
                                                      indent=1, default=_jsonable))
    print("->", str(local))
