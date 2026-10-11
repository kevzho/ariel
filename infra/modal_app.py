"""Modal jobs for Ariel. All physics, rendering and training run here.

Launch from the repository root (Guhan's workspace, profile tejas-16842):

    uv run modal run infra/modal_app.py::body_check_job
    uv run modal run infra/modal_app.py::body_check_job --renderer cpu

Outputs land in the ``ariel-runs`` volume and are also copied to ``runs/<run_id>/``
locally (ignored by Git).
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import modal

MUJOCO_VERSION = "3.15.0"

_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libegl1", "libgl1", "libglvnd0", "libopengl0", "libosmesa6", "ffmpeg")
    .pip_install(f"mujoco=={MUJOCO_VERSION}", "numpy>=2.0", "imageio>=2.34", "imageio-ffmpeg>=0.5")
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        "echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}'"
        " > /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
)
egl_image = _base.env({"MUJOCO_GL": "egl", "NVIDIA_DRIVER_CAPABILITIES": "all"}).add_local_python_source("ariel")
cpu_image = _base.env({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}).add_local_python_source("ariel")

BLENDER_VERSION = "4.5.14"
_EARTH = "https://eoimages.gsfc.nasa.gov/images/imagerecords"
blender_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "xz-utils", "ffmpeg", "libxrender1", "libxi6", "libxkbcommon0", "libsm6",
                 "libxxf86vm1", "libxfixes3", "libgl1", "libegl1", "libx11-6", "libxext6", "libglu1-mesa")
    .run_commands(
        f"wget -q https://download.blender.org/release/Blender4.5/blender-{BLENDER_VERSION}-linux-x64.tar.xz"
        " -O /tmp/blender.tar.xz",
        "mkdir -p /opt/blender && tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1"
        " && rm /tmp/blender.tar.xz",
        # NASA Visible Earth imagery (public domain) for the Earth on the horizon
        "mkdir -p /assets/earth",
        f"wget -q {_EARTH}/74000/74393/world.topo.200407.3x5400x2700.jpg -O /assets/earth/world.topo.5400x2700.jpg",
        f"wget -q {_EARTH}/57000/57747/cloud_combined_2048.jpg -O /assets/earth/clouds_2048.jpg",
    )
    .pip_install("numpy>=2.0")
    .add_local_dir("assets", "/assets/repo")
    .add_local_python_source("ariel")
)

app = modal.App("ariel")
runs = modal.Volume.from_name("ariel-runs", create_if_missing=True)
RUNS = Path("/runs")


def _save(run_id: str, files: dict[str, bytes]) -> None:
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)
    runs.commit()


@app.function(image=cpu_image, cpu=2.0, memory=2048, timeout=900, volumes={RUNS: runs})
def body_check(run_id: str) -> dict[str, bytes]:
    import numpy as np

    from ariel.robot.scripted import run_body_check

    res = run_body_check()
    buf = io.BytesIO()
    np.savez_compressed(buf, qpos=res["qpos"], ctrl=res["ctrl"], fps=res["fps"],
                        xml=np.array(res["xml"]), actuators=np.array(res["actuators"]))
    files = {
        "trace.npz": buf.getvalue(),
        "metrics.json": json.dumps(res["metrics"], indent=2).encode(),
        "model.xml": res["xml"].encode(),
    }
    _save(run_id, files)
    return files


def _render(trace: bytes, cameras: list[str], width: int, height: int,
            stills: list[float], every: int) -> dict[str, bytes]:
    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    from ariel.render.mjrender import TraceRenderer, write_video

    z = np.load(io.BytesIO(trace))
    model = mujoco.MjModel.from_xml_string(str(z["xml"]))
    fps = float(z["fps"])
    qpos = z["qpos"][::every]
    out_fps = fps / every
    r = TraceRenderer(model, width=width, height=height)
    files: dict[str, bytes] = {}
    timing = {}
    for cam in cameras:
        frames = []
        t0 = time.perf_counter()
        for f in r.frames(qpos, out_fps, camera=cam):
            frames.append(f)
        timing[cam] = round((time.perf_counter() - t0) / max(len(frames), 1) * 1000, 2)
        path = Path(f"/tmp/{cam}.mp4")
        write_video(frames, path, out_fps)
        files[f"{cam}.mp4"] = path.read_bytes()
        for t in stills:
            i = min(len(frames) - 1, int(t * out_fps))
            b = io.BytesIO()
            imageio.imwrite(b, frames[i], format="png")
            files[f"{cam}_t{t:05.2f}.png"] = b.getvalue()
    files["render.json"] = json.dumps({"ms_per_frame": timing, "width": width, "height": height,
                                       "frames": len(qpos), "fps": out_fps,
                                       "gl": __import__("os").environ.get("MUJOCO_GL")}).encode()
    return files


@app.function(image=egl_image, gpu="L4", cpu=4.0, memory=8192, timeout=1800, volumes={RUNS: runs})
def render_gpu(run_id: str, trace: bytes, cameras: list[str], width: int = 1280, height: int = 720,
               stills: list[float] = (), every: int = 2) -> dict[str, bytes]:
    files = _render(trace, cameras, width, height, list(stills), every)
    _save(run_id, files)
    return files


@app.function(image=cpu_image, cpu=8.0, memory=8192, timeout=3600, volumes={RUNS: runs})
def render_cpu(run_id: str, trace: bytes, cameras: list[str], width: int = 1280, height: int = 720,
               stills: list[float] = (), every: int = 2) -> dict[str, bytes]:
    files = _render(trace, cameras, width, height, list(stills), every)
    _save(run_id, files)
    return files


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=900)
def export_rig(trace: bytes, every: int = 1) -> dict[str, bytes]:
    """Body geometry and per-frame body poses (forward kinematics only) for Blender."""
    import mujoco
    import numpy as np

    from ariel.render.rig import body_poses
    from ariel.render.rig import export_rig as rig_of

    from ariel.world.settle import load_model

    z = np.load(io.BytesIO(trace))
    model = load_model(z)
    poses = body_poses(model, z["qpos"][::every])
    buf = io.BytesIO()
    np.savez_compressed(buf, names=np.array(poses["names"]), pos=poses["pos"], quat=poses["quat"],
                        fps=float(z["fps"]) / every)
    return {"rig.json": json.dumps(rig_of(model)).encode(), "poses.npz": buf.getvalue()}


@app.function(image=blender_image, gpu="L40S", cpu=8.0, memory=32768, timeout=3600, volumes={RUNS: runs})
def render_blender(run_id: str, rig: bytes, poses: bytes, frames: list[int], cameras: list[str],
                   res: str = "1920x1080", samples: int = 128, save_blend: bool = False,
                   scene: bytes = b"", terrain: bytes = b"") -> dict[str, bytes]:
    import subprocess

    work = Path("/tmp/blender_job")
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    (work / "rig.json").write_bytes(rig)
    (work / "poses.npz").write_bytes(poses)
    boot = ("import sys; sys.path.insert(0, '/root'); import runpy; "
            "runpy.run_module('ariel.art.render', run_name='__main__')")
    cmd = ["/opt/blender/blender", "-b", "--python-exit-code", "1", "--python-expr", boot, "--",
           "--rig", str(work / "rig.json"), "--poses", str(work / "poses.npz"), "--out", str(out),
           "--assets", "/assets", "--frames", ",".join(map(str, frames)), "--cameras", ",".join(cameras),
           "--res", res, "--samples", str(samples)]
    if save_blend:
        cmd.append("--save-blend")
    if scene:
        (work / "scene.json").write_bytes(scene)
        (work / "terrain.npz").write_bytes(terrain)
        cmd += ["--scene", str(work / "scene.json"), "--terrain", str(work / "terrain.npz")]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-8000:]
    files = {"blender.log": log.encode()}
    for f in out.iterdir():
        files[f.name] = f.read_bytes()
    files["blender_wall.json"] = json.dumps({"seconds": round(time.time() - t0, 1),
                                             "returncode": proc.returncode}).encode()
    _save(run_id, files)
    return files


@app.local_entrypoint()
def lookdev(trace_dir: str = "", frames: str = "50,225,450", cameras: str = "hero,side",
            res: str = "1920x1080", samples: int = 128):
    """Photoreal Blender stills of recorded robot states (default: the latest body check)."""
    src = Path(trace_dir) if trace_dir else sorted(Path("runs").glob("body-check-*"))[-1]
    trace = (src / "trace.npz").read_bytes()
    run_id = time.strftime("lookdev-%Y%m%d-%H%M%S")
    exported = export_rig.remote(trace)
    files = render_blender.remote(run_id, exported["rig.json"], exported["poses.npz"],
                                  [int(x) for x in frames.split(",")], cameras.split(","), res, samples)
    out = _write_local(run_id, {**exported, **files})
    print(files.get("blender_wall.json", b"").decode())
    print(files.get("blender_report.json", b"{}").decode())
    if b"Error" in files["blender.log"] or b"Traceback" in files["blender.log"]:
        print(files["blender.log"].decode()[-4000:])
    print(f"saved to {out} (source trace: {src})")


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=900, volumes={RUNS: runs})
def course_settle(run_id: str, params: dict | None = None) -> dict[str, bytes]:
    from ariel.world.course import CourseParams
    from ariel.world.settle import pack_terrain, pack_trace, settle_course

    res = settle_course(CourseParams(**(params or {})))
    files = {"trace.npz": pack_trace(res), "terrain.npz": pack_terrain(res["course"]),
             "scene.json": json.dumps(res["course"].scene_json(), indent=1).encode(),
             "metrics.json": json.dumps(res["metrics"], indent=2).encode()}
    _save(run_id, files)
    return files


@app.local_entrypoint()
def course_lookdev(shots: str = "establish,habitat,crawlspace,fissure,scarp,astronaut,overview",
                   res: str = "1920x1080", samples: int = 192):
    """Settle the showcase course on Modal, then render it photoreal in Blender."""
    run_id = time.strftime("course-%Y%m%d-%H%M%S")
    settled = course_settle.remote(run_id)
    print(settled["metrics.json"].decode())
    exported = export_rig.remote(settled["trace.npz"])
    import numpy as np
    n_frames = len(np.load(io.BytesIO(settled["trace.npz"]))["qpos"])
    files = render_blender.remote(run_id, exported["rig.json"], exported["poses.npz"], [n_frames - 1],
                                  shots.split(","), res, samples, False,
                                  settled["scene.json"], settled["terrain.npz"])
    out = _write_local(run_id, {**settled, **exported, **files})
    print(files.get("blender_wall.json", b"").decode())
    log = files["blender.log"].decode(errors="replace")
    if "Traceback" in log or "Error" in log:
        print(log[-5000:])
    print(f"saved to {out}")


def _write_local(run_id: str, files: dict[str, bytes]) -> Path:
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (out / name).write_bytes(blob)
    return out


@app.local_entrypoint()
def body_check_job(renderer: str = "gpu", width: int = 1280, height: int = 720):
    run_id = time.strftime("body-check-%Y%m%d-%H%M%S")
    files = body_check.remote(run_id)
    print(files["metrics.json"].decode())
    stills = [1.0, 4.5, 9.0, 13.5]
    fn = render_gpu if renderer == "gpu" else render_cpu
    rendered = fn.remote(run_id, files["trace.npz"], ["side", "cinematic"], width, height, stills)
    out = _write_local(run_id, {**files, **rendered})
    print(rendered["render.json"].decode())
    print(f"saved to {out}")


@app.function(image=blender_image, gpu="L40S", cpu=8.0, memory=32768, timeout=3600)
def render_video_chunk(rig: bytes, poses: bytes, scene: bytes, terrain: bytes, frames: list[int],
                       camera: str, res: str, samples: int) -> dict[str, bytes]:
    import subprocess

    work = Path("/tmp/vid_job")
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    for n, b in (("rig.json", rig), ("poses.npz", poses), ("scene.json", scene), ("terrain.npz", terrain)):
        (work / n).write_bytes(b)
    (work / "frames.txt").write_text(" ".join(map(str, frames)))
    boot = ("import sys; sys.path.insert(0, '/root'); import runpy; "
            "runpy.run_module('ariel.art.render', run_name='__main__')")
    cmd = ["/opt/blender/blender", "-b", "--python-exit-code", "1", "--python-expr", boot, "--",
           "--rig", str(work / "rig.json"), "--poses", str(work / "poses.npz"), "--out", str(out),
           "--assets", "/assets", "--scene", str(work / "scene.json"), "--terrain", str(work / "terrain.npz"),
           "--cameras", camera, "--res", res, "--samples", str(samples), "--frame-list", str(work / "frames.txt")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    files = {f.name: f.read_bytes() for f in out.iterdir() if f.suffix == ".png"}
    files["log"] = (proc.stdout[-3000:] + "\n" + proc.stderr[-3000:]).encode()
    return files


@app.function(image=blender_image, cpu=4.0, memory=8192, timeout=1800, volumes={RUNS: runs})
def encode_video(run_id: str, name: str, frames: dict[str, bytes], fps: float) -> bytes:
    import subprocess
    import tempfile

    d = Path(tempfile.mkdtemp())
    for i, k in enumerate(sorted(frames)):
        (d / f"f_{i:05d}.png").write_bytes(frames[k])
    mp4 = d / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(d / "f_%05d.png"),
                    "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(mp4)], check=True)
    _save(run_id, {f"{name}.mp4": mp4.read_bytes()})
    return mp4.read_bytes()


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=900, volumes={RUNS: runs})
def course_body_check(run_id: str) -> dict[str, bytes]:
    """The scripted reshaping program, performed on the showcase course's landing pad."""
    from ariel.robot.scripted import run_body_check
    from ariel.world.course import CourseParams, generate_course
    from ariel.world.settle import pack_terrain, pack_trace

    course = generate_course(CourseParams())
    res = run_body_check(course=course)
    files = {"trace.npz": pack_trace(res), "terrain.npz": pack_terrain(course),
             "scene.json": json.dumps(course.scene_json()).encode(),
             "metrics.json": json.dumps(res["metrics"], indent=2).encode()}
    _save(run_id, files)
    return files


@app.local_entrypoint()
def showcase_videos(res: str = "1920x1080", samples: int = 96, chunks: int = 10, every: int = 2,
                    clips: str = "flythrough,track"):
    """Photoreal clips: a course flythrough and the scripted reshaping demo on the pad."""
    run_id = time.strftime("videos-%Y%m%d-%H%M%S")
    src = course_body_check.remote(run_id)
    print(src["metrics.json"].decode())
    exported = export_rig.remote(src["trace.npz"], every)
    import numpy as np
    n = len(np.load(io.BytesIO(exported["poses.npz"]))["pos"])
    fps = float(np.load(io.BytesIO(exported["poses.npz"]))["fps"])
    frames = list(range(n))
    out = {}
    for camera, name in (("flythrough", "base_flythrough"), ("track", "reshaping_demo")):
        if camera not in clips.split(","):
            continue
        parts = [frames[i::chunks] for i in range(chunks)]
        args = [(exported["rig.json"], exported["poses.npz"], src["scene.json"], src["terrain.npz"], p,
                 camera, res, samples) for p in parts]
        pngs = {}
        for files in render_video_chunk.starmap(args):
            pngs.update({k: v for k, v in files.items() if k.endswith(".png")})
        print(f"{name}: {len(pngs)} frames")
        out[f"{name}.mp4"] = encode_video.remote(run_id, name, pngs, fps)
        for t in (0.1, 0.35, 0.6, 0.85):
            key = sorted(pngs)[int(t * (len(pngs) - 1))]
            out[f"{name}_{key}"] = pngs[key]
    local = _write_local(run_id, {**out, "metrics.json": src["metrics.json"]})
    print(f"saved to {local}")


VISUAL_VERSION = "v1"


@app.function(image=blender_image, cpu=8.0, memory=32768, timeout=3600, volumes={RUNS: runs})
def export_visual_assets(rig: bytes, poses: bytes, scene: bytes, terrain: bytes) -> dict[str, bytes]:
    """Run Blender headless to export MuJoCo visual meshes; store them under /runs/visual/<version>."""
    import subprocess

    work = Path("/tmp/vis_job")
    out = RUNS / "visual" / VISUAL_VERSION
    if out.exists():
        import shutil
        shutil.rmtree(out)
    work.mkdir(parents=True, exist_ok=True)
    for n, b in (("rig.json", rig), ("poses.npz", poses), ("scene.json", scene), ("terrain.npz", terrain)):
        (work / n).write_bytes(b)
    boot = ("import sys; sys.path.insert(0, '/root'); import runpy; "
            "runpy.run_module('ariel.art.render', run_name='__main__')")
    cmd = ["/opt/blender/blender", "-b", "--python-exit-code", "1", "--python-expr", boot, "--",
           "--rig", str(work / "rig.json"), "--poses", str(work / "poses.npz"), "--out", "/tmp/vis_out",
           "--assets", "/assets", "--scene", str(work / "scene.json"), "--terrain", str(work / "terrain.npz"),
           "--device", "CPU", "--export-visual", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    files = {"export.log": (proc.stdout[-6000:] + "\n--- stderr ---\n" + proc.stderr[-4000:]).encode()}
    if out.exists():
        import shutil
        for png in ("mj-logo-badge.png", "mj-logo-pad.png"):
            shutil.copy(f"/assets/repo/branding/{png}", out / png)
        for f in out.iterdir():
            files["visual/" + f.name] = f.read_bytes()
    runs.commit()
    return files


@app.local_entrypoint()
def visual_export():
    run_id = time.strftime("visual-%Y%m%d-%H%M%S")
    settled = course_settle.remote(run_id)
    exported = export_rig.remote(settled["trace.npz"])
    files = export_visual_assets.remote(exported["rig.json"], exported["poses.npz"], settled["scene.json"],
                                        settled["terrain.npz"])
    out = _write_local(run_id, {k.replace("visual/", "visual__"): v for k, v in files.items()})
    log = files["export.log"].decode(errors="replace")
    print(log[-2500:] if ("Traceback" in log or "Error" in log) else log[-600:])
    n = sum(1 for k in files if k.endswith(".obj"))
    print(f"{n} meshes, {sum(len(v) for k, v in files.items() if k.endswith('.obj')) / 1e6:.1f} MB -> {out}")


@app.function(image=egl_image, gpu="L4", cpu=4.0, memory=16384, timeout=1800, volumes={RUNS: runs})
def mj_render_trace(run_id: str, trace: bytes, cameras: list[str], width: int = 1280, height: int = 720,
                    every: int = 2, stills: list[float] = ()) -> dict[str, bytes]:
    """Per-episode recording renderer: MuJoCo + the detailed visual meshes from the volume."""
    import imageio.v2 as imageio
    import numpy as np

    from ariel.render.mjrender import TraceRenderer, model_with_visuals, write_video
    from ariel.world.visual import load_library

    lib_dir = RUNS / "visual" / VISUAL_VERSION
    lib = load_library(lib_dir) if lib_dir.exists() else None
    z = np.load(io.BytesIO(trace))
    t0 = time.perf_counter()
    model = model_with_visuals(z, lib)
    compile_s = time.perf_counter() - t0
    fps = float(z["fps"]) / every
    qpos = z["qpos"][::every]
    r = TraceRenderer(model, width=width, height=height)
    files: dict[str, bytes] = {}
    timing = {}
    for cam in cameras:
        frames = []
        t0 = time.perf_counter()
        for f in r.frames(qpos, fps, camera=cam):
            frames.append(f)
        timing[cam] = round((time.perf_counter() - t0) / max(len(frames), 1) * 1000, 2)
        path = Path(f"/tmp/{cam}.mp4")
        write_video(frames, path, fps)
        files[f"mujoco_{cam}.mp4"] = path.read_bytes()
        for t in stills:
            b = io.BytesIO()
            imageio.imwrite(b, frames[min(len(frames) - 1, int(t * fps))], format="png")
            files[f"mujoco_{cam}_t{t:05.2f}.png"] = b.getvalue()
    files["mujoco_render.json"] = json.dumps({"ms_per_frame": timing, "compile_seconds": round(compile_s, 2),
                                              "visual_library": lib is not None, "fps": fps}).encode()
    _save(run_id, files)
    return files


@app.local_entrypoint()
def recording_sample(trace_dir: str = ""):
    """Render a recorded trace the way every training episode will be recorded."""
    src = Path(trace_dir) if trace_dir else sorted(Path("runs").glob("videos-*"))[-1]
    trace = (src / "trace.npz").read_bytes() if (src / "trace.npz").exists() else None
    run_id = time.strftime("recsample-%Y%m%d-%H%M%S")
    if trace is None:
        trace = course_body_check.remote(run_id)["trace.npz"]
    files = mj_render_trace.remote(run_id, trace, ["side", "cinematic"], 1280, 720, 2, [1.0, 4.5, 9.0, 13.0])
    out = _write_local(run_id, files)
    print(files["mujoco_render.json"].decode())
    print(f"saved to {out}")


@app.local_entrypoint()
def settle_only():
    """Physics settle of the showcase course; prints metrics (no rendering)."""
    run_id = time.strftime("settle-%Y%m%d-%H%M%S")
    files = course_settle.remote(run_id)
    print(files["metrics.json"].decode())


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=1200)
def env_smoke(steps: int = 1500, level: float = 0.0) -> dict:
    """Random-action smoke test of the RL environment (correctness and throughput)."""
    import numpy as np

    from ariel.env.lunar import OBSTACLES, LunarRescueEnv

    t0 = time.perf_counter()
    env = LunarRescueEnv(seed=1, pool_size=2, level={k: level for k in OBSTACLES})
    build = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    obs = env.reset()
    dims = (len(obs["actor"]), len(obs["critic"]))
    reasons, returns, ret, terms_sum = {}, [], 0.0, {}
    t0 = time.perf_counter()
    for _ in range(steps):
        a = np.clip(rng.normal(0, 0.5, env.slot.model.nu), -1, 1)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        for k, v in info["terms"].items():
            terms_sum[k] = terms_sum.get(k, 0.0) + v
        assert np.all(np.isfinite(obs["actor"])), "non-finite observation"
        if term or trunc:
            reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
            returns.append(round(ret, 2))
            ret = 0.0
            obs = env.reset()
    dt = time.perf_counter() - t0
    return {"obs_dims": dims, "build_s": round(build, 1), "steps_per_s": round(steps / dt, 1),
            "episodes": len(returns), "reasons": reasons, "returns": returns[:10],
            "reward_terms_total": {k: round(v, 2) for k, v in terms_sum.items()}}


@app.local_entrypoint()
def smoke():
    print(json.dumps(env_smoke.remote(), indent=1))


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=1200)
def env_profile() -> dict:
    import numpy as np
    import mujoco

    from ariel.env.lunar import OBSTACLES, LunarRescueEnv

    env = LunarRescueEnv(seed=1, pool_size=1, level={k: 0.0 for k in OBSTACLES})
    env.reset(spawn=0)
    s = env.slot
    m, d = s.model, s.data
    out = {}
    t = time.perf_counter()
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=s.nstep)
    out["physics_ms_per_step"] = (time.perf_counter() - t) / 200 * 1000
    t = time.perf_counter()
    for _ in range(200):
        env._obs()
    out["obs_ms"] = (time.perf_counter() - t) / 200 * 1000
    t = time.perf_counter()
    for _ in range(200):
        env._contacts()
    out["contacts_ms"] = (time.perf_counter() - t) / 200 * 1000
    gid = np.array([-1], np.int32)
    t = time.perf_counter()
    for _ in range(1000):
        mujoco.mj_ray(m, d, np.array([1.0, 0.0, 2.0]), np.array([0, 0, -1.0]), s.rays_group, 1, -1, gid)
    out["ray_us"] = (time.perf_counter() - t) / 1000 * 1e6
    mujoco.mj_forward(m, d)
    out["ncon"] = int(d.ncon)
    out["ngeom"] = int(m.ngeom)
    t = time.perf_counter()
    for _ in range(2000):
        mujoco.mj_step(m, d)
    out["single_substep_ms"] = (time.perf_counter() - t) / 2000 * 1000
    m2 = m
    m2.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    t = time.perf_counter()
    for _ in range(2000):
        mujoco.mj_step(m2, d)
    out["substep_no_contact_ms"] = (time.perf_counter() - t) / 2000 * 1000
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}


@app.local_entrypoint()
def profile():
    print(json.dumps(env_profile.remote(), indent=1))


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=1200)
def contact_census() -> dict:
    from collections import Counter

    import mujoco

    from ariel.env.lunar import OBSTACLES, LunarRescueEnv

    env = LunarRescueEnv(seed=1, pool_size=1, level={k: 1.0 for k in OBSTACLES})
    env.reset(spawn=0)
    m, d = env.slot.model, env.slot.data
    for _ in range(50):
        mujoco.mj_step(m, d, nstep=10)
    pairs = Counter()
    for c in d.contact[:d.ncon]:
        a, b = sorted((m.geom(c.geom1).name.split("_")[0], m.geom(c.geom2).name.split("_")[0]))
        pairs[f"{a}-{b}"] += 1
    return {"ncon": int(d.ncon), "pairs": dict(pairs.most_common())}


@app.local_entrypoint()
def census():
    print(json.dumps(contact_census.remote(), indent=1))


train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libegl1", "libgl1", "libglvnd0", "libopengl0", "ffmpeg")
    .pip_install(f"mujoco=={MUJOCO_VERSION}", "numpy>=2.0", "imageio>=2.34", "imageio-ffmpeg>=0.5")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .env({"MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1"})
    .add_local_python_source("ariel")
)


@app.function(image=train_image, gpu="L4", cpu=32.0, memory=65536, timeout=24 * 3600, volumes={RUNS: runs},
              retries=modal.Retries(max_retries=3, initial_delay=15.0, backoff_coefficient=1.0))
def train_job(run_id: str, overrides: dict | None = None, note: str = "") -> dict:
    """PPO training. Every episode from every worker is recorded to /runs/<run_id>/episodes."""
    import threading

    if note:
        runs.reload()
        log = RUNS / run_id / "train_log.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps({"event": "note", "note": note, "time": time.time()}) + "\n")

    from ariel.train.ppo import TrainConfig, train

    runs.reload()                                   # see checkpoints written before a restart
    cfg = TrainConfig(run_id=run_id, out_root=str(RUNS), **(overrides or {}))
    stop = threading.Event()

    def committer():
        # persist recordings and checkpoints to the volume every minute (transient errors retried)
        while not stop.wait(60):
            for k in range(3):
                try:
                    runs.commit()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(5 * (k + 1))

    th = threading.Thread(target=committer, daemon=True)
    th.start()

    def on_log(row):
        print(json.dumps(row), flush=True)

    try:
        res = train(cfg, on_log)
    finally:
        stop.set()
        runs.commit()
    return res


@app.local_entrypoint()
def train_smoke(minutes: float = 4.0, workers: int = 8):
    """Short end-to-end check: rollouts, PPO updates, checkpoints and complete recordings."""
    run_id = time.strftime("smoke-%Y%m%d-%H%M%S")
    res = train_job.remote(run_id, {"workers": workers, "envs_per_worker": 4, "hours": minutes / 60,
                                    "checkpoint_every": 10, "log_every": 2})
    print(json.dumps(res, indent=1, default=str))
    print("run id:", run_id)


@app.function(image=cpu_image, cpu=4.0, memory=16384, timeout=1800, volumes={RUNS: runs})
def audit_run(run_id: str, sample: str = "best") -> dict:
    """Verify every recorded episode is indexed and readable; return a sample episode as a trace."""
    from collections import Counter

    import numpy as np

    from ariel.record.recorder import load_episode
    from ariel.world.course import CourseParams, generate_course
    from ariel.world.scene import build_world_xml, hfield_bytes

    runs.reload()
    run_dir = RUNS / run_id
    rows = []
    for idx in sorted((run_dir / "episodes").glob("w*/index.jsonl")):
        rows += [json.loads(line) for line in idx.read_text().splitlines() if line.strip()]
    missing, chunks_checked = 0, set()
    for r in rows:
        p = run_dir / "episodes" / f"w{r['worker']:03d}" / r["chunk"]
        if not p.exists():
            missing += 1
            continue
        if p not in chunks_checked:
            with np.load(p) as z:
                keys = set(z.files)
            chunks_checked.add(p)
        if f"{r['key']}_qpos" not in keys and r["steps"] > 0:
            missing += 1
    reasons = Counter(r["reason"] for r in rows)
    status = Counter(r["status"] for r in rows)
    fell = [round(r["max_x"], 2) for r in rows if r["reason"] == "fell_into_fissure"][:20]
    out = {"episodes": len(rows), "missing_or_unreadable": missing, "chunks": len(chunks_checked),
           "status": dict(status), "reasons": dict(reasons),
           "steps_recorded": int(sum(r["steps"] for r in rows)),
           "policy_versions": [min(r["policy_version_min"] for r in rows), max(r["policy_version_max"] for r in rows)],
           "fissure_fall_max_x": fell, "workers": len({r["worker"] for r in rows})}
    if sample == "recent":
        newest = sorted(rows, key=lambda r: r["policy_version_max"])[-600:]
        pick = max(newest, key=lambda r: r["max_x"] - r.get("spawn_x", 0))
    elif sample == "best":
        pick = max(rows, key=lambda r: r["max_x"] - r.get("spawn_x", 0))
    else:
        pick = rows[-1]
    ep = load_episode(run_dir, pick)
    course = generate_course(CourseParams(**pick["course_params"]))
    xml, _ = build_world_xml(course=course)
    import io as _io
    buf = _io.BytesIO()
    np.savez_compressed(buf, qpos=ep["qpos"], fps=pick["playback_hz"], xml=np.array(xml),
                        hfield=np.frombuffer(hfield_bytes(course), dtype=np.uint8),
                        scene=np.array(json.dumps(course.scene_json())))
    out["sample"] = {k: pick[k] for k in ("episode_id", "reason", "steps", "return", "max_x", "spawn_x",
                                           "policy_version_min", "policy_version_max")}
    return {"summary": out, "trace": buf.getvalue()}


@app.local_entrypoint()
def audit(run_id: str, render: bool = True, sample: str = "best"):
    res = audit_run.remote(run_id, sample)
    print(json.dumps(res["summary"], indent=1))
    local = _write_local(f"audit-{run_id}", {"summary.json": json.dumps(res["summary"], indent=1).encode(),
                                             "sample_trace.npz": res["trace"]})
    if render:
        files = mj_render_trace.remote(f"audit-{run_id}", res["trace"], ["side", "cinematic"], 1280, 720, 1, [1.0, 3.0])
        _write_local(f"audit-{run_id}", files)
    print("saved to", local)


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=1800)
def debug_sink(max_episodes: int = 60) -> dict:
    import numpy as np
    import mujoco

    from ariel.env.lunar import OBSTACLES, LunarRescueEnv

    env = LunarRescueEnv(seed=3, pool_size=1, level={k: 0.0 for k in OBSTACLES})
    rng = np.random.default_rng(1)
    for ep in range(max_episodes):
        env.reset(spawn=0)
        m, d = env.slot.model, env.slot.data
        hist = []
        while True:
            a = np.clip(rng.normal(0, 0.6, m.nu), -1, 1)
            _, _, term, trunc, info = env.step(a)
            touching = sorted({m.geom(c.geom2 if env.slot.robot_geom[c.geom2] else c.geom1).name
                               for c in d.contact[:d.ncon]
                               if env.slot.robot_geom[c.geom1] != env.slot.robot_geom[c.geom2]})
            deepest = max([-c.dist for c in d.contact[:d.ncon]] + [0.0])
            hist.append({"t": env.t, "pelvis": [round(float(v), 3) for v in d.qpos[:3]],
                         "head_z": round(float(d.geom_xpos[m.geom("head").id][2]), 3),
                         "chest_z": round(float(d.geom_xpos[m.geom("chest").id][2]), 3),
                         "touching": touching, "deepest": round(deepest, 4), "ncon": int(d.ncon)})
            if term or trunc:
                break
        if info["reason"] == "fell_into_fissure" and info["x"] < 2.0:
            ground = env._ground(float(d.qpos[0]), float(d.qpos[1]))
            return {"episode": ep, "ground_under_pelvis": round(ground, 3), "last": hist[-12:],
                    "first_below_0": next((h for h in hist if h["pelvis"][2] < 0), None),
                    "warnings": [int(w.number) for w in d.warning]}
    return {"no_repro": True}


@app.local_entrypoint()
def sink():
    print(json.dumps(debug_sink.remote(), indent=1))


@app.local_entrypoint()
def train_full(hours: float = 6.0, workers: int = 30, envs_per_worker: int = 6, comparison: bool = True):
    """Launch the main reshaping run (and the locked-telescoping comparison) detached on Modal.

    Use with ``modal run --detach`` so training continues if this terminal disconnects.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    common = {"workers": workers, "envs_per_worker": envs_per_worker, "hours": hours, "checkpoint_every": 100}
    calls = {f"reshape-{stamp}": train_job.spawn(f"reshape-{stamp}", common)}
    if comparison:
        calls[f"locked-{stamp}"] = train_job.spawn(f"locked-{stamp}", {**common, "lock_telescoping": True, "seed": 1})
    for run_id, call in calls.items():
        print(f"{run_id}: function call {call.object_id}")
    Path("runs").mkdir(exist_ok=True)
    (Path("runs") / f"launch-{stamp}.json").write_text(json.dumps({k: v.object_id for k, v in calls.items()}, indent=1))


HELDOUT_SEEDS = (90001, 90002, 90003)   # never used for training (training seeds are random 30-bit draws)


@app.function(image=train_image, cpu=8.0, memory=16384, timeout=3 * 3600, volumes={RUNS: runs})
def eval_checkpoint(run_id: str, checkpoint: str, lock_telescoping: bool = False) -> list[dict]:
    from dataclasses import asdict

    import numpy as np

    from ariel.train.evaluate import evaluate
    from ariel.world.course import CourseParams, random_params

    runs.reload()
    showcase = asdict(CourseParams())
    heldout = []
    for s in HELDOUT_SEEDS:
        p = random_params(np.random.default_rng(s))
        p.seed = s
        heldout.append(asdict(p))
    res = evaluate(str(RUNS / run_id / "checkpoints" / checkpoint), [showcase] + heldout,
                   episodes_per_course=1, lock_telescoping=lock_telescoping)
    out_dir = RUNS / run_id / "evaluations" / checkpoint.replace(".pt", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(res):
        (out_dir / f"course{i}_trace.npz").write_bytes(r.pop("trace"))
        r["trace_file"] = f"course{i}_trace.npz"
        r["course_role"] = "showcase" if i == 0 else "heldout"
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    runs.commit()
    return res


@app.local_entrypoint()
def compare(run_id: str, picks: int = 3, render: bool = True):
    """Evaluate early / middle / late checkpoints of a run on the showcase and held-out courses."""
    import subprocess

    names = subprocess.run(["modal", "volume", "ls", "ariel-runs", f"{run_id}/checkpoints"],
                           capture_output=True, text=True).stdout.split()
    ckpts = sorted({n.split("/")[-1] for n in names if n.endswith(".pt")})
    chosen = [ckpts[int(round(i * (len(ckpts) - 1) / max(picks - 1, 1)))] for i in range(picks)] if ckpts else []
    locked = "locked" in run_id
    summary = {}
    for ck, res in zip(chosen, eval_checkpoint.starmap([(run_id, c, locked) for c in chosen])):
        summary[ck] = [{k: r[k] for k in ("course_role", "course_id", "reason", "max_x", "success", "return", "steps")}
                       for r in res]
        print(ck, json.dumps(summary[ck]))
    if render and chosen:
        # render each checkpoint's showcase attempt with the recording renderer
        tmp = Path("runs") / f"compare-{run_id}"
        tmp.mkdir(parents=True, exist_ok=True)
        for ck in chosen:
            local_trace = tmp / f"{ck.replace('.pt', '')}_showcase_trace.npz"
            tr = subprocess.run(["modal", "volume", "get", "--force", "ariel-runs",
                                 f"{run_id}/evaluations/{ck.replace('.pt', '')}/course0_trace.npz", str(local_trace)],
                                capture_output=True)
            if tr.returncode == 0 and local_trace.exists():
                files = mj_render_trace.remote(f"compare-{run_id}", local_trace.read_bytes(), ["cinematic"])
                _write_local(f"compare-{run_id}", {f"{ck.replace('.pt', '')}_{k}": v for k, v in files.items()})
    _write_local(f"compare-{run_id}", {"summary.json": json.dumps(summary, indent=1).encode()})


@app.function(image=cpu_image, cpu=2.0, memory=4096, timeout=900)
def balance_trial(balance: dict) -> dict:
    from ariel.robot.scripted import run_body_check
    from ariel.world.course import CourseParams, generate_course

    res = run_body_check(course=generate_course(CourseParams()), balance=balance)
    m = res["metrics"]
    return {"balance": balance, "min_upright": m["min_uprightness"], "drift": m["pelvis_xy_drift_m"],
            "head": m["head_height_min_max"]}


@app.local_entrypoint()
def balance_sweep():
    grid = [{"ankle_kp": kp, "ankle_kd": kd, "hip_k": hk, "crouch": cr}
            for kp in (2.0, 3.5) for kd in (0.4, 0.8) for hk in (0.0, 1.5) for cr in (0.55, 0.7)]
    rows = sorted(balance_trial.map(grid), key=lambda r: (-r["min_upright"], r["drift"]))
    for r in rows[:8]:
        print(json.dumps(r))


@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=3 * 3600, volumes={RUNS: runs})
def showcase_remote(run_id: str, clips: str = "track", chunks: int = 6, res: str = "1920x1080",
                    samples: int = 96, every: int = 2) -> dict:
    """Server-side version of showcase_videos: survives the local client going away."""
    import numpy as np

    src = course_body_check.remote(run_id)
    exported = export_rig.remote(src["trace.npz"], every)
    z = np.load(io.BytesIO(exported["poses.npz"]))
    n, fps = len(z["pos"]), float(z["fps"])
    frames = list(range(n))
    made = {}
    for camera, name in (("flythrough", "base_flythrough"), ("track", "reshaping_demo")):
        if camera not in clips.split(","):
            continue
        parts = [frames[i::chunks] for i in range(chunks)]
        args = [(exported["rig.json"], exported["poses.npz"], src["scene.json"], src["terrain.npz"], p,
                 camera, res, samples) for p in parts]
        pngs = {}
        for files in render_video_chunk.starmap(args):
            pngs.update({k: v for k, v in files.items() if k.endswith(".png")})
        encode_video.remote(run_id, name, pngs, fps)
        stills = {}
        for t in (0.1, 0.35, 0.6, 0.85):
            key = sorted(pngs)[int(t * (len(pngs) - 1))]
            stills[f"{name}_{key}"] = pngs[key]
        _save(run_id, stills)
        made[name] = len(pngs)
    return {"run_id": run_id, "clips": made, "metrics": json.loads(src["metrics.json"])}


@app.local_entrypoint()
def showcase_detached(clips: str = "track", chunks: int = 6):
    run_id = time.strftime("videos-%Y%m%d-%H%M%S")
    call = showcase_remote.spawn(run_id, clips, chunks)
    print(f"spawned {run_id} as {call.object_id}")



@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=1800)
def crawl_check(clearances: list[float] = (0.72, 0.62)) -> dict:
    """Hold the compressed army-crawl posture under the collapsed module and measure the fit (physics)."""
    from dataclasses import replace

    import numpy as np
    import mujoco

    from ariel.env.lunar import LunarRescueEnv
    from ariel.robot.poses import place, robot_top_and_bottom
    from ariel.world.course import CourseParams
    from ariel.world.scene import hfield_bytes

    out, traces = {}, {}
    for clr in clearances:
        params = replace(CourseParams(), hab_clearance=float(clr))
        env = LunarRescueEnv(seed=0, pool_size=1, fixed_params=params)
        env.reset(spawn=0)
        s = env.slot
        m, d = s.model, s.data
        hab = {m.geom(f"hab_shell_{k}").id for k in range(10)}
        vec = place(m, d, s.amap, s.robot_geom, "crawl", params.hab_x, 0.0, lambda x, y: s.course.height_at(x, y), env.spec)
        mujoco.mj_forward(m, d)
        top0, _ = robot_top_and_bottom(m, d, s.robot_geom)
        s.cmd.reset(d)
        qs, tops, hab_contacts = [], [], 0
        p0 = d.qpos[:3].copy()
        for t in range(150):                      # 3 s holding the posture under physics
            s.cmd.set_targets(vec, d)
            mujoco.mj_step(m, d, nstep=s.nstep)
            top, _ = robot_top_and_bottom(m, d, s.robot_geom)
            tops.append(top)
            hab_contacts += sum(1 for c in d.contact[:d.ncon] if (c.geom1 in hab) != (c.geom2 in hab)
                                and (s.robot_geom[c.geom1] or s.robot_geom[c.geom2]))
            if t % 2 == 0:
                qs.append(d.qpos.astype(np.float32).copy())
        under = params.hab_clearance + s.course.height_at(params.hab_x, 0.0)
        out[f"clearance_{clr}"] = {
            "gap_underside_z": round(under, 3), "robot_top_initial": round(top0, 3),
            "robot_top_max_during_hold": round(max(tops), 3), "margin_m": round(under - max(tops), 3),
            "hab_contacts_during_hold": int(hab_contacts),
            "pelvis_moved_m": round(float(np.linalg.norm(d.qpos[:3] - p0)), 3),
        }
        buf = io.BytesIO()
        np.savez_compressed(buf, qpos=np.stack(qs), fps=25.0, xml=np.array(s.xml),
                            hfield=np.frombuffer(hfield_bytes(s.course), dtype=np.uint8),
                            scene=np.array(json.dumps(s.course.scene_json())))
        traces[clr] = buf.getvalue()
    return {"results": out, "trace_072": traces[clearances[0]]}


@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=1800, volumes={RUNS: runs})
def verify_traverse(run_id: str) -> dict:
    from ariel.verify.traverse import record_and_reload
    from ariel.world.course import CourseParams, generate_course
    from ariel.world.settle import pack_terrain

    trace, info = record_and_reload(str(RUNS), run_id)
    runs.commit()
    return {"trace": trace, "info": info, "terrain": pack_terrain(generate_course(CourseParams()))}


@app.function(image=cpu_image, cpu=2.0, memory=8192, timeout=3 * 3600, volumes={RUNS: runs})
def verify_remote(run_id: str) -> dict:
    """Crawl fit + recording round trip + every camera (MuJoCo and Blender), saved to the volume."""
    import numpy as np

    from ariel.world.course import CourseParams

    crawl_call = crawl_check.spawn([0.78, 0.70])
    trav = verify_traverse.remote(run_id)
    mj = mj_render_trace.spawn(run_id, trav["trace"], ["side", "cinematic"], 1280, 720, 1, [])
    exported = export_rig.remote(trav["trace"], 1)
    z = np.load(io.BytesIO(trav["trace"]))
    scene = str(z["scene"]).encode()
    params = CourseParams()
    xs = z["qpos"][:, 0]
    picks = {}
    for label, x in (("approach", params.hab_x - 3.0), ("under_module", params.hab_x - 0.3),
                     ("exit_module", params.hab_x + params.hab_radius + 0.4), ("fissure", params.fissure_x + 0.4),
                     ("scarp", params.scarp_x + 0.1), ("goal", params.goal_x - 1.3)):
        picks[label] = int(np.argmin(np.abs(xs - x)))
    blender = render_video_chunk.spawn(exported["rig.json"], exported["poses.npz"], scene, trav["terrain"],
                                       sorted(picks.values()), "track", "1920x1080", 128)
    crawl = crawl_call.get()
    files = {"crawl_fit.json": json.dumps(crawl["results"], indent=1).encode(),
             "traverse_info.json": json.dumps(trav["info"], indent=1).encode()}
    crawl_mj = mj_render_trace.spawn(run_id + "/crawl", crawl["trace_072"], ["cinematic", "side"], 1280, 720, 1,
                                     [1.0, 2.5])
    bl = blender.get()
    inv = {v: k for k, v in picks.items()}
    for k, v in bl.items():
        if k.endswith(".png"):
            f = int(k.split("_")[1].split(".")[0])
            files[f"blender_{inv.get(f, f)}.png"] = v
    files["blender_log.txt"] = bl.get("log", b"")
    files.update({f"traverse_{k}": v for k, v in mj.get().items()})
    files.update({f"crawl_{k}": v for k, v in crawl_mj.get().items()})
    _save(run_id, files)
    return {"crawl": crawl["results"], "traverse": trav["info"], "files": sorted(files)}


@app.local_entrypoint()
def verify_all():
    run_id = time.strftime("verify-%Y%m%d-%H%M%S")
    call = verify_remote.spawn(run_id)
    print(f"spawned {run_id} as {call.object_id}")


@app.local_entrypoint()
def train_smoke_detached(minutes: float = 4.0, workers: int = 8):
    run_id = time.strftime("smoke-%Y%m%d-%H%M%S")
    call = train_job.spawn(run_id, {"workers": workers, "envs_per_worker": 4, "hours": minutes / 60,
                                    "checkpoint_every": 10, "log_every": 2})
    print(f"spawned {run_id} as {call.object_id}")


@app.local_entrypoint()
def resume_test():
    """Train briefly, stop, resume into the same run; then the audit checks continuity."""
    run_id = time.strftime("resume-%Y%m%d-%H%M%S")
    first = train_job.remote(run_id, {"workers": 6, "envs_per_worker": 4, "hours": 2.0 / 60,
                                      "checkpoint_every": 10, "log_every": 5})
    second = train_job.remote(run_id, {"workers": 6, "envs_per_worker": 4, "hours": 4.0 / 60,
                                       "checkpoint_every": 10, "log_every": 5})
    print(json.dumps({"run_id": run_id, "first_end": [first["version"], first["global_step"]],
                      "second_end": [second["version"], second["global_step"]]}))


@app.local_entrypoint()
def train_resume(launch_file: str, hours: float = 5.5, note: str = "", level_cap: float = -1.0):
    """Relaunch runs into their existing run ids; each resumes from its latest checkpoint."""
    runs_ = json.loads(Path(launch_file).read_text())
    calls = {}
    for run_id in runs_:
        over = {"workers": 30, "envs_per_worker": 6, "hours": hours, "checkpoint_every": 100,
                "level_cap_on_resume": level_cap}
        if run_id.startswith("locked"):
            over.update({"lock_telescoping": True, "seed": 1})
        calls[run_id] = train_job.spawn(run_id, over, note)
        print(f"{run_id}: function call {calls[run_id].object_id}")
    Path(launch_file).write_text(json.dumps({k: v.object_id for k, v in calls.items()}, indent=1))


# ---------------------------------------------------------------------- CMU walker (MoCapAct stack) renders
def _load_export(src: str) -> dict:
    d = Path(src)
    return {n: (d / n).read_bytes() for n in ("rig.json", "poses.npz", "scene.json", "terrain.npz")}


@app.local_entrypoint()
def cmu_stills(src: str, xs: str = "", cameras: str = "hero,side", shots: str = "establish,overview",
               res: str = "1280x720", samples: int = 64):
    """Blender stills of an exported showcase episode (see mocapact_app.py::showcase_export).

    ``xs``: course positions (m) at which to render the robot cameras; ``shots``: fixed course views."""
    import numpy as np

    ex = _load_export(src)
    z = np.load(io.BytesIO(ex["poses.npz"]))
    names = [str(n) for n in z["names"]]
    x = z["pos"][:, names.index("root"), 0]
    want = [float(v) for v in xs.split(",")] if xs else list(np.linspace(x.min() + 1.0, x.max() - 0.5, 4))
    frames = sorted({int(np.argmin(np.abs(x - v))) for v in want})
    run_id = time.strftime("cmu-stills-%Y%m%d-%H%M%S")
    calls = [render_blender.spawn(run_id + f"/f{f:05d}", ex["rig.json"], ex["poses.npz"], [f], cameras.split(","),
                                  res, samples, False, ex["scene.json"], ex["terrain.npz"]) for f in frames]
    if shots:
        calls.append(render_blender.spawn(run_id + "/shots", ex["rig.json"], ex["poses.npz"], [frames[0]],
                                          shots.split(","), res, samples, False, ex["scene.json"], ex["terrain.npz"]))
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    for c in calls:
        files = c.get()
        for k, v in files.items():
            if k.endswith(".png"):
                (out / k).write_bytes(v)
        log = files.get("blender.log", b"").decode(errors="replace")
        if "Traceback" in log or json.loads(files.get("blender_wall.json", b'{"returncode": 0}'))["returncode"]:
            print(log[-3500:])
        print(files.get("blender_wall.json", b"").decode())
    print(f"saved to {out} (frames {frames}, source {src})")


@app.local_entrypoint()
def cmu_video(src: str, camera: str = "track", chunks: int = 16, every: int = 1, res: str = "1920x1080",
              samples: int = 64, start: int = 0, stop: int = 0, name: str = "showcase"):
    """Blender video of an exported showcase episode, rendered in parallel chunks."""
    import numpy as np

    ex = _load_export(src)
    z = np.load(io.BytesIO(ex["poses.npz"]))
    n, fps = len(z["pos"]), float(z["fps"])
    frames = list(range(start, stop or n, every))
    run_id = time.strftime("cmu-video-%Y%m%d-%H%M%S")
    parts = [frames[i::chunks] for i in range(chunks)]
    args = [(ex["rig.json"], ex["poses.npz"], ex["scene.json"], ex["terrain.npz"], p, camera, res, samples)
            for p in parts if p]
    pngs = {}
    for files in render_video_chunk.starmap(args):
        got = {k: v for k, v in files.items() if k.endswith(".png")}
        pngs.update(got)
        if not got:
            print(files.get("log", b"").decode(errors="replace")[-3000:])
    print(f"{len(pngs)} of {len(frames)} frames rendered")
    mp4 = encode_video.remote(run_id, name, pngs, fps / every)
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.mp4").write_bytes(mp4)
    for t in (0.15, 0.45, 0.75):
        key = sorted(pngs)[int(t * (len(pngs) - 1))]
        (out / key).write_bytes(pngs[key])
    print(f"saved to {out}")


def _render_chunk(run_id: str, rig: bytes, poses: bytes, scene: bytes, terrain: bytes, frames: list[int],
                  camera: str, res: str, samples: int) -> dict:
    """Render frames and leave them on the volume (frames never travel through the local client)."""
    import shutil
    import subprocess

    import tempfile
    work = Path(tempfile.mkdtemp(prefix="vid_job_"))
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    for n, b in (("rig.json", rig), ("poses.npz", poses), ("scene.json", scene), ("terrain.npz", terrain)):
        (work / n).write_bytes(b)
    (work / "frames.txt").write_text(" ".join(map(str, frames)))
    boot = ("import sys; sys.path.insert(0, '/root'); import runpy; "
            "runpy.run_module('ariel.art.render', run_name='__main__')")
    cmd = ["/opt/blender/blender", "-b", "--python-exit-code", "1", "--python-expr", boot, "--",
           "--rig", str(work / "rig.json"), "--poses", str(work / "poses.npz"), "--out", str(out),
           "--assets", "/assets", "--scene", str(work / "scene.json"), "--terrain", str(work / "terrain.npz"),
           "--cameras", camera, "--res", res, "--samples", str(samples), "--frame-list", str(work / "frames.txt")]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dest = RUNS / run_id / "frames"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in out.glob("frame_*.png"):
        shutil.copy(f, dest / f.name)
        n += 1
    runs.commit()
    return {"frames": n, "wanted": len(frames), "seconds": round(time.time() - t0, 1), "returncode": proc.returncode,
            "log": (proc.stdout[-2500:] + "\n" + proc.stderr[-2500:]) if (proc.returncode or n < len(frames)) else ""}


@app.function(image=blender_image, gpu="L40S", cpu=8.0, memory=32768, timeout=3600, volumes={RUNS: runs})
def render_chunk_to_volume(run_id: str, rig: bytes, poses: bytes, scene: bytes, terrain: bytes, frames: list[int],
                           camera: str, res: str, samples: int) -> dict:
    return _render_chunk(run_id, rig, poses, scene, terrain, frames, camera, res, samples)


@app.function(image=blender_image, gpu="A100", cpu=8.0, memory=32768, timeout=3600, volumes={RUNS: runs})
def render_chunk_to_volume_b(run_id: str, rig: bytes, poses: bytes, scene: bytes, terrain: bytes, frames: list[int],
                             camera: str, res: str, samples: int) -> dict:
    """Second GPU pool (A100) for the same work, so more chunks render at once."""
    return _render_chunk(run_id, rig, poses, scene, terrain, frames, camera, res, samples)


@app.function(image=blender_image, cpu=8.0, memory=16384, timeout=3600, volumes={RUNS: runs})
def encode_from_volume(run_id: str, name: str, fps: float, crf: int = 17) -> bytes:
    import subprocess

    runs.reload()
    src = RUNS / run_id / "frames"
    mp4 = RUNS / run_id / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-pattern_type", "glob",
                    "-i", str(src / "frame_*.png"), "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    runs.commit()
    return mp4.read_bytes()


@app.local_entrypoint()
def cmu_film(src: str, camera: str = "track", chunks: int = 16, every: int = 1, res: str = "1920x1080",
             samples: int = 64, start: int = 0, stop: int = 0, name: str = "showcase", crf: int = 17,
             pool_b: int = 0):
    """Blender video of an exported episode; frames stay on the volume and only the mp4 comes back."""
    import numpy as np

    ex = _load_export(src)
    z = np.load(io.BytesIO(ex["poses.npz"]))
    n, fps = len(z["pos"]), float(z["fps"])
    frames = list(range(start, stop or n, every))
    run_id = time.strftime("cmu-film-%Y%m%d-%H%M%S")
    parts = [p for p in (frames[i::chunks] for i in range(chunks)) if p]
    args = [(run_id, ex["rig.json"], ex["poses.npz"], ex["scene.json"], ex["terrain.npz"], p, camera, res, samples)
            for p in parts]
    done = 0
    half = len(args) * pool_b // 100
    calls = [render_chunk_to_volume_b.spawn(*a_) for a_ in args[:half]] + [render_chunk_to_volume.spawn(*a_) for a_ in args[half:]]
    for c in calls:
        r = c.get()
        done += r["wanted"] if r["frames"] >= r["wanted"] else r["frames"]
        if r["log"]:
            print(r["log"][-2500:])
        print(f"chunk: {r['frames']}/{r['wanted']} frames in {r['seconds']} s (total {done}/{len(frames)})", flush=True)
    mp4 = encode_from_volume.remote(run_id, name, fps / every, crf)
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.mp4").write_bytes(mp4)
    print(f"saved to {out / (name + '.mp4')} ({done} frames, source {src})")
