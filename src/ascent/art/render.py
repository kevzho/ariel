"""Blender entry point: build a scene and render recorded robot states.

    blender -b --python-exit-code 1 -P render.py -- --rig rig.json --poses poses.npz \
        --out /tmp/out --frames 50,225 --camera hero --res 1920x1080 --samples 128

Runs inside Blender only; ``ascent`` must be importable on ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from ascent.art import environment as env
from ascent.art.robot import build_robot, set_pose


def setup_render(res: tuple[int, int], samples: int, device: str = "GPU") -> str:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.glossy_bounces = 4
    scene.cycles.transparent_max_bounces = 8
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    used = "CPU"
    if device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    scene.cycles.device = "GPU"
                    used = kind
                    break
            except Exception:  # noqa: BLE001
                continue
        if used in ("OPTIX", "CUDA"):
            scene.cycles.denoiser = "OPENIMAGEDENOISE"  # the OptiX denoiser is unavailable on Modal
    return used


def add_headlamps(empties: dict) -> None:
    """Two spotlights in the head's light bars, aimed ahead and slightly down."""
    chest = empties.get("chest")
    if chest is None:
        return
    for s in (1, -1):
        light = bpy.data.lights.new(f"headlamp_{s}", "SPOT")
        light.energy = 140.0
        light.spot_size = math.radians(70)
        light.spot_blend = 0.45
        light.color = (0.92, 0.96, 1.0)
        light.shadow_soft_size = 0.02
        o = bpy.data.objects.new(f"headlamp_{s}", light)
        bpy.context.scene.collection.objects.link(o)
        o.parent = chest
        o.location = (0.1, s * 0.045, 0.385)
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = Vector((1.0, 0.0, -0.45)).to_track_quat("-Z", "Y")


def add_camera(name: str, lens: float = 35.0):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start = 0.05
    cam.clip_end = 8000
    cam.dof.use_dof = False
    o = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(o)
    bpy.context.scene.camera = o
    return o


def aim(cam, eye: Vector, target: Vector):
    cam.location = eye
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (target - eye).to_track_quat("-Z", "Y")


CAMERA_RIGS = {
    # (azimuth of the camera around the robot, elevation, distance, lens, target height factor)
    "hero": (-38.0, 4.0, 3.3, 35.0),
    "front": (-12.0, 6.0, 3.6, 40.0),
    "side": (-90.0, 6.0, 6.5, 50.0),
    "cinematic": (-40.0, 7.0, 4.6, 32.0),
}


def place_camera(cam, kind: str, pelvis: Vector, lookat_z: float | None = None):
    az, el, dist, lens = CAMERA_RIGS[kind]
    cam.data.lens = lens
    # frame the whole body: the robot's height scales with its pelvis height
    scale = max(1.0, pelvis.z / 0.89)
    dist *= scale
    tz = lookat_z if lookat_z is not None else max(0.55, 0.78 * pelvis.z)
    target = Vector((pelvis.x, pelvis.y if kind != "side" else 0.0, tz))
    eye = target + env.direction(az, el) * dist
    eye.z = max(eye.z, 0.25)
    aim(cam, eye, target)


# Fixed shots along the course: (eye, target, lens). Positions follow the course layout.
def course_shots(scene: dict) -> dict:
    P = scene["params"]
    gx, gy, gz = scene["goal"]
    hab_x, fx, sx, mx = P["hab_x"], P["fissure_x"], P["scarp_x"], P["mast_x"]
    return {
        "establish": ((-4.6, -3.2, 2.1), (6.5, 0.4, 0.6), 24.0),
        "debris": ((2.6, -3.3, 1.2), (6.0, 0.2, 0.3), 28.0),
        "habitat": ((hab_x - 5.2, -3.6, 1.0), (hab_x, 0.2, 1.1), 26.0),
        "over_hab": ((hab_x + 0.4, -7.2, 5.4 + max(0.0, P["hab_clearance"] - 0.8) * 1.6), (hab_x + 2.8, 0.0, 0.2), 24.0),
        "crawlspace": ((hab_x - 2.6, -0.5, 0.35), (hab_x + 1.0, 0.0, 0.45), 20.0),
        "fissure": ((fx - 0.9, -3.6, 1.3), (fx + 0.5, 0.6, -0.4), 22.0),
        "scarp": ((sx - 2.4, -3.0, 1.1), (mx, 0.3, 0.9), 26.0),
        "astronaut": ((gx - 2.0, gy - 1.6, gz + 0.9), (gx, gy, gz + 0.5), 32.0),
        "overview": ((gx * 0.45, -17.0, 7.0), (gx * 0.5, 0.0, 0.0), 22.0),
        "blocks": ((P["pad_end"] + 0.4, -2.8, 1.3), (P["pad_end"] + 4.6, 0.0, 0.7), 28.0),
        "deck": ((hab_x - 4.4, -2.7, 1.25), (hab_x, 0.1, 1.2), 26.0),
        "climb": ((sx + 0.2, -4.6, 1.5), (sx + 3.0, 0.2, 0.9), 26.0),
    }


def camera_path(kind: str, pelvis: np.ndarray, fps: float, shots: dict, low: np.ndarray | None = None):
    """Deterministic per-frame camera (eye, target, lens), identical in every render worker."""
    n = len(pelvis)
    if low is None:
        low = pelvis[:, 2] - 0.89
    # heavily smoothed robot track (centred moving average) so the camera glides
    k = max(1, int(fps * 0.8))
    pad = np.pad(np.column_stack([pelvis, low]), ((k, k), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for i in range(4)], 1)
    if kind == "flythrough":
        # a crane move from behind the pad, over the habitat, down the course and up into an overview
        keys = [shots["establish"], shots["debris"], shots["habitat"], shots["over_hab"], shots["fissure"],
                shots["scarp"], shots["astronaut"], shots["overview"]]
        times = np.linspace(0, n - 1, len(keys))

        def f(i):
            j = int(np.clip(np.searchsorted(times, i, side="right") - 1, 0, len(keys) - 2))
            u = (i - times[j]) / (times[j + 1] - times[j])
            u = u * u * (3 - 2 * u)
            e = np.array(keys[j][0]) * (1 - u) + np.array(keys[j + 1][0]) * u
            g = np.array(keys[j][1]) * (1 - u) + np.array(keys[j + 1][1]) * u
            lens = keys[j][2] * (1 - u) + keys[j + 1][2] * u
            # never pass through structures: pull the camera toward its target if something is in between
            tv, ev = Vector(g), Vector(e)
            d = (ev - tv)
            want = d.length
            clear = spring_arm(tv, d.normalized(), want)
            if clear < want:
                e = np.array(tv + d.normalized() * clear)
            return e, g, lens
        return f
    if kind == "side":
        # side-on tracking view from outside the corridor: constant distance, no orbit
        def side(i):
            p = smooth[min(i, n - 1)]
            body_h = max(0.4, p[2] - p[3])
            target = Vector((p[0] + 0.4, 0.0, p[3] + 0.80 * body_h))
            eye = target + env.direction(-90.0, 6.0) * 6.4
            return tuple(eye), tuple(target), 40.0
        return side
    az0, el, dist, lens0 = CAMERA_RIGS["cinematic" if kind == "track" else kind]
    state = {"d": None, "az": None}

    def f(i):
        p = smooth[min(i, n - 1)]
        body_h = max(0.4, p[2] - p[3])
        target = Vector((p[0], p[1], p[3] + 0.85 * body_h))
        want = dist * max(1.0, body_h / 0.89)
        az_pref = az0 + 25.0 * np.sin(2 * np.pi * i / max(n, 1) * 0.5)   # slow orbit for parallax
        best = None
        for dz in (0, 20, -20, 40, -40, 65, -65, 95, -95):
            d = spring_arm(target, env.direction(az_pref + dz, el), want)
            score = d - 0.012 * abs(dz)
            if best is None or score > best[0]:
                best = (score, d, az_pref + dz)
        _, clear, az = best
        state["az"] = az if state["az"] is None else state["az"] + (az - state["az"]) * 0.06
        state["d"] = clear if state["d"] is None else state["d"] + (clear - state["d"]) * (0.5 if clear < state["d"] else 0.08)
        eye = target + env.direction(state["az"], el) * state["d"]
        # widen the lens when the camera has to sit close, so the whole body stays in frame
        lens = lens0 * min(1.0, max(0.62, state["d"] / want))
        return tuple(eye), tuple(target), lens
    return f


_OBSTACLE_TREES: list = []
OBSTACLE_COLLECTIONS = ("habitat", "solar_tower", "terrain", "astronaut", "boulders", "decor", "lander", "pad")


def _obstacle_trees():
    """World-space BVH trees of the static set (never the robot), built once."""
    if _OBSTACLE_TREES:
        return _OBSTACLE_TREES
    import bmesh
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    for name in OBSTACLE_COLLECTIONS:
        coll = bpy.data.collections.get(name)
        if coll is None:
            continue
        for o in coll.all_objects:
            if o.type != "MESH" or o.hide_render:
                continue
            bm = bmesh.new()
            bm.from_object(o, dg)
            bm.transform(o.matrix_world)
            if len(bm.faces):
                _OBSTACLE_TREES.append(BVHTree.FromBMesh(bm))
            bm.free()
    return _OBSTACLE_TREES


def spring_arm(target: Vector, direction: Vector, want: float) -> float:
    """Distance the camera can sit from ``target`` along ``direction`` without being inside geometry."""
    best = want
    for tree in _obstacle_trees():
        hit, _, _, dist = tree.ray_cast(target, direction, want + 0.4)
        if hit is not None and dist is not None:
            best = min(best, max(0.8, dist - 0.35))
    return best


def build_lookdev(assets: Path, earth_azimuth: float):
    env.make_world_sky()
    env.make_sun(azimuth=15.0, elevation=9.0, strength=5.0)
    ground = env.make_ground()
    env.make_horizon()
    env.scatter_rocks(ground, count=2600, extent=44.0)
    tex = assets / "earth" / "world.topo.5400x2700.jpg"
    clouds = assets / "earth" / "clouds_2048.jpg"
    if tex.exists():
        env.make_earth(str(tex), str(clouds) if clouds.exists() else None, azimuth=earth_azimuth)
    env.make_lander(location=(-24.0, 30.0, 0.0))
    return ground


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rig", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--assets", default="/assets")
    p.add_argument("--frames", default="0")
    p.add_argument("--cameras", default="hero")
    p.add_argument("--res", default="1920x1080")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--device", default="GPU")
    p.add_argument("--save-blend", action="store_true")
    p.add_argument("--scene", default="")
    p.add_argument("--terrain", default="")
    p.add_argument("--export-visual", default="")
    p.add_argument("--frame-list", default="")
    a = p.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = tuple(int(x) for x in a.res.split("x"))
    device = setup_render(res, a.samples, a.device)
    t0 = time.time()

    rig = json.loads(Path(a.rig).read_text())
    poses = np.load(a.poses)
    names = [str(n) for n in poses["names"]]
    shots = {}
    if a.scene:
        from ascent.art.base import build_base
        scene = json.loads(Path(a.scene).read_text())
        terr = np.load(a.terrain)
        loose_names = {b["name"] for b in scene["loose"]}
        rig = {**rig, "bodies": [b for b in rig["bodies"] if b["name"] not in loose_names]}
        if rig.get("kind") == "cmu":
            from ascent.art.robot_cmu import add_headlamps_cmu, build_robot_cmu
            empties = build_robot_cmu(rig)
            add_headlamps_cmu(empties)
        else:
            empties = build_robot(rig)
            add_headlamps(empties)
        env.make_world_sky(fill=0.02)
        env.make_sun(azimuth=-128.0, elevation=10.0, strength=5.0, bounce=0.4)
        env.make_horizon()
        tex = Path(a.assets) / "earth" / "world.topo.5400x2700.jpg"
        clouds = Path(a.assets) / "earth" / "clouds_2048.jpg"
        if tex.exists():
            env.make_earth(str(tex), str(clouds) if clouds.exists() else None, azimuth=14.0)
        lx, ly = -9.0, 14.0
        vxs, vys, vz = terr["vista_xs"], terr["vista_ys"], terr["vista_z"]
        foot = [float(vz[np.argmin(abs(vys - (ly + dy))), np.argmin(abs(vxs - (lx + dx)))])
                for dx in (-2.6, 0, 2.6) for dy in (-2.6, 0, 2.6)]
        env.make_lander(location=(lx, ly, min(foot) + 0.02))
        empties.update(build_base(scene, terr["heights"],
                                  {"xs": terr["vista_xs"], "ys": terr["vista_ys"], "z": terr["vista_z"],
                                   **{k: terr[k] for k in terr.files if k.startswith("near_")}},
                                  str(Path(a.assets) / "repo" / "branding" / "hackphs-logo-2048.png")))
        shots = course_shots(scene)
        if rig.get("kind") == "cmu" and "thorax" in empties:
            try:                                             # HackPHS badge on the power pack
                from ascent.art.base import place_logo
                place_logo("robot_logo", 0.2, (0.0, 0.06, -0.2285), (0.0, math.pi, 0.0), "robot",
                           parent=empties["thorax"])
            except Exception as exc:  # noqa: BLE001
                print("robot logo skipped:", repr(exc)[:200], flush=True)
        if a.export_visual:
            from ascent.art.export_mj import export_visuals
            robot_names = {b["name"] for b in rig["bodies"]}
            set_pose(empties, names, poses["pos"][0], poses["quat"][0])
            bpy.context.view_layer.update()
            info = export_visuals(a.export_visual, {k: v for k, v in empties.items() if k in robot_names},
                                  {k: v for k, v in empties.items() if k in loose_names}, scene)
            print(f"exported {len(info['meshes'])} visual meshes", flush=True)
            return
    else:
        if rig.get("kind") == "cmu":
            from ascent.art.robot_cmu import build_robot_cmu
            empties = build_robot_cmu(rig)
        else:
            empties = build_robot(rig)
        build_lookdev(Path(a.assets), earth_azimuth=118.0)
    cam = add_camera("camera")
    build_s = time.time() - t0

    report = {"device": device, "build_seconds": round(build_s, 1), "renders": []}
    pelvis_i = names.index("pelvis") if "pelvis" in names else names.index("root")
    if a.frame_list:
        # video mode: every listed frame, one moving camera, numbered output
        frames = [int(x) for x in Path(a.frame_list).read_text().split()]
        kind = a.cameras.split(",")[0]
        low_ids = [names.index(n) for n in ("left_foot", "right_foot", "left_hand", "right_hand", "ltoes", "rtoes",
                                            "lfoot", "rfoot") if n in names]
        low = poses["pos"][:, low_ids, 2].min(axis=1) - 0.08
        path_fn = camera_path(kind, poses["pos"][:, pelvis_i], float(poses["fps"]), shots, low)
        for f in frames:
            set_pose(empties, names, poses["pos"][f], poses["quat"][f])
            eye, target, lens = path_fn(f)
            cam.data.lens = lens
            aim(cam, Vector(eye), Vector(target))
            bpy.context.scene.render.filepath = str(out / f"frame_{f:05d}.png")
            t = time.time()
            bpy.ops.render.render(write_still=True)
            report["renders"].append({"frame": f, "seconds": round(time.time() - t, 1)})
        (out / "blender_report.json").write_text(json.dumps(report, indent=2))
        return
    for f in [int(x) for x in a.frames.split(",")]:
        set_pose(empties, names, poses["pos"][f], poses["quat"][f])
        pelvis = Vector(poses["pos"][f][pelvis_i])
        for kind in a.cameras.split(","):
            if kind in shots:
                eye, target, lens = shots[kind]
                cam.data.lens = lens
                aim(cam, Vector(eye), Vector(target))
            else:
                place_camera(cam, kind, pelvis)
            path = out / f"{kind}_f{f:05d}.png"
            bpy.context.scene.render.filepath = str(path)
            t = time.time()
            bpy.ops.render.render(write_still=True)
            report["renders"].append({"file": path.name, "seconds": round(time.time() - t, 1)})
            print(f"rendered {path.name} in {time.time() - t:.1f}s", flush=True)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    (out / "blender_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
