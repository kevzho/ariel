"""Render recorded states with MuJoCo's renderer (no physics is re-run).

A trace is a sequence of ``qpos`` vectors. Each frame sets the state, runs
forward kinematics only, and draws it from a tracking camera.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

CAMERAS = {
    # side-on tracking: every training recording, consistent across episodes
    "side": dict(azimuth=90.0, elevation=-6.0, distance=5.2, height=0.75, drift=0.0),
    # cinematic 3/4 chase with a slow drift
    "cinematic": dict(azimuth=148.0, elevation=-8.0, distance=4.2, height=None, drift=6.0),
}


def model_with_visuals(trace, library=None):
    """Compile the recorded model, optionally with the detailed visual meshes attached."""
    import json
    import xml.etree.ElementTree as ET

    from ..world.scene import HFIELD_FILE
    assets = {HFIELD_FILE: trace["hfield"].tobytes()} if "hfield" in trace.files else {}
    xml = str(trace["xml"])
    if library is not None and "scene" in trace.files:
        from ..world.course import course_from_scene
        root = ET.fromstring(xml)
        course = course_from_scene(json.loads(str(trace["scene"])))
        assets.update(library.attach(root, course))
        xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml, assets)


class TraceRenderer:
    def __init__(self, model: mujoco.MjModel, width: int = 1280, height: int = 720,
                 track_body: str = "pelvis"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.track = model.body(track_body).id
        self.opt = mujoco.MjvOption()
        has_visual = bool(np.any(model.geom_group == 5))
        self.opt.geomgroup[3] = 0 if has_visual else 1      # robot collision shapes
        self.opt.geomgroup[5] = 1                           # robot visual meshes
        self.feet = [model.body(n).id for n in ("left_foot", "right_foot", "left_hand", "right_hand")]
        # camera rays see terrain, world visuals and world collision shapes, never the robot
        self.ray_groups = np.array([1, 1, 1, 0, 1, 0], np.uint8)
        self.gid = np.array([-1], np.int32)

    def _clear(self, origin: np.ndarray, az: float, el: float, want: float) -> float:
        a, e = math.radians(az), math.radians(el)
        back = -np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        start = origin + back * 0.45                        # skip the robot's own volume
        hit = mujoco.mj_ray(self.model, self.data, start, back, self.ray_groups, 1, -1, self.gid)
        if hit < 0 or hit + 0.45 > want + 0.3:
            return want
        return max(0.6, hit + 0.45 - 0.35)

    def frames(self, qpos: np.ndarray, fps: float, camera: str = "side"):
        cfg = CAMERAS[camera]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.elevation = cfg["elevation"]
        smooth, arm, az_s = None, None, None
        for i, q in enumerate(qpos):
            self.data.qpos[:] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_camlight(self.model, self.data)
            pel = self.data.xpos[self.track].copy()
            low = min(self.data.xpos[b][2] for b in self.feet) - 0.08     # roughly where it stands
            body_h = max(0.4, pel[2] - low)
            target = np.array([pel[0], 0.0 if camera == "side" else pel[1], low + 0.85 * body_h])
            smooth = target if smooth is None else smooth + (target - smooth) * (1 - math.exp(-3.0 / fps))
            cam.lookat[:] = smooth
            t = i / fps
            want = cfg["distance"] * max(1.0, body_h / 0.89)
            az = cfg["azimuth"] + cfg["drift"] * math.sin(0.25 * t)
            if camera != "side":
                # spring arm with a clear-angle search: swing to the nearest unobstructed view
                best = None
                for dz in (0, 20, -20, 40, -40, 65, -65, 95, -95):
                    d = self._clear(smooth, az + dz, cfg["elevation"], want)
                    score = d - 0.012 * abs(dz)
                    if best is None or score > best[0]:
                        best = (score, d, az + dz)
                _, want, az = best
            az_s = az if az_s is None else az_s + (az - az_s) * 0.06
            arm = want if arm is None else arm + (want - arm) * (0.5 if want < arm else 0.08)
            cam.azimuth = az_s
            cam.distance = arm
            # side views cut away the near half of the habitat so the crawlspace stays visible
            self.opt.geomgroup[2] = 0 if camera == "side" else 1
            self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
            yield self.renderer.render()


def write_video(frames, path: str | Path, fps: float) -> int:
    import imageio.v2 as imageio
    n = 0
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=8, pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
            n += 1
    return n
