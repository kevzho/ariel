"""Attach the Blender-exported visual meshes to a MuJoCo world.

Training recordings are rendered with MuJoCo from recorded states. With this
library attached they show the detailed robot and base instead of collision
primitives. Visual geoms carry no mass and never collide.

Geom groups:
    0  terrain heightfield (visible, used by sensors)
    1  visual meshes (visible)
    2  near-side habitat visuals (hidden by the side camera as a cutaway)
    3  robot collision primitives (hidden)
    4  world collision primitives once visuals are attached (hidden, used by sensors)
    5  robot visual meshes (shown by the renderer; kept apart so camera rays ignore the robot)
"""

from __future__ import annotations

import io
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from .course import Course, vista_grid

# MuJoCo stand-ins for the Blender materials (no node graphs in MuJoCo's renderer).
MATERIALS = {
    "panel_white": dict(rgba=(0.86, 0.86, 0.84, 1), specular=0.35, shininess=0.4),
    "foil_gold": dict(rgba=(0.86, 0.64, 0.26, 1), specular=0.9, shininess=0.8, reflectance=0.1),
    "anodized_black": dict(rgba=(0.07, 0.07, 0.08, 1), specular=0.5, shininess=0.6),
    "chrome_tube": dict(rgba=(0.8, 0.82, 0.85, 1), specular=1.0, shininess=1.0, reflectance=0.3),
    "visor": dict(rgba=(0.03, 0.03, 0.04, 1), specular=1.0, shininess=1.0, reflectance=0.4),
    "rubber": dict(rgba=(0.04, 0.04, 0.045, 1), specular=0.1, shininess=0.1),
    "accent_orange": dict(rgba=(0.95, 0.36, 0.06, 1), specular=0.3, shininess=0.4),
    "lamp": dict(rgba=(1, 0.98, 0.92, 1), emission=1.0),
    "lamp_cool": dict(rgba=(0.8, 0.9, 1, 1), emission=1.0),
    "beacon_glow": dict(rgba=(1, 0.45, 0.1, 1), emission=1.0),
    "beacon_red": dict(rgba=(1, 0.15, 0.05, 1), emission=1.0),
    "glass_dark": dict(rgba=(0.04, 0.05, 0.06, 1), specular=1.0, shininess=1.0, reflectance=0.3),
    "decal_black": dict(rgba=(0.03, 0.03, 0.03, 1), specular=0.2),
    "patch_navy": dict(rgba=(0.03, 0.06, 0.18, 1), specular=0.3),
    "badge_navy": dict(rgba=(0.02, 0.04, 0.11, 1), specular=0.4),
    "beta_cloth": dict(rgba=(0.83, 0.82, 0.78, 1), specular=0.15, shininess=0.2),
    "aluminum": dict(rgba=(0.74, 0.75, 0.77, 1), specular=0.8, shininess=0.6, reflectance=0.1),
    "solar_cells": dict(rgba=(0.05, 0.08, 0.2, 1), specular=0.9, shininess=0.9, reflectance=0.15),
    "sintered": dict(rgba=(0.2, 0.195, 0.19, 1), specular=0.25, shininess=0.3),
    "paint_white": dict(rgba=(0.84, 0.84, 0.8, 1), specular=0.2),
    "hazard_orange": dict(rgba=(0.92, 0.33, 0.04, 1), specular=0.3),
    "suit_fabric": dict(rgba=(0.9, 0.9, 0.87, 1), specular=0.15),
    "gold_visor": dict(rgba=(1.0, 0.76, 0.36, 1), specular=1.0, shininess=1.0, reflectance=0.5),
    "cable_orange": dict(rgba=(0.85, 0.3, 0.04, 1), specular=0.4),
    "rock": dict(rgba=(0.3, 0.29, 0.275, 1), specular=0.05, shininess=0.05),
    "terrain": dict(rgba=(0.36, 0.35, 0.33, 1), specular=0.05, shininess=0.05),
    "logo_decal": dict(texture="hackphs_badge", rgba=(1, 1, 1, 1), specular=0.3),
    "logo_painted": dict(texture="hackphs_pad", rgba=(1, 1, 1, 1), specular=0.2),
    "default": dict(rgba=(0.6, 0.6, 0.6, 1)),
}
TEXTURES = {"hackphs_badge": "mj-logo-badge.png", "hackphs_pad": "mj-logo-pad.png"}
HAB_EXPORT_RADIUS = 1.45


def _f(*v: float) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _quat_z(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _split_obj(data: bytes, keep) -> tuple[bytes, bytes]:
    """Split an OBJ into (kept, removed) triangles by a predicate on triangle centroids."""
    lines = data.decode().splitlines()
    v = [list(map(float, ln.split()[1:4])) for ln in lines if ln.startswith("v ")]
    vt = [ln for ln in lines if ln.startswith("vt ")]
    faces = [ln for ln in lines if ln.startswith("f ")]
    head = [ln for ln in lines if ln.startswith("v ")] + vt
    a, b = [], []
    for fl in faces:
        idx = [int(tok.split("/")[0]) - 1 for tok in fl.split()[1:]]
        c = np.mean([v[i] for i in idx], axis=0)
        (a if keep(c) else b).append(fl)
    return ("\n".join(head + a) + "\n").encode(), ("\n".join(head + b) + "\n").encode()


def vista_obj(course: Course, res: float = 0.5, half: float = 45.0) -> bytes:
    """A coarse background terrain mesh around the course (visual only), tucked under the heightfield."""
    xs, ys, z = vista_grid(course.params, res=res, x_range=(-half + 10, half + 10), y_range=(-half, half))
    x0, y0, x1, y1 = course.extent
    X, Y = np.meshgrid(xs, ys)
    inside = (X > x0 + 0.4) & (X < x1 - 0.4) & (Y > y0 + 0.4) & (Y < y1 - 0.4)
    z = np.where(inside, z - 0.3, z)
    nrow, ncol = z.shape
    out = io.StringIO()
    for r in range(nrow):
        for c in range(ncol):
            out.write(f"v {xs[c]:.3f} {ys[r]:.3f} {z[r, c]:.3f}\n")
    for r in range(nrow - 1):
        for c in range(ncol - 1):
            a = r * ncol + c + 1
            out.write(f"f {a} {a + 1} {a + ncol + 1}\nf {a} {a + ncol + 1} {a + ncol}\n")
    return out.getvalue().encode()


class VisualLibrary:
    def __init__(self, files: dict[str, bytes]):
        """``files``: exported OBJ meshes plus ``manifest.json`` and the logo PNGs."""
        self.manifest = json.loads(files["manifest.json"])
        self.files = dict(files)
        self.by_owner: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for m in self.manifest["meshes"]:
            self.by_owner[m["owner"]].append((m["mesh"], m["material"]))
        self.assets: dict[str, bytes] = {}

    # ------------------------------------------------------------------ helpers
    def _mesh_asset(self, asset: ET.Element, file: str, scale=(1, 1, 1), data: bytes | None = None) -> str:
        key = f"{file[:-4]}@{_f(*scale).replace(' ', '_')}"
        if key not in self._declared:
            self._declared.add(key)
            self.assets[file] = data if data is not None else self.files[file]
            ET.SubElement(asset, "mesh", name=key, file=file, scale=_f(*scale), inertia="shell")
        return key

    def _geom(self, parent: ET.Element, mesh: str, material: str, group: int = 1, pos=None, quat=None) -> None:
        attrs = {"type": "mesh", "mesh": mesh, "material": "vis_" + (material if material in MATERIALS else "default"),
                 "group": str(group), "contype": "0", "conaffinity": "0", "density": "0"}
        if pos is not None:
            attrs["pos"] = _f(*pos)
        if quat is not None:
            attrs["quat"] = _f(*quat)
        ET.SubElement(parent, "geom", attrs)

    def _place(self, asset, parent, owner: str, pos=None, quat=None, scale=(1, 1, 1), group: int = 1) -> None:
        for file, mat in self.by_owner.get(owner, []):
            mesh = self._mesh_asset(asset, file, scale)
            self._geom(parent, mesh, mat, group, pos, quat)

    # ------------------------------------------------------------------ main
    def attach(self, root: ET.Element, course: Course | None) -> dict[str, bytes]:
        self._declared: set[str] = set()
        asset = root.find("asset")
        world = root.find("worldbody")
        for name, fname in TEXTURES.items():
            if fname in self.files:
                self.assets[fname] = self.files[fname]
                ET.SubElement(asset, "texture", name=name, type="2d", file=fname)
        for name, props in MATERIALS.items():
            attrs = {"name": "vis_" + name}
            for k, v in props.items():
                if k == "texture" and TEXTURES[v] not in self.files:
                    continue
                attrs[k] = _f(*v) if isinstance(v, tuple) else (v if isinstance(v, str) else f"{v:.6g}")
            ET.SubElement(asset, "material", attrs)

        bodies = {b.get("name"): b for b in root.iter("body")}
        # robot bodies
        for owner in self.by_owner:
            if owner.startswith("robot."):
                body = bodies.get(owner[len("robot."):])
                if body is not None:
                    self._place(asset, body, owner, group=5)
        if course is None:
            return self.assets

        # world collision primitives become sensor-only geometry
        for g in world.findall("geom"):
            if g.get("type") != "hfield" and g.get("name") != "ground":
                g.set("group", "4")
        for b in course.loose:
            body = bodies.get(b["name"])
            if body is None:
                continue
            for g in body.findall("geom"):
                g.set("group", "4")
            if b["kind"] == "rock":
                r = b["size"][0]
                self._place(asset, body, f"proto.rock{abs(int(b.get('seed', 0))) % 4}", scale=(r * 1.05, r, r))
            elif b["kind"] in ("crate", "panel"):
                self._place(asset, body, f"proto.{b['kind']}", scale=tuple(b["size"]))

        # structures, placed per this course's layout
        for d in course.decor:
            k = d["kind"]
            if k == "hab_module":
                s = d["radius"] / HAB_EXPORT_RADIUS
                self._place_hab(asset, world, d["pos"], (s, 1.0, s))
            elif k == "solar_tower":
                self._place(asset, world, "struct.solar_tower", d["pos"], _quat_z(d["yaw"]))
            elif k == "astronaut":
                self._place(asset, world, "struct.astronaut", d["pos"], _quat_z(d.get("yaw", math.pi)))
            elif k == "comms_mast":
                self._place(asset, world, "struct.comms_tower", d["pos"], _quat_z(d["yaw"]))
            elif k == "cable_reel":
                p = list(d["pos"])
                p[2] += 0.45
                self._place(asset, world, "struct.cable_reel", p, _quat_z(d["yaw"]))
            elif k == "lander":
                fr = self.manifest["frames"].get("lander")
                if fr:
                    self._place(asset, world, "struct.lander", fr["location"], fr["quaternion"])
            elif k == "landing_pad":
                self._place(asset, world, "struct.pad")
        for s in course.static:
            if s["kind"] == "boulder":
                r = s["size"][0]
                self._place(asset, world, f"proto.rock{abs(int(s.get('seed', 0))) % 4}", s["pos"],
                            scale=(r * 1.05, r, r * 1.02))
            elif s["kind"] == "solar_panel":
                self._place(asset, world, "proto.solar_panel", s["pos"], s["quat"], scale=tuple(s["size"]))
        # background terrain
        self.assets["vista.obj"] = vista_obj(course)
        ET.SubElement(asset, "mesh", name="vista", file="vista.obj", inertia="shell")
        self._geom(world, "vista", "terrain", 1)
        return self.assets

    def _place_hab(self, asset, world, pos, scale) -> None:
        """Habitat visuals split into a near-side half (group 2, cutaway) and the rest."""
        for file, mat in self.by_owner.get("struct.habitat", []):
            near, far = _split_obj(self.files[file], lambda c: c[1] < -1.2)
            for part, data, group in (("near", near, 2), ("far", far, 1)):
                fname = f"{file[:-4]}_{part}.obj"
                mesh = self._mesh_asset(asset, fname, scale, data)
                self._geom(world, mesh, mat, group, pos)


def load_library(directory) -> VisualLibrary:
    from pathlib import Path
    d = Path(directory)
    files = {p.name: p.read_bytes() for p in d.iterdir() if p.suffix in (".obj", ".json", ".png")}
    return VisualLibrary(files)
