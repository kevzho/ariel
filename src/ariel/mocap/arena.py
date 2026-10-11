"""The Ariel lunar course as a dm_control arena.

Uses the same seeded generator as the MuJoCo 3 stack (``ariel.world.course``), so terrain,
habitat, fissure, scarp, tower, boulders and loose debris match the original layouts. Element
counts are fixed at the largest a layout can use; unused elements are parked off the course.
Composer recompiles the model on every reset, so a new layout takes effect at the next episode.
"""

from __future__ import annotations

import numpy as np
from dm_control import composer
from dm_control.mujoco.wrapper import mjbindings

from ..world.course import HFIELD_RES, Course, CourseParams, generate_course

mjlib = mjbindings.mjlib

LUNAR_GRAVITY = 1.62
EARTH_GRAVITY = 9.81          # Guhan, Oct 11: train at Earth gravity (the MoCapAct prior falls at 1.62)
MAX_LOOSE = {"crate": 6, "panel": 3, "rock": 8}
STATIC_NAMES = ([f"hab_shell_{i}" for i in range(10)] + ["hab_deck", "solar_mast", "solar_panel_0", "solar_panel_1"]
                + [f"talus_{k}" for k in range(6)] + [f"boulder_{k}" for k in range(11)]
                + [f"climb_rock_{k}" for k in range(16)]
                + ["rest_boulder", "astronaut"])
_STATIC_MAT = {"hab": "hab_white", "mast": "metal_grey", "solar_panel": "solar_cells", "boulder": "rock",
               "astronaut": "suit_white"}
_LOOSE_MAT = {"crate": "crate_alu", "panel": "metal_grey", "rock": "rock"}
PARK = np.array([-30.0, 0.0, -20.0])


def _f(v):
    return np.asarray(v, dtype=float)


class LunarCourse(composer.Arena):
    """Heightfield terrain plus the course structures (Earth gravity by default)."""

    def _build(self, course: Course, name: str = "course", gravity: float = EARTH_GRAVITY,
               njmax: int = 8000, nconmax: int = 3000, loose_static: bool = False):
        super()._build(name=name)
        self._course = course
        self._loose_static = loose_static
        root = self._mjcf_root
        root.option.gravity = (0.0, 0.0, -gravity)
        # MuJoCo 2.2.2 preallocates contacts; heightfield contacts with debris overflow the default
        root.size.njmax = njmax
        root.size.nconmax = nconmax
        root.visual.headlight.set_attributes(ambient=[0.22, 0.23, 0.25], diffuse=[0, 0, 0], specular=[0, 0, 0])
        root.visual.quality.shadowsize = 4096
        root.visual.map.znear = 0.005
        root.visual.map.zfar = 80.0
        getattr(root.visual, "global").offwidth = 1920
        getattr(root.visual, "global").offheight = 1080
        root.default.geom.set_attributes(friction=(0.8, 0.01, 0.001), solref=(0.008, 1), solimp=(0.95, 0.99, 0.001))

        a = root.asset
        a.add("texture", name="sky", type="skybox", builtin="gradient", rgb1=(0.01, 0.012, 0.02), rgb2=(0, 0, 0),
              width=512, height=512, mark="random", random=0.0012, markrgb=(0.9, 0.92, 1))
        tex = a.add("texture", name="regolith", type="2d", builtin="flat", width=512, height=512,
                    rgb1=(0.46, 0.45, 0.43), rgb2=(0.4, 0.39, 0.37), mark="random", random=0.35,
                    markrgb=(0.33, 0.32, 0.31))
        self._mats = {"regolith": a.add("material", name="regolith", texture=tex, texrepeat=(40, 40),
                                        specular=0.05, shininess=0.05)}
        for mname, rgba in {"hab_white": (0.85, 0.85, 0.83, 1), "metal_grey": (0.55, 0.56, 0.58, 1),
                            "solar_cells": (0.06, 0.09, 0.2, 1), "rock": (0.3, 0.29, 0.28, 1),
                            "suit_white": (0.9, 0.9, 0.88, 1), "crate_alu": (0.7, 0.71, 0.72, 1)}.items():
            self._mats[mname] = a.add("material", name=mname, rgba=rgba, specular=0.2, shininess=0.2)

        w = root.worldbody
        w.add("light", name="sun", directional=True, castshadow=True, dir=(0.606, 0.776, -0.174),
              diffuse=(1.25, 1.22, 1.15), specular=(0.6, 0.6, 0.6), pos=(0, 0, 30))
        w.add("light", name="earthshine", directional=True, castshadow=False, dir=(0.4, 0.5, -0.6),
              diffuse=(0.1, 0.12, 0.16), specular=(0, 0, 0))
        self._hab_lights = [w.add("light", name=f"hab_worklight_{k}", dir=(0, 0, -1), directional=False,
                                  castshadow=False, cutoff=75, exponent=1, attenuation=(1, 0.1, 0.05),
                                  diffuse=(0.55, 0.57, 0.6), specular=(0, 0, 0)) for k in range(2)]
        self._fissure_light = w.add("light", name="fissure_light", dir=(0, 0, -1), directional=False,
                                    castshadow=False, cutoff=80, exponent=1, attenuation=(1, 0.1, 0.04),
                                    diffuse=(0.45, 0.4, 0.32), specular=(0, 0, 0))

        nrow, ncol = course.heights.shape
        x0, y0, x1, y1 = course.extent
        self._hfield = a.add("hfield", name="terrain", nrow=nrow, ncol=ncol, size=((x1 - x0) / 2, (y1 - y0) / 2, 1.0, 0.5))
        self._terrain = w.add("geom", name="terrain", type="hfield", hfield=self._hfield,
                              material=self._mats["regolith"], pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.0))
        self._park = w.add("geom", name="parking", type="box", size=(4, 4, 0.5), pos=PARK - (0, 0, 0.5),
                           rgba=(0, 0, 0, 0), group=3)
        self._static = {}
        for n in STATIC_NAMES:
            self._static[n] = w.add("geom", name=n, type="box", size=(0.1, 0.1, 0.1), pos=PARK + (0, 0, 5))
        self._loose = {}
        for kind, cnt in MAX_LOOSE.items():
            for k in range(cnt):
                n = f"{kind}_{k}"
                b = w.add("body", name=n, pos=PARK)
                if not loose_static:
                    b.add("freejoint", name=n + "_free")
                extra = dict(condim=6, friction=(0.8, 0.03, 0.01)) if kind == "rock" else dict(friction=(0.7, 0.01, 0.001))
                g = b.add("geom", name=n, type="sphere" if kind == "rock" else "box", size=(0.1, 0.1, 0.1),
                          mass=1.0, material=self._mats[_LOOSE_MAT[kind]], **extra)
                self._loose[n] = (b, g)
        self.set_course(course)

    # ------------------------------------------------------------------ layout
    @property
    def course(self) -> Course:
        return self._course

    def set_course(self, course: Course) -> None:
        """Move every element to a new layout (takes effect at the next compile)."""
        self._course = course
        P = course.params
        for k, dx in enumerate((-0.7, 0.7)):
            self._hab_lights[k].pos = (P.hab_x + dx, 0, P.hab_clearance - 0.04)
        self._fissure_light.pos = (P.fissure_x + P.fissure_width / 2, 0, 0.6)
        x0, y0, x1, y1 = course.extent
        data, lo, span = course.hfield_data()
        self._hfield.size = ((x1 - x0) / 2, (y1 - y0) / 2, span, 0.5)
        self._terrain.pos = ((x0 + x1) / 2, (y0 + y1) / 2, lo)
        self._hdata = data
        used = set()
        for s in course.static:
            g = self._static.get(s["name"])
            if g is None:
                continue
            used.add(s["name"])
            g.type, g.size, g.pos = s["type"], _f(s["size"]), _f(s["pos"])
            g.quat = _f(s.get("quat", (1, 0, 0, 0)))
            g.material = self._mats[_STATIC_MAT.get(s["kind"], "rock")]
        for i, (n, g) in enumerate(self._static.items()):
            if n not in used:
                g.type, g.size, g.pos, g.quat = "sphere", (0.05, 0, 0), PARK + (i * 0.2 - 3, 3.5, 0.06), (1, 0, 0, 0)
        used = set()
        for b in course.loose:
            if b["name"] not in self._loose:
                continue
            body, g = self._loose[b["name"]]
            used.add(b["name"])
            body.pos, body.quat = _f(b["pos"]) - (0, 0, 0.03 if self._loose_static else 0.0), _f(b.get("quat", (1, 0, 0, 0)))
            g.size, g.mass = _f(b["size"]), float(b["mass"])
        for i, (n, (body, g)) in enumerate(self._loose.items()):
            if n not in used:
                body.pos, body.quat = PARK + (i * 0.4 - 3.5, -2.5, 0.2), (1, 0, 0, 0)
                g.size, g.mass = (0.08, 0.08, 0.08), 1.0

    def initialize_episode(self, physics, random_state):
        res = physics.bind(self._hfield)
        adr, n = int(res.adr), int(res.nrow) * int(res.ncol)
        if getattr(self, "_hfield_model", None) == id(physics.model.ptr) and not physics.contexts:
            return                           # same compiled model as last episode: heights are already in place
        self._hfield_model = id(physics.model.ptr)
        physics.model.hfield_data[adr:adr + n] = self._hdata.ravel()
        if physics.contexts:
            with physics.contexts.gl.make_current() as ctx:
                ctx.call(mjlib.mjr_uploadHField, physics.model.ptr, physics.contexts.mujoco.ptr,
                         physics.bind(self._hfield).element_id)

    @property
    def ground_geoms(self):
        return (self._terrain,) + tuple(self._static.values()) + tuple(g for _, g in self._loose.values())

    @property
    def terrain_geom(self):
        return self._terrain

    def height_at(self, x: float, y: float) -> float:
        return self._course.height_at(x, y)


def course_pool(n: int, seed: int, showcase_first: bool = False) -> list[Course]:
    """Random training layouts (the generator's own distribution)."""
    from ..world.course import random_params
    rng = np.random.default_rng(seed)
    pool = [generate_course(CourseParams())] if showcase_first else []
    while len(pool) < n:
        pool.append(generate_course(random_params(rng)))
    return pool


__all__ = ["LunarCourse", "course_pool", "EARTH_GRAVITY", "LUNAR_GRAVITY", "HFIELD_RES"]
