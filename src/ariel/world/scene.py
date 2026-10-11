"""Assemble a complete MuJoCo model: physics settings, lighting, sky, course and robot."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from ..robot.humanoid import RobotSpec, _f, robot_xml_parts
from .course import Course

LUNAR_GRAVITY = 1.62
HFIELD_FILE = "course_hfield.bin"


@dataclass(frozen=True)
class PhysicsSpec:
    gravity: float = LUNAR_GRAVITY
    timestep: float = 0.004
    regolith_friction: float = 0.8
    sun_direction: tuple = (0.606, 0.776, -0.174)  # direction sunlight travels (sun low, az -128 deg)


def hfield_bytes(course: Course) -> bytes:
    data, _, _ = course.hfield_data()
    nrow, ncol = data.shape
    return struct.pack("<ii", nrow, ncol) + data.astype("<f4").tobytes()


def _course_xml(course: Course, asset: ET.Element, world: ET.Element) -> None:
    # emergency work lights under the collapsed module and down in the fissure
    P = course.params
    zc = P.hab_clearance + P.hab_radius
    for k, dx in enumerate((-0.7, 0.7)):
        ET.SubElement(world, "light", name=f"hab_worklight_{k}", pos=_f(P.hab_x + dx, 0, P.hab_clearance - 0.04),
                      dir="0 0 -1", directional="false", castshadow="false", cutoff="75", exponent="1",
                      attenuation="1 0.1 0.05", diffuse="0.55 0.57 0.6", specular="0 0 0")
    ET.SubElement(world, "light", name="fissure_light", pos=_f(P.fissure_x + P.fissure_width / 2, 0, 0.6),
                  dir="0 0 -1", directional="false", castshadow="false", cutoff="80", exponent="1",
                  attenuation="1 0.1 0.04", diffuse="0.45 0.4 0.32", specular="0 0 0")
    del zc
    x0, y0, x1, y1 = course.extent
    _, lo, span = course.hfield_data()
    ET.SubElement(asset, "hfield", name="terrain", file=HFIELD_FILE,
                  size=_f((x1 - x0) / 2, (y1 - y0) / 2, span, 0.5))
    ET.SubElement(world, "geom", name="terrain", type="hfield", hfield="terrain", material="regolith",
                  pos=_f((x0 + x1) / 2, (y0 + y1) / 2, lo))
    mats = {"hab": "hab_white", "mast": "metal_grey", "solar_panel": "solar_cells", "boulder": "rock",
            "astronaut": "suit_white"}
    for s in course.static:
        attrs = {"name": s["name"], "type": s["type"], "size": _f(*s["size"]), "pos": _f(*s["pos"]),
                 "material": mats.get(s["kind"], "rock")}
        if "quat" in s:
            attrs["quat"] = _f(*s["quat"])
        if s["kind"] == "hab":
            attrs["group"] = "2"            # near-side cutaway can hide group 2 in side views
        ET.SubElement(world, "geom", attrs)
    loose_mats = {"crate": "crate_alu", "panel": "metal_grey", "rock": "rock"}
    for b in course.loose:
        body = ET.SubElement(world, "body", name=b["name"], pos=_f(*b["pos"]), quat=_f(*b.get("quat", (1, 0, 0, 0))))
        ET.SubElement(body, "freejoint", name=b["name"] + "_free")
        # rocks get rolling resistance (condim 6) so they come to rest like real fragments
        extra = {"condim": "6", "friction": "0.8 0.03 0.01"} if b["kind"] == "rock" else {"friction": "0.7 0.01 0.001"}
        ET.SubElement(body, "geom", name=b["name"], type=b["type"], size=_f(*b["size"]), mass=f"{b['mass']:.4g}",
                      material=loose_mats[b["kind"]], **extra)


def build_world_xml(robot: RobotSpec = RobotSpec(), physics: PhysicsSpec = PhysicsSpec(),
                    course: Course | None = None) -> tuple[str, dict]:
    """Return (xml, meta). With a course, MuJoCo needs ``assets()`` for the heightfield."""
    parts = robot_xml_parts(robot)
    m = ET.Element("mujoco", model=f"ariel-{robot.name}")
    ET.SubElement(m, "compiler", angle="radian", autolimits="true")
    ET.SubElement(m, "option", timestep=f"{physics.timestep}", gravity=_f(0, 0, -physics.gravity),
                  integrator="implicitfast", cone="pyramidal", iterations="15", ls_iterations="8")
    size = ET.SubElement(m, "size", memory="64M")
    del size

    visual = ET.SubElement(m, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080", fovy="40")
    ET.SubElement(visual, "quality", shadowsize="8192", offsamples="8")
    # clip planes are fractions of the model extent, which the background terrain makes large
    ET.SubElement(visual, "map", znear="0.002", zfar="60", shadowclip="10", shadowscale="0.6", haze="0.0")
    ET.SubElement(visual, "headlight", ambient="0.22 0.23 0.25", diffuse="0 0 0", specular="0 0 0")

    default = ET.SubElement(m, "default")
    ET.SubElement(default, "geom", condim="3", friction=_f(physics.regolith_friction, 0.01, 0.001),
                  solref="0.008 1", solimp="0.95 0.99 0.001")

    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.01 0.012 0.02", rgb2="0 0 0", width="1024", height="1024",
                  mark="random", random="0.0012", markrgb="0.9 0.92 1")
    ET.SubElement(asset, "texture", name="regolith", type="2d", builtin="flat", width="1024",
                  height="1024", rgb1="0.46 0.45 0.43", rgb2="0.4 0.39 0.37", mark="random",
                  random="0.35", markrgb="0.33 0.32 0.31")
    ET.SubElement(asset, "material", name="regolith", texture="regolith", texrepeat="40 40",
                  texuniform="false", specular="0.05", shininess="0.05")
    for name, rgba in {"hab_white": "0.85 0.85 0.83 1", "metal_grey": "0.55 0.56 0.58 1",
                       "solar_cells": "0.06 0.09 0.2 1", "rock": "0.3 0.29 0.28 1",
                       "suit_white": "0.9 0.9 0.88 1", "crate_alu": "0.7 0.71 0.72 1"}.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba, specular="0.2", shininess="0.2")
    for child in parts["asset"]:
        asset.append(child)

    world = ET.SubElement(m, "worldbody")
    ET.SubElement(world, "light", name="sun", directional="true", castshadow="true",
                  dir=_f(*physics.sun_direction), diffuse="1.25 1.22 1.15", specular="0.6 0.6 0.6",
                  pos="0 0 30")
    ET.SubElement(world, "light", name="earthshine", directional="true", castshadow="false",
                  dir=_f(0.4, 0.5, -0.6), diffuse="0.10 0.12 0.16", specular="0 0 0")
    # The robot comes first so its state occupies the start of qpos/qvel in every course.
    body = parts["body"]
    if course is not None:
        sx, sy = course.start
        z0 = course.height_at(sx, sy) + robot.pelvis_height + 0.005
        body.set("pos", _f(sx, sy, z0))
    world.append(body)
    if course is None:
        ET.SubElement(world, "geom", name="ground", type="plane", size="40 40 0.1", material="regolith")
    else:
        _course_xml(course, asset, world)

    m.append(parts["actuator"])
    m.append(parts["contact"])
    m.append(parts["sensor"])
    ET.indent(m)
    meta = parts["meta"]
    meta["loose"] = [b["name"] for b in course.loose] if course is not None else []
    return ET.tostring(m, encoding="unicode"), meta


def model_assets(course: Course | None) -> dict[str, bytes]:
    return {HFIELD_FILE: hfield_bytes(course)} if course is not None else {}


__all__ = ["build_world_xml", "model_assets", "PhysicsSpec", "LUNAR_GRAVITY", "np"]
