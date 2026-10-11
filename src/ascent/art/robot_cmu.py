"""Rescue-robot shells for the telescoping CMU walker (the MoCapAct body). Runs inside Blender only.

Same look as the first ASCENT robot (white thermal panels, gold-foil joint housings, black
anodized structure, polished inner tubes, dark visor), built around the CMU humanoid's bodies.
Each physics body is an Empty; its parts are modelled in that body's frame. The CMU torso and
head frames have x to the robot's left, y up and z forward; the feet have -y forward and -z down.

Each telescoping limb is drawn as three nested stages: a white outer housing fixed to the parent
bone, a narrower white stage fixed to the moving joint, and a polished tube between them that is
only visible when the segment is extended.
"""

from __future__ import annotations

import bpy
from mathutils import Matrix, Quaternion, Vector

from .robot import (COLL, _actuator, _box, _collection, _cyl, _finish, _ring, _sleeve, _sphere, _tube,
                    _visor_band)

# sleeve radius and the joint housing at the top of each telescoping bone
LIMBS = {
    "femur": dict(radius=0.078, joint=("drum", 0.082, 0.125), accent=True),
    "tibia": dict(radius=0.060, joint=("drum", 0.064, 0.140), accent=False),
    "humerus": dict(radius=0.052, joint=("ball", 0.066, 0.0), accent=True),
    "radius": dict(radius=0.043, joint=("drum", 0.046, 0.094), accent=False),
}


def _empty(name: str, parent=None, location=(0, 0, 0), quat=None):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.04
    e.rotation_mode = "QUATERNION"
    _collection(COLL).objects.link(e)
    if parent is not None:
        e.parent = parent
    e.location = location
    if quat is not None:
        e.rotation_quaternion = quat
    return e


def _aligned(parent, origin, z_dir, name: str):
    """A child frame at ``origin`` whose +z axis points along ``z_dir`` (given in the parent frame)."""
    q = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector(z_dir).normalized())
    return _empty(name, parent, origin, q)


def _geom_frame(parent, geom: dict, name: str):
    return _empty(name, parent, geom["pos"], Quaternion(geom["quat"]))


def _side(name: str) -> int:
    return 1 if name[0] == "l" else -1


def _limb(e, empties: dict, tel: dict, body: str) -> None:
    """Outer housing on the parent bone, inner stage and polished tube on the moving joint."""
    kind = body[1:]
    cfg = LIMBS[kind]
    side = _side(body)
    d = Vector(tel["axis"])
    L, lo, hi, r = tel["length"], tel["lo"], tel["hi"], cfg["radius"]
    A = _aligned(e, (0, 0, 0), -d, f"{body}_axis")            # +z back toward the bone's own joint
    _sleeve(A, top=(0, 0, -0.015), bottom=(0, 0, -(lo * L - 0.03)), radius=r, accent=cfg["accent"],
            stripe_side=side)
    if kind == "femur":                                       # hip fairing
        _sphere(A, "panel_white", (0, 0, -0.12), 0.090, scale=(1.0, 1.0, 1.6))
    if kind == "tibia":                                       # knee guard (the leg frames have +z forward)
        _box(e, "panel_white", (0, -0.005, 0.070), (0.052, 0.066, 0.012), bevel=0.012, segments=4)
        _box(e, "accent_orange", (0, -0.05, 0.083), (0.03, 0.006, 0.002), bevel=0.001)
    if kind == "humerus":                                     # shoulder cap
        _sphere(A, "panel_white", (0, 0, 0.012), 0.074, scale=(1.0, 1.0, 0.72))
    jk, jr, jw = cfg["joint"]
    if jk == "drum":
        _actuator(e, (0, 0, 0), jr, jw, axis="x", cap_side=side)
    else:
        _sphere(e, "foil_gold", (0, 0, 0), jr)
        _ring(A, "anodized_black", (0, 0, -0.035), jr * 0.86, 0.008)
    T = empties.get(f"tele_{tel['label']}")
    if T is None:
        return
    TA = _aligned(T, (0, 0, 0), -d, f"tele_{tel['label']}_axis")
    stage = (1.0 - lo) * L
    _tube(TA, top=(0, 0, (hi - lo) * L + 0.03), bottom=(0, 0, 0.02), radius=r * 0.60)
    _sleeve(TA, top=(0, 0, stage + 0.01), bottom=(0, 0, 0.03), radius=r * 0.80)


def _capsule_shell(e, geom: dict, mat: str, scale: float = 1.0, bevel: float = 0.006) -> None:
    g = _geom_frame(e, geom, geom["name"] + "_f")
    r, h = geom["size"][0] * scale, geom["size"][1]
    _cyl(g, mat, (0, 0, 0), r, 2 * h + r, bevel=bevel)


def _pelvis(e, body: dict) -> None:
    _box(e, "panel_white", (0, -0.06, 0.0), (0.172, 0.088, 0.108), bevel=0.04, segments=5)
    _box(e, "anodized_black", (0, 0.035, 0.0), (0.150, 0.022, 0.098), bevel=0.012)
    _box(e, "foil_gold", (0, -0.05, -0.108), (0.105, 0.055, 0.008), bevel=0.004)
    _box(e, "accent_orange", (0, -0.03, 0.109), (0.045, 0.012, 0.004), bevel=0.002)


def _lowerback(e, body: dict) -> None:
    # flexible waist: dark bellows with clamp rings
    _cyl(e, "rubber", (0, 0.068, 0.008), 0.108, 0.125, axis="y", bevel=0.012)
    for y in (0.02, 0.068, 0.116):
        _ring(e, "anodized_black", (0, y, 0.008), 0.11, 0.007, axis="y")


def _upperback(e, body: dict) -> None:
    _box(e, "panel_white", (0, 0.068, 0.016), (0.160, 0.072, 0.116), bevel=0.04, segments=5)
    _box(e, "anodized_black", (0, 0.068, 0.133), (0.085, 0.045, 0.004), bevel=0.003)


def _thorax(e, body: dict) -> None:
    _box(e, "panel_white", (0, 0.078, 0.018), (0.186, 0.090, 0.128), bevel=0.05, segments=5)
    # chest instrument panel and light bars
    _box(e, "anodized_black", (0, 0.095, 0.147), (0.105, 0.040, 0.004), bevel=0.003)
    for s in (1, -1):
        _box(e, "lamp_cool", (s * 0.06, 0.118, 0.151), (0.028, 0.006, 0.002), bevel=0.001)
    _box(e, "accent_orange", (0, 0.045, 0.147), (0.07, 0.006, 0.003), bevel=0.002)
    # power pack on the back with a gold-foil radiator
    _box(e, "panel_white", (0, 0.06, -0.155), (0.145, 0.135, 0.062), bevel=0.03, segments=4)
    _box(e, "foil_gold", (0, 0.06, -0.221), (0.115, 0.10, 0.006), bevel=0.003)
    for s in (1, -1):
        _cyl(e, "chrome_tube", (s * 0.10, 0.20, -0.15), 0.012, 0.05, axis="y")


def _neck(e, body: dict) -> None:
    lower = body["name"] == "lowerneck"
    _cyl(e, "anodized_black" if lower else "rubber", (0, 0.054, 0.0), 0.078 if lower else 0.055, 0.075 if lower else 0.10,
         axis="y", bevel=0.008)


def _head(e, body: dict) -> None:
    # a frame with x forward, y left, z up (the head body has x left, y up, z forward)
    # the CMU head body rests pitched forward; the helmet is levelled by tipping it back about its left axis
    level = Quaternion((0.0, 1.0, 0.0), -0.33)
    H = _empty("head_frame", e, (0, 0.078, 0.012),
               Matrix(((0, 1, 0), (0, 0, 1), (1, 0, 0))).to_quaternion() @ level)
    # rounded sensor helmet, narrower at the jaw (same construction as the first ASCENT robot)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    o = bpy.context.active_object
    o.scale = (0.118, 0.104, 0.112)
    bpy.ops.object.transform_apply(scale=True)
    for v in o.data.vertices:
        if v.co.z < 0:
            v.co.y *= 0.84
            v.co.x *= 0.92
    helmet = _finish(o, H, "panel_white", bevel=0.04, segments=4, subsurf=2)
    _visor_band(H, helmet, z_offset=0.014, height=0.080, span_deg=82)
    for s in (1, -1):
        _box(H, "lamp", (0.110, s * 0.052, -0.052), (0.003, 0.022, 0.004), bevel=0.0015)
        _cyl(H, "anodized_black", (0.0, s * 0.106, 0.0), 0.036, 0.016, axis="y", bevel=0.004)
    # lidar puck on top
    _cyl(H, "anodized_black", (0, 0, 0.118), 0.046, 0.022, bevel=0.004)
    _cyl(H, "glass_dark", (0, 0, 0.136), 0.040, 0.018, verts=64)
    _cyl(H, "anodized_black", (0, 0, 0.148), 0.042, 0.008, bevel=0.003)


def _clavicle(e, body: dict) -> None:
    g = next((g for g in body["geoms"] if g["type"] == "capsule"), None)
    p = Vector(g["pos"]) if g else Vector((0.11 * _side(body["name"]), 0.046, 0.006))
    _box(e, "panel_white", p + Vector((0, 0.012, 0)), (0.100, 0.066, 0.094), bevel=0.035, segments=5)
    _box(e, "accent_orange", p + Vector((0, 0.076, 0)), (0.05, 0.003, 0.012), bevel=0.002)


def _foot(e, body: dict) -> None:
    s = _side(body["name"])
    _actuator(e, (0, 0, 0), 0.040, 0.092, axis="x", cap_side=s)
    _box(e, "panel_white", (0, -0.078, -0.004), (0.060, 0.112, 0.033), bevel=0.022, segments=4)
    _box(e, "rubber", (0, -0.078, -0.040), (0.064, 0.118, 0.011), bevel=0.006)
    _box(e, "accent_orange", (0, 0.036, -0.004), (0.035, 0.004, 0.012), bevel=0.002)


def _toes(e, body: dict) -> None:
    _box(e, "panel_white", (0, -0.014, -0.008), (0.060, 0.040, 0.027), bevel=0.018, segments=4)
    _box(e, "rubber", (0, -0.014, -0.036), (0.063, 0.044, 0.010), bevel=0.005)


def _wrist(e, body: dict) -> None:
    _sphere(e, "foil_gold", (0, 0, 0), 0.036)
    for g in body["geoms"]:
        _capsule_shell(e, g, "anodized_black", scale=1.25)


def _hand(e, body: dict) -> None:
    # mitt: a padded palm (the hand body extends along -y)
    _box(e, "rubber", (0, -0.035, 0.0), (0.046, 0.050, 0.026), bevel=0.02, segments=4)
    _box(e, "panel_white", (0, -0.028, 0.0), (0.049, 0.030, 0.029), bevel=0.012, segments=3)


def _digits(e, body: dict) -> None:
    if body["name"][1:] == "fingers":
        _box(e, "rubber", (0, -0.040, 0.0), (0.042, 0.045, 0.020), bevel=0.018, segments=4)
    else:
        for g in body["geoms"]:
            if g["type"] == "capsule":
                _capsule_shell(e, g, "rubber", scale=1.6, bevel=0.004)


BUILDERS = {"root": _pelvis, "lowerback": _lowerback, "upperback": _upperback, "thorax": _thorax,
            "lowerneck": _neck, "upperneck": _neck, "head": _head}
SIDED = {"clavicle": _clavicle, "foot": _foot, "toes": _toes, "wrist": _wrist, "hand": _hand,
         "fingers": _digits, "thumb": _digits}
SKIP = {"face", "lhipjoint", "rhipjoint"}


def build_robot_cmu(rig: dict) -> dict[str, bpy.types.Object]:
    """Create one Empty per body with its shells. Returns body name -> Empty."""
    empties = {b["name"]: _empty(b["name"]) for b in rig["bodies"]}
    limb_of = {t["parent"]: t for t in rig.get("telescopes", []) if t.get("parent_geom")}
    for body in rig["bodies"]:
        name = body["name"]
        e = empties[name]
        if name in SKIP or name.startswith("tele_"):
            continue
        if name in limb_of and name[1:] in LIMBS:
            _limb(e, empties, limb_of[name], name)
        elif name in BUILDERS:
            BUILDERS[name](e, body)
        elif name[0] in "lr" and name[1:] in SIDED:
            SIDED[name[1:]](e, body)
        else:
            for g in body["geoms"]:
                if g["type"] == "capsule":
                    _capsule_shell(e, g, "panel_white")
    # torso telescope: a polished spine section under the chest, visible when the torso extends
    spine = empties.get("tele_torso")
    if spine is not None:
        _cyl(spine, "chrome_tube", (0, -0.035, 0.008), 0.062, 0.15, axis="y", bevel=0.004)
        _ring(spine, "anodized_black", (0, 0.03, 0.008), 0.068, 0.008, axis="y")
    return empties


def add_headlamps_cmu(empties: dict) -> None:
    """Two spotlights on the chest, aimed ahead and slightly down."""
    import math
    chest = empties.get("thorax")
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
        o.location = (s * 0.06, 0.118, 0.158)
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = Vector((0.0, -0.45, 1.0)).to_track_quat("-Z", "Y")


__all__ = ["build_robot_cmu", "add_headlamps_cmu"]
