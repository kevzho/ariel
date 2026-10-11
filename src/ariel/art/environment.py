"""Lunar south-pole environment for Blender: ground, rocks, sky, sun, Earth, lander.

Runs inside Blender only. Distances in metres. Sun elevation is low, as at the
lunar south pole, and Earth always sits near the horizon there.
"""

from __future__ import annotations

import math
import random

import bpy
import numpy as np
from mathutils import Vector, noise

from .materials import _Graph, material


def _link(obj, coll_name="environment"):
    coll = bpy.data.collections.get(coll_name)
    if coll is None:
        coll = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def direction(azimuth_deg: float, elevation_deg: float) -> Vector:
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def crater_field(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, extent: float,
                 keep_clear: list[tuple[float, float, float]] = ()) -> np.ndarray:
    """Height of overlapping simple craters with raised rims (power-law sizes)."""
    z = np.zeros_like(x)
    n = int(extent * extent / 60)
    radii = np.clip(0.4 * (rng.pareto(1.6, n) + 1.0), 0.4, 9.0)
    cx = rng.uniform(-extent / 2, extent / 2, n)
    cy = rng.uniform(-extent / 2, extent / 2, n)
    for r, a, b in zip(radii, cx, cy):
        if any(math.hypot(a - kx, b - ky) < kr + r for kx, ky, kr in keep_clear):
            continue
        d = np.hypot(x - a, y - b) / r
        depth = 0.18 * r
        bowl = np.where(d < 1.0, -depth * (1 - d ** 2), 0.0)
        rim = 0.06 * r * np.exp(-((d - 1.0) / 0.25) ** 2)
        z += bowl + rim
    return z


def make_ground(size: float = 120.0, res: int = 480, seed: int = 3, keep_clear=((0, 0, 3.0),),
                flat_radius: float = 2.5):
    rng = np.random.default_rng(seed)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=res, y_subdivisions=res, size=size)
    g = bpy.context.active_object
    g.name = "ground"
    co = np.zeros(len(g.data.vertices) * 3)
    g.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    x, y = co[:, 0], co[:, 1]
    z = crater_field(x, y, rng, size, keep_clear)
    # long undulations and fine roughness
    for amp, freq in ((0.35, 0.03), (0.08, 0.12), (0.02, 0.6)):
        ph = rng.uniform(0, 100, 2)
        z += amp * np.sin(freq * 2 * np.pi * x + ph[0]) * np.cos(freq * 2 * np.pi * y * 0.8 + ph[1])
    r = np.hypot(x, y)
    z *= np.clip((r - flat_radius) / 3.0, 0.0, 1.0) ** 2 * 0.85 + 0.15
    co[:, 2] = z
    g.data.vertices.foreach_set("co", co.ravel())
    g.data.update()
    g.data.polygons.foreach_set("use_smooth", [True] * len(g.data.polygons))
    g.data.materials.append(material("regolith"))
    return _link(g)


def make_horizon(seed: int = 5):
    """Distant massifs and crater rims ringing the scene."""
    rng = np.random.default_rng(seed)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=260, y_subdivisions=260, size=6000)
    h = bpy.context.active_object
    h.name = "horizon"
    co = np.zeros(len(h.data.vertices) * 3)
    h.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    x, y = co[:, 0], co[:, 1]
    r = np.hypot(x, y)
    ph = rng.uniform(0, 100, 8)
    ridge = (np.sin(x / 310 + ph[0]) * np.cos(y / 270 + ph[1]) + 0.5 * np.sin(x / 120 + ph[2]) * np.sin(y / 150 + ph[3])
             + 0.25 * np.sin(x / 45 + ph[4]) * np.cos(y / 60 + ph[5]))
    z = np.clip((r - 150) / 900, 0, 1) ** 1.5 * (45 + 35 * ridge)
    co[:, 2] = z - 0.6
    h.data.vertices.foreach_set("co", co.ravel())
    h.data.update()
    h.data.polygons.foreach_set("use_smooth", [True] * len(h.data.polygons))
    h.data.materials.append(material("regolith"))
    return _link(h)


def _rock_mesh(seed: int, name: str):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
    o = bpy.context.active_object
    o.name = name
    rnd = random.Random(seed)
    off = Vector((rnd.uniform(0, 100), rnd.uniform(0, 100), rnd.uniform(0, 100)))
    squash = rnd.uniform(0.45, 0.8)
    for v in o.data.vertices:
        p = v.co.copy()
        d = 1.0 + 0.28 * noise.noise(p * 1.3 + off) + 0.08 * noise.noise(p * 4.0 + off)
        # facets: angular breaks like fractured basalt
        d += 0.06 * abs(noise.noise(p * 2.2 + off * 2))
        v.co = p * d
        v.co.z *= squash
        if v.co.z < -0.25:
            v.co.z = -0.25 - (v.co.z + 0.25) * 0.15
    o.data.polygons.foreach_set("use_smooth", [True] * len(o.data.polygons))
    o.data.materials.append(material("rock"))
    return o


def scatter_rocks(ground, count: int = 700, extent: float = 50.0, seed: int = 11,
                  keep_clear=((0, 0, 1.6),)):
    rng = np.random.default_rng(seed)
    protos = [_link(_rock_mesh(seed * 10 + i, f"rock_proto_{i}"), "rock_protos") for i in range(10)]
    for p in protos:
        p.hide_render = True
        p.hide_viewport = True
    depsgraph = bpy.context.evaluated_depsgraph_get()
    placed = 0
    tries = 0
    while placed < count and tries < count * 5:
        tries += 1
        # lunar rock sizes follow a steep power law: many small, few large
        s = float(np.clip(0.025 * (rng.pareto(2.2) + 1.0), 0.025, 0.9))
        x, y = rng.uniform(-extent / 2, extent / 2, 2)
        if any(math.hypot(x - kx, y - ky) < kr + s for kx, ky, kr in keep_clear):
            continue
        hit, loc, nrm, _, _, _ = bpy.context.scene.ray_cast(depsgraph, Vector((x, y, 50)), Vector((0, 0, -1)))
        if not hit:
            continue
        proto = protos[int(rng.integers(len(protos)))]
        o = bpy.data.objects.new(f"rock_{placed}", proto.data)
        o.location = loc - Vector((0, 0, s * rng.uniform(0.1, 0.35)))
        o.rotation_euler = (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, 2 * math.pi))
        o.scale = (s * rng.uniform(0.8, 1.3), s * rng.uniform(0.8, 1.3), s)
        _link(o, "rocks")
        placed += 1
    return placed


# ---------------------------------------------------------------------------
# Sky, lights and Earth
# ---------------------------------------------------------------------------

def make_world_sky(star_strength: float = 0.9, fill: float = 0.012):
    world = bpy.data.worlds.new("lunar_sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 900.0
    nt.links.new(tc.outputs["Generated"], vor.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = 0.035
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    nt.links.new(vor.outputs["Distance"], mr.inputs["Value"])
    # vary star brightness with the cell colour
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(vor.outputs["Color"], sep.inputs["Color"])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 6.0
    nt.links.new(sep.outputs["Red"], pw.inputs[0])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(mr.outputs["Result"], mul.inputs[0])
    nt.links.new(pw.outputs[0], mul.inputs[1])
    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    mul2.inputs[1].default_value = star_strength
    nt.links.new(mul.outputs[0], mul2.inputs[0])
    nt.links.new(mul2.outputs[0], bg.inputs["Strength"])
    bg.inputs["Color"].default_value = (0.9, 0.93, 1.0, 1.0)
    # light paths: stars are seen by the camera but do not light the scene
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    amb = nt.nodes.new("ShaderNodeBackground")
    amb.inputs["Color"].default_value = (0.55, 0.62, 0.8, 1.0)
    amb.inputs["Strength"].default_value = fill       # faint Earthshine/sky fill
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(amb.outputs["Background"], mix.inputs[1])
    nt.links.new(bg.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return world


def make_sun(azimuth: float = -35.0, elevation: float = 9.0, strength: float = 5.0, bounce: float = 0.25):
    """``azimuth``/``elevation`` give the direction *towards* the sun."""
    light = bpy.data.lights.new("sun", "SUN")
    light.energy = strength
    light.angle = math.radians(0.53)
    light.color = (1.0, 0.985, 0.96)
    o = bpy.data.objects.new("sun", light)
    to_sun = direction(azimuth, elevation)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = (-to_sun).to_track_quat("-Z", "Y")
    _link(o)
    # bounce from the sunlit regolith: a weak, shadowless upward fill
    bounce_light = bpy.data.lights.new("bounce", "SUN")
    bounce_light.energy = bounce
    bounce_light.color = (0.95, 0.93, 0.9)
    bounce_light.cycles.cast_shadow = False
    bo = bpy.data.objects.new("bounce", bounce_light)
    bo.rotation_mode = "QUATERNION"
    up_dir = direction(azimuth + 180, -35)
    bo.rotation_quaternion = (-up_dir).to_track_quat("-Z", "Y")
    _link(bo)
    # Earthshine: very weak bluish fill from Earth's direction
    es = bpy.data.lights.new("earthshine", "SUN")
    es.energy = 0.06
    es.color = (0.6, 0.72, 1.0)
    es.cycles.cast_shadow = False
    eo = bpy.data.objects.new("earthshine", es)
    eo.rotation_mode = "QUATERNION"
    eo.rotation_quaternion = (-direction(azimuth + 130, 6)).to_track_quat("-Z", "Y")
    _link(eo)
    return o


def make_earth(texture_path: str, clouds_path: str | None, azimuth: float, elevation: float = 8.0,
               distance: float = 3800.0):
    radius = distance * math.tan(math.radians(1.9 / 2))  # Earth spans ~1.9 degrees from the Moon
    centre = direction(azimuth, elevation) * distance
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=radius, location=centre)
    e = bpy.context.active_object
    e.name = "earth"
    e.rotation_euler = (math.radians(23.4), 0, math.radians(200))
    e.data.polygons.foreach_set("use_smooth", [True] * len(e.data.polygons))
    g = _Graph("earth")
    tex = g.node("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(texture_path)
    g.link(tex.outputs["Color"], g.bsdf.inputs["Base Color"])
    if clouds_path:
        ct = g.node("ShaderNodeTexImage")
        ct.image = bpy.data.images.load(clouds_path)
        col = g.mix_color(ct.outputs["Color"], tex.outputs["Color"], (1.0, 1.0, 1.0))
        g.link(col, g.bsdf.inputs["Base Color"])
    g.set(Roughness=0.55)
    e.data.materials.append(g.mat)
    _link(e)
    # thin atmosphere glow on the limb
    bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=radius * 1.018, location=centre)
    a = bpy.context.active_object
    a.name = "earth_atmosphere"
    a.data.polygons.foreach_set("use_smooth", [True] * len(a.data.polygons))
    mat = bpy.data.materials.new("atmosphere")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.25
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.35, 0.6, 1.0, 1)
    em.inputs["Strength"].default_value = 1.4
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Facing"], mix.inputs["Fac"])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    a.data.materials.append(mat)
    _link(a)
    return e


def make_lander(location=(-14.0, 22.0, 0.0), yaw: float = 0.4):
    """A crewed lunar lander parked beyond the pad: foil-wrapped descent stage, white cabin."""
    import ariel.art.robot as rb
    from .robot import _box, _cyl, _ring, _sphere
    root = bpy.data.objects.new("lander", None)
    root.location = location
    root.rotation_euler = (0, 0, yaw)
    _link(root, "lander")
    old = rb.COLL
    rb.COLL = "lander"
    try:
        # descent stage: octagonal foil-wrapped box with black structure bands
        _cyl(root, "foil_gold", (0, 0, 2.25), 2.25, 1.5, verts=8, bevel=0.04)
        _cyl(root, "anodized_black", (0, 0, 3.05), 2.3, 0.1, verts=8, bevel=0.02)
        _cyl(root, "anodized_black", (0, 0, 1.5), 2.3, 0.08, verts=8, bevel=0.02)
        for k in range(8):
            a = 2 * math.pi * (k + 0.5) / 8
            _box(root, "anodized_black", (2.08 * math.cos(a), 2.08 * math.sin(a), 2.25), (0.05, 0.05, 0.76),
                 rot=(0, 0, a), bevel=0.01)
        # main engine and RCS quads
        _cyl(root, "anodized_black", (0, 0, 1.05), 0.75, 0.9, r2=1.05, verts=64, bevel=0.01)
        _cyl(root, "chrome_tube", (0, 0, 0.62), 0.98, 0.03, verts=64)
        for k in range(4):
            a = 2 * math.pi * k / 4
            p = Vector((2.35 * math.cos(a), 2.35 * math.sin(a), 3.4))
            _box(root, "panel_white", p, (0.12, 0.12, 0.12), rot=(0, 0, a), bevel=0.02)
            for dz, dy in ((0.16, 0), (-0.16, 0), (0, 0.16), (0, -0.16)):
                off = Vector((0, dy, dz))
                _cyl(root, "anodized_black", p + off, 0.03, 0.08, axis="x", verts=12)
        # crew cabin: white pressure vessel with windows, hatch and docking port
        _cyl(root, "panel_white", (0, 0, 4.25), 1.4, 2.2, verts=48, bevel=0.08)
        _cyl(root, "panel_white", (0, 0, 5.5), 1.0, 0.35, r2=1.35, verts=48, bevel=0.04)
        _cyl(root, "aluminum", (0, 0, 5.8), 0.55, 0.3, verts=48, bevel=0.02)
        _ring(root, "anodized_black", (0, 0, 5.96), 0.55, 0.04)
        for k in range(2):
            a = math.radians(-20 + 40 * k)
            _cyl(root, "glass_dark", (1.39 * math.cos(a), 1.39 * math.sin(a), 4.7), 0.18, 0.04, axis="x", bevel=0.01)
        _box(root, "aluminum", (1.42, 0, 3.9), (0.03, 0.42, 0.55), bevel=0.04)
        _box(root, "accent_orange", (1.45, 0, 3.32), (0.01, 0.42, 0.02))
        _box(root, "anodized_black", (-1.1, 0.6, 4.6), (0.35, 0.05, 0.5), rot=(0, 0, 0.5), bevel=0.02)
        _cyl(root, "anodized_black", (0.4, -0.6, 6.4), 0.02, 1.0, verts=12)
        _sphere(root, "panel_white", (0.4, -0.6, 6.95), 0.25, scale=(1, 1, 0.3))
        # four legs with pads, secondary struts and a ladder on the hatch side
        for k in range(4):
            a = math.pi / 4 + k * math.pi / 2
            top = Vector((1.95 * math.cos(a), 1.95 * math.sin(a), 2.0))
            knee = Vector((3.0 * math.cos(a), 3.0 * math.sin(a), 1.2))
            foot = Vector((3.7 * math.cos(a), 3.7 * math.sin(a), 0.12))
            for p0, p1, r in ((top, knee, 0.08), (knee, foot, 0.065),
                              (Vector((1.7 * math.cos(a), 1.7 * math.sin(a), 1.5)), knee, 0.04)):
                d = p1 - p0
                bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=r, depth=d.length, location=(p0 + p1) / 2)
                leg = bpy.context.active_object
                leg.rotation_mode = "QUATERNION"
                leg.rotation_quaternion = d.to_track_quat("Z", "Y")
                rb._finish(leg, root, "foil_gold" if r > 0.07 else "chrome_tube")
            _cyl(root, "panel_white", foot, 0.5, 0.1, verts=32, bevel=0.03)
            _cyl(root, "foil_gold", foot + Vector((0, 0, 0.08)), 0.2, 0.08, verts=24)
        for k in range(9):
            _box(root, "aluminum", (2.0 + 0.15 * k, 0.0, 2.1 - 0.22 * k), (0.02, 0.22, 0.012), bevel=0.004)
        for s in (-1, 1):
            p0, p1 = Vector((1.95, s * 0.22, 2.2)), Vector((3.3, s * 0.22, 0.2))
            d = p1 - p0
            bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.02, depth=d.length, location=(p0 + p1) / 2)
            rail = bpy.context.active_object
            rail.rotation_mode = "QUATERNION"
            rail.rotation_quaternion = d.to_track_quat("Z", "Y")
            rb._finish(rail, root, "aluminum")
        _cyl(root, "lamp_cool", (2.32, 0.0, 3.0), 0.05, 0.02, axis="x")
    finally:
        rb.COLL = old
    return root
