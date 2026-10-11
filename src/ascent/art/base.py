"""The damaged HackPHS lunar base: terrain and structures built from a course scene.

Every structure the robot can touch is modelled around the same shape the physics
uses (``scene["static"]`` / ``scene["loose"]``). Purely decorative items sit outside
the search corridor. Runs inside Blender only.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Quaternion, Vector, noise

import ascent.art.robot as rb
from ascent.art.environment import _rock_mesh
from ascent.art.materials import material


@contextmanager
def collection(name: str):
    old = rb.COLL
    rb.COLL = name
    try:
        yield
    finally:
        rb.COLL = old


def basis_euler(right, up) -> Euler:
    """Rotation whose local x points along ``right``, y along ``up`` and z out of the surface."""
    from mathutils import Matrix
    r = Vector(right).normalized()
    u = Vector(up).normalized()
    n = r.cross(u).normalized()
    u = n.cross(r).normalized()
    return Matrix((r, u, n)).transposed().to_euler()


def _empty(name: str, coll: str, loc=(0, 0, 0), quat=(1, 0, 0, 0)) -> bpy.types.Object:
    e = bpy.data.objects.new(name, None)
    e.rotation_mode = "QUATERNION"
    e.location = loc
    e.rotation_quaternion = quat
    rb._collection(coll).objects.link(e)
    return e


def _grid_mesh(name: str, xs: np.ndarray, ys: np.ndarray, z: np.ndarray, skirt: float = 0.0) -> bpy.types.Object:
    """Triangulated height grid (rows = y), optionally with a skirt hanging from its border."""
    nrow, ncol = z.shape
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
    idx = np.arange(nrow * ncol).reshape(nrow, ncol)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = np.stack([a, b, c, d], axis=1)
    if skirt > 0:
        border = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
        low = verts[border].copy()
        low[:, 2] -= skirt
        base = len(verts)
        verts = np.concatenate([verts, low])
        n = len(border)
        sk = [(border[i], border[(i + 1) % n], base + (i + 1) % n, base + i) for i in range(n)]
        faces = np.concatenate([faces, np.array(sk)])
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 4, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 4, dtype=np.int32))
    mesh.update(calc_edges=True)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    o = bpy.data.objects.new(name, mesh)
    rb._collection("terrain").objects.link(o)
    o.data.materials.append(material("terrain"))
    return o


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------

def build_terrain(scene: dict, heights: np.ndarray, vista: dict) -> None:
    hf = scene["hfield"]
    nrow, ncol = heights.shape
    xs = hf["x0"] + hf["res"] * np.arange(ncol)
    ys = hf["y0"] + hf["res"] * np.arange(nrow)
    gy, gz = ys, heights
    if "near_xs" in vista:
        # one welded mesh: near-field strip, exact physics grid, near-field strip
        gy = np.concatenate([vista["near_lo_ys"][:-1], ys, vista["near_hi_ys"][1:]])
        gz = np.concatenate([vista["near_lo_z"][:-1], heights, vista["near_hi_z"][1:]], axis=0)
    terr = _grid_mesh("course_terrain", xs, gy, gz, skirt=0.3)
    _relief(terr, xs, gy, gz, scene["params"]["geofence"])
    vx, vy, vz = vista["xs"], vista["ys"], vista["z"].copy()
    VX, VY = np.meshgrid(vx, vy)
    inside = (VX > xs[0] + 0.3) & (VX < xs[-1] - 0.3) & (VY > gy[0] + 0.3) & (VY < gy[-1] - 0.3)
    vz[inside] -= 0.35                # tucked under the detailed mesh
    _grid_mesh("vista_terrain", vx, vy, vz)


def _relief(obj, xs, ys, heights, fence: float) -> None:
    """Push steep-face vertices sideways with noise so cliffs read as broken rock.

    Inside the search corridor the offset stays within 1.5 cm of the physics surface.
    """
    gx = np.gradient(heights, xs, axis=1)
    gy = np.gradient(heights, ys, axis=0)
    slope = np.hypot(gx, gy)
    steep = np.clip((slope - 1.0) / 2.0, 0.0, 1.0)
    X, Y = np.meshgrid(xs, ys)
    amp = np.where(np.abs(Y) > fence + 0.5, 0.045, 0.015) * steep
    nx = -gx / np.maximum(slope, 1e-6)
    ny = -gy / np.maximum(slope, 1e-6)
    rng = np.random.default_rng(11)
    offs = rng.uniform(0, 100, 4)
    n = np.zeros_like(heights)
    for i in range(heights.shape[0]):
        row = [noise.noise(Vector((float(x) * 6.0 + offs[0], float(ys[i]) * 6.0 + offs[1], float(z) * 6.0)))
               + 0.5 * noise.noise(Vector((float(x) * 17.0, float(ys[i]) * 17.0 + offs[2], float(z) * 17.0)))
               for x, z in zip(xs, heights[i])] if steep[i].max() > 0 else [0.0] * len(xs)
        n[i] = row
    co = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    k = heights.size
    co[:k, 0] += (amp * n * nx).ravel()
    co[:k, 1] += (amp * n * ny).ravel()
    obj.data.vertices.foreach_set("co", co.ravel())
    obj.data.update()


def scatter_pebbles(scene: dict, heights: np.ndarray, vista: dict, seed: int = 4) -> None:
    """Visual-only stones: pebbles in the corridor, larger rocks only outside it."""
    rng = np.random.default_rng(seed)
    hf = scene["hfield"]
    fence = scene["params"]["geofence"]
    protos = []
    with collection("rock_protos"):
        for i in range(12):
            p = _rock_mesh(1000 + i, f"pebble_proto_{i}")
            rb._finish(p, None, "rock", smooth=True)
            p.hide_render = True
            p.hide_viewport = True
            protos.append(p)

    def h_at(x, y):
        if hf["x0"] <= x <= hf["x0"] + hf["res"] * (heights.shape[1] - 1) and abs(y) < 4.9:
            c = int(round((x - hf["x0"]) / hf["res"]))
            r = int(round((y - hf["y0"]) / hf["res"]))
            return float(heights[r, c])
        c = int(round((x - vista["xs"][0]) / (vista["xs"][1] - vista["xs"][0])))
        r = int(round((y - vista["ys"][0]) / (vista["ys"][1] - vista["ys"][0])))
        return float(vista["z"][r, c])

    coll = rb._collection("pebbles")
    count = 0
    goal_x = scene["params"]["goal_x"]
    for _ in range(9000):
        x = rng.uniform(-8, goal_x + 8)
        y = rng.uniform(-14, 14)
        inside = abs(y) < fence + 0.3 and -1.5 < x < goal_x + 2
        if inside:
            s = float(np.clip(0.008 * (rng.pareto(2.6) + 1), 0.008, 0.025))
        else:
            s = float(np.clip(0.02 * (rng.pareto(2.0) + 1), 0.02, 0.7))
        if math.hypot(x, y) < scene["params"]["pad_end"]:
            continue          # the pad is swept clean
        z = h_at(x, y)
        if z < -0.5:
            continue          # not inside the fissure
        o = bpy.data.objects.new(f"pebble_{count}", protos[int(rng.integers(len(protos)))].data)
        o.location = (x, y, z - s * rng.uniform(0.15, 0.4))
        o.rotation_euler = (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(0, 2 * math.pi))
        o.scale = (s * rng.uniform(0.8, 1.3), s * rng.uniform(0.8, 1.3), s)
        coll.objects.link(o)
        count += 1


# ---------------------------------------------------------------------------
# HackPHS branding
# ---------------------------------------------------------------------------

_LOGO_PATH: str | None = None


def logo_mesh(path: str):
    """Register the HackPHS logo image (a transparent PNG rasterised from hackphs.tech's SVG)."""
    global _LOGO_PATH
    if Path(path).exists():
        _LOGO_PATH = path
    return _LOGO_PATH


def _logo_material(painted: bool) -> bpy.types.Material:
    name = "logo_painted" if painted else "logo_decal"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(_LOGO_PATH)
    tex.extension = "CLIP"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    alpha = tex.outputs["Alpha"]
    if painted:
        # worn paint on the pad: scuffs and dust break up the alpha
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 18.0
        n.inputs["Detail"].default_value = 8.0
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.35
        mr.inputs["From Max"].default_value = 0.5
        nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        nt.links.new(alpha, mul.inputs[0])
        nt.links.new(mr.outputs["Result"], mul.inputs[1])
        alpha = mul.outputs[0]
    nt.links.new(alpha, bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.55 if painted else 0.4
    return mat


def place_logo(name: str, size: float, loc, rot, coll: str, parent=None, badge: bool = True,
               shrink_target=None, painted: bool = False) -> None:
    if _LOGO_PATH is None:
        return
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=24, y_subdivisions=24, size=size)
    o = bpy.context.active_object
    o.name = name
    o.location = loc
    o.rotation_euler = rot
    o.data.materials.append(_logo_material(painted))
    for c in list(o.users_collection):
        c.objects.unlink(o)
    rb._collection(coll).objects.link(o)
    o.parent = parent
    n = Euler(rot).to_matrix() @ Vector((0, 0, 1))
    if badge:
        bpy.ops.mesh.primitive_circle_add(vertices=96, radius=0.64 * size, fill_type="TRIFAN")
        bd = bpy.context.active_object
        bd.location = loc
        bd.rotation_euler = rot
        rb._finish(bd, parent, "badge_navy", smooth=False)
        for c in list(bd.users_collection):
            c.objects.unlink(bd)
        rb._collection(coll).objects.link(bd)
        rim = rb._ring(parent, "foil_gold", loc, 0.64 * size, 0.012)
        rim.rotation_euler = rot
        o.location = Vector(loc) + n * 0.004
        if shrink_target is not None:
            for obj, off in ((bd, 0.004), (rim, 0.006)):
                m = obj.modifiers.new("wrap", "SHRINKWRAP")
                m.target = shrink_target
                m.offset = off
    if shrink_target is not None:
        m = o.modifiers.new("wrap", "SHRINKWRAP")
        m.target = shrink_target
        m.offset = 0.008


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

def build_landing_pad(item: dict, logo_path: str) -> None:
    r = item["radius"]
    with collection("pad"):
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=r, depth=0.05, location=(0, 0, -0.024))
        pad = bpy.context.active_object
        rb._finish(pad, None, "sintered", bevel=0.01)
        rb._ring(None, "paint_white", (0, 0, 0.002), r * 0.78, 0.035).scale = (1, 1, 0.08)
        rb._ring(None, "hazard_orange", (0, 0, 0.002), r * 0.93, 0.025).scale = (1, 1, 0.08)
        for k in range(12):
            a = 2 * math.pi * (k + 0.5) / 12
            rb._box(None, "anodized_black", (r * 0.985 * math.cos(a), r * 0.985 * math.sin(a), 0.008),
                    (0.05, 0.05, 0.012), bevel=0.004)
            rb._cyl(None, "lamp_cool", (r * 0.985 * math.cos(a), r * 0.985 * math.sin(a), 0.021), 0.025, 0.006)
        # moonquake crack running across the pad
        rng = random.Random(3)
        pts = []
        x, y = -r * 0.9, -0.8
        while x < r * 0.95:
            pts.append((x, y))
            x += rng.uniform(0.08, 0.2)
            y += rng.uniform(-0.12, 0.12)
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            mid = ((xa + xb) / 2, (ya + yb) / 2, 0.0015)
            ln = math.hypot(xb - xa, yb - ya)
            rb._box(None, "anodized_black", mid, (ln / 2 + 0.01, 0.012, 0.002),
                    rot=(0, 0, math.atan2(yb - ya, xb - xa)), bevel=0.0)
    if logo_mesh(logo_path) is not None:
        place_logo("pad_logo", 2.3, (0.0, 0.0, 0.003), (0, 0, -math.pi / 2), "pad", badge=False, painted=True)
    with collection("pad"):
        rb._text(None, "HACKPHS LUNAR BASE", (-r * 0.6, 0, 0.004), 0.22, rot=(0, 0, -math.pi / 2),
                 mat="paint_white")
        rb._text(None, "PAD 1", (r * 0.6, 0, 0.004), 0.26, rot=(0, 0, -math.pi / 2), mat="paint_white")


def build_hab(item: dict, logo_path: str) -> None:
    cx, _, cz = item["pos"]
    R, L = item["radius"], item["length"]
    with collection("habitat"):
        root = _empty("habitat", "habitat", (cx, 0, cz))
        # pressure hull with a crumpled top (the underside matches the physics shell exactly)
        bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=R, depth=L, location=(0, 0, 0),
                                            rotation=(math.pi / 2, 0, 0))
        hull = bpy.context.active_object
        bpy.ops.object.transform_apply(rotation=True)
        bm = bmesh.new()
        bm.from_mesh(hull.data)
        bmesh.ops.subdivide_edges(bm, edges=[e for e in bm.edges if abs(e.verts[0].co.y - e.verts[1].co.y) > 1.0],
                                  cuts=48, use_grid_fill=True)
        for v in bm.verts:
            if v.co.z > 0.2 * R:
                d = noise.noise(Vector((v.co.x * 1.4, v.co.y * 0.9, 3.0)))
                dent = 0.18 * max(0.0, d) * (v.co.z / R) ** 2
                crush = 0.25 * max(0.0, (v.co.y - L * 0.25) / (L * 0.25)) * (v.co.z / R)
                v.co -= Vector((v.co.x, 0, v.co.z)).normalized() * (dent + crush)
        bm.to_mesh(hull.data)
        bm.free()
        rb._finish(hull, root, "beta_cloth", subsurf=1)
        hull_obj = hull
        # structural ribs and end rings
        for k in range(8):
            y = -L / 2 + 0.35 + k * (L - 0.7) / 7
            rb._ring(root, "aluminum", (0, y, 0), R + 0.012, 0.035, axis="y")
        for s in (-1, 1):
            rb._ring(root, "anodized_black", (0, s * L / 2, 0), R * 0.98, 0.07, axis="y")
            bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=R * 0.96,
                                                 location=(0, s * L / 2, 0))
            cap = bpy.context.active_object
            cap.scale = (1, 0.28, 1)
            rb._finish(cap, root, "panel_white")
            if s < 0:
                near_cap = cap
            # crushed end cradles resting on rubble mounds, outside the corridor
            rb._box(root, "aluminum", (0, s * (L / 2 - 0.4), -R - 0.1), (R * 0.8, 0.08, 0.12), bevel=0.02)
            rb._box(root, "aluminum", (R * 0.6, s * (L / 2 - 0.4), -R * 0.55), (0.06, 0.06, R * 0.5),
                    rot=(0, 0.35, 0), bevel=0.01)
            rb._box(root, "aluminum", (-R * 0.6, s * (L / 2 - 0.4), -R * 0.55), (0.06, 0.06, R * 0.5),
                    rot=(0, -0.35, 0), bevel=0.01)
        # docking hatch on the -y end, windows, handrails
        rb._cyl(root, "aluminum", (0, -L / 2 - 0.25, 0.1), 0.55, 0.12, axis="y", bevel=0.02)
        rb._cyl(root, "anodized_black", (0, -L / 2 - 0.32, 0.1), 0.4, 0.04, axis="y", bevel=0.01)
        for wy in (-1.6, 1.4):
            a = math.radians(160)
            p = Vector((R * math.cos(a), wy, R * math.sin(a)))
            rb._cyl(root, "glass_dark", p * 1.005, 0.16, 0.03, axis="x", bevel=0.01)
            rb._ring(root, "anodized_black", p * 1.01, 0.16, 0.02, axis="x")
        for k in range(10):
            a = math.radians(200 + 0)
            y = -2.6 + k * 0.55
            rb._cyl(root, "hazard_orange", (R * 1.02 * math.cos(math.radians(205)), y,
                                            R * 1.02 * math.sin(math.radians(205))), 0.012, 0.5, axis="y")
        # torn insulation flap hanging over the crawl gap
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-R * 0.45, -0.8, -R * 0.82))
        flap = bpy.context.active_object
        flap.scale = (0.5, 0.7, 1)
        flap.rotation_euler = (0.2, 1.2, 0.3)
        rb._finish(flap, root, "beta_cloth")
        flap.modifiers.new("solid", "SOLIDIFY").thickness = 0.01
        # rubble mounds under the end cradles
        for s in (-1, 1):
            for k in range(14):
                rr = random.Random(k * 7 + (s > 0))
                rk = _rock_mesh(500 + k + (50 if s > 0 else 0), f"mound_{s}_{k}")
                rk.location = (cx + rr.uniform(-1.2, 1.2), s * rr.uniform(L / 2 - 0.9, L / 2 + 0.6),
                               rr.uniform(0.0, 0.5))
                sc = rr.uniform(0.25, 0.6)
                rk.scale = (sc, sc, sc * 0.8)
                for c in list(rk.users_collection):
                    c.objects.unlink(rk)
                rb._collection("habitat").objects.link(rk)
    if logo_mesh(logo_path) is not None:
        # big badge on the side facing the approaching robot, and on the near end cap
        a = math.radians(165)
        normal = Vector((math.cos(a), 0, math.sin(a)))
        up = Vector((-math.sin(a), 0, math.cos(a))) * -1
        p = Vector((cx, 0.9, cz)) + normal * (R + 0.02)
        place_logo("hab_logo_side", 0.95, p, basis_euler((0, -1, 0), up), "habitat", shrink_target=hull_obj)
        place_logo("hab_logo_end", 0.8, (cx, -L / 2 - 0.45, cz + 0.78), (math.pi / 2, 0, 0), "habitat",
                   shrink_target=near_cap)
    with collection("habitat"):
        for a_deg, text, size, mat, dy in ((168, "HACKPHS", 0.42, "badge_navy", -1.05),
                                            (186, "HAB-2  ·  CREW MODULE", 0.12, "hazard_orange", -1.05)):
            a = math.radians(a_deg)
            normal = Vector((math.cos(a), 0, math.sin(a)))
            up = Vector((math.sin(a), 0, -math.cos(a)))
            p = normal * (R + 0.012) + Vector((0, dy, 0))
            rb._text(root, text, p, size, rot=basis_euler((0, -1, 0), up), mat=mat, extrude=0.002)


def _area_light(name: str, loc, size: tuple, energy: float, color, coll: str, aim=(0, 0, -1)) -> None:
    light = bpy.data.lights.new(name, "AREA")
    light.shape = "RECTANGLE"
    light.size, light.size_y = size
    light.energy = energy
    light.color = color
    o = bpy.data.objects.new(name, light)
    o.location = loc
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector(aim).to_track_quat("-Z", "Y")
    rb._collection(coll).objects.link(o)


def build_work_lights(scene: dict) -> None:
    """Emergency strip lights under the collapsed module and safety beacons along the fissure."""
    P = scene["params"]
    hab_x, R, clr = P["hab_x"], P["hab_radius"], P["hab_clearance"]
    with collection("work_lights"):
        for k, dx in enumerate((-0.75, 0.0, 0.75)):
            # strip fixture on the hull's underside, glowing downward
            z = clr + R - (R ** 2 - dx ** 2) ** 0.5
            rb._box(None, "anodized_black", (hab_x + dx, 0.0, z + 0.012), (0.04, 1.3, 0.012), bevel=0.004)
            rb._box(None, "lamp_cool", (hab_x + dx, 0.0, z), (0.018, 1.2, 0.004))
            _area_light(f"hab_strip_{k}", (hab_x + dx, 0.0, z - 0.02), (0.05, 2.4), 90.0, (0.85, 0.92, 1.0),
                        "work_lights")
        fx, fw = P["fissure_x"], P["fissure_width"]
        for k, y in enumerate((-2.4, -1.2, 0.0, 1.2, 2.4)):
            for side, x in ((0, fx - 0.25), (1, fx + fw + 0.25)):
                rb._cyl(None, "anodized_black", (x, y, 0.005), 0.035, 0.02, bevel=0.004)  # flush ground marker
                rb._cyl(None, "beacon_glow", (x, y, 0.017), 0.022, 0.006, bevel=0.002)
        _area_light("fissure_fill", (fx + fw / 2, 0.0, 0.35), (fw * 0.8, 5.0), 120.0, (1.0, 0.86, 0.66), "work_lights")


def build_solar_tower(item: dict, statics: list[dict]) -> None:
    px, py, pz = item["pos"]
    yaw = item["yaw"]
    L = item["length"]
    with collection("solar_tower"):
        root = _empty("solar_tower", "solar_tower", (px, py, pz), Quaternion((0, 0, 1), yaw))
        h = 0.18
        bays = int(L / 0.4)
        for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rb._cyl(root, "aluminum", (sx * h, 0, sz * h), 0.022, L, axis="y", verts=16)
        for b in range(bays + 1):
            y = -L / 2 + b * L / bays
            for (ax, az), (bx, bz) in (((1, 1), (1, -1)), ((-1, 1), (-1, -1)), ((1, 1), (-1, 1)), ((1, -1), (-1, -1))):
                p0, p1 = Vector((ax * h, y, az * h)), Vector((bx * h, y, bz * h))
                _strut(root, p0, p1, 0.012)
            if b < bays:
                y1 = y + L / bays
                for (ax, az), (bx, bz) in (((1, 1), (1, -1)), ((-1, 1), (-1, -1)), ((1, 1), (-1, 1)), ((1, -1), (-1, -1))):
                    _strut(root, Vector((ax * h, y, az * h)), Vector((bx * h, y1, bz * h)), 0.01)
        # ballast foot at one end, broken gimbal and crumpled array at the other
        bal = Vector((0, -L / 2 - 0.5, -0.05))
        rb._box(root, "panel_white", bal, (0.6, 0.5, 0.35), bevel=0.04, segments=3)
        rb._box(root, "foil_gold", bal + Vector((0, 0, 0.36)), (0.52, 0.42, 0.012), bevel=0.006)
        for sx in (-1, 1):
            rb._box(root, "anodized_black", bal + Vector((sx * 0.605, 0, 0.0)), (0.006, 0.42, 0.26), bevel=0.004)
            rb._box(root, "hazard_orange", bal + Vector((sx * 0.612, 0, -0.2)), (0.004, 0.42, 0.04))
        rb._cyl(root, "anodized_black", bal + Vector((0, 0.5, 0.1)), 0.16, 0.14, axis="y", bevel=0.02)
        rb._text(root, "VSAT-3", bal + Vector((0.613, 0, 0.12)), 0.11, rot=(math.pi / 2, 0, math.pi / 2),
                 mat="badge_navy", extrude=0.001)
        rb._cyl(root, "anodized_black", (0, L / 2 + 0.1, 0), 0.22, 0.25, axis="y", bevel=0.02)
        for k, (dx, rot) in enumerate(((1.4, (0.25, 0.1, 0.3)), (-1.5, (-0.4, 0.15, -0.2)))):
            rb._box(root, "solar_cells", (dx, L / 2 + 1.2, -0.05 + 0.25 * k), (1.1, 1.6, 0.015),
                    rot=rot, bevel=0.005)
    # physics panels lying on the terrace
    with collection("solar_tower"):
        for s in statics:
            if s["kind"] != "solar_panel":
                continue
            q = Quaternion(s["quat"])
            hx, hy, hz = s["size"]
            o = rb._box(None, "solar_cells", (0, 0, 0), (hx, hy, hz), bevel=0.004)
            o.rotation_mode = "QUATERNION"
            o.location = s["pos"]
            o.rotation_quaternion = q
            fr = rb._box(None, "aluminum", (0, 0, 0), (hx + 0.02, hy + 0.02, hz * 0.7), bevel=0.004)
            fr.rotation_mode = "QUATERNION"
            fr.location = Vector(s["pos"]) - Vector((0, 0, 0.003))
            fr.rotation_quaternion = q


def _strut(parent, p0: Vector, p1: Vector, r: float) -> None:
    d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=r, depth=d.length, location=(p0 + p1) / 2)
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    rb._finish(o, parent, "aluminum")


def scatter_talus(scene: dict, heights: np.ndarray, vista: dict, seed: int = 21) -> None:
    """Fallen blocks at the foot of the scarp and along the fissure lips (outside the corridor)."""
    rng = np.random.default_rng(seed)
    P = scene["params"]
    fence = P["geofence"]
    vx, vy, vz = vista["xs"], vista["ys"], vista["z"]
    dx = vx[1] - vx[0]
    gzx = np.gradient(vz, dx, axis=1)
    gzy = np.gradient(vz, dx, axis=0)
    slope = np.hypot(gzx, gzy)
    near = (np.abs(np.meshgrid(vx, vy)[0] - P["scarp_x"]) < 12) | (np.abs(np.meshgrid(vx, vy)[0] - P["fissure_x"]) < 12)
    cand = np.argwhere((slope > 1.2) & near)
    if len(cand) == 0:
        return
    protos = []
    with collection("rock_protos"):
        for i in range(10):
            pr = _rock_mesh(3000 + i, f"talus_proto_{i}")
            pr.hide_render = True
            pr.hide_viewport = True
            protos.append(pr)
    with collection("talus"):
        for k in range(min(900, len(cand))):
            i, j = cand[rng.integers(len(cand))]
            x, y = float(vx[j]) + rng.uniform(-0.3, 0.3), float(vy[i]) + rng.uniform(-0.3, 0.3)
            if abs(y) < fence + 0.6:
                continue
            z = float(vz[i, j])
            if z < -0.6:
                continue
            sc = float(np.clip(0.06 * (rng.pareto(1.8) + 1), 0.06, 0.6))
            o = bpy.data.objects.new(f"talus_{k}", protos[k % len(protos)].data)
            bpy.context.scene.collection.objects.link(o)
            o.location = (x, y, z - sc * 0.3)
            o.scale = (sc * rng.uniform(0.8, 1.4), sc * rng.uniform(0.8, 1.3), sc)
            o.rotation_euler = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0, 6.28))
            for c in list(o.users_collection):
                c.objects.unlink(o)
            rb._collection("talus").objects.link(o)


def build_boulders(statics: list[dict]) -> None:
    with collection("boulders"):
        for s in statics:
            if s["kind"] != "boulder":
                continue
            r = s["size"][0]
            o = _rock_mesh(int(s.get("seed", 1)) % 100000, s["name"])
            o.location = s["pos"]
            o.scale = (r * 1.05, r * 1.0, r * 1.02)
            o.rotation_euler = (0, 0, (s.get("seed", 0) % 628) / 100)
            for c in list(o.users_collection):
                c.objects.unlink(o)
            rb._collection("boulders").objects.link(o)


def build_loose(scene: dict) -> dict[str, bpy.types.Object]:
    """Empties for loose physics bodies (animated from recorded poses) with detailed parts."""
    empties = {}
    for b in scene["loose"]:
        e = _empty(b["name"], "loose", b["pos"], b.get("quat", (1, 0, 0, 0)))
        empties[b["name"]] = e
        with collection("loose"):
            if b["kind"] == "crate":
                hx, hy, hz = b["size"]
                rb._box(e, "aluminum", (0, 0, 0), (hx, hy, hz), bevel=0.015, segments=3)
                # stiffening ribs and carry handles on the long sides
                for k in (-1, 1):
                    rb._box(e, "aluminum", (0, 0, k * hz * 0.45), (hx + 0.006, hy + 0.006, 0.01), bevel=0.003)
                for sy in (-1, 1):
                    rb._box(e, "anodized_black", (0, sy * (hy + 0.012), hz * 0.25), (hx * 0.25, 0.008, 0.012), bevel=0.004)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        rb._box(e, "hazard_orange", (sx * (hx - 0.012), sy * (hy - 0.012), 0),
                                (0.016, 0.016, hz + 0.004), bevel=0.006)
                rb._box(e, "anodized_black", (0, 0, hz - 0.006), (hx * 0.92, hy * 0.92, 0.004))
                rb._box(e, "aluminum", (0, 0, hz + 0.005), (hx * 0.3, 0.012, 0.006), bevel=0.003)
                rb._box(e, "paint_white", (hx + 0.001, 0, 0.02), (0.001, hy * 0.6, hz * 0.35))
                rb._text(e, "HACKPHS", (hx + 0.003, 0, 0.04), min(hy, 0.25) * 0.42,
                         rot=(math.pi / 2, 0, math.pi / 2), mat="decal_black")
                rb._text(e, f"CARGO {b['name'][-1]}", (hx + 0.003, 0, -0.03), min(hy, 0.25) * 0.2,
                         rot=(math.pi / 2, 0, math.pi / 2), mat="hazard_orange")
            elif b["kind"] == "panel":
                hx, hy, hz = b["size"]
                rb._box(e, "aluminum", (0, 0, 0), (hx, hy, hz), bevel=0.004)
                for k in range(4):
                    rb._box(e, "aluminum", (-hx + (k + 0.5) * hx / 2, 0, hz + 0.006), (0.008, hy * 0.95, 0.006))
                rb._box(e, "hazard_orange", (hx - 0.05, 0, hz + 0.001), (0.03, hy * 0.9, 0.001))
            elif b["kind"] == "rock":
                r = b["size"][0]
                o = _rock_mesh(int(b.get("seed", 3)) % 100000, b["name"] + "_mesh")
                o.scale = (r * 1.05, r, r)
                rb._finish(o, e, "rock")
    return empties


def build_astronaut(item: dict) -> None:
    """A stranded crew member sitting against a boulder, suit beacon flashing."""
    gx, gy, gz = item["pos"]
    with collection("astronaut"):
        root = _empty("astronaut", "astronaut", (gx, gy, gz), Quaternion((0, 0, 1), item.get("yaw", math.pi)))
        torso_c = Vector((0.05, 0, 0.47))
        torso = _empty("astronaut_torso", "astronaut", torso_c, Euler((0, -0.35, 0)).to_quaternion())
        torso.parent = root
        rb._box(torso, "suit_fabric", (0, 0, 0), (0.13, 0.2, 0.25), bevel=0.09, segments=5)
        rb._box(torso, "panel_white", (0.08, 0, 0.12), (0.06, 0.17, 0.12), bevel=0.04)
        rb._box(torso, "panel_white", (0.18, 0, 0.0), (0.035, 0.1, 0.06), bevel=0.012)
        rb._box(torso, "anodized_black", (0.215, 0, 0.0), (0.004, 0.075, 0.04))
        rb._box(torso, "accent_orange", (0.14, 0.12, 0.13), (0.004, 0.03, 0.012))
        # life-support backpack against the boulder
        rb._box(torso, "panel_white", (-0.2, 0, 0.08), (0.09, 0.2, 0.27), bevel=0.04)
        # helmet with gold visor
        hc = Vector((-0.02, 0, 0.36))
        rb._sphere(torso, "suit_fabric", hc, 0.17)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.155, location=hc + Vector((0.035, 0, 0.0)))
        v = bpy.context.active_object
        v.scale = (1, 0.92, 0.85)
        rb._finish(v, torso, "gold_visor")
        rb._ring(torso, "aluminum", hc + Vector((0.0, 0, -0.14)), 0.12, 0.02)
        for s_ in (-1, 1):
            rb._cyl(torso, "lamp_cool", hc + Vector((0.09, s_ * 0.13, 0.09)), 0.02, 0.03, axis="x")
        # legs stretched out along the ground
        for s in (-1, 1):
            hip = Vector((0.12, s * 0.11, 0.2))
            knee = Vector((0.5, s * 0.13, 0.27))
            ankle = Vector((0.88, s * 0.14, 0.1))
            _limb(root, hip, knee, 0.085)
            _limb(root, knee, ankle, 0.072)
            rb._ring(root, "aluminum", knee, 0.08, 0.012, axis="x")
            rb._box(root, "suit_fabric", ankle + Vector((0.07, 0, -0.03)), (0.13, 0.065, 0.07), bevel=0.04)
            rb._box(root, "rubber", ankle + Vector((0.07, 0, -0.095)), (0.135, 0.068, 0.014), bevel=0.01)
        # one arm rests on a knee, the other holds up a flashing beacon
        sh_l = torso_c + Vector((0.0, 0.24, 0.17))
        sh_r = torso_c + Vector((0.0, -0.24, 0.17))
        _limb(root, sh_l, Vector((0.32, 0.25, 0.36)), 0.06)
        _limb(root, Vector((0.32, 0.25, 0.36)), Vector((0.5, 0.18, 0.33)), 0.052)
        rb._sphere(root, "suit_fabric", Vector((0.55, 0.18, 0.33)), 0.05)
        el = sh_r + Vector((0.16, -0.12, 0.12))
        hand = el + Vector((0.12, -0.02, 0.26))
        _limb(root, sh_r, el, 0.06)
        _limb(root, el, hand, 0.052)
        rb._sphere(root, "suit_fabric", hand, 0.05)
        rb._cyl(root, "anodized_black", hand + Vector((0, 0, 0.07)), 0.025, 0.08)
        rb._sphere(root, "beacon_red", hand + Vector((0, 0, 0.13)), 0.03)
        rb._cyl(torso, "beacon_red", (-0.2, 0.15, 0.37), 0.015, 0.04)
        light = bpy.data.lights.new("astronaut_beacon", "POINT")
        light.energy = 6.0
        light.color = (1.0, 0.15, 0.05)
        light.shadow_soft_size = 0.02
        lo = bpy.data.objects.new("astronaut_beacon", light)
        lo.parent = root
        lo.location = hand + Vector((0, 0, 0.16))
        rb._collection("astronaut").objects.link(lo)


def _limb(parent, a: Vector, b: Vector, r: float) -> None:
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=r, depth=d.length, location=(a + b) / 2)
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    rb._finish(o, parent, "suit_fabric", bevel=0.02, segments=3)
    rb._sphere(parent, "suit_fabric", a, r * 1.02)


def build_cable_reel(item: dict) -> None:
    x, y, z = item["pos"]
    with collection("decor"):
        root = _empty("cable_reel", "decor", (x, y, z + 0.45), Quaternion((0, 0, 1), item["yaw"]))
        for s in (-1, 1):
            rb._cyl(root, "aluminum", (0, s * 0.32, 0), 0.45, 0.03, axis="y", bevel=0.005)
        rb._cyl(root, "cable_orange", (0, 0, 0), 0.32, 0.6, axis="y", bevel=0.02)
        rb._cyl(root, "anodized_black", (0, 0, 0), 0.06, 0.72, axis="y")


def build_comms_mast(item: dict) -> None:
    """A lattice comms tower, knocked askew by the quake, with its dish still pointing at Earth."""
    x, y, z = item["pos"]
    with collection("decor"):
        root = _empty("comms_tower", "decor", (x, y, z), Quaternion((0, 0, 1), item["yaw"]))
        tilt = _empty("comms_tilt", "decor", (0, 0, 0.18), Euler((0.22, -0.08, 0)).to_quaternion())
        tilt.parent = root
        rb._box(root, "panel_white", (0, 0, 0.09), (0.45, 0.45, 0.09), bevel=0.02)
        rb._box(root, "hazard_orange", (0.451, 0, 0.09), (0.002, 0.3, 0.03))
        h, w = 3.2, 0.16
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            _strut(tilt, Vector((sx * w, sy * w, 0)), Vector((sx * w * 0.6, sy * w * 0.6, h)), 0.016)
        bays = 8
        for k in range(bays):
            z0, z1 = h * k / bays, h * (k + 1) / bays
            f0, f1 = 1 - 0.4 * k / bays, 1 - 0.4 * (k + 1) / bays
            c = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
            for i in range(4):
                a, bb = c[i], c[(i + 1) % 4]
                _strut(tilt, Vector((a[0] * w * f0, a[1] * w * f0, z0)), Vector((bb[0] * w * f1, bb[1] * w * f1, z1)), 0.008)
                _strut(tilt, Vector((a[0] * w * f1, a[1] * w * f1, z1)), Vector((bb[0] * w * f1, bb[1] * w * f1, z1)), 0.008)
        # dish on a gimbal at the top
        top = Vector((0, 0, h + 0.1))
        rb._cyl(tilt, "anodized_black", top, 0.06, 0.2, bevel=0.01)
        dish = _empty("comms_dish", "decor", top + Vector((0.0, 0.0, 0.25)), Euler((0.0, -1.1, 0.4)).to_quaternion())
        dish.parent = tilt
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.7, location=(0, 0, 0))
        d = bpy.context.active_object
        for v in d.data.vertices:
            v.co.z = 0.18 * (v.co.x ** 2 + v.co.y ** 2) / 0.49 * 0.7 if v.co.z < 0 else 0.18 * (v.co.x ** 2 + v.co.y ** 2) / 0.49 * 0.7 + 0.01
        rb._finish(d, dish, "panel_white")
        d.modifiers.new("solid", "SOLIDIFY").thickness = 0.02
        for k in range(3):
            a = 2 * math.pi * k / 3
            _strut(dish, Vector((0.6 * math.cos(a), 0.6 * math.sin(a), 0.13)), Vector((0, 0, 0.55)), 0.006)
        rb._cyl(dish, "anodized_black", (0, 0, 0.56), 0.04, 0.06)
        rb._cyl(dish, "lamp_cool", (0, 0, 0.6), 0.012, 0.02)


def build_deck(item: dict) -> None:
    """The service deck hanging under the collapsed module (the low passage)."""
    hx, hy, hz = item["size"]
    with collection("habitat"):
        root = _empty("hab_deck", "habitat", item["pos"])
        rb._box(root, "aluminum", (0, 0, 0), (hx, hy, hz), bevel=0.008)
        # underside ribs and edge beams
        n = max(2, int(hx * 2 / 0.6))
        for k in range(n + 1):
            x = -hx + 2 * hx * k / n
            rb._box(root, "anodized_black", (x, 0, -hz - 0.012), (0.02, hy * 0.98, 0.014), bevel=0.004)
        for s in (-1, 1):
            rb._box(root, "paint_white", (0, s * hy, 0.0), (hx, 0.03, hz + 0.03), bevel=0.01)
            rb._box(root, "hazard_orange", (s * hx, 0, 0.0), (0.012, hy, hz + 0.012), bevel=0.004)
        # hangers up to the hull
        for sx in (-0.75, 0.75):
            for sy in (-0.8, -0.3, 0.3, 0.8):
                p0 = Vector((sx * hx, sy * hy, hz))
                _strut(root, p0, p0 + Vector((-0.25 * sx * hx, 0, 0.55)), 0.018)


def scatter_slope_rocks(scene: dict, heights: np.ndarray, seed: int = 33) -> None:
    """Visual rocks on the rocky slope, outside the walking lane (the physics rocks are separate)."""
    P = scene["params"]
    if not P.get("ramp_run"):
        return
    hf = scene["hfield"]
    rng = np.random.default_rng(seed)
    nrow, ncol = heights.shape
    with collection("boulders"):
        for k in range(90):
            x = float(rng.uniform(P["scarp_x"] - 0.8, P["scarp_x"] + P["ramp_run"] + 1.5))
            y = float(rng.uniform(1.25, 4.9)) * (1 if rng.random() < 0.5 else -1)
            r = float(np.clip(0.07 * (rng.pareto(2.2) + 1), 0.07, 0.5))
            c = int(np.clip(round((x - hf["x0"]) / hf["res"]), 0, ncol - 1))
            row = int(np.clip(round((y - hf["y0"]) / hf["res"]), 0, nrow - 1))
            o = _rock_mesh(7000 + k, f"slope_rock_{k}")
            o.location = (x, y, float(heights[row, c]) + r * 0.15)
            o.scale = (r * rng.uniform(0.9, 1.3), r * rng.uniform(0.8, 1.2), r * rng.uniform(0.7, 1.0))
            o.rotation_euler = (0, 0, float(rng.uniform(0, 6.28)))
            for coll in list(o.users_collection):
                coll.objects.unlink(o)
            rb._collection("boulders").objects.link(o)


def build_base(scene: dict, heights: np.ndarray, vista: dict, logo_path: str) -> dict[str, bpy.types.Object]:
    build_terrain(scene, heights, vista)
    scatter_slope_rocks(scene, heights)
    for s in scene["static"]:
        if s["name"] == "hab_deck":
            build_deck(s)
    for item in scene["decor"]:
        k = item["kind"]
        if k == "landing_pad":
            build_landing_pad(item, logo_path)
        elif k == "hab_module":
            build_hab(item, logo_path)
        elif k == "solar_tower":
            build_solar_tower(item, scene["static"])
        elif k == "astronaut":
            build_astronaut(item)
        elif k == "cable_reel":
            build_cable_reel(item)
        elif k == "comms_mast":
            build_comms_mast(item)
    build_boulders(scene["static"])
    build_work_lights(scene)
    scatter_pebbles(scene, heights, vista)
    scatter_talus(scene, heights, vista)
    return build_loose(scene)


__all__ = ["build_base", "random"]
