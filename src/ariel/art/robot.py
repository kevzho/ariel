"""Build the detailed Ariel humanoid in Blender around the physics geoms.

Each MuJoCo body becomes an Empty; its detailed parts are children modelled in
that body's frame, wrapping the collision shapes of the same body. Animation
then only needs one world pose per body per frame. Runs inside Blender only.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Euler, Quaternion, Vector

from .materials import material

COLL = "robot"
_SHELLS: dict = {}


def _collection(name: str) -> bpy.types.Collection:
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def _finish(obj, parent, mat: str, bevel: float = 0.0, segments: int = 3, smooth: bool = True,
            subsurf: int = 0, harden: bool = True):
    coll = _collection(COLL)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    obj.parent = parent
    obj.data.materials.clear()
    obj.data.materials.append(material(mat))
    if smooth and obj.type == "MESH":
        obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    if bevel > 0:
        m = obj.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = segments
        m.limit_method = "ANGLE"
        m.angle_limit = math.radians(35)
        m.harden_normals = harden
    if subsurf:
        m = obj.modifiers.new("subsurf", "SUBSURF")
        m.levels = subsurf
        m.render_levels = subsurf
    return obj


def _box(parent, mat, center, half, bevel=0.01, rot=(0, 0, 0), segments=3):
    bpy.ops.mesh.primitive_cube_add(size=2, location=center, rotation=rot)
    o = bpy.context.active_object
    o.scale = half
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return _finish(o, parent, mat, bevel=bevel, segments=segments)


def _cyl(parent, mat, center, radius, depth, axis="z", verts=48, bevel=0.0, r2=None, segments=3):
    rot = {"z": (0, 0, 0), "y": (math.pi / 2, 0, 0), "x": (0, math.pi / 2, 0)}[axis]
    if r2 is None:
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth,
                                            location=center, rotation=rot)
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r2, radius2=radius, depth=depth,
                                        location=center, rotation=rot)
    return _finish(bpy.context.active_object, parent, mat, bevel=bevel, segments=segments)


def _ring(parent, mat, center, radius, minor, axis="z"):
    rot = {"z": (0, 0, 0), "y": (math.pi / 2, 0, 0), "x": (0, math.pi / 2, 0)}[axis]
    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=minor, major_segments=64,
                                     minor_segments=12, location=center, rotation=rot)
    return _finish(bpy.context.active_object, parent, mat)


def _sphere(parent, mat, center, radius, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=center)
    o = bpy.context.active_object
    o.scale = scale
    return _finish(o, parent, mat)


def _text(parent, text, center, size, rot=(math.pi / 2, 0, math.pi / 2), mat="decal_black", extrude=0.0006):
    cu = bpy.data.curves.new(f"txt_{text}", "FONT")
    cu.body = text
    cu.size = size
    cu.extrude = extrude
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.space_character = 1.15
    o = bpy.data.objects.new(f"txt_{text}", cu)
    bpy.context.scene.collection.objects.link(o)
    o.location = center
    o.rotation_euler = rot
    o.data.materials.append(material(mat))
    coll = _collection(COLL)
    for c in list(o.users_collection):
        c.objects.unlink(o)
    coll.objects.link(o)
    o.parent = parent
    return o


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _actuator(parent, center, radius, width, axis="y", cap_side=1, cap=True):
    """A joint actuator drum: anodized housing, gold MLI wrap and a white end cap."""
    c = Vector(center)
    _cyl(parent, "anodized_black", c, radius, width, axis=axis, bevel=0.004)
    _cyl(parent, "foil_gold", c, radius * 1.03, width * 0.55, axis=axis, verts=64)
    if cap:
        off = Vector({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]) * (width / 2 + 0.004) * cap_side
        _cyl(parent, "panel_white", c + off, radius * 0.82, 0.012, axis=axis, bevel=0.003)
        _cyl(parent, "anodized_black", c + off * 1.12, radius * 0.35, 0.006, axis=axis, bevel=0.0015)
        for k in range(8):
            a = 2 * math.pi * k / 8
            r = radius * 0.62
            if axis == "y":
                p = c + off * 1.1 + Vector((r * math.cos(a), 0, r * math.sin(a)))
            elif axis == "x":
                p = c + off * 1.1 + Vector((0, r * math.cos(a), r * math.sin(a)))
            else:
                p = c + off * 1.1 + Vector((r * math.cos(a), r * math.sin(a), 0))
            _cyl(parent, "chrome_tube", p, 0.0045, 0.006, axis=axis, verts=12)


def _sleeve(parent, top, bottom, radius, accent=False, stripe_side=1, guard=None):
    """Telescoping outer sleeve: tapered white shell, clamp bands and a wiper seal."""
    top, bottom = Vector(top), Vector(bottom)
    length = (top - bottom).length
    mid = (top + bottom) / 2
    _cyl(parent, "panel_white", mid, radius * 1.02, length, r2=radius * 0.9, bevel=0.008, segments=4)
    _ring(parent, "anodized_black", top + Vector((0, 0, -0.012)), radius * 1.0, 0.004)
    _ring(parent, "anodized_black", bottom + Vector((0, 0, 0.02)), radius * 0.92, 0.0035)
    # wiper seal at the mouth where the polished tube emerges
    _cyl(parent, "rubber", bottom + Vector((0, 0, -0.004)), radius * 0.86, 0.016, bevel=0.004)
    if accent:
        _box(parent, "accent_orange", mid + Vector((0, stripe_side * radius * 0.97, 0.01)),
             (0.012, 0.004, length * 0.32), bevel=0.002)
    if guard:
        _box(parent, "panel_white", mid + Vector((radius * 0.95, 0, 0.0)),
             (0.012, radius * 0.75, length * 0.42), bevel=0.01, segments=4)


def _tube(parent, top, bottom, radius):
    top, bottom = Vector(top), Vector(bottom)
    _cyl(parent, "chrome_tube", (top + bottom) / 2, radius, (top - bottom).length, bevel=0.003, verts=64)
    _ring(parent, "anodized_black", bottom + Vector((0, 0, 0.012)), radius * 1.05, 0.004)


def _capsule_ends(geom):
    p = Vector(geom["pos"])
    half = geom["size"][1]
    q = Quaternion(geom["quat"])
    axis = q @ Vector((0, 0, 1))
    return p + axis * half, p - axis * half, geom["size"][0]


def build_part(parent, body_name: str, geom: dict) -> None:
    n = geom["name"]
    side = 1 if n.startswith("left_") else -1
    base = n.replace("left_", "").replace("right_", "")
    pos = Vector(geom["pos"])
    size = geom["size"]

    if base == "pelvis":
        # sculpted hip girdle: wider at the top, white covers over a black core
        bpy.ops.mesh.primitive_cube_add(size=2, location=pos)
        o = bpy.context.active_object
        o.scale = (0.092, 0.148, 0.066)
        bpy.ops.object.transform_apply(scale=True)
        for v in o.data.vertices:
            if v.co.z < pos.z:
                v.co.y = pos.y + (v.co.y - pos.y) * 0.72
                v.co.x = pos.x + (v.co.x - pos.x) * 0.9
        _finish(o, parent, "panel_white", bevel=0.028, segments=4, subsurf=1)
        _box(parent, "anodized_black", pos + Vector((0.0, 0, -0.066)), (0.06, 0.1, 0.008), bevel=0.006)
        _box(parent, "accent_orange", pos + Vector((0.093, 0, 0.042)), (0.003, 0.06, 0.006), bevel=0.002)
        # outer stage of the telescoping spine (the chest's chrome stage slides inside it)
        _cyl(parent, "panel_white", pos + Vector((0, 0, 0.115)), 0.072, 0.09, bevel=0.01, segments=4)
        _cyl(parent, "foil_gold", pos + Vector((0, 0, 0.1)), 0.0735, 0.03, verts=64)
        _cyl(parent, "rubber", pos + Vector((0, 0, 0.162)), 0.064, 0.012, bevel=0.004)
    elif base == "torso_tube":
        pass  # hidden inside the two visual spine stages
    elif base == "chest":
        c = pos
        # V-tapered torso shell with a slightly domed front
        bpy.ops.mesh.primitive_cube_add(size=2, location=c)
        o = bpy.context.active_object
        o.scale = (0.12, 0.18, 0.155)
        bpy.ops.object.transform_apply(scale=True)
        for v in o.data.vertices:
            dz = (v.co.z - c.z) / 0.155
            k = 0.74 + 0.26 * (dz + 1) / 2
            v.co.y = c.y + (v.co.y - c.y) * k
            v.co.x = c.x + (v.co.x - c.x) * (0.88 + 0.12 * (dz + 1) / 2)
        _finish(o, parent, "panel_white", bevel=0.04, segments=5, subsurf=1)
        # raised chest plate
        _box(parent, "panel_white", c + Vector((0.112, 0, 0.055)), (0.012, 0.13, 0.07), bevel=0.012, segments=4)
        _box(parent, "anodized_black", c + Vector((0.108, 0, -0.03)), (0.006, 0.1, 0.004), bevel=0.002)
        # mission patch on the left chest, name below it
        _cyl(parent, "patch_navy", c + Vector((0.125, 0.07, 0.07)), 0.03, 0.003, axis="x", verts=64)
        _ring(parent, "foil_gold", c + Vector((0.1265, 0.07, 0.07)), 0.03, 0.0022, axis="x")
        _box(parent, "panel_white", c + Vector((0.127, 0.07, 0.07)), (0.0008, 0.018, 0.0035))
        _box(parent, "accent_orange", c + Vector((0.127, 0.07, 0.078)), (0.0008, 0.011, 0.0025))
        _text(parent, "Ariel", c + Vector((0.1255, 0.07, 0.022)), 0.016)
        for k in range(3):
            _box(parent, "accent_orange", c + Vector((0.1255, -0.075, 0.09 - 0.012 * k)),
                 (0.0015, 0.022 - 0.004 * k, 0.0035), bevel=0.001)
        # ribbed abdomen and shoulder yoke
        for k in range(3):
            _ring(parent, "rubber", c + Vector((0, 0, 0.005 + 0.014 * k)), 0.082, 0.007).scale = (1.05, 1.4, 1)
        _box(parent, "anodized_black", c + Vector((-0.01, 0, 0.13)), (0.075, 0.185, 0.018), bevel=0.012, segments=3)
        for s in (1, -1):
            for k in range(6):
                _box(parent, "anodized_black", c + Vector((-0.045 + 0.017 * k, s * 0.16, -0.0)),
                     (0.0028, 0.008, 0.07), bevel=0.0012)
    elif base == "waist_housing":
        _cyl(parent, "anodized_black", pos, 0.078, 0.06, bevel=0.006)
        _cyl(parent, "foil_gold", pos, 0.0805, 0.028, verts=64)
        # inner chrome stage of the telescoping spine
        _cyl(parent, "chrome_tube", pos + Vector((0, 0, -0.1)), 0.057, 0.14, bevel=0.003, verts=64)
        for k in range(2):
            _ring(parent, "anodized_black", pos + Vector((0, 0, -0.06 - 0.05 * k)), 0.058, 0.0035)
    elif base == "power_pack":
        _box(parent, "panel_white", pos, (0.054, 0.132, 0.122), bevel=0.02, segments=4)
        _box(parent, "foil_gold", pos + Vector((-0.012, 0, -0.01)), (0.046, 0.118, 0.1), bevel=0.012, segments=3)
        for k in range(9):
            _box(parent, "anodized_black", pos + Vector((-0.06, -0.096 + 0.024 * k, -0.01)),
                 (0.004, 0.006, 0.085), bevel=0.0015)
        _cyl(parent, "anodized_black", pos + Vector((0.0, -0.1, 0.22)), 0.004, 0.2, verts=12)
        _sphere(parent, "anodized_black", pos + Vector((0.0, -0.1, 0.32)), 0.007)
    elif base == "beacon":
        _cyl(parent, "anodized_black", pos + Vector((0, 0, -0.025)), 0.024, 0.012, bevel=0.003)
        _cyl(parent, "beacon_glow", pos, 0.019, 0.04, bevel=0.006)
    elif base == "head":
        c = pos
        # rounded sensor helmet, narrower at the jaw
        bpy.ops.mesh.primitive_cube_add(size=2, location=c)
        o = bpy.context.active_object
        o.scale = (0.1, 0.088, 0.094)
        bpy.ops.object.transform_apply(scale=True)
        for v in o.data.vertices:
            if v.co.z < c.z:
                v.co.y = c.y + (v.co.y - c.y) * 0.84
                v.co.x = c.x + (v.co.x - c.x) * 0.92
        _finish(o, parent, "panel_white", bevel=0.035, segments=4, subsurf=2)
        _SHELLS["helmet"] = o
        # neck
        _cyl(parent, "anodized_black", c + Vector((0, 0, -0.125)), 0.048, 0.05, bevel=0.004)
        for k in range(3):
            _ring(parent, "rubber", c + Vector((0, 0, -0.14 + 0.014 * k)), 0.049, 0.005)
        # light bars under the visor
        for s in (1, -1):
            _box(parent, "lamp", c + Vector((0.093, s * 0.045, -0.045)), (0.003, 0.02, 0.004), bevel=0.0015)
    elif base == "visor":
        # wraparound visor band built directly as geometry around the helmet front
        _visor_band(parent, _SHELLS["helmet"], z_offset=pos.z + 0.008 - _SHELLS["helmet"].location.z,
                    height=0.068, span_deg=80)
    elif base == "lidar":
        _cyl(parent, "anodized_black", pos + Vector((0, 0, -0.006)), 0.04, 0.02, bevel=0.004)
        _cyl(parent, "glass_dark", pos + Vector((0, 0, 0.011)), 0.035, 0.016, verts=64)
        _cyl(parent, "anodized_black", pos + Vector((0, 0, 0.022)), 0.037, 0.007, bevel=0.003)
    elif base == "shoulder_housing":
        _actuator(parent, pos, 0.058, 0.1, axis="y", cap_side=side)
        _sphere(parent, "panel_white", pos + Vector((0, side * 0.01, 0.02)), 0.07, scale=(1.0, 0.9, 0.75))
    elif base in ("upper_arm_sleeve", "forearm_sleeve", "thigh_sleeve", "shin_sleeve"):
        top, bottom, r = _capsule_ends(geom)
        top = top + Vector((0, 0, -0.01))
        _sleeve(parent, top, bottom, r, accent=base in ("thigh_sleeve", "upper_arm_sleeve"),
                stripe_side=side, guard=base == "shin_sleeve")
    elif base in ("upper_arm_tube", "forearm_tube", "thigh_tube", "shin_tube"):
        top, bottom, r = _capsule_ends(geom)
        _tube(parent, top, bottom, r)
    elif base == "elbow_housing":
        _actuator(parent, pos, 0.044, 0.08, axis="y", cap_side=side)
    elif base == "hip_housing":
        _actuator(parent, pos, 0.066, 0.12, axis="y", cap_side=side)
        _cyl(parent, "anodized_black", pos + Vector((0, 0, 0.05)), 0.05, 0.04, bevel=0.005)
    elif base == "knee_housing":
        _actuator(parent, pos, 0.052, 0.1, axis="y", cap_side=side, cap=False)
        _sphere(parent, "panel_white", pos + Vector((0.035, 0, 0.0)), 0.05, scale=(0.6, 1.0, 1.15))
    elif base == "hand":
        c = pos
        _box(parent, "panel_white", c + Vector((-0.005, 0, 0.01)), (0.04, 0.03, 0.045), bevel=0.018, segments=4)
        _box(parent, "rubber", c + Vector((0.012, 0, -0.025)), (0.035, 0.027, 0.04), bevel=0.016, segments=4)
        for k in range(3):
            _box(parent, "anodized_black", c + Vector((0.046, -0.016 + 0.016 * k, -0.035)),
                 (0.002, 0.0055, 0.03), bevel=0.0015)
        _cyl(parent, "anodized_black", c + Vector((0, 0, 0.05)), 0.03, 0.02, bevel=0.003)
    elif base == "foot":
        c = pos
        _box(parent, "rubber", c + Vector((0, 0, -0.012)), (0.125, 0.058, 0.014), bevel=0.012, segments=3)
        for k in range(9):
            _box(parent, "rubber", c + Vector((-0.1 + 0.025 * k, 0, -0.027)), (0.006, 0.05, 0.003), bevel=0.002)
        _box(parent, "panel_white", c + Vector((0.01, 0, 0.012)), (0.108, 0.052, 0.018), bevel=0.016, segments=4)
        _box(parent, "accent_orange", c + Vector((0.113, 0, 0.0)), (0.004, 0.035, 0.008), bevel=0.002)
        _actuator(parent, Vector((0, 0, 0)), 0.04, 0.09, axis="y", cap_side=side)


def _visor_band(parent, helmet, z_offset: float, height: float, span_deg: float,
                rows: int = 10, cols: int = 80, standoff: float = 0.0025):
    """A visor strip projected onto the helmet surface, standing slightly proud of it."""
    import bmesh
    from mathutils.bvhtree import BVHTree

    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(helmet, dg)     # helmet-local coordinates
    centre = Vector((0.0, 0.0, 0.0))
    mesh = bpy.data.meshes.new("visor_band")
    bm = bmesh.new()
    grid = []
    for r in range(rows + 1):
        v = r / rows - 0.5
        row = []
        for c in range(cols + 1):
            a = math.radians(-span_deg + 2 * span_deg * c / cols)
            d = Vector((math.cos(a), math.sin(a), 0.0))
            origin = centre + Vector((0, 0, z_offset + v * height))
            hit, normal, _, _ = tree.ray_cast(origin, d, 1.0)
            if hit is None:
                hit, normal = origin + d * 0.1, d
            bulge = standoff + 0.004 * (1 - (2 * v) ** 2)
            p = hit + normal.normalized() * bulge + helmet.location
            row.append(bm.verts.new(p))
        grid.append(row)
    for r in range(rows):
        for c in range(cols):
            bm.faces.new((grid[r][c], grid[r][c + 1], grid[r + 1][c + 1], grid[r + 1][c]))
    frame_edges = []
    for r in (0, rows):
        frame_edges.append([grid[r][c].co.copy() for c in range(cols + 1)])
    bm.to_mesh(mesh)
    bm.free()
    o = bpy.data.objects.new("visor_band", mesh)
    bpy.context.scene.collection.objects.link(o)
    _finish(o, parent, "visor")
    sol = o.modifiers.new("solid", "SOLIDIFY")
    sol.thickness = 0.004
    sol.offset = -1.0
    # dark frame along the top and bottom edges of the visor
    frame = bpy.data.meshes.new("visor_frame")
    bm = bmesh.new()
    for line in frame_edges:
        prev = None
        for co in line:
            vtx = bm.verts.new(co)
            if prev is not None:
                bm.edges.new((prev, vtx))
            prev = vtx
    bm.to_mesh(frame)
    bm.free()
    fo = bpy.data.objects.new("visor_frame", frame)
    bpy.context.scene.collection.objects.link(fo)
    _finish(fo, parent, "anodized_black", smooth=False)
    skin = fo.modifiers.new("tube", "SKIN")
    for vtx in fo.data.skin_vertices[0].data:
        vtx.radius = (0.0028, 0.0028)
    sub = fo.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1


def build_robot(rig: dict) -> dict[str, bpy.types.Object]:
    """Create one Empty per body and its detailed parts. Returns body name -> Empty."""
    empties = {}
    coll = _collection(COLL)
    for body in rig["bodies"]:
        e = bpy.data.objects.new(body["name"], None)
        e.empty_display_size = 0.05
        e.rotation_mode = "QUATERNION"
        coll.objects.link(e)
        empties[body["name"]] = e
        for geom in body["geoms"]:
            build_part(e, body["name"], geom)
    return empties


def set_pose(empties: dict, names: list[str], pos, quat, frame: int | None = None) -> None:
    for i, name in enumerate(names):
        e = empties.get(name)
        if e is None:
            continue
        e.location = tuple(float(x) for x in pos[i])
        e.rotation_quaternion = tuple(float(x) for x in quat[i])
        if frame is not None:
            e.keyframe_insert("location", frame=frame)
            e.keyframe_insert("rotation_quaternion", frame=frame)


__all__ = ["build_robot", "set_pose", "Euler"]
