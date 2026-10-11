"""Export the Blender models as visual meshes for MuJoCo's renderer.

Every training episode is recorded with MuJoCo (fast, from recorded states). These
meshes make those recordings show the same robot and structures as the Blender
showcase. Physics is unaffected: the meshes become visual-only geoms.

Meshes are written per (owner, material) in the owner's local frame:
* robot bodies: the body frame;
* structures: the root frame of each structure kind, so any course layout can
  place them;
* rocks, crates and panels: unit-sized prototypes scaled per instance.

Runs inside Blender only.
"""

from __future__ import annotations

import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

STRUCTURE_ROOTS = ("habitat", "solar_tower", "astronaut", "comms_tower", "lander", "cable_reel")


def _mesh_arrays(obj, to_local: Matrix, depsgraph):
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    if n == 0:
        ev.to_mesh_clear()
        return None
    m = to_local @ obj.matrix_world
    co = np.array([(m @ v.co)[:] for v in me.vertices], np.float32)
    tri = np.zeros(n * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    uv = None
    if me.uv_layers.active is not None:
        loops = np.zeros(n * 3, np.int32)
        me.loop_triangles.foreach_get("loops", loops)
        luv = np.zeros(len(me.loops) * 2, np.float32)
        me.uv_layers.active.data.foreach_get("uv", luv)
        luv = luv.reshape(-1, 2)
        # unweld so every triangle corner has its own UV
        co = co[tri.ravel()]
        uv = luv[loops]
        tri = np.arange(n * 3, dtype=np.int32).reshape(-1, 3)
    mats = np.zeros(n, np.int32)
    me.loop_triangles.foreach_get("material_index", mats)
    names = [s.material.name if s.material else "default" for s in obj.material_slots] or ["default"]
    ev.to_mesh_clear()
    return co, tri, uv, mats, names


def _write_obj(path: Path, co, tri, uv) -> None:
    lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in co]
    if uv is not None:
        lines += [f"vt {u:.5f} {v:.5f}" for u, v in uv]
        lines += [f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}" for a, b, c in tri]
    else:
        lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in tri]
    path.write_text("\n".join(lines) + "\n")


class _Collector:
    def __init__(self):
        self.parts: dict[tuple[str, str], list] = {}

    def add(self, owner: str, obj, to_local: Matrix, depsgraph, force: bool = False) -> None:
        if obj.type not in ("MESH", "FONT", "CURVE") or (obj.hide_render and not force):
            return
        arr = _mesh_arrays(obj, to_local, depsgraph)
        if arr is None:
            return
        co, tri, uv, mats, names = arr
        for mi in np.unique(mats):
            sel = tri[mats == mi]
            used = np.unique(sel)
            remap = -np.ones(len(co), np.int64)
            remap[used] = np.arange(len(used))
            key = (owner, names[min(mi, len(names) - 1)])
            self.parts.setdefault(key, []).append((co[used], remap[sel], uv[used] if uv is not None else None))

    def write(self, out: Path, decimate_above: int = 60000) -> list[dict]:
        out.mkdir(parents=True, exist_ok=True)
        manifest = []
        for (owner, mat), chunks in self.parts.items():
            vs, ts, uvs, off = [], [], [], 0
            has_uv = all(c[2] is not None for c in chunks)
            for co, tri, uv in chunks:
                vs.append(co)
                ts.append(tri + off)
                if has_uv:
                    uvs.append(uv)
                off += len(co)
            co, tri = np.concatenate(vs), np.concatenate(ts)
            uv = np.concatenate(uvs) if has_uv else None
            if len(tri) > decimate_above:
                keep = np.random.default_rng(0).choice(len(tri), decimate_above, replace=False)
                tri = tri[np.sort(keep)]  # coarse fallback; meshes are pre-decimated in Blender
            name = f"{owner}__{mat}".replace(".", "_")
            _write_obj(out / f"{name}.obj", co, tri, uv)
            manifest.append({"mesh": f"{name}.obj", "owner": owner, "material": mat,
                             "triangles": int(len(tri)), "uv": uv is not None})
        return manifest


def _decimate_heavy(objs, limit: int = 3000) -> None:
    dg = bpy.context.evaluated_depsgraph_get()
    counts = [(o, len(o.evaluated_get(dg).data.polygons)) for o in objs if o.type == "MESH"]
    for o, n in counts:
        if n > limit:
            m = o.modifiers.new("export_decimate", "DECIMATE")
            m.ratio = max(0.08, limit / n)


def export_visuals(out_dir: str, robot_empties: dict, loose: dict, scene: dict) -> dict:
    out = Path(out_dir)
    col = _Collector()
    wanted = {"robot", "habitat", "solar_tower", "astronaut", "decor", "lander", "pad", "loose"}
    all_objs = [o for c in bpy.data.collections if c.name in wanted for o in c.objects if o.type == "MESH"]
    _decimate_heavy(all_objs)
    dg = bpy.context.evaluated_depsgraph_get()

    # robot: each body's parts in the body frame
    for name, e in robot_empties.items():
        inv = e.matrix_world.inverted()
        for child in e.children:
            col.add(f"robot.{name}", child, inv, dg)

    # structures with a root empty
    for root_name in STRUCTURE_ROOTS:
        roots = [o for o in bpy.data.objects if o.name == root_name or o.name.startswith(root_name + ".")]
        if not roots:
            continue
        root = roots[0]
        inv = root.matrix_world.inverted()
        for child in root.children_recursive:
            col.add(f"struct.{root_name}", child, inv, dg)
    # habitat rubble mounds and the pad are not parented: export in their frames
    hab = bpy.data.objects.get("habitat")
    if hab is not None:
        inv = hab.matrix_world.inverted()
        inside = set(hab.children_recursive)
        for o in bpy.data.collections["habitat"].objects:
            if o.parent is None and o not in inside and o is not hab:
                col.add("struct.habitat", o, inv, dg)
    for o in bpy.data.collections.get("pad", bpy.data.collections.new("_")).objects:
        col.add("struct.pad", o, Matrix.Identity(4), dg)

    # unit prototypes: rock (radius 1), crate (half extents 1), panel (half extents 1)
    rock_protos = sorted((o for o in bpy.data.objects if o.name.startswith("talus_proto_")), key=lambda o: o.name)
    for i, o in enumerate(rock_protos[:4]):
        col.add(f"proto.rock{i}", o, o.matrix_world.inverted(), dg, force=True)
    for b in scene["loose"]:
        e = loose.get(b["name"])
        if e is None:
            continue
        if b["kind"] == "crate" and not any(k[0] == "proto.crate" for k in col.parts):
            hx, hy, hz = b["size"]
            inv = Matrix.Diagonal((1 / hx, 1 / hy, 1 / hz, 1)) @ e.matrix_world.inverted()
            for child in e.children:
                col.add("proto.crate", child, inv, dg)
        if b["kind"] == "panel" and not any(k[0] == "proto.panel" for k in col.parts):
            hx, hy, hz = b["size"]
            inv = Matrix.Diagonal((1 / hx, 1 / hy, 1 / (hz * 1.0), 1)) @ e.matrix_world.inverted()
            for child in e.children:
                col.add("proto.panel", child, inv, dg)
    sp = [s for s in scene["static"] if s["kind"] == "solar_panel"]
    if sp:
        s = sp[0]
        hx, hy, hz = s["size"]
        from mathutils import Quaternion, Vector
        frame = Matrix.Translation(Vector(s["pos"])) @ Quaternion(s["quat"]).to_matrix().to_4x4()
        inv = Matrix.Diagonal((1 / hx, 1 / hy, 1 / hz, 1)) @ frame.inverted()
        for o in bpy.data.collections["solar_tower"].objects:
            if o.parent is None and o.type == "MESH":
                d = (o.matrix_world.translation - Vector(s["pos"])).length
                if d < 0.02:
                    col.add("proto.solar_panel", o, inv, dg)

    manifest = col.write(out)
    # reference frames used when exporting, so placement can be checked
    frames = {}
    for root_name in STRUCTURE_ROOTS + ("habitat",):
        o = bpy.data.objects.get(root_name)
        if o is not None:
            frames[root_name] = {"location": list(o.matrix_world.translation),
                                 "quaternion": list(o.matrix_world.to_quaternion())}
    info = {"meshes": manifest, "frames": frames,
            "materials": sorted({m["material"] for m in manifest}),
            "export_params": {"hab_radius": scene["params"]["hab_radius"]}}
    (out / "manifest.json").write_text(json.dumps(info, indent=1))
    return info
