"""Physically based materials for Blender (Cycles). Runs inside Blender only."""

from __future__ import annotations

import bpy

_CACHE: dict[str, bpy.types.Material] = {}


def _sock(node, name, kind=None, out=False):
    socks = node.outputs if out else node.inputs
    for s in socks:
        if s.name == name and (kind is None or s.type == kind):
            return s
    raise KeyError(f"{node.bl_idname} has no socket {name!r} ({kind})")


class _Graph:
    def __init__(self, name: str):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new("ShaderNodeOutputMaterial")
        self.bsdf = self.nt.nodes.new("ShaderNodeBsdfPrincipled")
        self.link(self.bsdf.outputs["BSDF"], self.out.inputs["Surface"])

    def node(self, kind: str, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def set(self, **inputs):
        for k, v in inputs.items():
            self.bsdf.inputs[k.replace("_", " ")].default_value = v

    def noise(self, scale, detail=6.0, roughness=0.55, coords="Object"):
        tc = self.node("ShaderNodeTexCoord")
        n = self.node("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = roughness
        self.link(tc.outputs[coords], n.inputs["Vector"])
        return n

    def map_range(self, value, a, b, c, d):
        m = self.node("ShaderNodeMapRange")
        self.link(value, m.inputs["Value"])
        m.inputs["From Min"].default_value = a
        m.inputs["From Max"].default_value = b
        m.inputs["To Min"].default_value = c
        m.inputs["To Max"].default_value = d
        return m.outputs["Result"]

    def mix_color(self, fac, col_a, col_b):
        m = self.node("ShaderNodeMix", data_type="RGBA")
        if isinstance(fac, float):
            _sock(m, "Factor", "VALUE").default_value = fac
        else:
            self.link(fac, _sock(m, "Factor", "VALUE"))
        for sock_name, col in (("A", col_a), ("B", col_b)):
            s = _sock(m, sock_name, "RGBA")
            if isinstance(col, tuple):
                s.default_value = (*col, 1.0) if len(col) == 3 else col
            else:
                self.link(col, s)
        return _sock(m, "Result", "RGBA", out=True)

    def bump(self, height, strength, distance=0.002, normal=None):
        b = self.node("ShaderNodeBump")
        b.inputs["Strength"].default_value = strength
        b.inputs["Distance"].default_value = distance
        self.link(height, b.inputs["Height"])
        if normal is not None:
            self.link(normal, b.inputs["Normal"])
        return b.outputs["Normal"]

    def dust(self, base_color, amount=0.35):
        """Regolith dust collects on upward-facing surfaces, broken up by noise."""
        geo = self.node("ShaderNodeNewGeometry")
        sep = self.node("ShaderNodeSeparateXYZ")
        self.link(geo.outputs["Normal"], sep.inputs["Vector"])
        up = self.map_range(sep.outputs["Z"], 0.2, 1.0, 0.0, 1.0)
        n = self.noise(9.0, detail=8.0, roughness=0.65)
        patch = self.map_range(n.outputs["Fac"], 0.42, 0.68, 0.0, 1.0)
        mul = self.node("ShaderNodeMath", operation="MULTIPLY")
        self.link(up, mul.inputs[0])
        self.link(patch, mul.inputs[1])
        scale = self.node("ShaderNodeMath", operation="MULTIPLY")
        self.link(mul.outputs[0], scale.inputs[0])
        scale.inputs[1].default_value = amount
        col = self.mix_color(scale.outputs[0], base_color, (0.42, 0.40, 0.37))
        return col, scale.outputs[0]


def material(name: str) -> bpy.types.Material:
    if name in _CACHE:
        return _CACHE[name]
    g = _Graph(name)
    if name == "panel_white":
        # white thermal-control paint: slightly warm, satin, with dust and wear
        n = g.noise(40.0, detail=4.0)
        tint = g.mix_color(g.map_range(n.outputs["Fac"], 0.3, 0.7, 0.0, 1.0),
                           (0.80, 0.80, 0.78), (0.86, 0.86, 0.84))
        col, dust = g.dust(tint, 0.45)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.link(g.map_range(dust, 0.0, 1.0, 0.38, 0.85), g.bsdf.inputs["Roughness"])
        g.set(Coat_Weight=0.12, Coat_Roughness=0.25)
        g.link(g.bump(g.noise(300.0, 2.0).outputs["Fac"], 0.04), g.bsdf.inputs["Normal"])
    elif name == "foil_gold":
        # multi-layer insulation: crinkled metallised film
        tc = g.node("ShaderNodeTexCoord")
        vor = g.node("ShaderNodeTexVoronoi", feature="SMOOTH_F1")
        vor.inputs["Scale"].default_value = 55.0
        vor.inputs["Randomness"].default_value = 1.0
        g.link(tc.outputs["Object"], vor.inputs["Vector"])
        n = g.noise(120.0, detail=3.0)
        add = g.node("ShaderNodeMath", operation="ADD")
        g.link(vor.outputs["Distance"], add.inputs[0])
        g.link(n.outputs["Fac"], add.inputs[1])
        g.link(g.bump(add.outputs[0], 0.35, 0.003), g.bsdf.inputs["Normal"])
        col = g.mix_color(g.map_range(n.outputs["Fac"], 0.35, 0.65, 0, 1), (1.0, 0.70, 0.30), (0.95, 0.78, 0.42))
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Metallic=1.0, Roughness=0.24)
    elif name == "anodized_black":
        n = g.noise(60.0, 3.0)
        g.link(g.map_range(n.outputs["Fac"], 0.3, 0.7, 0.28, 0.42), g.bsdf.inputs["Roughness"])
        col, _ = g.dust((0.035, 0.035, 0.04), 0.3)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Metallic=0.75)
    elif name == "chrome_tube":
        g.set(Base_Color=(0.92, 0.92, 0.94, 1), Metallic=1.0, Roughness=0.07, Anisotropic=0.4)
        wave = g.node("ShaderNodeTexWave", wave_type="RINGS")
        wave.inputs["Scale"].default_value = 400.0
        g.link(g.bump(wave.outputs["Fac"], 0.015), g.bsdf.inputs["Normal"])
    elif name == "visor":
        # dark tinted glass with a faint gold interference coating
        g.set(Base_Color=(0.004, 0.004, 0.006, 1), Metallic=0.0, Roughness=0.04, IOR=1.52,
              Coat_Weight=1.0, Coat_Roughness=0.02, Thin_Film_Thickness=560.0, Thin_Film_IOR=2.0)
    elif name == "rubber":
        n = g.noise(80.0, 4.0)
        g.link(g.map_range(n.outputs["Fac"], 0.3, 0.7, 0.6, 0.85), g.bsdf.inputs["Roughness"])
        col, _ = g.dust((0.02, 0.02, 0.022), 0.5)
        g.link(col, g.bsdf.inputs["Base Color"])
    elif name == "accent_orange":
        col, _ = g.dust((0.95, 0.30, 0.04), 0.3)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Roughness=0.45, Coat_Weight=0.1)
    elif name == "lamp":
        g.set(Base_Color=(1, 1, 1, 1), Emission_Color=(1.0, 0.97, 0.9, 1), Emission_Strength=25.0)
    elif name == "beacon_glow":
        g.set(Base_Color=(1, 0.4, 0.05, 1), Emission_Color=(1.0, 0.35, 0.05, 1), Emission_Strength=12.0)
    elif name == "glass_dark":
        g.set(Base_Color=(0.02, 0.03, 0.04, 1), Metallic=0.0, Roughness=0.02, IOR=1.5,
              Transmission_Weight=0.3, Coat_Weight=1.0)
    elif name == "patch_navy":
        g.set(Base_Color=(0.02, 0.05, 0.16, 1), Roughness=0.6)
    elif name == "decal_black":
        g.set(Base_Color=(0.02, 0.02, 0.02, 1), Roughness=0.5)
    elif name == "regolith":
        # fine grey lunar soil, rough and dark with subtle colour variation
        n1 = g.noise(2.0, 8.0, 0.6)
        n2 = g.noise(35.0, 6.0, 0.7)
        col = g.mix_color(g.map_range(n1.outputs["Fac"], 0.35, 0.7, 0, 1), (0.30, 0.29, 0.275), (0.43, 0.42, 0.40))
        col = g.mix_color(g.map_range(n2.outputs["Fac"], 0.4, 0.65, 0, 0.35), col, (0.22, 0.215, 0.21))
        g.link(col, g.bsdf.inputs["Base Color"])
        h = g.node("ShaderNodeMath", operation="ADD")
        g.link(n2.outputs["Fac"], h.inputs[0])
        n3 = g.noise(240.0, 4.0, 0.7)
        g.link(n3.outputs["Fac"], h.inputs[1])
        g.link(g.bump(h.outputs[0], 0.6, 0.01), g.bsdf.inputs["Normal"])
        g.set(Roughness=0.97, Specular_IOR_Level=0.2)
    elif name == "rock":
        n1 = g.noise(6.0, 8.0, 0.65)
        col = g.mix_color(g.map_range(n1.outputs["Fac"], 0.35, 0.7, 0, 1), (0.18, 0.175, 0.17), (0.34, 0.33, 0.31))
        col, _ = g.dust(col, 0.6)
        g.link(col, g.bsdf.inputs["Base Color"])
        n2 = g.noise(25.0, 10.0, 0.7)
        g.link(g.bump(n2.outputs["Fac"], 0.8, 0.02), g.bsdf.inputs["Normal"])
        g.set(Roughness=0.93)
    elif name == "terrain":
        # regolith on gentle ground, exposed fractured rock on steep faces
        geo = g.node("ShaderNodeNewGeometry")
        sep = g.node("ShaderNodeSeparateXYZ")
        g.link(geo.outputs["Normal"], sep.inputs["Vector"])
        n0 = g.noise(3.0, 6.0, 0.6)
        slope = g.node("ShaderNodeMath", operation="ADD")
        g.link(sep.outputs["Z"], slope.inputs[0])
        g.link(g.map_range(n0.outputs["Fac"], 0.3, 0.7, -0.08, 0.08), slope.inputs[1])
        steep = g.map_range(slope.outputs[0], 0.55, 0.75, 1.0, 0.0)
        n1 = g.noise(1.5, 8.0, 0.6)
        n2 = g.noise(30.0, 6.0, 0.7)
        reg = g.mix_color(g.map_range(n1.outputs["Fac"], 0.35, 0.7, 0, 1), (0.30, 0.29, 0.275), (0.42, 0.41, 0.39))
        reg = g.mix_color(g.map_range(n2.outputs["Fac"], 0.4, 0.65, 0, 0.35), reg, (0.22, 0.215, 0.21))
        n3 = g.noise(9.0, 10.0, 0.75)
        rock = g.mix_color(g.map_range(n3.outputs["Fac"], 0.35, 0.7, 0, 1), (0.13, 0.125, 0.12), (0.27, 0.26, 0.25))
        # fractures: dark cracks along Voronoi cell edges; strata: faint horizontal banding
        tc = g.node("ShaderNodeTexCoord")
        # warp the cell pattern with noise so fractures wander, and show only some of them
        warp = g.noise(1.7, 4.0, 0.6)
        vadd = g.node("ShaderNodeVectorMath", operation="MULTIPLY_ADD")
        g.link(warp.outputs["Color"], vadd.inputs[0])
        vadd.inputs[1].default_value = (0.35, 0.35, 0.35)
        g.link(tc.outputs["Object"], vadd.inputs[2])
        vor = g.node("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
        vor.inputs["Scale"].default_value = 3.2
        vor.inputs["Randomness"].default_value = 1.0
        g.link(vadd.outputs["Vector"], vor.inputs["Vector"])
        crack = g.map_range(vor.outputs["Distance"], 0.0, 0.014, 1.0, 0.0)
        keep = g.map_range(g.noise(2.3, 3.0, 0.5).outputs["Fac"], 0.48, 0.58, 0.0, 1.0)
        cm = g.node("ShaderNodeMath", operation="MULTIPLY")
        g.link(crack, cm.inputs[0])
        g.link(keep, cm.inputs[1])
        crack = cm.outputs[0]
        rock = g.mix_color(g.map_range(crack, 0, 1, 0, 0.7), rock, (0.05, 0.05, 0.05))
        wave = g.node("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Z")
        wave.inputs["Scale"].default_value = 3.0
        wave.inputs["Distortion"].default_value = 6.0
        wave.inputs["Detail"].default_value = 4.0
        g.link(tc.outputs["Object"], wave.inputs["Vector"])
        rock = g.mix_color(g.map_range(wave.outputs["Fac"], 0.3, 0.9, 0.0, 0.25), rock, (0.33, 0.32, 0.3))
        g.link(g.mix_color(steep, reg, rock), g.bsdf.inputs["Base Color"])
        h = g.node("ShaderNodeMath", operation="ADD")
        g.link(n2.outputs["Fac"], h.inputs[0])
        g.link(g.noise(260.0, 4.0, 0.7).outputs["Fac"], h.inputs[1])
        hr = g.node("ShaderNodeMath", operation="ADD")
        g.link(h.outputs[0], hr.inputs[0])
        rh = g.node("ShaderNodeMath", operation="SUBTRACT")
        g.link(n3.outputs["Fac"], rh.inputs[0])
        g.link(crack, rh.inputs[1])
        mr = g.node("ShaderNodeMath", operation="MULTIPLY")
        g.link(rh.outputs[0], mr.inputs[0])
        g.link(steep, mr.inputs[1])
        g.link(mr.outputs[0], hr.inputs[1])
        g.link(g.bump(hr.outputs[0], 0.8, 0.02), g.bsdf.inputs["Normal"])
        g.set(Roughness=0.96, Specular_IOR_Level=0.2)
    elif name == "beta_cloth":
        # quilted multi-layer insulation blanket on the habitat
        tc = g.node("ShaderNodeTexCoord")
        brick = g.node("ShaderNodeTexBrick")
        brick.inputs["Scale"].default_value = 1.6
        brick.inputs["Mortar Size"].default_value = 0.012
        brick.inputs["Mortar Smooth"].default_value = 0.6
        brick.offset = 0.0
        g.link(tc.outputs["Object"], brick.inputs["Vector"])
        wr = g.noise(14.0, 6.0, 0.6)
        h = g.node("ShaderNodeMath", operation="ADD")
        g.link(brick.outputs["Fac"], h.inputs[0])
        sc = g.node("ShaderNodeMath", operation="MULTIPLY")
        g.link(wr.outputs["Fac"], sc.inputs[0])
        sc.inputs[1].default_value = 0.6
        g.link(sc.outputs[0], h.inputs[1])
        g.link(g.bump(h.outputs[0], 0.25, 0.01), g.bsdf.inputs["Normal"])
        col, dust = g.dust((0.84, 0.83, 0.79), 0.6)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Roughness=0.82, Sheen_Weight=0.3)
    elif name == "aluminum":
        n = g.noise(25.0, 4.0)
        g.link(g.map_range(n.outputs["Fac"], 0.3, 0.7, 0.28, 0.42), g.bsdf.inputs["Roughness"])
        col, _ = g.dust((0.78, 0.79, 0.8), 0.45)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Metallic=1.0, Anisotropic=0.3)
    elif name == "solar_cells":
        tc = g.node("ShaderNodeTexCoord")
        brick = g.node("ShaderNodeTexBrick")
        brick.inputs["Scale"].default_value = 6.0
        brick.inputs["Mortar Size"].default_value = 0.03
        brick.offset = 0.0
        brick.squash = 1.0
        g.link(tc.outputs["Object"], brick.inputs["Vector"])
        col = g.mix_color(brick.outputs["Fac"], (0.015, 0.03, 0.09), (0.55, 0.57, 0.6))
        col, _ = g.dust(col, 0.5)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.link(g.map_range(brick.outputs["Fac"], 0, 1, 0.12, 0.35), g.bsdf.inputs["Roughness"])
        g.set(Metallic=0.4, Coat_Weight=1.0, Coat_Roughness=0.05)
    elif name == "sintered":
        # sintered regolith pavers: dark glassy tiles with cracks
        tc = g.node("ShaderNodeTexCoord")
        vor = g.node("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE")
        vor.inputs["Scale"].default_value = 1.2
        g.link(tc.outputs["Object"], vor.inputs["Vector"])
        grout = g.map_range(vor.outputs["Distance"], 0.0, 0.025, 0.0, 1.0)
        n = g.noise(12.0, 8.0, 0.65)
        base = g.mix_color(g.map_range(n.outputs["Fac"], 0.35, 0.7, 0, 1), (0.13, 0.125, 0.12), (0.21, 0.2, 0.19))
        col = g.mix_color(grout, (0.05, 0.05, 0.05), base)
        col, _ = g.dust(col, 0.55)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.link(g.bump(grout, 0.5, 0.01), g.bsdf.inputs["Normal"])
        g.set(Roughness=0.55)
    elif name == "paint_white":
        col, _ = g.dust((0.82, 0.82, 0.78), 0.6)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Roughness=0.6)
    elif name == "hazard_orange":
        col, _ = g.dust((0.92, 0.32, 0.03), 0.5)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Roughness=0.5)
    elif name == "suit_fabric":
        wr = g.noise(40.0, 6.0, 0.6)
        g.link(g.bump(wr.outputs["Fac"], 0.3, 0.004), g.bsdf.inputs["Normal"])
        col, _ = g.dust((0.88, 0.88, 0.85), 0.7)
        g.link(col, g.bsdf.inputs["Base Color"])
        g.set(Roughness=0.78, Sheen_Weight=0.4)
    elif name == "gold_visor":
        g.set(Base_Color=(1.0, 0.76, 0.36, 1), Metallic=1.0, Roughness=0.04)
    elif name == "cable_orange":
        g.set(Base_Color=(0.85, 0.28, 0.03, 1), Roughness=0.5, Coat_Weight=0.3)
    elif name == "beacon_red":
        g.set(Base_Color=(1, 0.15, 0.05, 1), Emission_Color=(1.0, 0.12, 0.03, 1), Emission_Strength=40.0)
    elif name == "lamp_cool":
        g.set(Base_Color=(1, 1, 1, 1), Emission_Color=(0.75, 0.88, 1.0, 1), Emission_Strength=18.0)
    elif name == "badge_navy":
        g.set(Base_Color=(0.012, 0.03, 0.09, 1), Roughness=0.45, Coat_Weight=0.4)
    else:
        raise KeyError(name)
    _CACHE[name] = g.mat
    return g.mat
