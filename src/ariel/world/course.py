"""Seeded obstacle courses for the damaged lunar base.

One ``Course`` drives both the physics (MuJoCo heightfield, static geoms and
loose debris bodies) and the scene description that Blender turns into
detailed models. Coordinates: x along the course, y across it, z up; metres.

The robot starts on the landing pad at x = 0 and must reach the stranded
astronaut. Every obstacle spans the whole search corridor; leaving the
corridor (|y| > ``geofence``) ends the episode, so there is no way around.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

HFIELD_RES = 0.05           # m per heightfield cell


@dataclass
class CourseParams:
    """Layout parameters. Training draws these at random; showcase fixes them."""
    seed: int = 7
    pad_end: float = 3.2
    debris_end: float = 7.8
    crates: int = 6
    panels: int = 3
    loose_rocks: int = 8
    hab_x: float = 10.2
    hab_radius: float = 1.45
    hab_clearance: float = 0.78           # lowest gap under the collapsed module (army crawl is 0.65 m tall)
    fissure_x: float = 13.0
    fissure_width: float = 0.85
    scarp_x: float = 15.4
    scarp_height: float = 0.8
    mast_x: float = 17.2
    mast_yaw_deg: float = 14.0
    broken_end: float = 23.5
    boulders: int = 7
    goal_x: float = 24.4
    geofence: float = 3.0
    roof_len: float = 0.0                 # > 0: a flat deck of this length hangs under the module (low passage)
    mast_blocks: bool = True              # False: the toppled mast does not lie across the route
    talus: int = 6                        # fallen blocks at the foot of the scarp
    clear_lane: float = 0.0               # > 0: no debris or boulders within this distance of the centre line
    rough: float = 1.0                    # scale of the craters and ripples on the broken ground
    lane_blocks: int = 0                  # crates standing in the walking lane (must be walked around)
    block_spacing: float = 3.3            # distance between lane blocks along the course (m)
    block_start: float = 1.4              # distance from the pad edge to the first lane block (m)
    block_offset: float = 0.45            # how far each lane block sits from the centre line (m), alternating sides
    ramp_run: float = 0.0                 # > 0: the scarp is a rocky slope of this length rising scarp_height
    ramp_rocks: int = 0                   # rocks scattered on the slope
    ramp_linear: bool = False             # True: a constant-gradient slope (False: an S-shaped rise)
    craters: int = 9                      # craters on the broken ground beyond the slope


@dataclass
class Course:
    params: CourseParams
    x0: float
    y0: float
    heights: np.ndarray                 # (nrow=y, ncol=x) terrain height, m
    static: list[dict] = field(default_factory=list)
    loose: list[dict] = field(default_factory=list)
    decor: list[dict] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    start: tuple = (0.0, 0.0)
    goal: tuple = (0.0, 0.0, 0.0)
    roof: tuple | None = None           # (x0, x1, underside z) of the low deck, if any
    blocks: list = field(default_factory=list)   # lane blocks as (x, y, half width)

    def ceiling_at(self, x: float) -> float:
        """Height (z) of the lowest overhead structure above course position x (inf if open sky)."""
        P = self.params
        if self.roof is not None and self.roof[0] <= x <= self.roof[1]:
            return self.roof[2]
        dx = abs(x - P.hab_x)
        if dx < P.hab_radius:
            base = (self.roof[2] + 0.08) if self.roof is not None else P.hab_clearance
            return base + P.hab_radius - math.sqrt(P.hab_radius ** 2 - dx ** 2)
        return math.inf

    @property
    def extent(self) -> tuple[float, float, float, float]:
        nrow, ncol = self.heights.shape
        return (self.x0, self.y0, self.x0 + (ncol - 1) * HFIELD_RES, self.y0 + (nrow - 1) * HFIELD_RES)

    @property
    def course_id(self) -> str:
        h = hashlib.sha256(json.dumps(asdict(self.params), sort_keys=True).encode()).hexdigest()[:10]
        return f"base-{self.params.seed}-{h}"

    def height_at(self, x: float, y: float) -> float:
        nrow, ncol = self.heights.shape
        c = (x - self.x0) / HFIELD_RES
        r = (y - self.y0) / HFIELD_RES
        c0 = int(np.clip(np.floor(c), 0, ncol - 2))
        r0 = int(np.clip(np.floor(r), 0, nrow - 2))
        fc, fr = c - c0, r - r0
        h = self.heights
        return float((h[r0, c0] * (1 - fc) + h[r0, c0 + 1] * fc) * (1 - fr)
                     + (h[r0 + 1, c0] * (1 - fc) + h[r0 + 1, c0 + 1] * fc) * fr)

    def hfield_data(self) -> tuple[np.ndarray, float, float]:
        """Heights normalised to exactly [0, 1] for MuJoCo (row 0 at the -y edge).

        MuJoCo rescales heightfield data to its own min/max, so the geom is placed at
        the true minimum and the elevation span equals the true range.
        """
        lo, hi = float(self.heights.min()), float(self.heights.max())
        return ((self.heights - lo) / (hi - lo)).astype(np.float32), lo, hi - lo

    def scene_json(self) -> dict:
        """Everything Blender needs to build the visible world (no physics engine required)."""
        return {
            "course_id": self.course_id,
            "params": asdict(self.params),
            "hfield": {"x0": self.x0, "y0": self.y0, "res": HFIELD_RES,
                       "shape": list(self.heights.shape)},
            "static": self.static, "loose": self.loose, "decor": self.decor,
            "segments": self.segments, "start": list(self.start), "goal": list(self.goal),
        }


def _quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _quat_y(pitch: float) -> list[float]:
    return [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise1(rng, n_terms=5, scale=1.0):
    """A random smooth 1-D function (sum of sines) for jagged edges."""
    f = rng.uniform(0.6, 4.0, n_terms) * scale
    p = rng.uniform(0, 2 * math.pi, n_terms)
    a = rng.uniform(0.3, 1.0, n_terms) / np.arange(1, n_terms + 1)
    return lambda t: sum(ai * np.sin(fi * t + pi) for ai, fi, pi in zip(a, f, p)) / a.sum()


def _value_noise(rng: np.random.Generator, octaves: int = 4, size: int = 256):
    """Smooth fractal value noise in [-1, 1]: irregular, unlike sums of sines."""
    tables = [rng.uniform(-1.0, 1.0, (size, size)) for _ in range(octaves)]
    offsets = rng.uniform(0, size, (octaves, 2))

    def f(X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        out = np.zeros(np.broadcast(X, Y).shape)
        amp, total = 1.0, 0.0
        for o, t in enumerate(tables):
            fx = X * (2 ** o) + offsets[o, 0]
            fy = Y * (2 ** o) + offsets[o, 1]
            ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
            tx, ty = fx - ix, fy - iy
            tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
            i0, i1 = ix % size, (ix + 1) % size
            j0, j1 = iy % size, (iy + 1) % size
            v = ((t[j0, i0] * (1 - tx) + t[j0, i1] * tx) * (1 - ty)
                 + (t[j1, i0] * (1 - tx) + t[j1, i1] * tx) * ty)
            out += amp * v
            total += amp
            amp *= 0.5
        return out / total
    return f


def terrain_function(params: CourseParams):
    """Height z(X, Y) of the base terrain. The same function feeds the physics grid
    along the course and the wider visual vista, so features continue seamlessly."""
    rng = np.random.default_rng([params.seed, 1])
    base_ph = [rng.uniform(0, 2 * math.pi, 2) for _ in range(3)]
    craters = [(rng.uniform(params.mast_x + 1.0, params.broken_end), rng.uniform(-4.5, 4.5),
                rng.uniform(0.4, 1.3)) for _ in range(params.craters)]
    rough_ph = rng.uniform(0, 2 * math.pi, 2)
    edge = _noise1(rng, scale=0.6)
    e1, e2 = _noise1(rng, scale=0.8), _noise1(rng, scale=0.8)
    # vista: bigger craters and rolling ground away from the corridor
    vista_craters = [(rng.uniform(-60, 90), rng.uniform(-60, 60), float(np.clip(0.8 * (rng.pareto(1.5) + 1), 0.8, 14)))
                     for _ in range(220)]
    vista_ph = rng.uniform(0, 2 * math.pi, 6)
    vn = _value_noise(np.random.default_rng([params.seed, 2]))
    vn2 = _value_noise(np.random.default_rng([params.seed, 3]))

    def height(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        z = np.zeros_like(X, dtype=np.float64)
        for (amp, k), ph in zip(((0.05, 0.35), (0.02, 1.4), (0.006, 6.0)), base_ph):
            z += amp * np.sin(k * X + ph[0]) * np.cos(0.8 * k * Y + ph[1])
        # beyond the corridor the ground rolls and is cratered (none inside the corridor)
        # the corridor blends into the landscape over ~10 m along a wavering boundary
        wobble = 2.5 * vn(X / 9.0 + 5.0, Y / 9.0)
        away = (_smoothstep(5.5, 15.0, np.abs(Y) + wobble) + _smoothstep(4.0, 12.0, -X + wobble)
                + _smoothstep(4.0, 12.0, X - params.goal_x - 3.5 + wobble))
        away = np.clip(away, 0.0, 1.0)
        if np.any(away > 0):
            v = 0.7 * vn(X / 14.0, Y / 14.0 + 30.0) + 0.25 * vn2(X / 4.0 + 10.0, Y / 4.0)
            for cx, cy, r in vista_craters:
                d = np.hypot(X - cx, Y - cy) / r
                if np.all(d > 1.6):
                    continue
                v += np.where(d < 1, -0.17 * r * (1 - d ** 2), 0) + 0.05 * r * np.exp(-((d - 1) / 0.25) ** 2)
            z += away * v

        pad_r = params.pad_end
        rpad = np.hypot(X, Y)
        z = np.where(rpad < pad_r, 0.0, z * _smoothstep(pad_r, pad_r + 1.0, rpad))

        r0 = params.scarp_x + max(2.0, params.ramp_run + 0.5)
        rough = _smoothstep(r0, r0 + 1.5, X) * (1 - away) * params.rough
        for cx, cy, r in craters:
            d = np.hypot(X - cx, Y - cy) / r
            z += rough * (np.where(d < 1, -0.16 * r * (1 - d ** 2), 0) + 0.05 * r * np.exp(-((d - 1) / 0.25) ** 2))
        z += rough * 0.05 * np.sin(2.1 * X + rough_ph[0]) * np.sin(1.7 * Y + rough_ph[1])

        # fault scarp: the ground beyond it is raised by a moonquake. Away from the corridor it
        # meanders, varies in height and dies out; its face is broken into blocks and ledges.
        wide = np.clip((np.abs(Y) - 3.5) / 25.0, 0.0, 1.0)
        scarp_line = (params.scarp_x + 0.15 * edge(Y) + 2.5 * wide * edge(Y / 7.0 + 3.0)
                      + 0.06 * vn(Y * 1.7, 11.3))
        h_scarp = params.scarp_height * (1.0 + 0.12 * edge(Y / 3.0)) * np.clip(1.0 - wide * 1.1, 0.0, 1.0) ** 0.7
        # broken face: the slope position wanders with height noise, leaving blocks and ledges
        jitter = 0.07 * vn2(X * 4.0, Y * 4.0)
        rise = _smoothstep(scarp_line - 0.1 + jitter, scarp_line + 0.1 + jitter, X)
        face = np.exp(-((X - scarp_line) / 0.22) ** 2)
        if params.ramp_run > 0:
            # a rocky slope instead of a step: a smooth climb with small ledges
            t = np.clip((X - scarp_line) / params.ramp_run, 0.0, 1.0)
            rise = t if params.ramp_linear else t * t * (3 - 2 * t)
            face = 0.12 * np.sin(np.pi * t) ** 2
        blocks = 0.08 * vn(X * 3.0 + 40.0, Y * 3.0) + 0.04 * vn2(X * 9.0, Y * 9.0 + 20.0)
        z += h_scarp * rise + face * blocks * (h_scarp / max(params.scarp_height, 1e-6))

        # fissure: a deep jagged crack across the corridor, meandering and narrowing far away
        taper = np.clip(1.0 - (np.abs(Y) - 6.0) / 30.0, 0.12, 1.0)
        near = (params.fissure_x + 0.06 * e1(Y) + 0.02 * Y + 1.8 * wide * e1(Y / 6.0 + 2.0)
                + 0.035 * vn(Y * 2.5, 3.7))
        far = (near + params.fissure_width * taper * (1.0 + 0.25 * wide * e2(Y / 4.0))
               + 0.06 * (e2(Y) - e1(Y)) * taper + 0.035 * vn(Y * 2.5, 9.1))
        inside = (X > near) & (X < far)
        # steep, fractured walls falling to a rubble floor
        dist = np.minimum(X - near, far - X)
        floor = -2.2 + 0.25 * vn2(X * 2.0, Y * 2.0)
        wall = z - 0.05 - dist * (9.0 + 3.0 * vn(X * 6.0, Y * 6.0))
        lip = np.exp(-np.minimum(np.abs(X - near), np.abs(X - far)) / 0.12)
        z = np.where(inside, np.maximum(floor, wall), z - 0.04 * lip * (1 + vn2(Y * 6.0, 1.0)))
        return z

    return height


def vista_grid(params: CourseParams, res: float = 0.2, x_range=(-60.0, 90.0), y_range=(-60.0, 60.0)):
    """A coarse terrain grid around the course for rendering (not used by physics)."""
    xs = np.arange(x_range[0], x_range[1] + 1e-9, res)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, res)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, terrain_function(params)(X, Y).astype(np.float32)


def generate_course(params: CourseParams = CourseParams()) -> Course:
    rng = np.random.default_rng(params.seed)
    x0, x1 = -6.0, params.goal_x + 3.5
    y0, y1 = -5.0, 5.0
    xs = np.arange(x0, x1 + 1e-9, HFIELD_RES)
    ys = np.arange(y0, y1 + 1e-9, HFIELD_RES)
    X, Y = np.meshgrid(xs, ys)                   # rows = y
    z = terrain_function(params)(X, Y)
    pad_r = params.pad_end

    course = Course(params=params, x0=float(xs[0]), y0=float(ys[0]), heights=z.astype(np.float32))

    # ------------------------------------------------------------------ static
    # collapsed habitat module lying across the corridor, propped on its end rings
    R = params.hab_radius
    zc = params.hab_clearance + R
    if params.roof_len > 0:
        # low passage: a flat service deck hangs under the module; its underside is hab_clearance
        # above the highest ground beneath it
        xa, xb = params.hab_x - params.roof_len / 2, params.hab_x + params.roof_len / 2
        under = z[(np.abs(Y) < 0.6) & (X > xa) & (X < xb)]
        deck_z = float(under.max()) + params.hab_clearance
        course.roof = (xa, xb, deck_z)
        course.static.append({"name": "hab_deck", "kind": "hab", "type": "box",
                              "size": [params.roof_len / 2, 3.6, 0.04], "pos": [params.hab_x, 0.0, deck_z + 0.04]})
        zc = deck_z + 0.08 + R
    seg = 10
    for i in range(seg):
        a0 = math.pi + math.pi * i / seg
        a1 = math.pi + math.pi * (i + 1) / seg
        am = (a0 + a1) / 2
        chord = 2 * (R + 0.02) * math.sin((a1 - a0) / 2) + 0.02
        px = params.hab_x + (R - 0.1) * math.cos(am)
        pz = zc + (R - 0.1) * math.sin(am)
        # box tangent to the arc; rotate about y so its long side follows the hull
        course.static.append({
            "name": f"hab_shell_{i}", "kind": "hab", "type": "box",
            "size": [chord / 2, 3.6, 0.1], "pos": [px, 0.0, pz],
            "quat": _quat_y(-(am - 3 * math.pi / 2)),
        })
    course.decor.append({"kind": "hab_module", "pos": [params.hab_x, 0.0, zc], "radius": R,
                         "length": 7.4, "clearance": params.hab_clearance})

    # toppled solar tower: lattice mast across the terrace, panels beside it
    yaw = math.radians(params.mast_yaw_deg)
    mz = course.height_at(params.mast_x, 0.0) + 0.2
    if params.mast_blocks:
        course.static.append({"name": "solar_mast", "kind": "mast", "type": "box",
                              "size": [0.2, 4.6, 0.2], "pos": [params.mast_x, 0.0, mz], "quat": _quat_z(yaw)})
    if params.mast_blocks:
        course.decor.append({"kind": "solar_tower", "pos": [params.mast_x, 0.0, mz], "yaw": yaw, "length": 9.2})
    else:
        # the toppled tower lies beside the route, on the flat ground before the slope
        tx, ty = params.scarp_x - 5.5, 5.4
        course.decor.append({"kind": "solar_tower", "pos": [tx, ty, course.height_at(tx, 4.9) + 0.2],
                             "yaw": math.pi / 2 + 0.1, "length": 9.2})
    for k, (dx, dy) in enumerate(((1.4, -1.9), (-1.2, 2.3))):
        px, py = params.mast_x + dx, dy
        pz = course.height_at(px, py) + 0.02
        panel_quat = _quat_z(yaw + rng.uniform(-0.4, 0.4))
        if params.mast_blocks:              # with the tower moved aside, its panels are not on the route either
            course.static.append({"name": f"solar_panel_{k}", "kind": "solar_panel", "type": "box",
                                  "size": [0.9, 1.6, 0.02], "pos": [px, py, pz], "quat": panel_quat})

    # fallen blocks at the foot of the scarp: small, sparse, real collision geometry
    for k in range(params.talus):
        ty = float(rng.uniform(-2.8, 2.8))
        tx = params.scarp_x - float(rng.uniform(0.18, 0.6))
        r = float(rng.uniform(0.05, 0.12))
        course.static.append({"name": f"talus_{k}", "kind": "boulder", "type": "sphere", "size": [r],
                              "pos": [tx, ty, course.height_at(tx, ty) + r * 0.25],
                              "seed": int(rng.integers(1 << 30))})

    # rocks on the slope: larger ones beside the lane, small ones anywhere
    for k in range(params.ramp_rocks if params.ramp_run > 0 else 0):
        small = k % 3 == 0
        r = float(rng.uniform(0.04, 0.07) if small else rng.uniform(0.12, 0.32))
        rx = float(rng.uniform(params.scarp_x + 0.2, params.scarp_x + params.ramp_run + 0.6))
        ry = float(rng.uniform(-2.8, 2.8))
        if not small and abs(ry) < 0.75 + r:
            ry = math.copysign(0.75 + r + abs(ry) * 0.5, ry if ry else 1.0)
        course.static.append({"name": f"climb_rock_{k}", "kind": "boulder", "type": "sphere", "size": [r],
                              "pos": [rx, ry, course.height_at(rx, ry) + r * (0.0 if small else 0.2)],
                              "seed": int(rng.integers(1 << 30))})

    # boulders on the broken ground and one behind the astronaut
    for k in range(params.boulders):
        bx = rng.uniform(params.mast_x + 1.2, params.broken_end)
        by = rng.uniform(-2.8, 2.8)
        if abs(bx - params.goal_x) < 1.6 and abs(by) < 1.5:
            continue
        if params.clear_lane > 0 and abs(by) < params.clear_lane + 0.5:
            continue
        r = float(rng.uniform(0.18, 0.5))
        bz = course.height_at(bx, by) + r * rng.uniform(-0.3, 0.2)
        course.static.append({"name": f"boulder_{k}", "kind": "boulder", "type": "sphere",
                              "size": [r], "pos": [bx, by, bz], "seed": int(rng.integers(1 << 30))})
    gz = course.height_at(params.goal_x, 0.6)
    course.static.append({"name": "rest_boulder", "kind": "boulder", "type": "sphere", "size": [0.85],
                          "pos": [params.goal_x + 0.75, 0.6, gz + 0.2], "seed": 99})
    course.static.append({"name": "astronaut", "kind": "astronaut", "type": "capsule",
                          "size": [0.22, 0.3], "pos": [params.goal_x, 0.6, gz + 0.45],
                          "quat": _quat_y(-0.35)})
    course.decor.append({"kind": "astronaut", "pos": [params.goal_x, 0.6, gz], "yaw": math.pi})
    course.goal = (params.goal_x, 0.6, gz)

    # ------------------------------------------------------------------ loose debris
    placed: list[tuple[float, float, float]] = []
    side = 1.0 if rng.random() < 0.5 else -1.0
    for k in range(min(params.lane_blocks, params.crates)):
        hx = float(rng.uniform(0.22, 0.28))
        x = params.pad_end + params.block_start + k * params.block_spacing
        y = side * params.block_offset
        side = -side
        placed.append((x, y, hx * 1.3))
        course.blocks.append((x, y, hx))
        course.loose.append({"name": f"crate_{k}", "kind": "crate", "type": "box", "size": [hx, hx, 0.3],
                             "mass": 18.0, "pos": [x, y, course.height_at(x, y) + 0.3 + 0.03],
                             "quat": _quat_z(rng.uniform(-0.2, 0.2)), "seed": int(rng.integers(1 << 30))})

    def free_spot(r, xa, xb):
        for _ in range(200):
            x, y = rng.uniform(xa, xb), rng.uniform(-2.5, 2.5)
            if params.clear_lane > 0 and abs(y) < params.clear_lane + r:
                continue
            if all(math.hypot(x - a, y - b) > r + c + 0.15 for a, b, c in placed):
                placed.append((x, y, r))
                return x, y
        return None

    for k in range(min(params.lane_blocks, params.crates), params.crates):
        hx, hy, hz = rng.uniform(0.18, 0.28), rng.uniform(0.18, 0.28), rng.uniform(0.15, 0.24)
        spot = free_spot(max(hx, hy) * 1.3, params.pad_end + 0.4, params.debris_end)
        if spot is None:
            continue
        x, y = spot
        course.loose.append({"name": f"crate_{k}", "kind": "crate", "type": "box", "size": [hx, hy, hz],
                             "mass": float(rng.uniform(8, 20)),
                             "pos": [x, y, course.height_at(x, y) + hz + 0.03],
                             "quat": _quat_z(rng.uniform(0, math.pi)), "seed": int(rng.integers(1 << 30))})
    for k in range(params.panels):
        spot = free_spot(0.55, params.pad_end + 0.4, params.debris_end)
        if spot is None:
            continue
        x, y = spot
        course.loose.append({"name": f"panel_{k}", "kind": "panel", "type": "box", "size": [0.5, 0.32, 0.015],
                             "mass": 5.0, "pos": [x, y, course.height_at(x, y) + 0.05],
                             "quat": _quat_z(rng.uniform(0, math.pi))})
    for k in range(params.loose_rocks):
        r = float(rng.uniform(0.05, 0.12))
        spot = free_spot(r, params.pad_end + 0.3, params.hab_x - R)
        if spot is None:
            continue
        x, y = spot
        course.loose.append({"name": f"rock_{k}", "kind": "rock", "type": "sphere", "size": [r],
                             "mass": float(2700 * 4 / 3 * math.pi * r ** 3),
                             "pos": [x, y, course.height_at(x, y) + r + 0.03], "seed": int(rng.integers(1 << 30))})

    # ------------------------------------------------------------------ set dressing (visual only)
    course.decor += [
        {"kind": "landing_pad", "pos": [0.0, 0.0, 0.0], "radius": pad_r},
        {"kind": "cable_reel", "pos": [5.4, 3.6, course.height_at(5.4, 3.6)], "yaw": 0.4},
        {"kind": "cable_reel", "pos": [7.2, -3.8, course.height_at(7.2, -3.8)], "yaw": 1.3},
        {"kind": "comms_mast", "pos": [4.6, 4.2, course.height_at(4.6, 4.2)], "yaw": 0.9},
        {"kind": "lander", "pos": [-9.0, 14.0, 0.0], "yaw": 0.4},
    ]
    course.segments = [
        {"name": "landing pad", "x0": -1.0, "x1": params.pad_end},
        {"name": "debris field", "x0": params.pad_end, "x1": params.debris_end},
        {"name": "collapsed habitat", "x0": params.hab_x - R, "x1": params.hab_x + R},
        {"name": "fissure", "x0": params.fissure_x, "x1": params.fissure_x + params.fissure_width},
        {"name": "scarp", "x0": params.scarp_x - 0.2, "x1": params.scarp_x + 0.4},
        {"name": "toppled solar tower", "x0": params.mast_x - 0.6, "x1": params.mast_x + 0.6},
        {"name": "broken ground", "x0": params.mast_x + 0.6, "x1": params.goal_x},
    ]
    course.start = (0.0, 0.0)
    return course


def random_params(rng: np.random.Generator) -> CourseParams:
    """Training distribution over layouts (the showcase uses ``CourseParams()``)."""
    return CourseParams(
        seed=int(rng.integers(1 << 30)),
        crates=int(rng.integers(3, 6)), panels=int(rng.integers(1, 3)), loose_rocks=int(rng.integers(3, 7)),
        hab_x=float(rng.uniform(9.6, 10.8)), hab_radius=float(rng.uniform(1.3, 1.6)),
        hab_clearance=float(rng.uniform(0.70, 0.95)),
        fissure_width=float(rng.uniform(0.5, 1.1)), scarp_height=float(rng.uniform(0.45, 0.95)),
        mast_yaw_deg=float(rng.uniform(-25, 25)), boulders=int(rng.integers(4, 11)),
    )


def course_from_scene(scene: dict) -> Course:
    """Rebuild a Course (without heights) from ``scene_json()`` for visual placement."""
    hf = scene["hfield"]
    nrow, ncol = hf["shape"]
    course = Course(params=CourseParams(**scene["params"]), x0=hf["x0"], y0=hf["y0"],
                    heights=np.zeros((nrow, ncol), np.float32), static=scene["static"], loose=scene["loose"],
                    decor=scene["decor"], segments=scene["segments"], start=tuple(scene["start"]),
                    goal=tuple(scene["goal"]))
    return course
