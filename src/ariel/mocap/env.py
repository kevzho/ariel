"""RL environment: a learned layer on top of the frozen MoCapAct walking prior.

The frozen prior (MoCapAct's velocity-control policy and its low-level decoder) turns a speed
and heading command into 56 joint targets. The learned policy sits above it and outputs, on
every control step:

* a speed and a heading adjustment for the prior,
* a correction ``dz`` to the prior's 60-d motor intention,
* small joint-target corrections ``dq`` on the legs and spine,
* one length command for each telescoping group (thighs, shins, upper arms, forearms, torso).

Two modes share the same observations and actions:

* ``course``: the damaged-base course. The policy chooses the body shape. Early in training a
  hand-written rule ("teacher") suggests a shape for each obstacle and the reward includes an
  imitation term toward it; that term is scaled by ``teach`` which the trainer anneals to zero.
* ``flat``: a flat floor where the shape follows a random schedule the policy cannot change.
  This teaches the layer to keep the human gait stable on bodies the prior was never trained on.

Physics runs at Earth gravity (the prior was trained there). The lunar setting is visual.
"""

from __future__ import annotations

import glob
import math
import os
import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..world.course import HFIELD_RES, CourseParams, generate_course
from .walker import GROUPS, SLIDE_SPEED

CONTROL_DT = 0.03
NOMINAL_HEAD = 1.90          # m, top of the head while walking at nominal lengths (run mocapact-tolerance-20261011-021258)
NOMINAL_ROOT = 1.07
DQ_JOINTS = ("lfemurrx", "rfemurrx", "ltibiarx", "rtibiarx", "lfootrx", "rfootrx", "lowerbackrx", "upperbackrx")
SCAN_X = np.array([-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0])
SCAN_Y = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
CEIL_X = np.array([-0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8, 2.4, 3.0])
TERM_KEYS = ("progress", "alive", "teach", "bump", "dz", "dq", "lateral", "success", "terminal")
COURSE_LEVELS = ("blocks", "low", "fissure", "scarp")
FLAT_LEVELS = ("shape",)


@dataclass
class EnvCfg:
    mode: str = "course"                 # "course" or "flat"
    lock_telescoping: bool = False       # comparison body: every segment held at nominal length
    ranges: dict = field(default_factory=lambda: {"thigh": (0.70, 1.40), "shin": (0.70, 1.40)})
    # teacher shapes (length ratios); frozen-prior-safe values from the tolerance sweeps
    short_shape: dict = field(default_factory=lambda: {"thigh": 0.70, "torso": 0.80})
    long_shape: dict = field(default_factory=lambda: {"shin": 1.40})
    low_clearance: float = 1.82          # deck height at full difficulty (m)
    fissure_width: tuple = (0.40, 0.55)  # at full difficulty
    scarp_height: tuple = (0.8, 1.0)     # rise of the rocky slope at full difficulty (m)
    ramp_run: float = 5.0                # length of the slope (m)
    ramp_rocks: int = 14
    ramp_linear: bool = False            # constant-gradient slope
    climb_shape: dict = field(default_factory=dict)   # teacher shape on the slope (empty = nominal)
    lane_blocks: int = 2
    block_spacing: float = 4.0
    block_start: float = 2.6
    fissure_gap: float = 4.7             # distance from the deck centre to the fissure (m)
    stop_task: bool = False              # True: the attempt succeeds only by stopping and standing at the astronaut
    goal_gap: float = 3.4                # distance from the top of the slope to the astronaut (m)
    brake_dist: float = 4.0              # the commanded speed ramps to zero over this distance before the stop point
    brake_gain: float = 0.9              # commanded speed limit (m/s) per metre left to the stop point
    stop_runup: float = 5.0              # the finish spawn point is this far before the stop point (m)
    brake_fade: bool = True              # fade the learned corrections inside the braking zone
    relative_height: bool = False        # give the frozen prior its body height above the local ground (it was
                                         # trained on a floor at z = 0 and reads raised ground as extra height)
    plateau_height: bool = False         # the same correction, blended in over the first 2 m beyond the slope only
    w_brake: float = 0.0                 # cost per step on speed above the braking limit, (m/s)^2
    stand_steps: int = 40                # control steps the robot must stand still at the astronaut
    w_stand: float = 0.5                 # reward per step while standing still at the astronaut
    fissure_runup: float = 2.6           # the fissure spawn point is this far before the fissure (m)
    block_offset: float = 0.45
    turn_span: float = 0.8               # heading change (rad) the policy's turn action can add
    w_steer: float = 1.0                 # weight of the steering teacher (scaled by teach)
    steer_floor: float = 0.0             # the steering and pace hints keep at least this share of their weight
    block_lead: float = 3.2              # the teacher's path starts to swing this far before a crate (m)
    block_pace: float | None = None      # pace action the teacher suggests while among the crates (None: no hint)
    stall_steps: int = 250
    speed_base: float = 1.0
    speed_span: float = 0.6
    heading_rate_gain: float = 0.5       # heading command (rad) per m/s of lateral velocity
    heading_gain: float = 0.5            # heading command (rad) per metre of lateral offset
    heading_bias: float = 0.0
    spawn_yaw: float | None = None       # facing at spawn (rad); None keeps the reference clip facing, which the prior expects
    dz_scale: float = 0.5
    dq_scale: float = 0.2
    w_progress: float = 20.0
    w_alive: float = 0.0                 # flat mode sets this through flat_alive
    flat_alive: float = 0.3
    w_teach: float = 1.5
    w_bump: float = 0.5
    w_dz: float = 0.02
    w_dq: float = 0.05
    w_lateral: float = 0.05
    success_bonus: float = 100.0
    fall_penalty: float = 15.0
    spawn_weights: tuple = (0.35, 0.20, 0.25, 0.20)
    episode_steps: int = 1800
    flat_shape_seconds: tuple = (3.0, 7.0)   # how long each random shape is held
    njmax: int = 1500
    nconmax: int = 500
    clear_lane: float = 1.3              # no debris or boulders this close to the centre line (m)
    rough: float = 0.5                   # broken-ground crater scale
    craters: int = 9                     # craters on the broken ground
    static_debris: bool = True           # debris is fixed in place (free bodies resting on the heightfield are slow)
    record_every: int = 2


def _ratios_to_command(shape: dict, ranges: dict) -> np.ndarray:
    from .walker import DEFAULT_RANGE
    rng = {**DEFAULT_RANGE, **ranges}
    out = []
    for g in GROUPS:
        r, (lo, hi) = float(shape.get(g, 1.0)), rng[g]
        out.append((r - 1.0) / (hi - 1.0) if r >= 1.0 else (r - 1.0) / (1.0 - lo))
    return np.clip(np.array(out), -1.0, 1.0)


def showcase_params(cfg: EnvCfg, level: dict | None = None) -> CourseParams:
    """The fixed course used for demonstrations and checkpoint comparisons. ``level`` sets each
    obstacle's difficulty the same way training does (default: full difficulty)."""
    level = level or {}
    lerp = lambda a, b, k: a + (b - a) * float(np.clip(level.get(k, 1.0), 0.0, 1.0))  # noqa: E731
    L = _layout(cfg)
    return CourseParams(seed=7, crates=6, panels=2, loose_rocks=6, hab_radius=1.45, roof_len=2.4,
                        hab_clearance=lerp(2.30, cfg.low_clearance, "low"),
                        fissure_width=lerp(0.10, cfg.fissure_width[1], "fissure"),
                        scarp_height=lerp(0.15, cfg.scarp_height[1], "scarp"), boulders=6,
                        block_offset=lerp(1.3, cfg.block_offset, "blocks"), **L)


def _layout(cfg: EnvCfg) -> dict:
    """Positions along the course: pad, lane blocks, low deck, fissure, rocky slope, goal."""
    debris_end = 3.2 + cfg.block_start + max(cfg.lane_blocks - 1, 0) * cfg.block_spacing + 1.2
    hab_x = debris_end + 3.2
    fissure_x = hab_x + cfg.fissure_gap
    scarp_x = fissure_x + 3.6
    top = scarp_x + cfg.ramp_run
    return dict(debris_end=debris_end, hab_x=hab_x, fissure_x=fissure_x, scarp_x=scarp_x, mast_x=top + 0.8,
                broken_end=top + max(2.4, cfg.goal_gap - 1.5), goal_x=top + cfg.goal_gap, mast_blocks=False, talus=0,
                clear_lane=cfg.clear_lane,
                rough=cfg.rough, lane_blocks=cfg.lane_blocks, block_spacing=cfg.block_spacing,
                block_start=cfg.block_start,
                ramp_run=cfg.ramp_run, ramp_rocks=cfg.ramp_rocks, ramp_linear=cfg.ramp_linear, craters=cfg.craters)


def sample_params(rng: np.random.Generator, level: dict, cfg: EnvCfg) -> CourseParams:
    """Training layouts. ``level`` eases each obstacle from trivial (0) to full difficulty (1)."""
    lerp = lambda a, b, t: a + (b - a) * float(np.clip(t, 0.0, 1.0))  # noqa: E731
    L = _layout(cfg)
    L["hab_x"] += float(rng.uniform(-0.3, 0.3))
    L["fissure_x"] += float(rng.uniform(-0.3, 0.3))
    return CourseParams(
        seed=int(rng.integers(1 << 30)), crates=int(rng.integers(cfg.lane_blocks + 1, 7)),
        panels=int(rng.integers(0, 3)), loose_rocks=int(rng.integers(2, 6)),
        hab_radius=1.45, roof_len=float(rng.uniform(1.8, 2.6)),
        hab_clearance=lerp(2.30, cfg.low_clearance + float(rng.uniform(0.0, 0.04)), level.get("low", 1.0)),
        fissure_width=lerp(0.10, float(rng.uniform(*cfg.fissure_width)), level.get("fissure", 1.0)),
        scarp_height=lerp(0.15, float(rng.uniform(*cfg.scarp_height)), level.get("scarp", 1.0)),
        block_offset=lerp(1.3, cfg.block_offset + float(rng.uniform(0.0, 0.1)), level.get("blocks", 1.0)),
        boulders=int(rng.integers(3, 8)), **L)


class _FlatCourse:
    """Stand-in for a Course on the flat floor."""
    params = None
    roof = None
    course_id = "flat"

    def height_at(self, x, y):
        return 0.0

    def ceiling_at(self, x):
        return math.inf


# ---------------------------------------------------------------------- frozen prior
class FrozenWalk:
    """MoCapAct's velocity-control policy and low-level decoder, never updated."""

    def __init__(self):
        import torch
        from mocapact.distillation import model as mmodel
        from mocapact.sb3 import utils

        from .task import WALK_POLICY_KEYS

        roots = sorted({os.path.dirname(p) for p in glob.glob("/models/**/best_model.zip", recursive=True)})
        root = next(r for r in roots if "velocity_control" in r and "general_low_level" in r)
        self.keys = list(WALK_POLICY_KEYS)
        self.high = utils.load_policy(root, self.keys, device="cpu")
        self.low = mmodel.NpmpPolicy.load_from_checkpoint(os.path.join(root, "low_level_policy.ckpt"),
                                                          map_location="cpu")
        self.torch = torch

    def __call__(self, walk_obs: list[dict], dz: np.ndarray) -> np.ndarray:
        """Joint targets for a batch of environments. ``dz`` is added to the prior's intention."""
        torch = self.torch
        # the high-level policy concatenates the dict in iteration order: the order that reproduces
        # MoCapAct's walking is the sorted walker keys followed by target_obs (as in course_smoke)
        order = [k for k in self.keys if k != "target_obs"] + ["target_obs"]
        t_obs = {k: torch.from_numpy(np.stack([o[k] for o in walk_obs])) for k in order}
        with torch.no_grad():
            z = self.high.policy._predict(t_obs, deterministic=True).clamp(-3, 3)
            z = (z + torch.from_numpy(dz.astype(np.float32))).clamp(-3, 3)
            return self.low.low_level_policy(t_obs, z).numpy()


_FROZEN = None


def frozen_walk() -> FrozenWalk:
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = FrozenWalk()
    return _FROZEN


# ---------------------------------------------------------------------- environment
class ReshapeEnv:
    N_TELE = len(GROUPS)
    N_Z = 60

    def __init__(self, cfg: EnvCfg = EnvCfg(), seed: int = 0, level: dict | None = None,
                 fixed_params: CourseParams | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        names = FLAT_LEVELS if cfg.mode == "flat" else COURSE_LEVELS
        self.level = {k: float((level or {}).get(k, 0.0)) for k in names}
        self.fixed_params = fixed_params
        self.teach = 1.0
        self.act_dim = 2 + self.N_TELE + self.N_Z + len(DQ_JOINTS)
        self.tele_slice = slice(2, 2 + self.N_TELE)
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        from dm_control import composer

        from .arena import LunarCourse
        from .task import WALK_POLICY_KEYS, CourseTask, StandInitializer
        from .walker import TelescopingCMUHumanoid

        cfg = self.cfg
        self.init = StandInitializer()
        self.walker = TelescopingCMUHumanoid(initializer=self.init, telescoping=True, ranges=cfg.ranges)
        if cfg.mode == "flat":
            from dm_control.locomotion.arenas import floors
            self.course = _FlatCourse()
            arena = floors.Floor(size=(150, 20))
            self.level_built = dict(self.level)
        else:
            params = self.fixed_params or sample_params(self.rng, self.level, cfg)
            self.course = generate_course(params)
            arena = LunarCourse(self.course, njmax=cfg.njmax, nconmax=cfg.nconmax, loose_static=cfg.static_debris)
            self.level_built = dict(self.level)
        self.task = CourseTask(self.walker, arena)
        try:
            self.env = composer.Environment(self.task, random_state=int(self.seed), strip_singleton_obs_buffer_dim=True,
                                            recompile_mjcf_every_episode=False)
            self.recompiles = False
        except TypeError:
            self.env = composer.Environment(self.task, random_state=int(self.seed), strip_singleton_obs_buffer_dim=True)
            self.recompiles = True
        self.walk_keys = [k for k in WALK_POLICY_KEYS if k != "target_obs"]
        root = self.walker.mjcf_model
        self._b_root = self.walker.root_body
        self._b_hips = [root.find("body", "lfemur"), root.find("body", "rfemur")]
        self._g_head = root.find("geom", "head")
        act_names = [a.name for a in self.walker.actuators]
        self.dq_idx = np.array([act_names.index(n) for n in DQ_JOINTS])
        self._walker_geoms = root.find_all("geom")
        self._roof_geoms = ([g for n, g in arena._static.items() if n.startswith("hab_")]
                            if cfg.mode != "flat" else [])
        self.short_cmd = _ratios_to_command(cfg.short_shape, cfg.ranges)
        self.long_cmd = _ratios_to_command(cfg.long_shape, cfg.ranges)
        self.climb_cmd = _ratios_to_command(cfg.climb_shape, cfg.ranges)
        bones = self.walker.bone_lengths
        g = np.asarray(self.walker._group)
        self._leg_slides = np.where((g == GROUPS.index("thigh")) | (g == GROUPS.index("shin")))[0]
        self._torso_slides = np.where(g == GROUPS.index("torso"))[0]
        self._bones = bones
        if cfg.mode != "flat":
            P = self.course.params
            xs = np.arange(-6.0, P.goal_x + 3.5, 0.05)
            ground = np.array([max(self.course.height_at(x, y) for y in (-0.3, 0.0, 0.3)) for x in xs])
            ceil = np.array([self.course.ceiling_at(x) for x in xs])
            self._cx = xs
            self._clear = np.clip(ceil - ground, 0.0, 3.0)
            R = P.roof_len / 2 if P.roof_len > 0 else 0.5
            self.spawns = [0.0, P.hab_x - R - 2.2, P.fissure_x - cfg.fissure_runup, P.scarp_x - 2.0]
            self.stop_x = P.goal_x - 1.2
            if cfg.stop_task:
                self.spawns.append(self.stop_x - cfg.stop_runup)
            bx = [b[0] for b in self.course.blocks] or [P.pad_end + 1.0]
            self.zones = {"blocks": (min(bx) - 0.4, max(bx) + 0.4), "low": (P.hab_x - R, P.hab_x + R),
                          "fissure": (P.fissure_x, P.fissure_x + P.fissure_width),
                          "scarp": (P.scarp_x, P.scarp_x + max(P.ramp_run, 0.4))}
            self._build_maps()
        else:
            self.spawns = [0.0]
            self.zones = {}
        self._ids_ready = False
        self._fresh = True
        self.prof = {"frozen": 0.0, "physics": 0.0, "state": 0.0, "obs": 0.0, "reset": 0.0, "steps": 0, "resets": 0}

    def _reset_env(self) -> None:
        """Reset without recompiling the model (composer recompiles on every reset by default)."""
        if not self._fresh and not self.recompiles:
            import dm_env
            env = self.env
            with env._physics.reset_context():
                env._hooks.initialize_episode(env._physics_proxy, env._random_state)
            env._observation_updater.reset(env._physics_proxy, env._random_state)
            env._reset_next_step = False
            self.ts = dm_env.TimeStep(dm_env.StepType.FIRST, None, None, env._observation_updater.get_observation())
            return
        self.ts = self.env.reset()
        if self._fresh:
            self._fresh = False
            if self.cfg.mode == "flat" or self.cfg.static_debris:
                # the walker's free joint comes first; park its default pose above the course so the
                # forward pass inside a reset does not collide a half-buried body with the terrain
                self.env.physics.model.qpos0[2] = 10.0
            self.recompiles = not all(hasattr(self.env, a) for a in ("_physics_proxy", "_hooks", "_observation_updater",
                                                                      "_reset_next_step", "_random_state"))

    def _build_maps(self) -> None:
        """Terrain heights with debris and rocks stamped in (what a height scan would see), and the
        steering teacher's path around the lane blocks."""
        c = self.course
        h = c.heights.astype(np.float32).copy()
        nrow, ncol = h.shape
        for e in list(c.loose) + [s for s in c.static if s["kind"] in ("boulder", "astronaut")]:
            px, py, pz = e["pos"]
            if e["type"] == "sphere":
                rx = ry = e["size"][0]
                top = pz + e["size"][0]
            elif len(e["size"]) >= 3:
                rx = ry = max(e["size"][0], e["size"][1])
                top = pz + e["size"][2]
            else:                                           # capsule: radius, half length
                rx = ry = e["size"][0]
                top = pz + e["size"][0] + e["size"][1]
            c0, c1 = int((px - rx - c.x0) / HFIELD_RES), int((px + rx - c.x0) / HFIELD_RES) + 1
            r0, r1 = int((py - ry - c.y0) / HFIELD_RES), int((py + ry - c.y0) / HFIELD_RES) + 1
            c0, c1, r0, r1 = max(c0, 0), min(c1, ncol - 1), max(r0, 0), min(r1, nrow - 1)
            if c1 > c0 and r1 > r0:
                h[r0:r1 + 1, c0:c1 + 1] = np.maximum(h[r0:r1 + 1, c0:c1 + 1], top)
        self._hmap = h
        path = np.zeros_like(self._cx)
        sm = lambda a, b, x: (lambda t: t * t * (3 - 2 * t))(np.clip((x - a) / (b - a), 0.0, 1.0))  # noqa: E731
        for bx, by, hw in c.blocks:
            if abs(by) > hw + 0.75:
                continue                                    # far enough from the centre line already
            target = by - math.copysign(hw + 0.75, by if by else 1.0)
            path += target * sm(bx - self.cfg.block_lead, bx - 0.4 * self.cfg.block_lead, self._cx)                 * (1.0 - sm(bx + 0.2, bx + 1.5, self._cx))
        self._path = path
        self._path_slope = np.gradient(path, self._cx)

    def steer_teacher(self) -> float:
        """Turn action that would follow the path around the lane blocks (course mode)."""
        if self.cfg.mode == "flat":
            return 0.0
        cfg = self.cfg
        y_ref = float(np.interp(self.x + 1.0, self._cx, self._path))
        slope = float(np.interp(self.x + 1.0, self._cx, self._path_slope))
        want = cfg.heading_gain * y_ref + math.atan(slope)
        return float(np.clip(want / cfg.turn_span, -1.0, 1.0))

    def _pace_error(self, pace_action: float) -> float:
        """Squared distance from the suggested pace while among the crates (0 elsewhere)."""
        if self.cfg.block_pace is None:
            return 0.0
        b0, b1 = self.zones["blocks"]
        if b0 - self.cfg.block_lead - 1.0 < self.x < b1 + 1.0:
            return float((pace_action - self.cfg.block_pace) ** 2)
        return 0.0

    def refresh(self, level: dict) -> None:
        """Rebuild with a layout drawn at a new curriculum level (course mode)."""
        self.level = {k: float(level.get(k, v)) for k, v in self.level.items()}
        if self.cfg.mode != "flat" and self.fixed_params is None:
            self._build()

    def _bind_ids(self) -> None:
        p = self.env.physics
        self._walker_ids = np.asarray(p.bind(self._walker_geoms).element_id)
        self._roof_ids = np.asarray(p.bind(self._roof_geoms).element_id) if self._roof_geoms else np.zeros(0, int)
        self._ids_ready = True

    # ------------------------------------------------------------------ episode
    def reset(self, spawn: int | None = None) -> dict:
        cfg = self.cfg
        if spawn is None:
            w = np.array(cfg.spawn_weights[:len(self.spawns)], dtype=float)
            spawn = int(self.rng.choice(len(self.spawns), p=w / w.sum()))
        self.spawn_index = spawn
        x0 = float(self.spawns[spawn])
        y0 = float(self.rng.uniform(-0.35, 0.35))
        self.init.spawn = (x0, y0, self.course.height_at(x0, y0))
        self.init.yaw = cfg.spawn_yaw
        t0 = _time.perf_counter()
        self._reset_env()
        self.prof["reset"] += _time.perf_counter() - t0
        self.prof["resets"] += 1
        if not self._ids_ready or self.recompiles:
            self._bind_ids()
        self.t = 0
        self.frontier = self.start_x = x0
        self.last_gain_t = 0
        self.stall_mark = x0
        self.tele_cmd = np.zeros(self.N_TELE)            # rate-limited command state, per group
        self.tele_target = np.zeros(self.walker.n_slides)
        self.prev_action = np.zeros(self.act_dim)
        self.attempted = {k: False for k in self.level}
        self.passed = {k: False for k in self.level}
        self.max_pen = 0.0
        self.stand_count = 0
        self._fade = 1.0
        self._flat_goal = np.zeros(self.N_TELE)
        self._flat_until = int(2.0 / CONTROL_DT)         # walk at nominal length for 2 s first
        self._read_state()
        return self._obs()

    def _read_state(self) -> None:
        p = self.env.physics
        pos = np.asarray(p.bind(self._b_root).xpos)
        hips = np.asarray(p.bind(self._b_hips).xpos)
        left = hips[0, :2] - hips[1, :2]
        self.x, self.y, self.z = float(pos[0]), float(pos[1]), float(pos[2])
        self.yaw = math.atan2(-left[0], left[1])
        self.ground = self.course.height_at(self.x, self.y)
        self.head_top = float(p.bind(self._g_head).xpos[2]) + 0.143
        self.slide_q = np.asarray(p.bind(self.walker.slide_joints).qpos).copy()
        self.slide_v = np.asarray(p.bind(self.walker.slide_joints).qvel).copy()
        self.ratios = self.walker.group_ratios(self.slide_q)
        self.leg_delta = float(self.slide_q[self._leg_slides].sum() / 2.0)
        self.expected_head = NOMINAL_HEAD + self.leg_delta + float(self.slide_q[self._torso_slides].sum())

    def teacher(self) -> np.ndarray:
        """The hand-written shape rule (course mode): short under low structure, long at the fissure."""
        if self.cfg.mode == "flat":
            return self._flat_goal
        near = (self._cx > self.x - 0.7) & (self._cx < self.x + 1.9)
        if near.any() and self._clear[near].min() < NOMINAL_HEAD + 0.15:
            return self.short_cmd
        f0, f1 = self.zones["fissure"]
        if f0 - 2.2 < self.x < f1 + 0.9:
            return self.long_cmd
        s0, s1 = self.zones["scarp"]
        if s0 - 1.2 < self.x < s1 + 0.3:
            return self.climb_cmd
        return np.zeros(self.N_TELE)

    def walk_obs(self, action: np.ndarray) -> dict:
        """Apply the speed/heading part of the action and return the frozen prior's observation."""
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._a = a
        speed = cfg.speed_base + cfg.speed_span * a[0]
        self._fade = 1.0
        if cfg.stop_task and cfg.mode != "flat":
            # braking zone before the astronaut: the commanded speed ramps to zero and the learned
            # corrections fade, so the frozen prior brings the body to a stand
            remaining = self.stop_x - self.x
            if remaining < cfg.brake_dist:
                speed = min(speed, max(0.0, cfg.brake_gain * remaining))
                if cfg.brake_fade:
                    self._fade = float(np.clip(remaining / (0.5 * cfg.brake_dist), 0.1, 1.0))
        vy = float(np.asarray(self.ts.observation["walker/torso_yvel"]).ravel()[0])
        heading = float(np.clip(-cfg.heading_gain * self.y - cfg.heading_rate_gain * vy, -0.8, 0.8)
                        + cfg.heading_bias + cfg.turn_span * a[1])
        self.task.command = np.array([speed, math.sin(heading), math.cos(heading), (self.t % 333) / 333.0])
        o = {k: np.asarray(self.ts.observation[k], np.float32).ravel() for k in self.walk_keys}
        o["target_obs"] = self.task.command.astype(np.float32)
        if cfg.relative_height and cfg.mode != "flat":
            o["walker/body_height"] = o["walker/body_height"] - np.float32(self.ground)
        elif cfg.plateau_height and cfg.mode != "flat":
            t = float(np.clip((self.x - self.zones["scarp"][1]) / 2.0, 0.0, 1.0))
            o["walker/body_height"] = o["walker/body_height"] - np.float32(self.ground * t * t * (3 - 2 * t))
        return o

    def dz(self) -> np.ndarray:
        return self._fade * self.cfg.dz_scale * self._a[2 + self.N_TELE:2 + self.N_TELE + self.N_Z]

    def step_physics(self, q: np.ndarray):
        """Finish the step started by ``walk_obs``: residuals, telescoping, physics, reward."""
        cfg, a = self.cfg, self._a
        q = np.asarray(q, dtype=np.float64).copy()
        dq = a[2 + self.N_TELE + self.N_Z:]
        q[self.dq_idx] += self._fade * cfg.dq_scale * dq
        want = a[self.tele_slice]
        if cfg.mode == "flat":
            if self.t >= self._flat_until:
                L = self.level["shape"]
                self._flat_goal = np.clip(self.rng.uniform(-1.0, 1.0, self.N_TELE) * (0.25 + 0.75 * L), -1, 1)
                if self.rng.random() < 0.2:
                    self._flat_goal[:] = 0.0
                self._flat_until = self.t + int(self.rng.uniform(*cfg.flat_shape_seconds) / CONTROL_DT)
            want = self._flat_goal
        if cfg.lock_telescoping:
            want = np.zeros(self.N_TELE)
        teacher = self.teacher()
        steer = self.steer_teacher()
        self.tele_cmd = np.clip(want, -1.0, 1.0)
        goal = self.walker.slide_targets(self.tele_cmd)
        self.tele_target = self.tele_target + np.clip(goal - self.tele_target, -SLIDE_SPEED * CONTROL_DT,
                                                      SLIDE_SPEED * CONTROL_DT)
        error = ""
        t0 = _time.perf_counter()
        try:
            self.ts = self.env.step(np.concatenate([np.clip(q, -1.0, 1.0), self.tele_target]))
        except Exception as exc:  # noqa: BLE001  (physics error: end this attempt, keep training)
            error = repr(exc)[:200]
        t1 = _time.perf_counter()
        self.t += 1
        if not error:
            self._read_state()
        self.prof["physics"] += t1 - t0
        self.prof["state"] += _time.perf_counter() - t1
        self.prof["steps"] += 1
        x = self.x
        stopping = cfg.stop_task and cfg.mode != "flat"
        x_paid = min(x, self.stop_x) if stopping else x          # progress stops paying at the stop point
        gain = min(max(0.0, x_paid - self.frontier), 0.15)
        self.frontier = max(self.frontier, x_paid)
        standing = False
        if stopping and not error:
            vx = float(np.asarray(self.ts.observation["walker/torso_xvel"]).ravel()[0])
            vy = float(np.asarray(self.ts.observation["walker/torso_yvel"]).ravel()[0])
            standing = (self.stop_x - 1.3 <= x <= self.stop_x + 0.9 and math.hypot(vx, vy) < 0.3
                        and self.head_top - self.ground > 0.85 * self.expected_head)
        self.stand_count = self.stand_count + 1 if standing else 0
        over_speed = 0.0
        if stopping and not error and cfg.w_brake > 0 and self.stop_x - x < cfg.brake_dist:
            limit = max(0.0, cfg.brake_gain * (self.stop_x - x))
            over_speed = max(0.0, math.hypot(vx, vy) - limit - 0.2) ** 2
        if self.frontier - self.stall_mark >= 0.3:
            self.stall_mark, self.last_gain_t = self.frontier, self.t
        bump = self._roof_contact() if (len(getattr(self, "_roof_ids", ())) and not error) else 0.0
        live_tele = cfg.mode != "flat" and not cfg.lock_telescoping
        terms = {
            "progress": cfg.w_progress * gain,
            "alive": (cfg.flat_alive if cfg.mode == "flat" else cfg.w_alive) + (cfg.w_stand if standing else 0.0)
                     - cfg.w_brake * over_speed,
            "teach": (-cfg.w_teach * self.teach * float(np.mean((a[self.tele_slice] - teacher) ** 2)) if live_tele else 0.0)
                     - (cfg.w_steer * max(self.teach, cfg.steer_floor) * ((a[1] - steer) ** 2 + self._pace_error(a[0]))
                        if cfg.mode != "flat" else 0.0),
            "bump": -cfg.w_bump * bump,
            "dz": -cfg.w_dz * float(np.mean(a[2 + self.N_TELE:2 + self.N_TELE + self.N_Z] ** 2)),
            "dq": -cfg.w_dq * float(np.mean(dq ** 2)),
            "lateral": -cfg.w_lateral * (self.y / 3.0) ** 2,
        }
        # ---- termination
        reason = ""
        head_h = self.head_top - self.ground
        P = self.course.params
        in_fissure = False
        if P is not None:
            f0, f1 = self.zones["fissure"]
            in_fissure = f0 - 0.4 < x < f1 + 0.4
        if error:
            reason = "physics_error"
        elif in_fissure and self.z < self.course.height_at(P.fissure_x - 0.6, self.y) + 0.45:
            reason = "fell_into_fissure"
        elif head_h < 0.62 * self.expected_head:
            reason = "fell"
        elif abs(self.y) > (P.geofence if P is not None else 8.0):
            reason = "out_of_corridor"
        elif stopping and self.stand_count >= cfg.stand_steps:
            reason = "success"
            terms["success"] = cfg.success_bonus
        elif stopping and x > self.stop_x + 1.3:
            reason = "overshot"
        elif not stopping and P is not None and x >= P.goal_x - 1.2:
            reason = "success"
            terms["success"] = cfg.success_bonus
        if reason in ("fell", "fell_into_fissure", "out_of_corridor", "overshot"):
            terms["terminal"] = -cfg.fall_penalty
        terminated, truncated = bool(reason), False
        if not terminated:
            if self.t >= cfg.episode_steps:
                reason, truncated = "timeout", True
            elif self.t - self.last_gain_t > cfg.stall_steps:
                reason, truncated = "stalled", True
        self._update_passed(reason)
        self.prev_action = a
        reward = float(sum(terms.values()))
        info = {"terms": terms, "reason": reason, "x": x, "frontier": self.frontier, "passed": dict(self.passed),
                "attempted": dict(self.attempted), "ratios": self.ratios.tolist(), "head_top": self.head_top,
                "teacher": teacher.tolist(), "steer": steer, "bump": bump, "max_penetration": 0.0, "error": error}
        t0 = _time.perf_counter()
        obs = self._obs()
        self.prof["obs"] += _time.perf_counter() - t0
        return obs, reward, terminated, truncated, info

    def step(self, action: np.ndarray):
        """Single-environment step (evaluation and tests); training batches the frozen prior."""
        o = self.walk_obs(action)
        t0 = _time.perf_counter()
        q = frozen_walk()([o], self.dz()[None])[0]
        self.prof["frozen"] += _time.perf_counter() - t0
        return self.step_physics(q)

    def _roof_contact(self) -> float:
        d = self.env.physics.data
        n = int(d.ncon)
        if n == 0:
            return 0.0
        g1, g2 = np.asarray(d.contact.geom1[:n]), np.asarray(d.contact.geom2[:n])
        hit = (np.isin(g1, self._roof_ids) & np.isin(g2, self._walker_ids)) | \
              (np.isin(g2, self._roof_ids) & np.isin(g1, self._walker_ids))
        return float(hit.any())

    def _update_passed(self, reason: str) -> None:
        if self.cfg.mode == "flat":
            self.attempted["shape"] = True
            self.passed["shape"] = reason in ("timeout", "")
            return
        for k, (a, b) in self.zones.items():
            if self.start_x < a - 0.3 and self.frontier > a - 0.6:
                self.attempted[k] = True
            if self.attempted[k] and self.x > b + 0.6:
                self.passed[k] = True

    # ------------------------------------------------------------------ observations
    def _scan(self) -> np.ndarray:
        if self.cfg.mode == "flat":
            return np.zeros(len(SCAN_X) * len(SCAN_Y))
        c = self.course
        h = self._hmap
        nrow, ncol = h.shape
        X = (self.x + SCAN_X[:, None] - c.x0) / HFIELD_RES
        Y = (self.y + SCAN_Y[None, :] - c.y0) / HFIELD_RES
        X, Y = np.broadcast_arrays(X, Y)
        c0 = np.clip(np.floor(X).astype(int), 0, ncol - 2)
        r0 = np.clip(np.floor(Y).astype(int), 0, nrow - 2)
        fc, fr = np.clip(X - c0, 0, 1), np.clip(Y - r0, 0, 1)
        z = ((h[r0, c0] * (1 - fc) + h[r0, c0 + 1] * fc) * (1 - fr)
             + (h[r0 + 1, c0] * (1 - fc) + h[r0 + 1, c0 + 1] * fc) * fr)
        return np.clip(z - self.ground, -1.5, 1.5).ravel()

    def _ceiling(self) -> np.ndarray:
        if self.cfg.mode == "flat":
            return np.full(len(CEIL_X), 3.0)
        return np.interp(self.x + CEIL_X, self._cx, self._clear)

    def _obs(self) -> dict:
        o = self.ts.observation
        f = lambda k: np.asarray(o["walker/" + k], np.float64).ravel()  # noqa: E731
        touch = np.log1p(np.clip(f("sensors_touch"), 0.0, None))
        bones = self._bones
        if self.cfg.mode == "flat":
            fis = np.array([3.0, 3.0])
        else:
            f0, f1 = self.zones["fissure"]
            fis = np.clip(np.array([f0 - self.x, f1 - self.x]), -3.0, 3.0)
        actor = np.concatenate([
            f("joints_pos"), 0.1 * f("joints_vel"), f("sensors_velocimeter"), f("sensors_gyro"), f("world_zaxis"),
            f("end_effectors_pos"), touch,
            self.ratios - 1.0, self.slide_v * 4.0, self.slide_q - self.tele_target, self.tele_cmd,
            [self.z - self.ground - NOMINAL_ROOT, self.head_top - self.ground - NOMINAL_HEAD],
            self._scan(), self._ceiling() - 2.0, fis,
            [self.y / 3.0, math.sin(self.yaw), math.cos(self.yaw)],
            self.prev_action[:2 + self.N_TELE], self.prev_action[2 + self.N_TELE + self.N_Z:],
        ]).astype(np.float32)
        zone = np.zeros(4)
        if self.cfg.mode != "flat":
            P = self.course.params
            zone[int(np.searchsorted([self.zones["low"][1], self.zones["fissure"][1], self.zones["scarp"][1]], self.x))] = 1.0
            extra = [self.x / 25.0, P.hab_clearance - 2.0, P.fissure_width, P.scarp_height, self.frontier - self.x,
                     self.steer_teacher(), P.block_offset]
        else:
            extra = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        critic = np.concatenate([actor, zone, extra, self.teacher(), [self.teach, self.t / self.cfg.episode_steps]]
                                ).astype(np.float32)
        return {"actor": np.nan_to_num(actor), "critic": np.nan_to_num(critic)}

    # ------------------------------------------------------------------ recording
    @property
    def qpos(self) -> np.ndarray:
        return np.asarray(self.env.physics.data.qpos)

    def episode_meta(self) -> dict:
        P = self.course.params
        return {"course_id": self.course.course_id, "course_params": asdict(P) if P is not None else None,
                "mode": self.cfg.mode, "level": dict(self.level_built), "spawn_index": self.spawn_index,
                "spawn_x": float(self.spawns[self.spawn_index]), "lock_telescoping": self.cfg.lock_telescoping,
                "ranges": {k: list(v) for k, v in self.cfg.ranges.items()}}


__all__ = ["ReshapeEnv", "EnvCfg", "FrozenWalk", "frozen_walk", "showcase_params", "sample_params", "TERM_KEYS",
           "COURSE_LEVELS", "FLAT_LEVELS"]
