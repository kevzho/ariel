"""The lunar rescue RL environment: one continuous attempt across the damaged base.

Observations use only onboard sensing. Actions command every rotary joint and
every telescoping segment. Reward pays for new progress of the pelvis toward the
stranded astronaut and for reaching them; small costs discourage wasted energy,
jitter, violent impacts and self-collision. Shape is never rewarded directly.

The environment exposes the information the recorder needs (state, actions,
reward terms, termination reason) on every step.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from ..robot.control import Commander, actuator_map
from ..robot.humanoid import RobotSpec
from ..world.course import CourseParams, generate_course, random_params
from ..world.scene import build_world_xml, model_assets


@dataclass
class EnvConfig:
    control_hz: float = 50.0
    episode_seconds: float = 60.0
    stall_seconds: float = 15.0
    stall_progress: float = 0.3
    geofence: float = 3.0
    rear_limit: float = -3.0
    # Task: new progress toward the astronaut and reaching them.
    w_progress: float = 20.0
    success_bonus: float = 300.0
    # Locomotion terms after legged_gym (Rudin et al., CoRL 2022) and Humanoid-Gym (Gu et al., 2024).
    v_target: float = 0.6            # m/s forward walking pace (lunar)
    w_track: float = 0.5             # exp(-(vx - v_target)^2 / sigma)
    track_sigma: float = 0.25
    w_orient: float = 2.0            # |projected gravity xy|^2 of the pelvis (upright)
    w_height: float = 1.0            # exp(-((h - h_ref) / 0.1)^2), h_ref from current leg length
    w_air: float = 1.0               # feet air time on touchdown (t_air - 0.4 s), clipped
    air_target: float = 0.4
    w_feet_dist: float = 0.5         # lateral foot spacing outside [0.14, 0.42] m
    w_default: float = 0.05          # arm roll/yaw, hip yaw/roll near their natural values
    w_fallen: float = 0.2            # per step while torso/pelvis/head rests on the ground
    w_lateral: float = 0.5           # exp-kernel tracking of zero sideways velocity
    w_heading: float = 0.3           # exp-kernel tracking of heading along the corridor
    w_energy: float = 3e-6
    w_action_rate: float = 0.005
    w_impact: float = 0.3
    w_self_contact: float = 0.02
    w_joint_limit: float = 0.005
    w_wobble: float = 0.004          # chest (sensor head) angular velocity: keep the cameras steady
    w_slip: float = 0.05             # feet sliding while loaded
    w_joint_acc: float = 2e-6        # joint accelerations: no flailing
    impact_force: float = 350.0
    exploit_penetration: float = 0.06
    goal_radius: float = 1.3
    terminal_penalty: float = 1000.0  # must exceed any penalty an early exit could avoid (no suicide exploit)
    fall_grace_s: float = 2.0        # torso/pelvis/head on the ground longer than this ends the attempt
    fall_penalty: float = 300.0     # larger than the posture costs a deliberate fall could escape
    spawn_weights: tuple = (0.4, 0.12, 0.12, 0.12, 0.12, 0.12)


# Curriculum: each obstacle eases from an easy setting toward the full training distribution.
EASY = {"hab_clearance": 1.75, "fissure_width": 0.15, "scarp_height": 0.1, "crates": 1, "panels": 0, "loose_rocks": 2}
OBSTACLES = ("hab", "fissure", "scarp", "debris")


def curriculum_params(rng: np.random.Generator, level: dict[str, float]) -> CourseParams:
    p = random_params(rng)
    lerp = lambda a, b, t: a + (b - a) * float(np.clip(t, 0, 1))  # noqa: E731
    p.hab_clearance = lerp(EASY["hab_clearance"], p.hab_clearance, level.get("hab", 1.0))
    p.fissure_width = lerp(EASY["fissure_width"], p.fissure_width, level.get("fissure", 1.0))
    p.scarp_height = lerp(EASY["scarp_height"], p.scarp_height, level.get("scarp", 1.0))
    d = level.get("debris", 1.0)
    p.crates = int(round(lerp(EASY["crates"], p.crates, d)))
    p.panels = int(round(lerp(EASY["panels"], p.panels, d)))
    p.loose_rocks = int(round(lerp(EASY["loose_rocks"], p.loose_rocks, d)))
    return p


@dataclass
class CourseSlot:
    params: CourseParams
    course: object
    model: mujoco.MjModel
    data: mujoco.MjData
    settled_qpos: np.ndarray
    xml: str
    meta: dict
    level: dict = field(default_factory=dict)


class LunarRescueEnv:
    ROBOT_NQ = 39
    ROBOT_NV = 38

    def __init__(self, cfg: EnvConfig = EnvConfig(), seed: int = 0, pool_size: int = 3,
                 level: dict | None = None, fixed_params: CourseParams | None = None):
        self.cfg = cfg
        self.spec = RobotSpec()
        self.rng = np.random.default_rng(seed)
        self.level = dict(level or {k: 0.0 for k in OBSTACLES})
        self.fixed_params = fixed_params
        self.pool: list[CourseSlot] = [self._make_slot() for _ in range(pool_size)]
        self.slot: CourseSlot | None = None
        self.episode_steps = int(cfg.episode_seconds * cfg.control_hz)

    # ------------------------------------------------------------------ courses
    def _make_slot(self) -> CourseSlot:
        params = self.fixed_params or curriculum_params(self.rng, self.level)
        course = generate_course(params)
        xml, meta = build_world_xml(self.spec, course=course)
        model = mujoco.MjModel.from_xml_string(xml, model_assets(course))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        amap = actuator_map(model, meta["actuators"], self.spec)
        cmd = Commander(amap, self.spec, 1.0 / self.cfg.control_hz)
        cmd.reset(data)
        steps = int(round(1.0 / (self.cfg.control_hz * model.opt.timestep)))
        for _ in range(int(3.0 * self.cfg.control_hz)):      # let loose debris come to rest
            cmd.set_targets(amap.default, data)
            mujoco.mj_step(model, data, nstep=steps)
        slot = CourseSlot(params, course, model, data, data.qpos.copy(), xml, meta, dict(self.level))
        self._prepare(slot)
        return slot

    def refresh(self, level: dict, n: int = 1) -> None:
        """Replace up to ``n`` pooled courses with ones drawn at the new curriculum level."""
        self.level = dict(level)
        order = sorted(range(len(self.pool)), key=lambda i: sum(self.pool[i].level.values()))
        for i in order[:n]:
            self.pool[i] = self._make_slot()

    def _prepare(self, s: CourseSlot) -> None:
        m = s.model
        s.amap = actuator_map(m, s.meta["actuators"], self.spec)
        s.cmd = Commander(s.amap, self.spec, 1.0 / self.cfg.control_hz)
        s.nstep = int(round(1.0 / (self.cfg.control_hz * m.opt.timestep)))
        s.pelvis = m.body("pelvis").id
        s.head_geoms = {m.geom(n).id for n in ("head", "chest", "pelvis", "visor", "lidar")}
        robot_bodies = set()
        for b in range(m.nbody):
            if m.body_rootid[b] == s.pelvis:
                robot_bodies.add(b)
        s.robot_bodies = robot_bodies
        s.robot_geom = np.array([m.geom_bodyid[g] in robot_bodies for g in range(m.ngeom)])
        s.chest = m.body("chest").id
        names = s.amap.names
        s.leg_slide_adr = np.array([m.joint(n).qposadr[0] for n in names
                                    if n.endswith(("thigh_ext", "shin_ext"))])
        reg = ("shoulder_roll", "shoulder_yaw", "hip_yaw", "hip_roll", "waist_yaw", "waist_roll", "ankle_roll")
        s.reg_ids = np.array([i for i, n in enumerate(names) if n.endswith(reg)])
        s.torso_geoms = {m.geom(n).id for n in ("pelvis", "chest", "head", "visor", "lidar", "power_pack")}
        s.foot_sites = [m.site("left_foot_site").id, m.site("right_foot_site").id]
        s.contact_geoms = [[m.geom(f"{side}_sole_{k}").id for k in range(4)] for side in ("left", "right")]
        s.contact_geoms += [[m.geom("left_hand").id], [m.geom("right_hand").id]]
        s.rays_group = np.array([1, 0, 1, 0, 1, 0], np.uint8)
        P = s.params
        s.spawns = [0.0, P.pad_end + 0.6, P.hab_x - P.hab_radius - 1.0, P.fissure_x - 1.2,
                    P.scarp_x - 1.1, P.mast_x - 0.9]
        s.goal = np.array(s.course.goal)

    # ------------------------------------------------------------------ episode
    def reset(self, spawn: int | None = None) -> dict:
        self.slot = s = self.pool[int(self.rng.integers(len(self.pool)))]
        m, d = s.model, s.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = s.settled_qpos
        w = np.array(self.cfg.spawn_weights)
        self.spawn_index = int(self.rng.choice(len(s.spawns), p=w / w.sum())) if spawn is None else spawn
        x0 = s.spawns[self.spawn_index]
        y0 = float(self.rng.uniform(-0.4, 0.4))
        yaw = float(self.rng.uniform(-0.25, 0.25))
        z0 = self._ground(x0, y0) + self.spec.pelvis_height + 0.02
        d.qpos[0:3] = (x0, y0, z0)
        d.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        d.qpos[7:self.ROBOT_NQ] = 0.0
        d.qpos[s.amap.qpos_adr] = s.amap.default
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        s.cmd.reset(d)
        self.t = 0
        self.frontier = x0
        self.start_x = x0
        self.last_gain_t = 0
        self.stall_mark = x0
        self.prev_action = np.zeros(m.nu)
        self.prev_qd = np.zeros(int(np.sum(~s.amap.is_slide)))
        self.air_time = np.zeros(2)
        self.down_steps = 0
        self.was_contact = np.ones(2, bool)
        self.goal_hold = 0
        self.max_pen = 0.0
        self.passed = {k: False for k in OBSTACLES}
        return self._obs()

    def _ground(self, x: float, y: float) -> float:
        s = self.slot
        pnt = np.array([x, y, 4.0])
        geomid = np.array([-1], np.int32)
        dist = mujoco.mj_ray(s.model, s.data, pnt, np.array([0.0, 0.0, -1.0]), s.rays_group, 1, -1, geomid)
        return 4.0 - dist if dist >= 0 else s.course.height_at(x, y)

    def step(self, action: np.ndarray):
        s, cfg = self.slot, self.cfg
        m, d = s.model, s.data
        action = np.asarray(action, dtype=np.float64)
        s.cmd.command(action, d)
        x_before = d.qpos[0]
        mujoco.mj_step(m, d, nstep=s.nstep)
        self.t += 1

        # ---- reward terms
        x = float(d.qpos[0])
        gain = max(0.0, x - self.frontier)
        if gain > 0:
            self.frontier = x
        if self.frontier - self.stall_mark >= cfg.stall_progress:
            self.stall_mark = self.frontier
            self.last_gain_t = self.t
        power = float(np.sum(np.abs(d.actuator_force * d.qvel[s.amap.qvel_adr])))
        rate = float(np.mean((action - self.prev_action) ** 2))
        impact, self_contacts, max_pen = self._contacts()
        self.max_pen = max(self.max_pen, max_pen)
        at_limit = int(np.sum((d.qpos[s.amap.qpos_adr] <= s.amap.lo + 0.01) | (d.qpos[s.amap.qpos_adr] >= s.amap.hi - 0.01)))
        # movement quality: steady sensor head, planted feet, smooth joints
        chest_w = d.cvel[s.chest][:3]
        wobble = float(chest_w @ chest_w)
        slip = 0.0
        for k in range(2):
            if self._foot_loaded(k):
                v = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, s.foot_sites[k], v, 0)
                slip += float(v[3] ** 2 + v[4] ** 2)
        qd_rot = d.qvel[s.amap.qvel_adr[~s.amap.is_slide]]
        acc = (qd_rot - self.prev_qd) * cfg.control_hz
        self.prev_qd = qd_rot.copy()
        joint_acc = float(np.mean(acc ** 2))
        # ---- locomotion quality (posture terms relax where crawling is the right answer)
        crawl = self._in_crawl_zone(x)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, d.qpos[3:7])
        Rm = rot.reshape(3, 3)
        g_b = Rm.T @ np.array([0.0, 0.0, -1.0])
        orient = float(g_b[0] ** 2 + g_b[1] ** 2)
        vx = float(d.qvel[0])
        # forward-speed tracking measured against standing still: standing earns nothing
        track = (math.exp(-((vx - cfg.v_target) ** 2) / cfg.track_sigma)
                 - math.exp(-(cfg.v_target ** 2) / cfg.track_sigma))
        ground_z = s.course.height_at(float(d.qpos[0]), float(d.qpos[1]))
        leg = sum(getattr(self.spec, k).nominal for k in ("thigh", "shin")) + float(
            np.sum(d.qpos[s.leg_slide_adr]) / 2.0)
        h_ref = 0.07 + leg + 0.06 - 0.04
        height = math.exp(-(((float(d.qpos[2]) - ground_z) - h_ref) / 0.1) ** 2)
        contact = np.array([self._foot_loaded(0), self._foot_loaded(1)])
        dt = 1.0 / cfg.control_hz
        first = contact & ~self.was_contact
        air_reward = float(np.sum(np.clip(self.air_time - cfg.air_target, -0.4, 0.3) * first)) if vx > 0.05 else 0.0
        self.air_time = np.where(contact, 0.0, self.air_time + dt)
        self.was_contact = contact
        fy = Rm.T @ (d.site_xpos[s.foot_sites[0]] - d.site_xpos[s.foot_sites[1]])
        spread = abs(float(fy[1]))
        feet_dist = max(0.0, 0.14 - spread) + max(0.0, spread - 0.42)
        q = d.qpos[s.amap.qpos_adr]
        default_dev = float(np.sum((q[s.reg_ids] - s.amap.default[s.reg_ids]) ** 2))
        fallen = self._torso_on_ground()
        # "down" as in Humanoid-Gym: torso contact, or (outside the crawlspace) a collapsed or tipped posture
        h_above = float(d.qpos[2]) - ground_z
        down = fallen or (not crawl and (h_above < 0.45 or orient > 0.75))
        self.down_steps = self.down_steps + 1 if down else 0
        vy = float(d.qvel[1])
        yaw = math.atan2(Rm[1, 0], Rm[0, 0]) if not crawl else math.atan2(Rm[1, 2], Rm[0, 2])
        # posture terms are costs for deviating, never a reward for holding still
        lateral = 1.0 - math.exp(-(vy ** 2) / 0.1)
        heading = 1.0 - math.exp(-(yaw ** 2) / 0.15)
        upright_scale = 0.0 if crawl else 1.0
        terms = {
            "progress": cfg.w_progress * gain,
            "track": cfg.w_track * track * upright_scale,
            "orient": -cfg.w_orient * orient * upright_scale,
            "height": -cfg.w_height * (1.0 - height) * upright_scale,
            "air_time": cfg.w_air * air_reward * upright_scale,
            "feet_dist": -cfg.w_feet_dist * feet_dist * upright_scale,
            "default_pose": -cfg.w_default * default_dev,
            "fallen": -cfg.w_fallen * float(fallen),
            "lateral": -cfg.w_lateral * lateral,
            "heading": -cfg.w_heading * heading,
            "wobble": -cfg.w_wobble * wobble,
            "slip": -cfg.w_slip * slip,
            "joint_acc": -cfg.w_joint_acc * joint_acc,
            "energy": -cfg.w_energy * power,
            "action_rate": -cfg.w_action_rate * rate,
            "impact": -cfg.w_impact * impact,
            "self_contact": -cfg.w_self_contact * self_contacts,
            "joint_limit": -cfg.w_joint_limit * at_limit,
        }
        # ---- termination
        reason = ""
        pel = d.qpos[0:3]
        to_goal = float(np.linalg.norm(pel[:2] - s.goal[:2]))
        speed = float(np.linalg.norm(d.qvel[0:3]))
        self.goal_hold = self.goal_hold + 1 if (to_goal < cfg.goal_radius and speed < 0.4) else 0
        if self.goal_hold >= int(0.5 * cfg.control_hz):
            reason = "success"
            terms["success"] = cfg.success_bonus
        elif abs(pel[1]) > cfg.geofence or pel[0] < cfg.rear_limit:
            reason = "out_of_corridor"
        elif pel[2] < -0.7:
            P = s.params
            near_fissure = P.fissure_x - 0.5 < pel[0] < P.fissure_x + P.fissure_width + 0.5
            reason = "fell_into_fissure" if near_fissure else "fell_off_course"
        elif not np.all(np.isfinite(d.qpos)) or np.max(np.abs(d.qvel)) > 200:
            reason = "physics_unstable"
        elif max_pen > cfg.exploit_penetration:
            reason = "penetration_exploit"
        elif self.down_steps > cfg.fall_grace_s * cfg.control_hz:
            reason = "fell"            # as in legged_gym / Humanoid-Gym: a fall ends the attempt
        if reason in ("out_of_corridor", "fell_into_fissure", "fell_off_course", "penetration_exploit", "physics_unstable"):
            terms["terminal"] = -cfg.terminal_penalty
        elif reason == "fell":
            terms["terminal"] = -cfg.fall_penalty
        terminated = bool(reason)
        truncated = False
        if not terminated:
            if self.t >= self.episode_steps:
                reason, truncated = "timeout", True
            elif (self.t - self.last_gain_t) > cfg.stall_seconds * cfg.control_hz:
                reason, truncated = "stalled", True
        self._update_passed(x)
        reward = float(sum(terms.values()))
        self.prev_action = action
        info = {"terms": terms, "reason": reason, "x": x, "frontier": self.frontier, "to_goal": to_goal,
                "power": power, "max_penetration": max_pen, "passed": dict(self.passed),
                "dx": x - x_before}
        return self._obs(), reward, terminated, truncated, info

    def _update_passed(self, x: float) -> None:
        P = self.slot.params
        for k, xe in (("debris", P.debris_end), ("hab", P.hab_x + P.hab_radius), ("fissure", P.fissure_x + P.fissure_width),
                      ("scarp", P.scarp_x + 0.4)):
            if x > xe + 0.2:
                self.passed[k] = True

    def _in_crawl_zone(self, x: float) -> bool:
        """Under (and just around) the collapsed module, where crawling is the correct posture."""
        P = self.slot.params
        return P.hab_x - P.hab_radius - 0.5 < x < P.hab_x + P.hab_radius + 0.4

    def _torso_on_ground(self) -> bool:
        s = self.slot
        d = s.data
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            if (a in s.torso_geoms and not s.robot_geom[b]) or (b in s.torso_geoms and not s.robot_geom[a]):
                return True
        return False

    def _foot_loaded(self, k: int) -> bool:
        s = self.slot
        d = s.data
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in s.contact_geoms[k] and not s.robot_geom[c.geom2]) or \
               (c.geom2 in s.contact_geoms[k] and not s.robot_geom[c.geom1]):
                return True
        return False

    def _contacts(self) -> tuple[float, int, float]:
        s = self.slot
        m, d = s.model, s.data
        impact, self_c, max_pen = 0.0, 0, 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            r1, r2 = s.robot_geom[g1], s.robot_geom[g2]
            if r1 and r2:
                self_c += 1
            if not (r1 or r2):
                continue
            max_pen = max(max_pen, -c.dist)
            if (g1 in s.head_geoms) != (g2 in s.head_geoms) and not (r1 and r2):
                mujoco.mj_contactForce(m, d, i, f6)
                if f6[0] > self.cfg.impact_force:
                    impact += (f6[0] - self.cfg.impact_force) / self.cfg.impact_force
        return impact, self_c, max_pen

    # ------------------------------------------------------------------ observations
    SCAN_X = np.arange(-0.3, 1.81, 0.35)       # 7 rows ahead
    SCAN_Y = np.array([-0.45, -0.15, 0.15, 0.45])
    CEIL_X = np.arange(0.0, 2.11, 0.35)        # 7 points ahead

    def _obs(self) -> dict:
        s = self.slot
        m, d = s.model, s.data
        am = s.amap
        q = d.qpos[am.qpos_adr]
        qd = d.qvel[am.qvel_adr]
        quat = d.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        R = rot.reshape(3, 3)
        grav_b = R.T @ np.array([0, 0, -1.0])
        angvel_b = d.qvel[3:6]                      # free joint angular velocity is in the body frame
        linvel_b = R.T @ d.qvel[0:3]
        yaw = math.atan2(R[1, 0], R[0, 0])
        cy, sy = math.cos(yaw), math.sin(yaw)
        pel = d.qpos[0:3]
        # downward lidar grid in the yaw-aligned frame, relative to the pelvis
        scan = np.zeros((len(self.SCAN_X), len(self.SCAN_Y)))
        gid = np.array([-1], np.int32)
        down = np.array([0.0, 0.0, -1.0])
        for i, ax in enumerate(self.SCAN_X):
            for j, ay in enumerate(self.SCAN_Y):
                px = pel[0] + cy * ax - sy * ay
                py = pel[1] + sy * ax + cy * ay
                dist = mujoco.mj_ray(m, d, np.array([px, py, pel[2] + 1.2]), down, s.rays_group, 1, -1, gid)
                h = (pel[2] + 1.2 - dist) - pel[2] if dist >= 0 else -2.0
                scan[i, j] = np.clip(h, -1.6, 1.2)
        # upward clearance scan: from just above the ground ahead, how much room is there
        up = np.array([0.0, 0.0, 1.0])
        ceil = np.zeros(len(self.CEIL_X))
        for i, ax in enumerate(self.CEIL_X):
            px, py = pel[0] + cy * ax, pel[1] + sy * ax
            g = scan[min(i, len(self.SCAN_X) - 1), 1] + pel[2] if i < len(self.SCAN_X) else pel[2] - 0.8
            dist = mujoco.mj_ray(m, d, np.array([px, py, g + 0.05]), up, s.rays_group, 1, -1, gid)
            ceil[i] = min(dist, 2.5) if dist >= 0 else 2.5
        contacts = np.zeros(4)
        for i in range(d.ncon):
            c = d.contact[i]
            for k, gs in enumerate(s.contact_geoms):
                if (c.geom1 in gs and not s.robot_geom[c.geom2]) or (c.geom2 in gs and not s.robot_geom[c.geom1]):
                    contacts[k] = 1.0
        g = s.goal - pel
        gb = np.array([cy * g[0] + sy * g[1], -sy * g[0] + cy * g[1]])
        gdist = float(np.linalg.norm(gb))
        actor = np.concatenate([
            (q - am.default) / s.cmd.scale,
            np.clip(qd * np.where(am.is_slide, 2.0, 0.1), -5, 5),
            self.prev_action if hasattr(self, "prev_action") else np.zeros(m.nu),
            grav_b, np.clip(angvel_b * 0.25, -5, 5), np.clip(linvel_b, -5, 5),
            contacts, scan.ravel(), ceil / 2.5 - 0.5,
            gb / max(gdist, 1e-6), [min(gdist, 30.0) / 30.0],
            # localisation against the marked rescue corridor (lateral offset and heading)
            [pel[1] / self.cfg.geofence, math.sin(yaw), math.cos(yaw)],
        ]).astype(np.float32)
        P = s.params
        seg = np.zeros(6)
        bounds = [P.pad_end, P.debris_end, P.hab_x + P.hab_radius, P.fissure_x + P.fissure_width, P.scarp_x + 0.4]
        seg[int(np.searchsorted(bounds, pel[0]))] = 1.0
        critic = np.concatenate([actor, [pel[0] / 25.0, pel[1] / 3.0, pel[2], (self.frontier - pel[0])],
                                 d.qvel[0:3], seg, [P.hab_clearance, P.fissure_width, P.scarp_height]]).astype(np.float32)
        return {"actor": actor, "critic": critic}

    @property
    def obs_dims(self) -> tuple[int, int]:
        o = self._obs()
        return len(o["actor"]), len(o["critic"])

    def episode_meta(self) -> dict:
        s = self.slot
        return {"course_id": s.course.course_id, "course_params": asdict(s.params), "level": s.level,
                "spawn_index": self.spawn_index, "spawn_x": s.spawns[self.spawn_index]}


__all__ = ["LunarRescueEnv", "EnvConfig", "curriculum_params", "OBSTACLES", "CourseParams"]
