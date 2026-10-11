"""The ASCENT telescoping rescue humanoid, generated as MuJoCo XML.

Frame convention: z up, the robot faces +x, +y is its left.

Every limb bone and the torso is a telescoping pair: an outer sleeve on the
proximal body and an inner tube on a child body attached by a sliding joint.
Extending the slide moves the next joint away, changing the bone length while
the physics keeps mass, momentum and contacts consistent.

All actuators are plain force/torque motors. The controller (see
``ascent.robot.control``) turns position targets into torques through a motor
model with torque-speed limits, so the policy cannot exceed realistic actuator
performance.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


def _f(*v: float) -> str:
    return " ".join(f"{x:.6g}" for x in v)


@dataclass(frozen=True)
class Telescope:
    nominal: float         # joint-to-joint length at slide = 0, m
    min_scale: float
    max_scale: float
    force: float           # N

    @property
    def range(self) -> tuple[float, float]:
        return (self.nominal * (self.min_scale - 1), self.nominal * (self.max_scale - 1))


@dataclass(frozen=True)
class RobotSpec:
    name: str = "ascent-humanoid-v1"
    pelvis_height: float = 0.89
    torso: Telescope = Telescope(0.45, 0.85, 1.25, 1200.0)
    thigh: Telescope = Telescope(0.38, 0.80, 1.40, 1500.0)
    shin: Telescope = Telescope(0.38, 0.80, 1.40, 1500.0)
    upper_arm: Telescope = Telescope(0.28, 0.80, 1.40, 500.0)
    forearm: Telescope = Telescope(0.26, 0.80, 1.40, 400.0)
    slide_speed: float = 0.25          # m/s cap on every telescoping segment
    hip_width: float = 0.18
    shoulder_width: float = 0.43
    # peak torque (N m) and no-load speed (rad/s) of each rotary joint type
    motors: dict = field(default_factory=lambda: {
        "waist_yaw": (80, 10), "waist_pitch": (150, 10), "waist_roll": (120, 10),
        "hip_yaw": (80, 12), "hip_roll": (120, 12), "hip_pitch": (160, 12),
        "knee": (160, 12), "ankle_pitch": (80, 14), "ankle_roll": (50, 14),
        "shoulder_pitch": (70, 14), "shoulder_roll": (70, 14), "shoulder_yaw": (40, 16),
        "elbow": (50, 16),
    })


# Joint ranges in radians. Positive conventions: hip/shoulder pitch = flex forward,
# knee/elbow = flex, ankle pitch = dorsiflex (toes up), roll = outward (abduction).
RANGES = {
    "waist_yaw": (-0.9, 0.9), "waist_pitch": (-0.5, 1.4), "waist_roll": (-0.5, 0.5),
    "hip_yaw": (-0.6, 0.6), "hip_roll": (-0.4, 0.8), "hip_pitch": (-0.8, 2.4),
    "knee": (0.0, 2.6), "ankle_pitch": (-0.7, 0.9), "ankle_roll": (-0.4, 0.4),
    "shoulder_pitch": (-1.0, 3.1), "shoulder_roll": (-0.3, 2.8), "shoulder_yaw": (-1.5, 1.5),
    "elbow": (0.0, 2.6),
}

# Materials: Artemis-era flight hardware palette.
MATERIALS = {
    "panel_white": dict(rgba=(0.92, 0.92, 0.9, 1), specular=0.35, shininess=0.4),
    "foil_gold": dict(rgba=(0.86, 0.66, 0.26, 1), specular=0.9, shininess=0.85, reflectance=0.15),
    "anodized_black": dict(rgba=(0.07, 0.07, 0.08, 1), specular=0.5, shininess=0.6),
    "chrome_tube": dict(rgba=(0.78, 0.8, 0.83, 1), specular=1.0, shininess=1.0, reflectance=0.35),
    "visor": dict(rgba=(0.05, 0.06, 0.08, 1), specular=1.0, shininess=1.0, reflectance=0.5),
    "accent_orange": dict(rgba=(0.95, 0.42, 0.08, 1), specular=0.4, shininess=0.5),
    "rubber": dict(rgba=(0.04, 0.04, 0.045, 1), specular=0.1, shininess=0.1),
}


class _Builder:
    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self.bodies: list[str] = []
        self.parent: dict[str, str | None] = {}
        self.rotary: list[str] = []
        self.slides: list[str] = []

    def body(self, parent: ET.Element, name: str, pos, parent_name: str | None) -> ET.Element:
        b = ET.SubElement(parent, "body", name=name, pos=_f(*pos))
        self.bodies.append(name)
        self.parent[name] = parent_name
        return b

    def hinge(self, body: ET.Element, name: str, kind: str, axis) -> None:
        lo, hi = RANGES[kind]
        ET.SubElement(body, "joint", name=name, type="hinge", axis=_f(*axis), range=_f(lo, hi),
                      armature="0.05" if kind in ("hip_pitch", "knee", "waist_pitch") else "0.02",
                      damping="0.6", frictionloss="0.3")
        self.rotary.append(name)

    def slide(self, body: ET.Element, name: str, tel: Telescope, axis=(0, 0, -1)) -> None:
        lo, hi = tel.range
        ET.SubElement(body, "joint", name=name, type="slide", axis=_f(*axis), range=_f(lo, hi),
                      armature="0.5", damping="40", frictionloss="2")
        self.slides.append(name)

    @staticmethod
    def geom(body: ET.Element, name: str, material: str, mass: float, **kw) -> None:
        attrs = {"name": name, "material": material, "mass": f"{mass:.4g}", "group": "3"}
        attrs.update({k: (_f(*v) if isinstance(v, (tuple, list)) else str(v)) for k, v in kw.items()})
        ET.SubElement(body, "geom", attrs)


def _arm(b: _Builder, chest: ET.Element, side: str, sgn: int, z_shoulder: float) -> None:
    s = b.spec
    pre = f"{side}_"
    ua = b.body(chest, pre + "upper_arm", (0, sgn * s.shoulder_width / 2, z_shoulder), "chest")
    b.hinge(ua, pre + "shoulder_pitch", "shoulder_pitch", (0, -1, 0))
    b.hinge(ua, pre + "shoulder_roll", "shoulder_roll", (sgn, 0, 0))
    b.hinge(ua, pre + "shoulder_yaw", "shoulder_yaw", (0, 0, sgn))
    b.geom(ua, pre + "shoulder_housing", "foil_gold", 0.6, type="sphere", size=(0.062,))
    b.geom(ua, pre + "upper_arm_sleeve", "panel_white", 1.2, type="capsule",
           fromto=(0, 0, -0.03, 0, 0, -0.19), size=(0.043,))

    tel = s.upper_arm
    inner_len = 0.25
    ui = b.body(ua, pre + "upper_arm_inner", (0, 0, -(tel.nominal - inner_len)), pre + "upper_arm")
    b.slide(ui, pre + "upper_arm_ext", tel)
    b.geom(ui, pre + "upper_arm_tube", "chrome_tube", 0.6, type="capsule",
           fromto=(0, 0, 0, 0, 0, -inner_len), size=(0.03,))

    fa = b.body(ui, pre + "forearm", (0, 0, -inner_len), pre + "upper_arm_inner")
    b.hinge(fa, pre + "elbow", "elbow", (0, -1, 0))
    b.geom(fa, pre + "elbow_housing", "foil_gold", 0.3, type="sphere", size=(0.046,))
    b.geom(fa, pre + "forearm_sleeve", "panel_white", 0.8, type="capsule",
           fromto=(0, 0, -0.025, 0, 0, -0.16), size=(0.037,))

    tel = s.forearm
    inner_len = 0.23
    fi = b.body(fa, pre + "forearm_inner", (0, 0, -(tel.nominal - inner_len)), pre + "forearm")
    b.slide(fi, pre + "forearm_ext", tel)
    b.geom(fi, pre + "forearm_tube", "chrome_tube", 0.4, type="capsule",
           fromto=(0, 0, 0, 0, 0, -inner_len), size=(0.026,))

    hand = b.body(fi, pre + "hand", (0, 0, -inner_len), pre + "forearm_inner")
    b.geom(hand, pre + "hand", "anodized_black", 0.5, type="box", pos=(0, 0, -0.05),
           size=(0.045, 0.028, 0.06), friction=(1.0, 0.02, 0.002))
    ET.SubElement(hand, "site", name=pre + "hand_site", pos=_f(0, 0, -0.1), size="0.01")


def _leg(b: _Builder, pelvis: ET.Element, side: str, sgn: int) -> None:
    s = b.spec
    pre = f"{side}_"
    th = b.body(pelvis, pre + "thigh", (0, sgn * s.hip_width / 2, -0.06), "pelvis")
    b.hinge(th, pre + "hip_yaw", "hip_yaw", (0, 0, sgn))
    b.hinge(th, pre + "hip_roll", "hip_roll", (sgn, 0, 0))
    b.hinge(th, pre + "hip_pitch", "hip_pitch", (0, -1, 0))
    b.geom(th, pre + "hip_housing", "foil_gold", 1.0, type="sphere", size=(0.07,))
    b.geom(th, pre + "thigh_sleeve", "panel_white", 3.5, type="capsule",
           fromto=(0, 0, -0.03, 0, 0, -0.26), size=(0.066,))

    tel = s.thigh
    inner_len = 0.33
    ti = b.body(th, pre + "thigh_inner", (0, 0, -(tel.nominal - inner_len)), pre + "thigh")
    b.slide(ti, pre + "thigh_ext", tel)
    b.geom(ti, pre + "thigh_tube", "chrome_tube", 2.0, type="capsule",
           fromto=(0, 0, 0, 0, 0, -inner_len), size=(0.046,))

    sh = b.body(ti, pre + "shin", (0, 0, -inner_len), pre + "thigh_inner")
    b.hinge(sh, pre + "knee", "knee", (0, 1, 0))
    b.geom(sh, pre + "knee_housing", "foil_gold", 0.5, type="sphere", size=(0.056,))
    b.geom(sh, pre + "shin_sleeve", "panel_white", 2.5, type="capsule",
           fromto=(0, 0, -0.03, 0, 0, -0.25), size=(0.056,))

    tel = s.shin
    inner_len = 0.33
    si = b.body(sh, pre + "shin_inner", (0, 0, -(tel.nominal - inner_len)), pre + "shin")
    b.slide(si, pre + "shin_ext", tel)
    b.geom(si, pre + "shin_tube", "chrome_tube", 1.2, type="capsule",
           fromto=(0, 0, 0, 0, 0, -inner_len), size=(0.04,))

    ft = b.body(si, pre + "foot", (0, 0, -inner_len), pre + "shin_inner")
    b.hinge(ft, pre + "ankle_pitch", "ankle_pitch", (0, -1, 0))
    b.hinge(ft, pre + "ankle_roll", "ankle_roll", (sgn, 0, 0))
    # The boot box carries the foot's mass and shape for rendering; four sole spheres do the
    # ground contact (far fewer contacts on a heightfield than a box, same support polygon).
    b.geom(ft, pre + "foot", "anodized_black", 1.2, type="box", pos=(0.035, 0, -0.045),
           size=(0.12, 0.055, 0.025), contype="0", conaffinity="0")
    for k, (sx, sy) in enumerate(((0.145, 0.04), (0.145, -0.04), (-0.075, 0.04), (-0.075, -0.04))):
        b.geom(ft, f"{pre}sole_{k}", "rubber", 0.01, type="sphere", pos=(sx, sy, -0.047), size=(0.023,),
               friction=(1.0, 0.02, 0.002))
    ET.SubElement(ft, "site", name=pre + "foot_site", pos=_f(0.035, 0, -0.07), size="0.01")


def build_robot(spec: RobotSpec = RobotSpec()) -> tuple[ET.Element, dict]:
    """Return the robot's worldbody subtree and metadata (joint, actuator, body lists)."""
    b = _Builder(spec)
    root = ET.Element("body", name="pelvis", pos=_f(0, 0, spec.pelvis_height))
    b.bodies.append("pelvis")
    b.parent["pelvis"] = None
    ET.SubElement(root, "freejoint", name="root")
    b.geom(root, "pelvis", "panel_white", 9.0, type="box", size=(0.09, 0.14, 0.07))
    ET.SubElement(root, "site", name="imu", pos="0 0 0", size="0.01")

    tel = spec.torso
    col = b.body(root, "torso_column", (0, 0, 0.08), "pelvis")
    b.slide(col, "torso_ext", tel, axis=(0, 0, 1))
    b.geom(col, "torso_tube", "chrome_tube", 2.0, type="capsule",
           fromto=(0, 0, -0.07, 0, 0, 0.06), size=(0.05,))

    chest = b.body(col, "chest", (0, 0, 0.08), "torso_column")
    b.hinge(chest, "waist_yaw", "waist_yaw", (0, 0, 1))
    b.hinge(chest, "waist_pitch", "waist_pitch", (0, 1, 0))
    b.hinge(chest, "waist_roll", "waist_roll", (1, 0, 0))
    z_sh = tel.nominal - 0.16          # shoulder line above the chest origin
    b.geom(chest, "chest", "panel_white", 14.0, type="box", pos=(0, 0, 0.17), size=(0.11, 0.17, 0.15))
    b.geom(chest, "waist_housing", "foil_gold", 0.8, type="sphere", pos=(0, 0, 0.0), size=(0.075,))
    b.geom(chest, "power_pack", "anodized_black", 2.0, type="box", pos=(-0.15, 0, 0.18),
           size=(0.05, 0.13, 0.12))
    b.geom(chest, "beacon", "accent_orange", 0.1, type="cylinder", pos=(-0.15, 0.1, 0.33),
           size=(0.02, 0.03))
    b.geom(chest, "head", "panel_white", 2.4, type="box", pos=(0.0, 0, z_sh + 0.14),
           size=(0.095, 0.085, 0.095))
    b.geom(chest, "visor", "visor", 0.3, type="box", pos=(0.07, 0, z_sh + 0.15),
           size=(0.035, 0.075, 0.05))
    b.geom(chest, "lidar", "anodized_black", 0.3, type="cylinder", pos=(0.0, 0, z_sh + 0.255),
           size=(0.045, 0.02))
    ET.SubElement(chest, "site", name="head_sensor", pos=_f(0.1, 0, z_sh + 0.15), size="0.01")
    # headlamps: a spotlight that moves with the chest and lights the way ahead
    ET.SubElement(chest, "light", name="headlamp", mode="fixed", pos=_f(0.14, 0, z_sh + 0.1), dir=_f(1, 0, -0.5),
                  directional="false", castshadow="false", cutoff="55", exponent="3",
                  attenuation="1 0.05 0.03", diffuse="0.9 0.94 1", specular="0.2 0.2 0.2")

    for side, sgn in (("left", 1), ("right", -1)):
        _arm(b, chest, side, sgn, z_sh)
        _leg(b, root, side, sgn)

    meta = {
        "bodies": b.bodies,
        "parent": b.parent,
        "rotary": b.rotary,
        "slides": b.slides,
        "spec": spec,
    }
    return root, meta


def _tree_distance(parent: dict, a: str, b: str) -> int:
    def chain(n):
        out = []
        while n is not None:
            out.append(n)
            n = parent[n]
        return out
    ca, cb = chain(a), chain(b)
    common = next(x for x in ca if x in cb)
    return ca.index(common) + cb.index(common)


def robot_xml_parts(spec: RobotSpec = RobotSpec()) -> dict:
    """XML elements to insert into a world model, plus metadata."""
    body, meta = build_robot(spec)

    assets = ET.Element("asset")
    for name, props in MATERIALS.items():
        ET.SubElement(assets, "material", name=name,
                      **{k: (_f(*v) if isinstance(v, tuple) else f"{v:.6g}") for k, v in props.items()})

    # Native position servos: force = kp (target - q) - kv qdot, clipped to the motor's peak.
    # kv includes a back-EMF term peak / no_load_speed, which reproduces a DC motor's linear
    # torque-speed line; for the telescoping segments it caps powered extension near slide_speed.
    actuators = ET.Element("actuator")
    for j in meta["rotary"]:
        kind = joint_kind(j)
        peak, no_load = spec.motors[kind]
        kp, kv = 3.0 * peak, 0.04 * peak + peak / no_load
        lo, hi = RANGES[kind]
        ET.SubElement(actuators, "general", name=j, joint=j, gaintype="fixed", biastype="affine",
                      gainprm=_f(kp), biasprm=_f(0, -kp, -kv), ctrlrange=_f(lo, hi), ctrllimited="true",
                      forcerange=_f(-peak, peak), forcelimited="true")
    for j in meta["slides"]:
        tel = _slide_telescope(spec, j)
        kp, kv = 20.0 * tel.force, tel.force / spec.slide_speed
        lo, hi = tel.range
        ET.SubElement(actuators, "general", name=j, joint=j, gaintype="fixed", biastype="affine",
                      gainprm=_f(kp), biasprm=_f(0, -kp, -kv), ctrlrange=_f(lo, hi), ctrllimited="true",
                      forcerange=_f(-tel.force, tel.force), forcelimited="true")

    # Exclude contacts only between bodies that are mechanically interlocked
    # (within two links in the tree); everything else, such as arm against
    # torso or leg against leg, collides.
    contact = ET.Element("contact")
    bodies = meta["bodies"]
    for i, a in enumerate(bodies):
        for c in bodies[i + 1:]:
            d = _tree_distance(meta["parent"], a, c)
            if d == 2:
                ET.SubElement(contact, "exclude", body1=a, body2=c)

    sensors = ET.Element("sensor")
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensors, "accelerometer", name="imu_acc", site="imu")
    ET.SubElement(sensors, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sensors, "velocimeter", name="imu_vel", site="imu")

    meta["actuators"] = meta["rotary"] + meta["slides"]
    return {"body": body, "asset": assets, "actuator": actuators, "contact": contact,
            "sensor": sensors, "meta": meta}


def _slide_telescope(spec: RobotSpec, joint: str) -> Telescope:
    key = joint.replace("left_", "").replace("right_", "").replace("_ext", "")
    return getattr(spec, key)


def joint_kind(joint: str) -> str:
    return joint.split("_", 1)[1] if joint.startswith(("left_", "right_")) else joint


def standing_pose(crouch: float = 0.0) -> dict[str, float]:
    """A balanced standing pose; ``crouch`` = hip flexion (knee = 2x, ankle = 1x keeps the
    hip over the ankle with flat feet)."""
    pose = {}
    for side in ("left", "right"):
        pose[f"{side}_hip_pitch"] = crouch
        pose[f"{side}_knee"] = 2 * crouch
        pose[f"{side}_ankle_pitch"] = crouch
        pose[f"{side}_shoulder_roll"] = 0.12
        pose[f"{side}_elbow"] = 0.35
        pose[f"{side}_shoulder_pitch"] = 0.1
    return pose


__all__ = ["RobotSpec", "Telescope", "build_robot", "robot_xml_parts", "standing_pose",
           "joint_kind", "RANGES", "MATERIALS", "math"]
