"""The CMU humanoid used by MoCapAct, with telescoping segments added.

The pretrained decoder drives the 56 original joints and only sees observations of those
joints, so the observables are pinned to the original actuators. Nine prismatic joints
lengthen or shorten the thighs, shins, upper arms, forearms and torso. Each slide is the
first joint of the segment's child body, so it acts along the parent bone before the child's
own rotations (the knee, ankle, elbow or wrist moves along the bone).

Each telescoping limb bone shows as a fixed outer housing (the original capsule, shortened to
the minimum length) and a thinner inner tube that follows the slide. Both are resized at run
time, so the masses and inertias of the pretrained body are unchanged.
"""

from __future__ import annotations

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import cmu_humanoid

# (child body that moves, parent capsule that is resized, label, control group)
TELESCOPES = (
    ("ltibia", "lfemur", "l_thigh", "thigh"),
    ("rtibia", "rfemur", "r_thigh", "thigh"),
    ("lfoot", "ltibia", "l_shin", "shin"),
    ("rfoot", "rtibia", "r_shin", "shin"),
    ("lradius", "lhumerus", "l_upper_arm", "upper_arm"),
    ("rradius", "rhumerus", "r_upper_arm", "upper_arm"),
    ("lwrist", "lradius", "l_forearm", "forearm"),
    ("rwrist", "rradius", "r_forearm", "forearm"),
    ("upperback", None, "torso", "torso"),
)
GROUPS = ("thigh", "shin", "upper_arm", "forearm", "torso")     # left and right move together
DEFAULT_RANGE = {"thigh": (0.70, 1.40), "shin": (0.70, 1.40), "upper_arm": (0.70, 1.40),
                 "forearm": (0.70, 1.40), "torso": (0.75, 1.45)}
SLIDE_SPEED = 0.25          # m/s, limit on how fast a commanded length may change
# Position servos. The leg and torso slides carry the body's weight (about 690 N), so they are
# stiff and strong; the armature keeps the light swing-side mass stable at the 5 ms time step.
DRIVE = {
    "thigh": dict(kp=60000.0, damping=3000.0, armature=3.0, force=3000.0),
    "shin": dict(kp=60000.0, damping=3000.0, armature=3.0, force=3000.0),
    "torso": dict(kp=60000.0, damping=3000.0, armature=3.0, force=2500.0),
    "upper_arm": dict(kp=10000.0, damping=500.0, armature=0.5, force=500.0),
    "forearm": dict(kp=10000.0, damping=500.0, armature=0.5, force=500.0),
}
TUBE_RGBA = (0.80, 0.82, 0.85, 1.0)


def _rotate_inverse(quat, v):
    """Express a parent-frame vector in a child frame whose orientation is ``quat`` (w, x, y, z)."""
    if quat is None:
        return np.asarray(v, dtype=float)
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return rot.T @ np.asarray(v, dtype=float)


class _Observables(cmu_humanoid.CMUHumanoidObservables):
    """Same observables as MoCapAct, restricted to the original 56 actuators."""

    @composer.observable
    def actuator_activation(self):
        return observable.MJCFFeature("act", self._entity.actuators)

    @composer.observable
    def telescope_pos(self):
        return observable.MJCFFeature("qpos", self._entity.slide_joints)

    @composer.observable
    def telescope_vel(self):
        return observable.MJCFFeature("qvel", self._entity.slide_joints)


class TelescopingCMUHumanoid(cmu_humanoid.CMUHumanoidPositionControlledV2020):
    """CMU humanoid (MoCapAct's V2020 body) plus nine telescoping segments."""

    def _build(self, telescoping: bool = True, ranges: dict | None = None, skin: bool = True, **kwargs):
        super()._build(**kwargs)
        self._original_actuators = tuple(self._mjcf_root.find_all("actuator"))
        if skin:
            self._apply_skin()
        self._slides, self._slide_acts, self._bone, self._group, self._ratio = [], [], [], [], []
        self._tubes = []            # (housing geom, tube geom, axis, start, gap, bone length, min ratio)
        if not telescoping:
            return
        ranges = {**DEFAULT_RANGE, **(ranges or {})}
        for body_name, parent_geom, label, group in TELESCOPES:
            lo, hi = ranges[group]
            drive = DRIVE[group]
            body = self._mjcf_root.find("body", body_name)
            offset = np.asarray(body.pos, dtype=float)
            length = float(np.linalg.norm(offset))
            axis = offset / length                                   # along the bone, parent frame
            j = body.insert("joint", 0, name=f"tele_{label}", type="slide",
                            axis=_rotate_inverse(body.quat, axis), limited=True,
                            range=((lo - 1.0) * length, (hi - 1.0) * length),
                            damping=drive["damping"], armature=drive["armature"], stiffness=0.0)
            a = self._mjcf_root.actuator.add("position", name=f"tele_{label}", joint=j, kp=drive["kp"],
                                             ctrllimited=True, ctrlrange=j.range, forcelimited=True,
                                             forcerange=(-drive["force"], drive["force"]))
            self._slides.append(j)
            self._slide_acts.append(a)
            self._bone.append(length)
            self._group.append(GROUPS.index(group))
            self._ratio.append((lo, hi))
            if parent_geom is None:
                continue
            housing = self._mjcf_root.find("geom", parent_geom)
            centre = float(np.linalg.norm(np.asarray(housing.pos, dtype=float)))
            radius, half = float(housing.size[0]), float(housing.size[1])
            start, gap = centre - half, length - (centre + half)
            tube = body.parent.add("geom", name=f"tube_{label}", type="capsule", size=(radius * 0.62, 0.02),
                                   pos=housing.pos, quat=housing.quat, mass=0.0, rgba=TUBE_RGBA,
                                   **({"material": "tube_chrome"} if skin else {}))
            self._tubes.append((len(self._slides) - 1, housing, tube, axis, start, gap, length, lo, radius))

    def _apply_skin(self) -> None:
        """Rescue-robot look (visual only): white panels, gold joint housings, black hands and feet, dark visor."""
        root = self._mjcf_root
        body_mat = root.find("material", "self")
        body_mat.set_attributes(rgba=(0.88, 0.88, 0.86, 1.0), specular=0.4, shininess=0.5)
        a = root.asset
        a.add("material", name="joint_gold", rgba=(0.86, 0.64, 0.26, 1.0), specular=0.9, shininess=0.8, reflectance=0.1)
        a.add("material", name="rubber_black", rgba=(0.06, 0.06, 0.07, 1.0), specular=0.3, shininess=0.3)
        a.add("material", name="visor_dark", rgba=(0.03, 0.03, 0.04, 1.0), specular=1.0, shininess=1.0, reflectance=0.3)
        a.add("material", name="tube_chrome", rgba=TUBE_RGBA, specular=1.0, shininess=1.0, reflectance=0.25)
        for g in root.find_all("geom"):
            n = g.name or ""
            if n[1:].startswith(("foot", "toes", "hand", "finger", "thumb")):
                g.material = "rubber_black"
            elif n == "face":
                g.material = "visor_dark"
        for names, r in ((("lfemur", "rfemur"), 0.095), (("ltibia", "rtibia"), 0.075), (("lfoot", "rfoot"), 0.055),
                         (("lhumerus", "rhumerus"), 0.07), (("lradius", "rradius"), 0.05), (("lwrist", "rwrist"), 0.036)):
            for n in names:
                root.find("body", n).add("geom", name=f"housing_{n}", type="sphere", size=(r,), mass=0.0,
                                         contype=0, conaffinity=0, material="joint_gold")

    def _build_observables(self):
        return _Observables(self)

    @property
    def actuators(self):
        # the pretrained decoder's 56 actuators only (apply_action, observables); while the base
        # class is still building, only those exist
        return getattr(self, "_original_actuators", None) or tuple(self._mjcf_root.find_all("actuator"))

    @property
    def slide_joints(self):
        return tuple(self._slides)

    @property
    def slide_actuators(self):
        return tuple(self._slide_acts)

    @property
    def bone_lengths(self):
        return np.asarray(self._bone)

    @property
    def n_slides(self) -> int:
        return len(self._slides)

    def slide_targets(self, group_command) -> np.ndarray:
        """Slide targets (m) for one command per group in [-1, 1]: -1 shortest, 0 nominal, +1 longest."""
        c = np.clip(np.asarray(group_command, dtype=float), -1.0, 1.0)[self._group]
        lo = np.array([r[0] for r in self._ratio])
        hi = np.array([r[1] for r in self._ratio])
        ratio = 1.0 + np.where(c > 0, c * (hi - 1.0), c * (1.0 - lo))
        return (ratio - 1.0) * self.bone_lengths

    def group_ratios(self, slide_qpos) -> np.ndarray:
        """Mean length ratio of each group from slide positions (m)."""
        r = 1.0 + np.asarray(slide_qpos, dtype=float) / self.bone_lengths
        g = np.asarray(self._group)
        return np.array([r[g == k].mean() for k in range(len(GROUPS))])

    def sync_geometry(self, physics, slide_qpos=None) -> None:
        """Resize each limb's housing and inner tube to the current slide positions."""
        if not self._slides:
            return
        q = physics.bind(self.slide_joints).qpos if slide_qpos is None else slide_qpos
        model = physics.model
        for i, housing, tube, axis, start, gap, length, lo, radius in self._tubes:
            end_h = lo * length - gap                      # the housing ends where the shortest limb ends
            end_t = length + float(q[i]) - gap             # the tube reaches the current joint
            for geom, a, b, r in ((housing, start, end_h, radius),
                                  (tube, max(start, end_h - 0.04), max(end_t, end_h - 0.03), radius * 0.62)):
                b = max(b, a + 0.01)
                bound = physics.bind(geom)
                bound.pos = axis * (a + b) / 2.0
                bound.size = (r, (b - a) / 2.0, 0.0)
                model.geom_rbound[bound.element_id] = (b - a) / 2.0 + r
