"""Composer task: spawn the walker on the course and apply joint + telescope targets.

Rewards, terminations and the high-level action live in ``env.py``; this task only owns the
physics episode (layout, spawn pose, actuation) so the gym layer can be tested on its own.
"""

from __future__ import annotations

import collections

import mujoco
import numpy as np
import tree
from dm_control import composer
from dm_control.composer.observation import observable as dm_observable
from dm_control.locomotion.mocap import cmu_mocap_data, loader
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils as ref_utils
from dm_control.locomotion.walkers import initializers

PHYSICS_TIMESTEP = tracking.DEFAULT_PHYSICS_TIMESTEP   # 0.005 s, as MoCapAct
CONTROL_TIMESTEP = 0.03                                 # the decoder's control rate


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class StandInitializer(initializers.WalkerInitializer):
    """MoCapAct's standing start (first frame of CMU_040_12), placed on the course."""

    def __init__(self):
        traj = loader.HDF5TrajectoryLoader(cmu_mocap_data.get_path_for_cmu(version="2020")).get_trajectory("CMU_040_12")
        feats = tracking._strip_reference_prefix(traj.as_dict(), "walker/")
        self._stand = tree.map_structure(lambda x: x[0], feats)
        self.spawn = (0.0, 0.0, 0.0)          # x, y, terrain height: set by the task before each reset
        self.yaw = 0.0                        # facing direction about z (rad); None keeps the clip's own facing

    def initialize_pose(self, physics, walker, random_state):
        ref_utils.set_walker_from_features(physics, walker, self._stand)
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
        # face +x: rotate about z so the face points along the course
        head = physics.bind(walker.head).xpos.copy()
        face = physics.bind(walker.mjcf_model.find("body", "face")).xpos.copy()
        fwd = face - head
        yaw_now = np.arctan2(fwd[1], fwd[0])
        self.natural_yaw = float(yaw_now)
        turn = 0.0 if self.yaw is None else self.yaw - yaw_now
        q_rot = np.array([np.cos(turn / 2), 0, 0, np.sin(turn / 2)])
        # the free joint sits on composer's attachment frame, so move the walker through its pose
        pos, quat = walker.get_pose(physics)
        quat = _quat_mul(q_rot, np.asarray(quat, dtype=float))
        # stand on the terrain under the spawn point (the mocap floor is z = 0)
        sx, sy, sz = self.spawn
        walker.set_pose(physics, position=np.array([sx, sy, float(pos[2]) + sz + 0.01]),
                        quaternion=quat / np.linalg.norm(quat))
        walker.set_velocity(physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))
        physics.data.qvel[:] = 0.0
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)


class CourseTask(composer.Task):
    """Physics episode on the lunar course: 56 joint targets + telescope targets per step."""

    def __init__(self, walker, arena, physics_timestep=PHYSICS_TIMESTEP, control_timestep=CONTROL_TIMESTEP):
        self._walker = walker
        self._arena = arena
        self._walker.create_root_joints(self._arena.attach(self._walker))
        self.command = np.zeros(4)            # target_obs for the frozen walking policy
        self._task_observables = collections.OrderedDict()
        self._task_observables["target_obs"] = dm_observable.Generic(lambda physics: self.command.copy())
        enabled = []
        enabled += self._walker.observables.proprioception
        enabled += self._walker.observables.kinematic_sensors
        enabled += self._walker.observables.dynamic_sensors
        enabled.append(self._walker.observables.sensors_touch)
        enabled.append(self._walker.observables.torso_xvel)
        enabled.append(self._walker.observables.torso_yvel)
        enabled += [self._walker.observables.telescope_pos, self._walker.observables.telescope_vel]
        enabled += list(self._task_observables.values())
        for o in enabled:
            o.enabled = True
        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)
        self.next_course = None

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    @property
    def walker(self):
        return self._walker

    @property
    def arena(self):
        return self._arena

    def action_spec(self, physics):
        n = len(self._walker.actuators) + self._walker.n_slides
        from dm_env import specs
        lo = np.concatenate([-np.ones(len(self._walker.actuators)),
                             [j.range[0] for j in self._walker.slide_joints]]) if self._walker.n_slides else -np.ones(n)
        hi = np.concatenate([np.ones(len(self._walker.actuators)),
                             [j.range[1] for j in self._walker.slide_joints]]) if self._walker.n_slides else np.ones(n)
        return specs.BoundedArray((n,), np.float64, lo, hi, name="joint_and_telescope_targets")

    def initialize_episode_mjcf(self, random_state):
        if self.next_course is not None:
            self._arena.set_course(self.next_course)
            self.next_course = None

    def initialize_episode(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)
        self._walker.sync_geometry(physics)

    def before_step(self, physics, action, random_state):
        n = len(self._walker.actuators)
        self._walker.apply_action(physics, action[:n], random_state)
        if self._walker.n_slides:
            physics.bind(self._walker.slide_actuators).ctrl = action[n:]
            self._walker.sync_geometry(physics)

    def get_reward(self, physics):
        return 0.0

    def should_terminate_episode(self, physics):
        return False

    def get_discount(self, physics):
        return 1.0


# Observation keys of MoCapAct's velocity_control high-level policy, in the sorted order its
# feature extractor was trained with (measured in run mocapact-probe-20261011-014*).
WALK_POLICY_KEYS = (
    "target_obs", "walker/actuator_activation", "walker/appendages_pos", "walker/body_height",
    "walker/end_effectors_pos", "walker/joints_pos", "walker/joints_vel", "walker/sensors_accelerometer",
    "walker/sensors_gyro", "walker/sensors_torque", "walker/sensors_touch", "walker/sensors_velocimeter",
    "walker/torso_xvel", "walker/torso_yvel", "walker/world_zaxis",
)
