"""Record env resets and steps through a RunRecorder without importing gym.

    env = RecordingEnv(make_env(), run, body_points=lambda e: e.x.to_numpy()[:n_robot],
                       course=course_config, training_step=total_numsteps,   # resumed runs: current count
                       info_fn=lambda e, info: {"success": ..., "failure_mode": ...})
    obs = env.reset()
    if warmup:
        a, version = env.action_space.sample(), EXPLORATION
    else:
        a, version = agent.select_action(obs), updates  # Capture the version before updating.
    if len(memory) > batch_size:
        agent.update_parameters(memory, updates)
        updates += 1
    obs, r, done, info = env.step(a, policy_version=version)

On resume, record the checkpoint and continue the step counter:

    run = RunRecorder(f"runs/{run_id}", run_id=run_id, worker_id="w0",
                      resumed_from={"checkpoint_sha256": sha, "training_step": n, "policy_version": v})
    env = RecordingEnv(make_env(), run, body_points=..., course=course_config, training_step=n)

Supports gym<0.26 and gym>=0.26, returning env outputs unchanged.
RecordingError must stop training.

`on_reset(env, output)` can refresh a stale reset observation before recording
the initial state, e.g. with DittoGym's `env.get_obs()`. Env steps taken in that
hook are not recorded.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .ledger import RecordingError
from .recorder import EXPLORATION, EpisodeRecorder, RunRecorder  # noqa: F401 - re-exported


def split_step(output: tuple) -> tuple[bool, bool]:
    """(terminated, truncated) from a 4-tuple (old gym) or 5-tuple (new gym) step output."""
    if len(output) == 5:
        return bool(output[2]), bool(output[3])
    if len(output) == 4:
        done, info = bool(output[2]), output[3]
        truncated = done and bool(isinstance(info, dict) and info.get("TimeLimit.truncated", False))
        return done and not truncated, truncated
    raise RecordingError(f"step() returned {len(output)} values; expected 4 or 5")


class RecordingEnv:
    """Start a recording on reset and append each env step.

    body_points(env) returns [P, 2] positions. course is a dict with `id` and
    `goal_x`, or a callable(env) evaluated at reset. frame(env) may supply an
    HxWx3 uint8 render; otherwise the built-in renderer is used.

    Policy version precedence: step(..., policy_version=v), then the
    policy_version() callable, then set_policy_version(). Default: EXPLORATION
    (-1). Capture the version when choosing the action, before learner updates.

    Seed precedence: reset(seed=s), then a pending seed(s) call, then the
    constructor seed (value or callable(env)). The pending seed is consumed by
    one reset. The constructor seed only labels the recording.

    Training episodes advance training_step once per step; other kinds record
    None and leave the counter unchanged. Resume with the current training count.

    info_fn(env, step_info) supplies summary info on a normal ending or reset
    mid-episode. It receives the last step's info ({} before any step); include
    `success` and `failure_mode` in its result.
    """

    def __init__(self, env, run: RunRecorder, *, body_points: Callable[[Any], np.ndarray],
                 course: dict | Callable[[Any], dict], policy_version: Callable[[], int] | None = None,
                 frame: Callable[[Any], np.ndarray] | None = None, kind: str = "training",
                 on_reset: Callable[[Any, Any], Any] | None = None, training_step: int = 0,
                 seed: Any = None, info_fn: Callable[[Any, dict], dict] | None = None):
        self.env = env
        self.run = run
        self.body_points = body_points
        self.course = course
        self.policy_version = policy_version
        self.frame = frame
        self.kind = kind
        self.on_reset = on_reset
        self.training_step = training_step
        self.seed_option = seed
        self.info_fn = info_fn
        self.episode: EpisodeRecorder | None = None
        self.last_summary: dict | None = None
        self._version = EXPLORATION
        self._pending_seed = None
        self._step_info: dict = {}

    def __getattr__(self, name: str):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def seed(self, seed=None):
        """Old gym API: forward to env.seed() and record `seed` for the next episode."""
        self._pending_seed = seed
        return self.env.seed(seed)

    def set_policy_version(self, version: int) -> None:
        self._version = int(version)

    def _current_version(self) -> int:
        return int(self.policy_version()) if self.policy_version is not None else self._version

    def _open(self) -> bool:
        return self.episode is not None and not self.episode.closed

    def _end(self, status: str | None = None, error: str | None = None) -> None:
        info = None
        if status is None and self.info_fn is not None:
            try:
                info = self.info_fn(self.env, self._step_info)
            except BaseException as exc:
                self.last_summary = self.episode.end("interrupted", error=f"info_fn failed: {exc!r}")
                raise
        self.last_summary = self.episode.end(status, info=info, error=error)

    def reset(self, **kwargs):
        """End any open episode (`stopped` unless it hit a terminal flag), reset, start a new episode."""
        if self._open():
            self._end()
        output = self.env.reset(**kwargs)
        if self.on_reset is not None:
            output = self.on_reset(self.env, output)
        seed = kwargs.get("seed")
        if seed is None:
            seed = self._pending_seed
        if seed is None:
            seed = self.seed_option(self.env) if callable(self.seed_option) else self.seed_option
        self._pending_seed, self._step_info = None, {}
        course = self.course(self.env) if callable(self.course) else self.course
        training = self.kind == "training"
        self.episode = self.run.episode(
            course, seed=seed, kind=self.kind, training_step=self.training_step if training else None,
            initial_points=self.body_points(self.env),
            initial_frame=self.frame(self.env) if self.frame is not None else None)
        return output

    def step(self, action, *, policy_version: int | None = None):
        """Forward one step, record it, end the episode if done; returns the env output unchanged.

        policy_version overrides the callable/setter for this step (-1 = not chosen by a policy).
        """
        if not self._open():
            raise RecordingError("step() without an open episode; call reset() first")
        version = int(policy_version) if policy_version is not None else self._current_version()
        try:
            output = self.env.step(action)
            terminated, truncated = split_step(output)
            info = output[-1]
            self._step_info = info if isinstance(info, dict) else {}
            training_step = None
            if self.kind == "training":
                self.training_step += 1
                training_step = self.training_step
            self.episode.step(self.body_points(self.env), action, output[1], policy_version=version,
                              terminated=terminated, truncated=truncated,
                              frame=self.frame(self.env) if self.frame is not None else None,
                              training_step=training_step)
            if terminated or truncated:
                self._end()
        except RecordingError as exc:
            if self._open():
                try:
                    self._end("failed", error=str(exc))
                except RecordingError:
                    pass
            raise
        except BaseException as exc:
            if self._open():
                self._end("interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        return output

    def close(self) -> None:
        """End an open episode as `interrupted` and close the inner env.

        The RunRecorder stays open (it may serve other envs); call run.close() yourself.
        """
        try:
            if self._open():
                self._end("interrupted", error="env closed while episode open")
        finally:
            if hasattr(self.env, "close"):
                self.env.close()
