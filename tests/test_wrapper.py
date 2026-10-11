"""RecordingEnv with fake envs in both gym API styles."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ariel.recording import RecordingError, RunRecorder, View, read_ledger
from ariel.recording.wrapper import EXPLORATION, RecordingEnv, split_step

COURSE = {"id": "test-step", "goal_x": 1.0, "ground_y": 0.05, "obstacles": [[0.5, 0.05, 0.6, 0.15]]}


class NewApiEnv:
    """gym>=0.26 style: reset -> (obs, info), step -> 5-tuple. Terminates at x >= goal, truncates at limit."""

    def __init__(self, limit: int = 5, goal: float = 10.0, fail_at: int | None = None):
        self.limit, self.goal, self.fail_at = limit, goal, fail_at
        self.x, self.t, self.closed, self.marker = 0.0, 0, False, "forwarded"
        self.last = None

    def points(self) -> np.ndarray:
        return np.array([[self.x, 0.1], [self.x + 0.05, 0.1], [self.x, 0.15]])

    def reset(self, seed=None, options=None):
        self.x, self.t = 0.0, 0
        self.last = (np.array([self.x]), {"seed": seed})
        return self.last

    def step(self, action):
        self.t += 1
        if self.fail_at == self.t:
            raise ValueError("simulator exploded")
        self.x += float(np.asarray(action).ravel()[0])
        terminated = self.x >= self.goal
        self.last = (np.array([self.x]), 1.0, terminated, self.t >= self.limit and not terminated, {})
        return self.last

    def close(self):
        self.closed = True


class OldApiEnv(NewApiEnv):
    """gym<0.26 style: reset -> obs, step -> 4-tuple with TimeLimit.truncated on time limit."""

    def seed(self, seed=None):
        self.seeded = seed
        return [seed]

    def reset(self):
        self.x, self.t = 0.0, 0
        self.last = np.array([self.x])
        return self.last

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if truncated:
            info["TimeLimit.truncated"] = True
        self.last = (obs, reward, terminated or truncated, info)
        return self.last


class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = RunRecorder(Path(self.tmp.name) / "r", run_id="r", video_every=0, chunk_steps=4)

    def tearDown(self):
        self.run.close()
        self.tmp.cleanup()

    def wrap(self, env, **kwargs) -> RecordingEnv:
        return RecordingEnv(env, self.run, body_points=lambda e: e.points(), course=COURSE, **kwargs)

    def finished(self) -> list[dict]:
        rows, _ = read_ledger(self.run.worker_dir / "episodes.jsonl")
        return [r for r in rows if r["event"] == "episode_finished"]

    def summary(self, episode_id: str) -> dict:
        return json.loads((self.run.worker_dir / "episodes" / episode_id / "episode.json").read_text())

    def trace(self, episode_id: str, name: str) -> np.ndarray:
        directory = self.run.worker_dir / "episodes" / episode_id
        return np.concatenate([np.load(p)[name] for p in sorted(directory.glob("trace-[0-9]*.npz"))])

    def play(self, env: RecordingEnv, action: float, new_api: bool) -> int:
        env.reset()
        steps = 0
        while True:
            out = env.step([action])
            steps += 1
            if (out[2] or out[3]) if new_api else out[2]:
                return steps

    def test_every_episode_recorded_new_api(self):
        env = self.wrap(NewApiEnv(limit=5, goal=1.0))
        self.assertEqual(self.play(env, 0.1, True), 5)        # time limit
        self.assertEqual(self.play(env, 0.5, True), 2)        # goal
        self.assertEqual(self.play(env, 0.1, True), 5)
        rows = self.finished()
        self.assertEqual([r["status"] for r in rows], ["truncated", "terminated", "truncated"])
        self.assertEqual([r["steps"] for r in rows], [5, 2, 5])
        self.assertTrue(env.episode.closed)
        self.assertEqual(len(self.trace(rows[0]["episode_id"], "step")), 5)

    def test_every_episode_recorded_old_api(self):
        env = self.wrap(OldApiEnv(limit=5, goal=1.0))
        self.assertEqual(self.play(env, 0.1, False), 5)
        self.assertEqual(self.play(env, 0.5, False), 2)
        self.assertEqual([r["status"] for r in self.finished()], ["truncated", "terminated"])

    def test_split_step(self):
        self.assertEqual(split_step((0, 0, True, {})), (True, False))
        self.assertEqual(split_step((0, 0, True, {"TimeLimit.truncated": True})), (False, True))
        self.assertEqual(split_step((0, 0, False, {"TimeLimit.truncated": True})), (False, False))
        self.assertEqual(split_step((0, 0, False, True, {})), (False, True))
        with self.assertRaises(RecordingError):
            split_step((0, 0, False))

    def test_reset_mid_episode_stops_previous(self):
        env = self.wrap(NewApiEnv(limit=50))
        env.reset()
        env.step([0.1])
        env.step([0.1])
        env.reset()
        env.step([0.1])
        env.close()
        rows = self.finished()
        self.assertEqual([(r["status"], r["steps"]) for r in rows], [("stopped", 2), ("interrupted", 1)])
        self.assertEqual(self.run._next_index, 2)

    def test_outputs_unchanged(self):
        for inner in (NewApiEnv(), OldApiEnv()):
            env = self.wrap(inner)
            self.assertIs(env.reset(), inner.last)
            self.assertIs(env.step([0.1]), inner.last)
            env.close()
        new = NewApiEnv()
        self.assertEqual(self.wrap(new).reset(seed=7)[1], {"seed": 7})

    def test_policy_version_lineage_callable(self):
        updates = [0]
        env = self.wrap(NewApiEnv(limit=6), policy_version=lambda: updates[0])
        env.reset()
        for _ in range(6):
            env.step([0.1])
            updates[0] += 1
        episode_id = self.finished()[0]["episode_id"]
        self.assertEqual(self.trace(episode_id, "policy_version").tolist(), [0, 1, 2, 3, 4, 5])
        summary = self.summary(episode_id)
        self.assertEqual((summary["policy_version_first"], summary["policy_version_last"],
                          summary["policy_versions_distinct"]), (0, 5, 6))

    def test_policy_version_setter_and_training_step(self):
        env = self.wrap(NewApiEnv(limit=3), training_step=100)
        env.set_policy_version(4)
        env.reset()
        env.step([0.1])
        env.set_policy_version(9)
        env.step([0.1])
        env.step([0.1])
        env.reset()
        env.step([0.1])
        env.close()
        first, second = (self.summary(r["episode_id"]) for r in self.finished())
        self.assertEqual(self.trace(first["episode_id"], "policy_version").tolist(), [4, 9, 9])
        self.assertEqual((first["training_step_start"], first["training_step_end"]), (100, 103))
        self.assertEqual((second["training_step_start"], second["training_step_end"]), (103, 104))
        self.assertEqual(env.training_step, 104)

    def test_step_without_open_episode_raises(self):
        env = self.wrap(NewApiEnv(limit=1))
        with self.assertRaises(RecordingError):
            env.step([0.1])
        env.reset()
        env.step([0.1])
        with self.assertRaises(RecordingError):
            env.step([0.1])

    def test_env_exception_interrupts_episode(self):
        env = self.wrap(NewApiEnv(limit=10, fail_at=3))
        env.reset()
        env.step([0.1])
        env.step([0.1])
        with self.assertRaises(ValueError):
            env.step([0.1])
        row = self.finished()[0]
        self.assertEqual((row["status"], row["steps"]), ("interrupted", 2))
        self.assertIn("simulator exploded", self.summary(row["episode_id"])["error"])

    def test_recording_error_fails_episode(self):
        env = RecordingEnv(NewApiEnv(), self.run, course=COURSE,
                           body_points=lambda e: e.points() if e.t < 2 else e.points()[:2])
        env.reset()
        env.step([0.1])
        with self.assertRaises(RecordingError):
            env.step([0.1])
        self.assertEqual(self.finished()[0]["status"], "failed")

    def test_policy_version_defaults_to_exploration(self):
        env = self.wrap(NewApiEnv(limit=2))
        env.reset()
        env.step([0.1])
        env.step([0.1])
        episode_id = self.finished()[0]["episode_id"]
        self.assertEqual(self.trace(episode_id, "policy_version").tolist(), [EXPLORATION, EXPLORATION])
        summary = self.summary(episode_id)
        self.assertEqual((summary["exploration_steps"], summary["policy_version_first"]), (2, None))

    def test_resumed_from_recorded(self):
        resumed = {"checkpoint_sha256": "c" * 64, "training_step": 40, "policy_version": 3}
        root = Path(self.tmp.name) / "resumed"
        with RunRecorder(root, run_id="r", video_every=0, resumed_from=resumed) as run:
            env = RecordingEnv(NewApiEnv(limit=1), run, body_points=lambda e: e.points(), course=COURSE,
                               training_step=40)
            env.reset()
            env.step([0.1], policy_version=3)
            self.assertEqual(env.last_summary["training_step_start"], 40)
        rows, _ = read_ledger(root / "workers" / "w0" / "episodes.jsonl")
        opened = [r for r in rows if r["event"] == "worker_opened"]
        self.assertEqual([r["resumed_from"] for r in opened], [resumed])

    def test_policy_version_keyword_overrides(self):
        env = self.wrap(NewApiEnv(limit=4), policy_version=lambda: 7)
        env.reset()
        env.step([0.1], policy_version=EXPLORATION)
        env.step([0.1], policy_version=EXPLORATION)
        env.step([0.1])
        env.step([0.1], policy_version=8)
        episode_id = self.finished()[0]["episode_id"]
        self.assertEqual(self.trace(episode_id, "policy_version").tolist(), [-1, -1, 7, 8])
        summary = self.summary(episode_id)
        self.assertEqual((summary["exploration_steps"], summary["policy_version_first"],
                          summary["policy_version_last"]), (2, 7, 8))

    def seeds(self) -> list:
        rows, _ = read_ledger(self.run.worker_dir / "episodes.jsonl")
        return [r["seed"] for r in rows if r["event"] == "episode_started"]

    def test_seed_capture(self):
        inner = OldApiEnv(limit=1)
        env = self.wrap(inner)
        self.assertEqual(env.seed(11), [11])                  # forwarded
        self.assertEqual(inner.seeded, 11)
        for _ in range(2):
            env.reset()
            env.step([0.1])
        new = self.wrap(NewApiEnv(limit=1), seed=lambda e: f"x={e.x}")
        new.reset(seed=5)
        new.step([0.1])
        new.reset()
        new.step([0.1])
        fixed = self.wrap(NewApiEnv(limit=1), seed=3)
        fixed.reset()
        fixed.step([0.1])
        self.assertEqual(self.seeds(), [11, None, 5, "x=0.0", 3])

    def test_non_training_kind_has_no_training_step(self):
        with RunRecorder(Path(self.tmp.name) / "f", run_id="f", video_every=0, fixture=True) as run:
            env = RecordingEnv(NewApiEnv(limit=2), run, body_points=lambda e: e.points(), course=COURSE,
                               kind="fixture", training_step=50)
            env.reset()
            env.step([0.1])
            env.step([0.1])
            self.assertEqual(env.training_step, 50)
            summary = env.last_summary
            self.assertEqual((summary["kind"], summary["training_step_start"], summary["training_step_end"]),
                             ("fixture", None, None))
            self.assertTrue(summary["fixture"])

    def test_info_fn_reaches_summary(self):
        seen = []

        def info_fn(e, step_info):
            seen.append(step_info)
            return {"success": e.x >= e.goal, "failure_mode": None if e.x >= e.goal else "timeout"}

        env = self.wrap(NewApiEnv(limit=3, goal=1.0), info_fn=info_fn)
        self.play(env, 0.1, True)
        self.play(env, 0.6, True)
        env.reset()
        env.step([0.1])
        env.reset()  # Resetting mid-episode still calls info_fn.
        env.close()                                           # interrupted: no info_fn
        infos = [self.summary(r["episode_id"])["info"] for r in self.finished()]
        self.assertEqual(infos, [{"success": False, "failure_mode": "timeout"},
                                 {"success": True, "failure_mode": None},
                                 {"success": False, "failure_mode": "timeout"}, {}])
        self.assertEqual(seen, [{}, {}, {}])

    def test_info_fn_error_interrupts_episode(self):
        env = self.wrap(NewApiEnv(limit=1), info_fn=lambda e, i: 1 / 0)
        env.reset()
        with self.assertRaises(ZeroDivisionError):
            env.step([0.1])
        row = self.finished()[0]
        self.assertEqual((row["status"], row["steps"]), ("interrupted", 1))
        self.assertIn("info_fn failed", self.summary(row["episode_id"])["error"])

    def test_on_reset_hook_and_attribute_forwarding(self):
        inner = NewApiEnv()
        calls = []

        def fix(e, output):
            calls.append(e)
            return (np.array([42.0]), output[1])

        env = self.wrap(inner, on_reset=fix)
        self.assertEqual(env.reset()[0][0], 42.0)
        self.assertEqual(calls, [inner])
        self.assertEqual(env.marker, "forwarded")
        env.close()
        self.assertTrue(inner.closed)
        self.assertEqual(self.finished()[0]["status"], "interrupted")

    def test_course_callable_and_seed(self):
        env = RecordingEnv(NewApiEnv(limit=1), self.run, body_points=lambda e: e.points(),
                           course=lambda e: {"id": "c", "goal_x": 1.0, "limit": e.limit})
        env.reset(seed=3)
        env.step([0.1])
        rows, _ = read_ledger(self.run.worker_dir / "episodes.jsonl")
        self.assertEqual(rows[0]["event"], "worker_opened")             # ledger rows are not only episodes
        [started] = [r for r in rows if r["event"] == "episode_started"]
        self.assertEqual((started["course"], started["seed"]), ({"id": "c", "goal_x": 1.0, "limit": 1}, 3))


class WrapperVideoTest(unittest.TestCase):
    def test_frame_callable_and_builtin_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunRecorder(Path(tmp) / "r", run_id="r", view=View(width_px=64, height_px=64)) as run:
                for frame in (None, lambda e: np.full((32, 48, 3), 200, dtype=np.uint8)):
                    env = RecordingEnv(NewApiEnv(limit=3), run, body_points=lambda e: e.points(),
                                       course=COURSE, frame=frame)
                    env.reset()
                    for _ in range(3):
                        env.step([0.1])
                    self.assertEqual(env.last_summary["video_frames"], 4)   # initial state + 3 steps
                    self.assertEqual(env.last_summary["status"], "truncated")


class CostsTest(unittest.TestCase):
    def test_costs_reads_recorder_settings(self):
        import contextlib
        import inspect
        import io
        from ariel.recording import costs
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            report = costs.main(["--points", "40", "--steps", "6", "--episodes", "1", "--size", "64",
                                 "--trace-max-points", "16", "--out", tmp])
        self.assertEqual(report["settings"]["trace_max_points"], 16)
        self.assertEqual(report["settings"]["trace_points_recorded"], 16)
        default = inspect.signature(RunRecorder).parameters["chunk_steps"].default
        self.assertEqual(report["settings"]["chunk_steps"], default)
        modes = report["modes"]
        self.assertEqual((modes["trace_only"]["settings"]["video_every"],
                          modes["trace_render_encode"]["settings"]["video_every"]), (0, 1))
        video = modes["trace_render_encode"]
        self.assertEqual(video["bytes_per_800_step_episode"]["bytes"], round(video["bytes_per_step"] * 800))
        self.assertGreater(video["video_bytes_per_step"], 0)


if __name__ == "__main__":
    unittest.main()
