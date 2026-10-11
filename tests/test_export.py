"""Viewer export tests using scripted fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import imageio_ffmpeg
import numpy as np

from ariel.recording import RecordingError, RunRecorder, View
from ariel.analysis.eval_plan import build_eval_plan, select_checkpoints
from ariel.recording.export import MAX_ACTION, MAX_FRAMES, MAX_POINTS, _numbered, export_viewer
from ariel.recording.fixture import COURSE, LABEL, make_fixture


def count_frames(path: Path) -> int:
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]     # decodes fully; no pipes left open


class FixtureExportTest(unittest.TestCase):
    RESUMED = {"checkpoint_sha256": "b" * 64, "training_step": 9_000, "policy_version": 6}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.run_dir = make_fixture(root / "run", episodes=4, steps=60, seed=1, chunk_steps=25)
        cls.simulate_crash(cls.run_dir)
        cls.out = root / "viewer"
        cls.index = export_viewer(cls.run_dir, cls.out)
        cls.by_id = {e["id"]: e for e in cls.index["episodes"]}

    @staticmethod
    def simulate_crash(run_dir: Path) -> None:
        """A second worker dies mid-episode; reopening it recovers the episode as interrupted."""
        run = RunRecorder(run_dir, run_id="fixture-seed1", worker_id="w1", fixture=True, chunk_steps=25)
        rng = np.random.default_rng(0)
        points = rng.random((500, 2)) * 0.1 + [0.2, 0.1]
        FixtureExportTest.crash_points = points
        ep = run.episode(COURSE, kind="fixture", training_step=10_000, initial_points=points)
        for t in range(60):
            ep.step(points + 0.001 * t, [0.0], 0.1, policy_version=7)
        ep.video.close()
        run._lock.close()
        RunRecorder(run_dir, run_id="fixture-seed1", worker_id="w1", fixture=True,
                    resumed_from=FixtureExportTest.RESUMED).close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture_labelled_everywhere(self):
        run = json.loads((self.run_dir / "run.json").read_text())
        self.assertTrue(run["fixture"])
        self.assertEqual(run["meta"]["label"], LABEL)
        index = json.loads((self.out / "index.json").read_text())
        self.assertTrue(index["fixture"])
        self.assertEqual(index["label"], LABEL)
        self.assertEqual(len(index["episodes"]), 5)
        for episode in index["episodes"]:
            self.assertEqual((episode["kind"], episode["fixture"], episode["label"]), ("fixture", True, LABEL))
            self.assertEqual(episode["course"]["id"], COURSE["id"])
            self.assertIn("goal_x", episode["course"])
            if episode["worker"] == "w0" and episode["status"] != "interrupted":
                info = episode["info"]
                self.assertTrue(info["scripted"])
                self.assertIsInstance(info["success"], bool)
                self.assertEqual(info["failure_mode"] is None, info["success"])
            trace = json.loads((self.out / episode["trace"]).read_text())
            self.assertTrue(trace["fixture"])
            self.assertEqual(trace["label"], LABEL)
        self.assertTrue(all(c["fixture"] for c in index["checkpoints"]))
        self.assertEqual([c["note"].split(":")[0] for c in index["checkpoints"]], ["early", "mid", "late"])
        steps = [c["training_step"] for c in index["checkpoints"]]
        self.assertEqual(steps, sorted(set(steps)))

    def test_eval_plan_default_labels_select_fixture_checkpoints(self):
        picks = select_checkpoints(self.run_dir, selected_at="2026-01-01T00:00:00+00:00")
        self.assertEqual([p["label"] for p in picks], ["early", "mid", "late"])
        self.assertEqual(len({p["sha256"] for p in picks}), 3)
        notes = {c["sha256"]: c["note"] for c in self.index["checkpoints"]}
        self.assertTrue(all(notes[p["sha256"]].startswith(p["label"]) for p in picks))

    def test_worker_sessions_and_runtime_identity(self):
        run = json.loads((self.run_dir / "run.json").read_text())
        self.assertIn("git_commit", self.index)
        self.assertEqual(self.index["git_commit"], run.get("git_commit"))
        workers = {w["worker"]: w for w in self.index["workers"]}
        self.assertEqual(set(workers), {"w0", "w1"})
        for worker in workers.values():
            identity = json.loads((self.run_dir / "workers" / worker["worker"] / "worker.json").read_text())
            self.assertEqual((worker["host"], worker["modal_task_id"]), (identity["host"], identity["modal_task_id"]))
        self.assertEqual(len(workers["w0"]["sessions"]), 1)
        self.assertIsNone(workers["w0"]["sessions"][0]["resumed_from"])
        first, second = workers["w1"]["sessions"]
        self.assertEqual((first["resumed_from"], first["recovered_episodes"]), (None, []))
        self.assertEqual(second["resumed_from"], self.RESUMED)
        [recovered] = [e for e in self.index["episodes"] if e["recovered"]]
        self.assertEqual(second["recovered_episodes"], [recovered["id"]])

    def test_order_and_lineage(self):
        keys = [(e["training_step_start"], e["worker"], e["index"]) for e in self.index["episodes"]]
        self.assertEqual(keys, sorted(keys))
        finished = [e for e in self.index["episodes"] if not e["recovered"]]
        firsts = [e["policy_version_first"] for e in finished]
        self.assertEqual(firsts, sorted(firsts))
        self.assertLess(firsts[0], firsts[-1])

    def test_exploration_steps_flagged(self):
        first = min((e for e in self.index["episodes"] if e["worker"] == "w0"), key=lambda e: e["index"])
        self.assertEqual(first["exploration_steps"], 15)               # 25% of 60 steps
        self.assertEqual(first["trace_exploration_steps"], 15)
        self.assertGreaterEqual(first["policy_version_first"], 0)      # -1 never counts as a policy version
        self.assertEqual(first["trace_policy_version_first"], first["policy_version_first"])
        frames = json.loads((self.out / first["trace"]).read_text())["frames"]
        self.assertIsNone(frames[0]["exploration"])
        for frame in frames[1:]:
            self.assertEqual(frame["exploration"], frame["step"] <= 15)
            self.assertEqual(frame["policy_version"] == -1, frame["exploration"])
        others = [e for e in self.index["episodes"] if e is not first]
        self.assertTrue(all(e["exploration_steps"] in (0, None) for e in others))

    def test_frame_fields(self):
        for episode in self.index["episodes"]:
            trace = json.loads((self.out / episode["trace"]).read_text())
            self.assertEqual(trace["action_dims"], 1 if episode["worker"] == "w1" else 2)
            for key in ("aspect", "action", "action_mean_abs", "exploration", "end"):
                self.assertIn(key, trace["fields"])
            for frame in trace["frames"][1:]:
                self.assertEqual(len(frame["action"]), trace["action_dims"])
                self.assertAlmostEqual(frame["action_mean_abs"], float(np.mean(np.abs(frame["action"]))), places=2)
            self.assertGreater(trace["frames"][0]["aspect"], 1.0)
            if episode["worker"] == "w0":
                self.assertGreater(max(f["aspect"] for f in trace["frames"]), trace["frames"][0]["aspect"])
            self.assertTrue(all("end" not in f for f in trace["frames"][:-1]))
            end = trace["frames"][-1]["end"]
            self.assertEqual(end["status"], episode["status"])
            self.assertEqual((end["terminated"], end["truncated"]),
                             (episode["status"] == "terminated", episode["status"] == "truncated"))
            self.assertEqual(end["final_step"], not episode["recovered"])

    def test_points_reconstructed_from_offsets(self):
        [episode] = [e for e in self.index["episodes"] if e["recovered"]]
        trace = json.loads((self.out / episode["trace"]).read_text())
        keep = np.unique(np.linspace(0, 499, MAX_POINTS).round().astype(int))
        for frame in trace["frames"]:
            expected = self.crash_points[keep] + 0.001 * max(frame["step"] - 1, 0)
            np.testing.assert_allclose(frame["points"], expected, atol=1e-3)

    def test_interrupted_and_terminated_episodes(self):
        statuses = [e["status"] for e in self.index["episodes"] if e["worker"] == "w0"]
        self.assertIn("interrupted", statuses)
        self.assertIn("terminated", statuses)
        raised = [e for e in self.index["episodes"] if e["status"] == "interrupted" and not e["recovered"]]
        self.assertEqual(len(raised), 1)
        self.assertIn("FixtureInterruption", raised[0]["error"])
        self.assertLess(raised[0]["steps"], 60)

    def test_recovered_episode_exported_without_invented_values(self):
        [episode] = [e for e in self.index["episodes"] if e["recovered"]]
        summary = json.loads((self.run_dir / "workers" / "w1" / "episodes" / episode["id"] / "episode.json")
                             .read_text())
        self.assertEqual(episode["status"], "interrupted")
        for key in ("steps", "return", "policy_version_first", "policy_version_last", "policy_versions_distinct",
                    "exploration_steps"):
            self.assertIsNone(episode[key], key)                     # lost with the crash, not recomputed
        self.assertTrue(episode["missing"])
        self.assertEqual(episode["trace_steps"], 50)                 # two complete chunks of 25
        self.assertEqual((episode["steps_recorded"], episode["last_step"]), (50, 50))
        for key in ("trace_return", "trace_policy_version_first", "trace_policy_version_last",
                    "trace_exploration_steps"):
            self.assertEqual(episode[key], summary[key], key)        # passed through from the recorder
        self.assertEqual((episode["trace_policy_version_first"], episode["trace_exploration_steps"]), (7, 0))
        self.assertAlmostEqual(episode["trace_return"], 5.0, places=4)
        self.assertEqual(episode["settings"]["chunk_steps"], 25)
        self.assertTrue(episode["initial_state"])
        self.assertTrue(episode["video"])

    def test_trace_bounded_and_videos_decodable(self):
        for episode in self.index["episodes"]:
            trace = json.loads((self.out / episode["trace"]).read_text())
            self.assertLessEqual(len(trace["frames"]), MAX_FRAMES)
            self.assertEqual(trace["frames"][0]["step"], 0)
            for frame in trace["frames"]:
                self.assertLessEqual(len(frame["points"]), MAX_POINTS)
            if not episode["recovered"]:
                self.assertEqual(trace["frames"][-1]["step"], episode["steps"])
                self.assertAlmostEqual(trace["frames"][-1]["reward"], episode["return"], places=3)
            self.assertTrue(episode["video"])
            frames = sum(count_frames(self.out / path) for path in episode["video"])
            self.assertGreater(frames, 0)
            if episode["video_frames"] is not None:
                self.assertEqual(frames, episode["video_frames"])

    def test_export_is_repeatable(self):
        again = export_viewer(self.run_dir, self.out)
        self.assertEqual([e["id"] for e in again["episodes"]], [e["id"] for e in self.index["episodes"]])

    def test_refuses_existing_run(self):
        with self.assertRaises(FileExistsError):
            make_fixture(self.run_dir)


class LongEpisodeTest(unittest.TestCase):
    def test_downsample_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = RunRecorder(Path(tmp) / "run", run_id="long", video_every=0, trace_max_points=400)
            rng = np.random.default_rng(0)
            with run.episode({"ground_y": 0.1}, training_step=0) as ep:
                for t in range(1500):
                    ep.step(rng.random((1000, 2)), [0.0, 1.0], 1.0, policy_version=t // 100,
                            truncated=t == 1499)
            run.close()
            index = export_viewer(Path(tmp) / "run", Path(tmp) / "out")
            self.assertFalse(index["fixture"])
            [episode] = index["episodes"]
            self.assertIsNone(episode["label"])
            trace = json.loads((Path(tmp) / "out" / episode["trace"]).read_text())
            self.assertLessEqual(len(trace["frames"]), MAX_FRAMES)
            self.assertEqual(trace["frames"][-1]["step"], 1500)
            self.assertEqual(trace["frames"][-1]["reward"], 1500.0)
            self.assertEqual(trace["frames"][-1]["policy_version"], 14)
            self.assertEqual(trace["points_exported"], MAX_POINTS)
            self.assertEqual(len(trace["frames"][5]["points"]), MAX_POINTS)

    def test_action_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunRecorder(Path(tmp) / "run", run_id="wide", video_every=0) as run:
                with run.episode({"id": "flat", "goal_x": 1.0, "ground_y": 0.1}) as ep:
                    ep.step(np.random.default_rng(0).random((20, 2)), np.full(100, -0.5), float("nan"),
                            policy_version=-1, terminated=True)
            index = export_viewer(Path(tmp) / "run", Path(tmp) / "out")
            [episode] = index["episodes"]
            self.assertIsNone(episode["return"])
            self.assertEqual((episode["nonfinite_rewards"], episode["trace_nonfinite_rewards"]), (1, 1))
            trace = json.loads((Path(tmp) / "out" / episode["trace"]).read_text())
            last = trace["frames"][-1]
            self.assertEqual(trace["action_dims"], 100)
            self.assertEqual(last["action"], [-0.5] * MAX_ACTION)
            self.assertEqual(last["action_mean_abs"], 0.5)
            self.assertIsNone(last["reward"])
            self.assertTrue(last["exploration"])
            self.assertEqual(last["end"], {"status": "terminated", "terminated": True, "truncated": False,
                                           "final_step": True})
            # trace-meta holds step 1 when no initial state is supplied.
            self.assertFalse(trace["initial_state"])
            self.assertEqual([f["step"] for f in trace["frames"]], [1])

    def test_numeric_sort_for_glob_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("trace-10.npz", "trace-9.npz", "trace-100.npz"):
                (Path(tmp) / name).touch()
            self.assertEqual([p.name for p in _numbered(Path(tmp), "trace-[0-9]*.npz")],
                             ["trace-9.npz", "trace-10.npz", "trace-100.npz"])

    def record(self, root: Path, run_id: str = "r", video_every: int = 1) -> Path:
        with RunRecorder(root / "run", run_id=run_id, video_every=video_every,
                         view=View(width_px=64, height_px=64)) as run:
            with run.episode({"id": "flat", "goal_x": 1.0, "ground_y": 0.1},
                             initial_points=np.random.default_rng(0).random((20, 2))) as ep:
                for t in range(3):
                    ep.step(np.random.default_rng(t + 1).random((20, 2)), [0.1], 1.0, policy_version=0,
                            terminated=t == 2)
        return root / "run"

    def test_tampered_video_segment_not_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.record(Path(tmp))
            [segment] = run_dir.glob("workers/w0/episodes/*/video.seg000.mp4")
            out = Path(tmp) / "out"
            export_viewer(run_dir, out)                                   # good copy first ...
            segment.write_bytes(segment.read_bytes() + b"tampered")
            [episode] = export_viewer(run_dir, out)["episodes"]           # ... then removed on re-export
            self.assertEqual(episode["video"], [])
            self.assertTrue(any("video.seg000.mp4" in note and "sha256" in note for note in episode["missing"]))
            self.assertFalse((out / "episodes" / episode["id"] / "video.seg000.mp4").exists())
            [meta] = run_dir.glob("workers/w0/episodes/*/trace-meta.npz")
            meta.write_bytes(meta.read_bytes() + b"x")
            [episode] = export_viewer(run_dir, out)["episodes"]
            self.assertTrue(any(note.startswith("trace-meta.npz does not match") for note in episode["missing"]))
            trace = json.loads((out / episode["trace"]).read_text())
            self.assertEqual((trace["initial_state"], trace["frames"][0]["step"]), (False, 1))

    def test_newer_schema_disclosed_or_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.record(Path(tmp), video_every=0)
            [summary_path] = run_dir.glob("workers/w0/episodes/*/episode.json")
            summary = json.loads(summary_path.read_text())
            summary_path.write_text(json.dumps({**summary, "schema_version": 3}))
            [episode] = export_viewer(run_dir, Path(tmp) / "out")["episodes"]
            self.assertIsNone(episode["trace"])
            self.assertEqual(episode["video"], [])
            self.assertTrue(any("schema_version 3" in note for note in episode["missing"]))
            run_json = run_dir / "run.json"
            run_json.write_text(json.dumps({**json.loads(run_json.read_text()), "schema_version": 3}))
            with self.assertRaisesRegex(RecordingError, "schema_version 3"):
                export_viewer(run_dir, Path(tmp) / "out2")

    def test_missing_run_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RecordingError):
                export_viewer(tmp, Path(tmp) / "out")


class HeldoutGuardTest(unittest.TestCase):
    """Held-out (final_only) evaluation runs are exported only when the plan's guard allows it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        checkpoint = {"label": "late", "path": "ckpt.pth", "sha256": "a" * 64, "policy_version": 5,
                      "selected_at": "2026-01-01T00:00:00+00:00"}
        self.plan = build_eval_plan([checkpoint], {"id": "show", "goal_x": 1.0}, [{"id": "hold", "goal_x": 1.0}],
                                    [0], 1, plan_id="t", created_at="2026-01-01T00:00:00+00:00")
        self.runs = {}
        for job in self.plan["jobs"]:
            spec = job["recorder"]
            run_dir = self.root / spec["run_id"]
            with RunRecorder(run_dir, run_id=spec["run_id"], meta=spec["meta"], video_every=1) as run:
                with run.episode(spec["episode"]["course"], seed=0, kind="evaluation") as ep:
                    for t in range(3):
                        ep.step(np.random.default_rng(t).random((30, 2)), [0.1], 1.0, policy_version=5,
                                terminated=t == 2)
                    ep.end(info={"success": True, "failure_mode": None})
            self.runs[job["course"]["id"]] = run_dir
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))

    def tearDown(self):
        self.tmp.cleanup()

    def test_withheld_without_plan(self):
        out = self.root / "out"
        index = export_viewer(self.runs["hold"], out)
        self.assertFalse(index["heldout"]["reportable"])
        self.assertEqual(index["withheld_episodes"], 1)
        [episode] = json.loads((out / "index.json").read_text())["episodes"]
        self.assertTrue(episode["withheld"])
        self.assertIn("held-out", episode["withheld_reason"])
        self.assertEqual((episode["trace"], episode["video"]), (None, []))
        for key in ("return", "status", "info", "steps", "trace_return"):
            self.assertNotIn(key, episode)
        self.assertFalse((out / "episodes").exists())

    def test_reportable_with_plan(self):
        out = self.root / "out"
        export_viewer(self.runs["hold"], out)                               # withheld first ...
        index = export_viewer(self.runs["hold"], out, plan=self.plan_path)  # ... then allowed by the plan
        self.assertTrue(index["heldout"]["reportable"])
        [episode] = index["episodes"]
        self.assertFalse(episode["withheld"])
        self.assertEqual(episode["return"], 3.0)
        self.assertTrue((out / episode["trace"]).exists() and episode["video"])

    def test_withheld_when_selected_after_viewing(self):
        plan = {**self.plan, "heldout_first_viewed_at": "2025-12-31T00:00:00+00:00"}
        out = self.root / "out"
        export_viewer(self.runs["hold"], out, plan=self.plan_path)
        index = export_viewer(self.runs["hold"], out, plan=plan)
        [episode] = index["episodes"]
        self.assertTrue(episode["withheld"])
        self.assertIn("selected after", episode["withheld_reason"])
        self.assertFalse((out / "episodes" / episode["id"]).exists())    # earlier export removed

    def test_showcase_not_guarded(self):
        index = export_viewer(self.runs["show"], self.root / "out")
        self.assertIsNone(index["heldout"])
        self.assertFalse(index["episodes"][0]["withheld"])


if __name__ == "__main__":
    unittest.main()
