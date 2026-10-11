"""Analysis tests using synthetic recordings, not learned behavior.

Fixture runs use fixture=True. Training and evaluation test runs use `synthetic-`
ids and meta {"synthetic_test_data": True}.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from ariel.analysis import (UnsupportedSchemaError, course_hash, course_id, curve_from_rows, episode_metrics,
                            learning_curve, load_episode, load_run)
from ariel.analysis.eval_plan import (HeldoutGuardError, build_eval_plan, heldout_reportable, main as plan_main,
                                       mark_heldout_viewed, select_checkpoints)
from ariel.analysis.metrics import aspect_series
from ariel.analysis.report import bootstrap_ci, build_report
from ariel.analysis.svg import bar_chart, line_chart
from ariel.recording import RunRecorder, read_ledger

SYNTHETIC = {"synthetic_test_data": True}
COURSE = {"id": "synthetic-showcase", "ground_y": 0.0, "goal_x": 0.495, "obstacles": []}


def body(cx: float, width: float = 0.1, height: float = 0.1, n: int = 6) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(-width / 2, width / 2, n), np.linspace(0, height, n))
    return np.stack([xs.ravel() + cx, ys.ravel() + 0.1], axis=1)


def recorder(root: Path, run_id: str, *, worker: str = "w0", fixture: bool = False, meta=None,
             resumed_from=None) -> RunRecorder:
    return RunRecorder(root / run_id, run_id=run_id, worker_id=worker, fixture=fixture, video_every=0,
                       chunk_steps=10, meta=meta if meta is not None else SYNTHETIC, resumed_from=resumed_from)


def play(run: RunRecorder, *, steps: int = 20, dx: float = 0.02, start: float = 0.1, kind: str = "training",
         course=COURSE, training_step=0, version=0, reshape=False, info=None, status=None, seed=0,
         truncate=False):
    with run.episode(course, seed=seed, kind=kind, training_step=training_step,
                     initial_points=body(start)) as ep:
        for i in range(1, steps + 1):
            shape = (0.2, 0.05) if reshape and i % 2 else (0.1, 0.1)
            ep.step(body(start + dx * i, *shape), np.full(3, 0.5), 1.0, policy_version=version + i // 10,
                    truncated=truncate and i == steps)
        return ep.end(status, info=info)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_progress_completion_and_shape(self):
        with recorder(self.root, "synthetic-train") as run:
            summary = play(run, reshape=True, status="terminated")
        ep = load_episode(self.root / "synthetic-train/workers/w0/episodes" / summary["episode_id"])
        self.assertTrue(ep.trace_complete)
        m = episode_metrics(ep)
        self.assertAlmostEqual(m["progress_x"], 0.4, places=5)
        self.assertAlmostEqual(m["max_progress"], 0.4, places=5)
        self.assertTrue(m["completed"])
        self.assertEqual(m["steps_to_goal"], 20)
        self.assertIsNone(m["failure_mode"])
        self.assertEqual(m["return"], 20.0)
        self.assertEqual(m["steps"], 20)
        self.assertAlmostEqual(m["aspect_min"], 1.0, places=4)
        self.assertAlmostEqual(m["aspect_max"], 4.0, places=4)
        self.assertAlmostEqual(m["aspect_range"], 3.0, places=4)
        self.assertAlmostEqual(m["height_min"], 0.05, places=5)
        self.assertAlmostEqual(m["width_max"], 0.2, places=5)
        self.assertAlmostEqual(m["mean_abs_action"], 0.5, places=3)
        self.assertEqual(m["policy_versions_distinct"], 3)
        self.assertEqual(m["course_id"], "synthetic-showcase")
        self.assertEqual(m["seed"], 0)

    def test_info_success_overrides_goal(self):
        with recorder(self.root, "synthetic-info") as run:
            s = play(run, steps=5, status="terminated", info={"success": True})
        m = episode_metrics(load_episode(self.root / "synthetic-info/workers/w0/episodes" / s["episode_id"]))
        self.assertTrue(m["completed"])
        self.assertIsNone(m["steps_to_goal"])

    def test_failure_modes(self):
        with recorder(self.root, "synthetic-fail") as run:
            play(run, steps=10, dx=-0.01, truncate=True)
            play(run, steps=10, dx=0.001, truncate=True)
            play(run, steps=5, status="terminated", info={"split": True})
            play(run, steps=5, course={"id": "no-goal"}, status="terminated")
            with self.assertRaises(KeyboardInterrupt):
                with run.episode(COURSE, kind="training", training_step=0, initial_points=body(0.1)) as ep:
                    ep.step(body(0.6), [0.0], 0.5, policy_version=0)
                    raise KeyboardInterrupt
        rows = [episode_metrics(ep) for ep in load_run(self.root / "synthetic-fail")]
        self.assertEqual([r["failure_mode"] for r in rows],
                         ["backward", "timeout", "split", "unknown", None])
        self.assertEqual([r["completed"] for r in rows], [False, False, False, None, True])
        self.assertEqual(rows[4]["status"], "interrupted")
        self.assertEqual(rows[4]["return"], 0.5)

    def test_recovered_episode_yields_none_not_estimates(self):
        run = recorder(self.root, "synthetic-crash")
        ep = run.episode(COURSE, training_step=0, initial_points=body(0.1))
        for i in range(25):
            ep.step(body(0.1 + 0.01 * i), [0.1], 1.0, policy_version=0)
        run._lock.close()                                    # simulate the process dying without closing
        reopened = recorder(self.root, "synthetic-crash")
        reopened.close()
        (episode,) = load_run(self.root / "synthetic-crash")
        m = episode_metrics(episode)
        self.assertEqual(m["status"], "interrupted")
        self.assertTrue(m["recovered"])
        self.assertFalse(m["trace_complete"])
        self.assertEqual(m["steps_recorded"], 20)
        for key in ("return", "progress_x", "max_progress", "completed", "steps_to_goal", "policy_versions_distinct",
                    "steps", "exploration_steps", "exploration_share"):
            self.assertIsNone(m[key], key)
        self.assertEqual(m["last_step"], 20)                  # salvage keys: disclosed, never "complete"
        self.assertEqual(m["trace_return"], 20.0)
        self.assertIsNone(m["policy_version_first"])
        self.assertEqual((m["trace_policy_version_first"], m["trace_policy_version_last"]), (0, 0))
        self.assertEqual(m["exploration_steps_in_partial_trace"], 0)
        self.assertTrue(any("true length unknown" in note for note in episode.summary["missing"]))
        events = [r["event"] for r in read_ledger(self.root / "synthetic-crash/workers/w0/episodes.jsonl")[0]]
        self.assertEqual(m["failure_mode"], "interrupted")
        self.assertIsNotNone(m["aspect_mean"])                # describes the recorded steps only
        self.assertEqual(events.count("worker_opened"), 2)        # one per session; analysis filters by event

    def test_intact_recovery_is_never_complete(self):
        with recorder(self.root, "synthetic-intact") as run:
            s = play(run, steps=12, status="terminated")
        ledger = self.root / "synthetic-intact/workers/w0/episodes.jsonl"
        ledger.write_text("".join(ledger.read_text().splitlines(keepends=True)[:-1]))   # drop the finish row
        recorder(self.root, "synthetic-intact").close()
        (episode,) = load_run(self.root / "synthetic-intact")
        m = episode_metrics(episode)
        self.assertEqual((m["status"], m["recovered"], m["trace_complete"]), ("terminated", True, False))
        self.assertEqual((m["steps"], m["return"]), (12, 12.0))           # as written in the intact summary
        self.assertIsNone(m["progress_x"])
        self.assertIsNone(m["policy_versions_distinct"])
        self.assertIsNone(m["problems"])
        self.assertTrue(any("summary intact" in note for note in episode.summary["missing"]))
        self.assertEqual(s["episode_id"], m["episode_id"])

    def test_newer_schema_is_refused(self):
        with recorder(self.root, "synthetic-schema") as run:
            s = play(run, steps=5, status="terminated")
        directory = self.root / "synthetic-schema/workers/w0/episodes" / s["episode_id"]
        summary = json.loads((directory / "episode.json").read_text())
        (directory / "episode.json").write_text(json.dumps({**summary, "schema_version": 3}))
        with self.assertRaises(UnsupportedSchemaError):
            load_episode(directory)
        with self.assertRaises(UnsupportedSchemaError):
            load_run(self.root / "synthetic-schema", problems=[])
        with self.assertRaises(UnsupportedSchemaError):
            build_report([self.root / "synthetic-schema"], self.root / "out")
        (directory / "episode.json").write_text(json.dumps(summary))
        run_json = self.root / "synthetic-schema/run.json"
        run_json.write_text(json.dumps({**json.loads(run_json.read_text()), "schema_version": 3}))
        with self.assertRaises(UnsupportedSchemaError):
            load_run(self.root / "synthetic-schema")

    def test_altered_trace_meta_is_disclosed(self):
        with recorder(self.root, "synthetic-meta") as run:
            s = play(run, steps=10, status="terminated")
        directory = self.root / "synthetic-meta/workers/w0/episodes" / s["episode_id"]
        np.savez_compressed(directory / "trace-meta.npz", initial_com=np.array([0.0, 0.0]))
        (episode,) = load_run(self.root / "synthetic-meta")
        self.assertTrue(any("trace-meta.npz does not match" in p for p in episode.problems))
        self.assertFalse(episode.trace_complete)

    def test_resume_marks_discarded_lineage(self):
        with recorder(self.root, "synthetic-resume") as run:
            play(run, steps=10, training_step=0, version=5)          # versions 5-6: kept
            play(run, steps=10, training_step=10, version=20)        # versions 20-21: beyond the resume point
        crashed = recorder(self.root, "synthetic-resume")
        ep = crashed.episode(COURSE, training_step=20, initial_points=body(0.1))
        for i in range(12):
            ep.step(body(0.1 + 0.01 * i), [0], 1.0, policy_version=30)
        crashed._lock.close()
        resumed = {"checkpoint_sha256": "0" * 64, "training_step": 10, "policy_version": 10}
        with recorder(self.root, "synthetic-resume", resumed_from=resumed) as run:
            play(run, steps=10, training_step=10, version=11)        # new branch: also above 10, but after the resume
        rows = [episode_metrics(e) for e in load_run(self.root / "synthetic-resume")]
        self.assertEqual([bool(r["lineage_discarded"]) for r in rows], [False, True, True, False])
        self.assertIsNone(rows[1]["policy_version_first"])
        self.assertIsNone(rows[2]["trace_policy_version_last"])
        self.assertEqual(rows[3]["policy_version_first"], 11)
        result = build_report([self.root / "synthetic-resume"], self.root / "out", n_boot=50)
        self.assertEqual(len(result["lineage_discarded"]), 2)
        self.assertIn("discarded learner branch", result["files"]["report"].read_text())

    def test_tampered_chunk_is_disclosed(self):
        with recorder(self.root, "synthetic-tamper") as run:
            s = play(run, steps=20, status="terminated")
        directory = self.root / "synthetic-tamper/workers/w0/episodes" / s["episode_id"]
        np.savez_compressed(directory / "trace-0001.npz", step=np.arange(11, 21, dtype=np.int32))
        m = episode_metrics(load_episode(directory))
        self.assertFalse(m["trace_complete"])
        self.assertIn("sha256", m["problems"])
        self.assertIsNone(m["progress_x"])
        self.assertEqual(m["return"], 20.0)

    def test_open_episode_is_listed_with_its_flushed_chunks(self):
        run = recorder(self.root, "synthetic-open")
        ep = run.episode(COURSE, training_step=0, initial_points=body(0.1))
        for i in range(13):
            ep.step(body(0.1 + 0.01 * i), [0], 1.0, policy_version=0)
        (episode,) = load_run(self.root / "synthetic-open")
        self.assertEqual(episode.status, "open")
        self.assertFalse(episode.trace_complete)
        self.assertEqual(episode.steps_recorded, 10)          # the one complete chunk; 3 steps still buffered
        self.assertTrue(any("without a summary" in p for p in episode.problems))
        m = episode_metrics(episode)
        self.assertEqual(m["failure_mode"], "open")
        self.assertEqual(m["trace_return"], 10.0)
        self.assertIsNone(m["progress_x"])
        run.close()

    def test_summary_without_finish_row_or_matching_hash_is_not_complete(self):
        with recorder(self.root, "synthetic-unfinished") as run:
            play(run, steps=5, status="terminated")
            s = play(run, steps=5, status="terminated")
        ledger = self.root / "synthetic-unfinished/workers/w0/episodes.jsonl"
        lines = ledger.read_text().splitlines(keepends=True)
        ledger.write_text("".join(lines[:-1]))               # killed after episode.json, before its ledger row
        first, second = load_run(self.root / "synthetic-unfinished")
        self.assertEqual((second.episode_id, second.status, second.trace_complete), (s["episode_id"], "open", False))
        summary = first.path / "episode.json"
        summary.write_text(summary.read_text().replace('"steps": 5', '"steps": 6'))
        first, _ = load_run(self.root / "synthetic-unfinished")
        self.assertFalse(first.trace_complete)
        self.assertTrue(any("hash recorded in the ledger" in p for p in first.problems))

    def test_schema_v2_points_from_offsets(self):
        with recorder(self.root, "synthetic-v2") as run:
            s = play(run, steps=12, start=1234.5, status="terminated")
        ep = load_episode(self.root / "synthetic-v2/workers/w0/episodes" / s["episode_id"])
        self.assertNotIn("points", np.load(ep.path / "trace-0000.npz").files)
        np.testing.assert_allclose(ep.trace["points"][-1], body(1234.5 + 0.02 * 12), atol=1e-3)
        np.testing.assert_allclose(ep.initial["initial_points"], body(1234.5), atol=1e-3)
        self.assertEqual(ep.trace["com"].dtype, np.float64)
        self.assertAlmostEqual(episode_metrics(ep)["progress_x"], 0.24, places=9)
        self.assertEqual(episode_metrics(ep)["point_count"], 36)

    def test_exploration_steps_are_reported_separately(self):
        with recorder(self.root, "synthetic-explore") as run:
            with run.episode(COURSE, training_step=0, initial_points=body(0.1)) as ep:
                for i in range(1, 11):
                    ep.step(body(0.1), [0], 0.0, policy_version=-1 if i <= 4 else 3)
            with run.episode(COURSE, training_step=10, initial_points=body(0.1)) as ep:
                for i in range(5):
                    ep.step(body(0.1), [0], 0.0, policy_version=-1)
        rows = [episode_metrics(e) for e in load_run(self.root / "synthetic-explore")]
        self.assertEqual([r["exploration_steps"] for r in rows], [4, 5])
        self.assertEqual([r["exploration_share"] for r in rows], [0.4, 1.0])
        self.assertEqual([r["policy_versions_distinct"] for r in rows], [1, 0])
        self.assertEqual(rows[0]["policy_version_first"], 3)
        curve = curve_from_rows(rows, window=2)
        self.assertEqual(curve["exploration"], {"episodes_with_exploration": 2, "episodes_all_exploration": 1,
                                                "episodes_share_unknown": 0, "exploration_steps": 9})
        self.assertAlmostEqual(curve["points"][1]["rolling"]["exploration_share"]["mean"], 0.7)

    def test_fixture_outcomes_are_scripted_not_unexplained(self):
        with recorder(self.root, "fixture-modes", fixture=True) as fx:
            play(fx, steps=3, kind="fixture", status="terminated", info={"scripted_ending": "terminated"})
            play(fx, steps=3, kind="fixture", status="terminated", info={"success": False, "failure_mode": "stuck",
                                                                       "scripted": True})
            play(fx, steps=3, kind="fixture", status="terminated", info={"scripted": True})
        modes = [episode_metrics(e)["failure_mode"] for e in load_run(self.root / "fixture-modes")]
        self.assertEqual(modes, ["scripted_terminated", "stuck", "scripted"])

    def test_aspect_and_course_id(self):
        ratio = aspect_series(np.array([[4.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
        self.assertAlmostEqual(ratio[0], 2.0)
        self.assertAlmostEqual(ratio[1], 1.0)
        self.assertTrue(np.isnan(ratio[2]))
        self.assertEqual(course_id({"id": "a"}), "a")
        self.assertTrue(course_id({"ground_y": 0}).startswith("sha-"))
        self.assertEqual(course_hash({"id": "a", "g": 1}), course_hash({"g": 1, "id": "a"}))
        self.assertNotEqual(course_hash({"id": "a", "g": 1}), course_hash({"id": "a", "g": 2}))


class CurveTest(unittest.TestCase):
    def test_curve_orders_across_workers_and_excludes_other_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with recorder(root, "synthetic-run", worker="w0") as w0:
                for step in (0, 200, 400):
                    play(w0, steps=10, training_step=step, truncate=True)
            with recorder(root, "synthetic-run", worker="w1") as w1:
                for step in (100, 300):
                    play(w1, steps=10, dx=0.05, training_step=step, status="terminated")
                play(w1, steps=5, kind="evaluation", training_step=None, status="terminated")
            with recorder(root, "fixture-run", fixture=True) as fx:
                play(fx, steps=5, kind="fixture", status="terminated")
            curve = learning_curve([root / "synthetic-run", root / "fixture-run"], window=2)
        self.assertEqual(curve["episodes"], 5)
        self.assertEqual([p["training_step_start"] for p in curve["points"]], [0, 100, 200, 300, 400])
        self.assertEqual([p["worker"] for p in curve["points"]], ["w0", "w1", "w0", "w1", "w0"])
        self.assertEqual(curve["excluded_by_kind"], {"evaluation": 1, "fixture": 1})
        self.assertEqual(curve["points"][0]["rolling"]["completed"], {"mean": 0.0, "median": 0.0, "n": 1})
        self.assertEqual(curve["points"][1]["rolling"]["completed"]["mean"], 0.5)
        self.assertEqual(curve["points"][1]["rolling"]["completed"]["n"], 2)

    def test_none_values_reduce_n(self):
        rows = [{"kind": "training", "episode_id": f"e{i}", "worker": "w0", "status": "terminated",
                 "training_step_start": i, "started_at": None, "return": r, "progress_x": None, "completed": None}
                for i, r in enumerate([1.0, None, 3.0])]
        curve = curve_from_rows(rows, window=3)
        self.assertEqual(curve["points"][-1]["rolling"]["return"], {"mean": 2.0, "median": 2.0, "n": 2})
        self.assertEqual(curve["points"][-1]["rolling"]["progress_x"]["n"], 0)


def checkpoints(root: Path, count: int = 4) -> list[dict]:
    with recorder(root, "synthetic-ckpt") as run:
        for i in range(count):
            path = root / f"ckpt-{i}.pth"
            path.write_bytes(f"synthetic weights {i}".encode())
            run.checkpoint(path, policy_version=10 * i, training_step=1000 * i)
    return select_checkpoints(root / "synthetic-ckpt", ("early", "mid", "late"),
                              selected_at="2026-10-01T00:00:00+00:00")


class EvalPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ckpts = checkpoints(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_jobs_and_flags(self):
        self.assertEqual([c["policy_version"] for c in self.ckpts], [0, 20, 30])
        plan = build_eval_plan(self.ckpts, COURSE, ["layout-b", {"id": "layout-c"}], [0, 1], 3,
                               plan_id="t1", baseline={"mechanism": "freeze shape actuators", "params": {"value": 0}})
        self.assertEqual(len(plan["jobs"]), 2 * 3 * 3 * 2)
        safe = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
        self.assertEqual(len({j["run_id"] for j in plan["jobs"]}), len(plan["jobs"]))
        for job in plan["jobs"]:
            self.assertRegex(job["run_id"], safe)
            self.assertEqual(job["kind"], "evaluation")
            self.assertTrue(job["deterministic"])
            self.assertEqual(len(job["checkpoint"]["sha256"]), 64)
            self.assertEqual(job["course"]["final_only"], job["course"]["id"] != "synthetic-showcase")
            self.assertEqual(len(job["episode_seeds"]), 3)
        restricted = [j for j in plan["jobs"] if j["arm"] == "no_reshape"]
        self.assertEqual(restricted[0]["restriction"]["mechanism"], "freeze shape actuators")
        self.assertEqual(plan["arms"][1]["design"], "restricted_at_evaluation")
        json.dumps(plan, allow_nan=False)

    def test_guards(self):
        with self.assertRaises(HeldoutGuardError):
            build_eval_plan(self.ckpts, COURSE, ["layout-b"], [0], 1, heldout_viewed_at="2026-09-30T00:00:00Z")
        build_eval_plan(self.ckpts, COURSE, [], [0], 1, heldout_viewed_at="2026-09-30T00:00:00Z")
        with self.assertRaises(ValueError):
            build_eval_plan(self.ckpts, COURSE, ["synthetic-showcase"], [0], 1)
        with self.assertRaises(ValueError):
            build_eval_plan([{**self.ckpts[0], "selected_at": None}], COURSE, [], [0], 1)
        plan = build_eval_plan(self.ckpts, COURSE, ["layout-b"], [0], 1)
        sha = self.ckpts[0]["sha256"]
        self.assertTrue(heldout_reportable(plan, sha)[0])
        self.assertFalse(heldout_reportable(plan, "0" * 64)[0])
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan))
        first = mark_heldout_viewed(path, "2026-10-02T00:00:00+00:00")
        self.assertEqual(mark_heldout_viewed(path, "2026-10-05T00:00:00+00:00"), first)
        viewed = json.loads(path.read_text())
        self.assertTrue(heldout_reportable(viewed, sha)[0])
        late = {**viewed, "checkpoints": [{**viewed["checkpoints"][0], "selected_at": "2026-10-03T00:00:00+00:00"}]}
        self.assertFalse(heldout_reportable(late, sha)[0])

    def test_too_few_checkpoints_hints_at_labels(self):
        with self.assertRaisesRegex(ValueError, "--labels"):
            select_checkpoints(self.root / "synthetic-ckpt", ("a", "b", "c", "d", "e"))

    def test_cli_writes_plan(self):
        out = self.root / "out/plan.json"
        with contextlib.redirect_stdout(io.StringIO()):
            plan_main(["--checkpoints-from", str(self.root / "synthetic-ckpt"), "--showcase", "s", "--heldout", "h",
                       "--seeds", "0", "1", "--episodes-per-seed", "2", "--out", str(out)])
        plan = json.loads(out.read_text())
        self.assertEqual(len(plan["jobs"]), 3 * 2 * 2)
        self.assertIsNone(plan["heldout_first_viewed_at"])


def run_plan(root: Path, plan: dict, *, episodes: int | None = None, meta: bool = True,
             course: dict | None = None) -> list[Path]:
    """Record synthetic evaluations with scripted progress for each checkpoint.

    meta=False omits eval_job metadata; course overrides the recorded config.
    """
    dirs = []
    for job in plan["jobs"]:
        spec = job["recorder"]
        with RunRecorder(root / spec["run_dir"], run_id=spec["run_id"], worker_id=spec["worker_id"],
                         meta={**(spec["meta"] if meta else {}), **SYNTHETIC}, video_every=0, chunk_steps=10) as run:
            for seed in job["episode_seeds"][:episodes]:
                dx = 0.005 + 0.0002 * job["checkpoint"]["policy_version"] + 0.0001 * (seed % 1000)
                with run.episode(course or spec["episode"]["course"], seed=seed, kind="evaluation",
                                 initial_points=body(0.1)) as ep:
                    for i in range(1, 21):
                        ep.step(body(0.1 + dx * i), [0.2], 0.1, policy_version=job["checkpoint"]["policy_version"],
                                truncated=i == 20)
        dirs.append(root / spec["run_dir"])
    return dirs


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bootstrap_ci_known_data(self):
        self.assertEqual(bootstrap_ci([2.0, 2.0, 2.0]), {"mean": 2.0, "lo": 2.0, "hi": 2.0, "n": 3})
        self.assertEqual(bootstrap_ci([None]), {"mean": None, "lo": None, "hi": None, "n": 0})
        self.assertEqual(bootstrap_ci([5.0, None]), {"mean": 5.0, "lo": None, "hi": None, "n": 1})
        data = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.percentile(data[np.random.default_rng(7).integers(0, 4, size=(500, 4))].mean(axis=1),
                                 [2.5, 97.5])
        ci = bootstrap_ci([1, 2, 3, 4], n_boot=500, seed=7)
        self.assertEqual(ci["mean"], 2.5)
        self.assertAlmostEqual(ci["lo"], expected[0])
        self.assertAlmostEqual(ci["hi"], expected[1])
        self.assertTrue(1.0 <= ci["lo"] < 2.5 < ci["hi"] <= 4.0)
        wide = bootstrap_ci(list(range(100)), n_boot=4000)
        self.assertAlmostEqual(wide["lo"], 49.5 - 1.96 * np.std(range(100)) / 10, delta=1.0)

    def test_report_inventory_curves_comparison_and_heldout_guard(self):
        root = self.root
        with recorder(root, "synthetic-train", worker="w0") as run:
            for step in range(6):
                play(run, steps=10, training_step=step * 100, reshape=step >= 3, truncate=True)
            path = root / "ckpt-a.pth"
            path.write_bytes(b"synthetic a")
            run.checkpoint(path, policy_version=0, training_step=0)
            path = root / "ckpt-b.pth"
            path.write_bytes(b"synthetic b")
            run.checkpoint(path, policy_version=50, training_step=500)
        crashed = recorder(root, "synthetic-train", worker="w1")
        ep = crashed.episode(COURSE, training_step=50, initial_points=body(0.1))
        for i in range(12):
            ep.step(body(0.1), [0], 0, policy_version=0)
        crashed._lock.close()
        recorder(root, "synthetic-train", worker="w1").close()
        with recorder(root, "fixture-demo", fixture=True) as fx:
            play(fx, steps=5, kind="fixture", status="terminated")
        ckpts = select_checkpoints(root / "synthetic-train", ("early", "late"),
                                   selected_at="2026-10-01T00:00:00+00:00")
        plan = build_eval_plan(ckpts, COURSE, [{"id": "synthetic-heldout", "goal_x": 0.3}], [0, 1], 2, plan_id="t")
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        eval_dirs = run_plan(root, plan)
        runs = [root / "synthetic-train", root / "fixture-demo", *eval_dirs]

        blind = build_report(runs, root / "report-no-plan", n_boot=200)
        self.assertEqual(blind["inventory"]["total"], 6 + 1 + 1 + 16)
        counts = blind["inventory"]["by_kind_status"]
        self.assertEqual(counts["training"]["truncated"], 6)
        self.assertEqual(counts["training"]["interrupted"], 1)
        self.assertEqual(counts["training"]["failed"], 0)
        self.assertEqual(counts["evaluation"]["truncated"], 16)
        self.assertEqual(counts["fixture"]["terminated"], 1)
        self.assertEqual(blind["inventory"]["recovered"], 1)
        self.assertEqual(len(blind["withheld_heldout"]), 8)
        self.assertEqual({c["course_id"] for c in blind["comparison"]}, {"synthetic-showcase"})
        self.assertEqual(blind["curve"]["episodes"], 7)
        self.assertEqual(blind["curve"]["excluded_by_kind"], {"evaluation": 16, "fixture": 1})
        self.assertIsNone(json.loads(plan_path.read_text())["heldout_first_viewed_at"])
        for name in ("report.md", "episodes.csv", "report.json", "curve_return.svg", "curve_completed.svg",
                     "compare_synthetic_showcase.svg"):
            self.assertTrue((root / "report-no-plan" / name).exists(), name)
        with (root / "report-no-plan/episodes.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 24)
        text = (root / "report-no-plan/report.md").read_text()
        for phrase in ("## Data inventory", "## Learning curve", "## Checkpoint comparison", "## Reshaping usage",
                       "## Failures", "## Not shown or untested", "small n", "withheld",
                       "No restricted (no-reshape) baseline"):
            self.assertIn(phrase, text)
        early, late = blind["comparison"][0], blind["comparison"][1]
        self.assertEqual((early["episodes"], late["episodes"]), (4, 4))
        self.assertEqual((early["seeds"], early["deterministic"], early["point_counts"]),
                         (["0", "1", "1000", "1001"], ["True"], ["36"]))
        withheld_ids = {w["episode_id"] for w in blind["withheld_heldout"]}
        for row in blind["rows"]:
            if row["episode_id"] in withheld_ids:          # identity and status only
                self.assertTrue(row["withheld"])
                for key in ("progress_x", "return", "completed", "steps", "failure_mode", "aspect_mean"):
                    self.assertNotIn(key, row)
        with (root / "report-no-plan/episodes.csv").open() as stream:
            leaked = [r for r in csv.DictReader(stream) if r["episode_id"] in withheld_ids and
                      any(r[k] for k in ("progress_x", "return", "completed", "steps", "trace_return"))]
        self.assertEqual(leaked, [])
        self.assertIn("1 fixture episode is counted", text)
        self.assertIn("all 7 episodes mean / median", text)               # 7 episodes <= window 20: one row
        self.assertNotIn("first window", text)
        self.assertIn("### Runtime identity of evaluation runs", text)
        identity = {i["run_id"]: i for i in blind["runtime_identity"]}
        self.assertIn("git_commit", identity[eval_dirs[0].name])
        self.assertEqual(identity["synthetic-train"]["workers"][1]["sessions"], 2)   # crashed w1 reopened once
        self.assertTrue(identity[eval_dirs[0].name]["workers"][0]["host"])
        self.assertEqual(blind["evaluation_policy_warnings"], [])
        windowed = build_report([root / "synthetic-train"], root / "report-window", window=2, n_boot=50)
        self.assertIn("first window mean / median", windowed["files"]["report"].read_text())
        self.assertLess(early["progress_x_ci"]["mean"], late["progress_x_ci"]["mean"])
        shape = blind["reshaping_early_late"]["aspect_range"]
        self.assertAlmostEqual(shape["early"]["mean"], 0.0, places=4)
        self.assertGreater(shape["late"]["mean"], 2.0)

        guarded = build_report(runs, root / "report-plan", plan=plan_path, n_boot=200)
        self.assertEqual(guarded["withheld_heldout"], [])
        self.assertEqual({c["course_id"] for c in guarded["comparison"]}, {"synthetic-showcase", "synthetic-heldout"})
        self.assertIsNotNone(json.loads(plan_path.read_text())["heldout_first_viewed_at"])

    def test_heldout_status_comes_from_the_plan_and_fails_closed(self):
        root = self.root
        with recorder(root, "synthetic-train") as run:
            for i in range(2):
                path = root / f"ckpt-{i}.pth"
                path.write_bytes(f"synthetic {i}".encode())
                run.checkpoint(path, policy_version=10 * i, training_step=100 * i)
        ckpts = select_checkpoints(root / "synthetic-train", ("early", "late"), selected_at="2026-10-01T00:00:00Z")
        heldout = {"id": "synthetic-heldout", "goal_x": 0.3}
        plan = build_eval_plan(ckpts, COURSE, [heldout], [0], 1, plan_id="h")
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        bare = run_plan(root / "bare", {**plan, "jobs": [j for j in plan["jobs"]
                                                         if j["course"]["id"] == "synthetic-heldout"]}, meta=False)
        stray = run_plan(root / "stray", {**plan, "jobs": plan["jobs"][:1]}, course={"id": "unplanned", "goal_x": 1})
        result = build_report([*bare, *stray], root / "out", plan=plan_path, n_boot=50)
        reasons = {w["course_id"]: w["reason"] for w in result["withheld_heldout"]}
        self.assertIn("not in the evaluation plan", reasons["unplanned"])
        self.assertIn("checkpoint is not in the evaluation plan", reasons["synthetic-heldout"])
        self.assertEqual(len(result["withheld_heldout"]), 3)
        self.assertEqual(result["comparison"], [])
        self.assertTrue(result["heldout_courses"]["synthetic-heldout"]["heldout"])
        self.assertIsNone(json.loads(plan_path.read_text())["heldout_first_viewed_at"])

    def test_different_configs_under_one_course_id_are_not_compared(self):
        plan = build_eval_plan(checkpoints(self.root), COURSE, [], [0], 2, plan_id="c")
        dirs = run_plan(self.root / "a", plan) + run_plan(self.root / "b", {**plan, "jobs": plan["jobs"][:1]},
                                                          course={**COURSE, "gravity": 1.62})
        result = build_report(dirs, self.root / "out", n_boot=50)
        self.assertEqual(result["comparison"], [])
        (conflict,) = result["course_config_conflicts"]
        self.assertEqual(sorted(conflict["configs"].values()), [2, 6])
        self.assertIn("not compared", result["files"]["report"].read_text())

    def test_policy_change_during_evaluation_is_flagged(self):
        with recorder(self.root, "synthetic-eval-drift") as run:
            play(run, steps=20, kind="evaluation", training_step=None, version=4, truncate=True)   # 4 -> 6
            with run.episode(COURSE, seed=1, kind="evaluation", initial_points=body(0.1)) as ep:
                for i in range(5):
                    ep.step(body(0.1 + 0.01 * i), [0], 0.0, policy_version=7, truncated=i == 4)
        result = build_report([self.root / "synthetic-eval-drift"], self.root / "out", n_boot=50)
        (warning,) = result["evaluation_policy_warnings"]
        self.assertEqual((warning["policy_versions_distinct"], warning["policy_version_first"],
                          warning["policy_version_last"]), (3, 4, 6))
        self.assertIn("policy changed during evaluation", result["files"]["report"].read_text())

    def test_fixture_only_report_is_labelled(self):
        with recorder(self.root, "fixture-only", fixture=True) as fx:
            play(fx, steps=5, kind="fixture", status="terminated")
        result = build_report([self.root / "fixture-only"], self.root / "out")
        text = result["files"]["report"].read_text()
        self.assertIn("FIXTURE DATA ONLY", text)
        self.assertIn("No training episodes", text)
        self.assertEqual(result["curve"]["episodes"], 0)


class SvgTest(unittest.TestCase):
    def test_charts_are_valid_xml(self):
        import xml.etree.ElementTree as ET
        ET.fromstring(line_chart([{"label": "a", "x": [0, 1, 2], "y": [1.0, None, 3.0]},
                                  {"label": "b<&>", "x": [0, 1], "y": [2.0, 2.0]}], title="t", xlabel="x", ylabel="y"))
        ET.fromstring(bar_chart([{"label": "a", "value": 1.0, "lo": 0.5, "hi": 1.5, "n": 3},
                                 {"label": "b", "value": None, "n": 0}], title="t", ylabel="y"))
        ET.fromstring(line_chart([], title="empty", xlabel="x", ylabel="y"))
        with self.assertRaises(ValueError):
            line_chart([{"label": str(i), "x": [0], "y": [0]} for i in range(4)], title="t", xlabel="x", ylabel="y")


if __name__ == "__main__":
    unittest.main()
