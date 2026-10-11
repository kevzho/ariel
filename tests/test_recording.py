"""Recorder tests with synthetic bodies; no physics or learned behavior."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import imageio_ffmpeg
import numpy as np

from ariel.recording import RecordingError, RunRecorder, View, read_ledger
from ariel.recording.ledger import Ledger
from ariel.recording.recorder import EXPLORATION, SCHEMA_VERSION
from ariel.recording.verify import verify_run

SRC = Path(__file__).resolve().parents[1] / "src"
COURSE = {"id": "flat-box", "goal_x": 0.9, "ground_y": 0.1, "obstacles": [[0.6, 0.1, 0.7, 0.2]]}
SMALL = View(64, 64)
INFO = {"success": False, "failure_mode": "synthetic"}      # synthetic test bodies, never learned behaviour


def body(count: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((count, 2)) * 0.5 + 0.25


def load_trace(directory: Path) -> dict[str, np.ndarray]:
    parts: dict[str, list] = {}
    for path in sorted(directory.glob("trace-[0-9]*.npz")):
        with np.load(path) as data:
            for key in data.files:
                parts.setdefault(key, []).append(data[key])
    return {key: np.concatenate(values) for key, values in parts.items()}


def points_of(trace: dict[str, np.ndarray]) -> np.ndarray:
    """Schema v2: recorded points are com + offsets."""
    return trace["com"][:, None, :] + trace["offsets"].astype(np.float64)


def count_frames(path: Path, view: View = SMALL) -> int:
    """Decode every frame (imageio_ffmpeg.read_frames leaks its pipes when ffmpeg exits first)."""
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return len(out) // (view.width_px * view.height_px)


class RecorderCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"

    def tearDown(self):
        self._tmp.cleanup()

    def recorder(self, **kwargs) -> RunRecorder:
        options = {"run_id": "r1", "chunk_steps": 10, "view": SMALL, "video_every": 0, **kwargs}
        return RunRecorder(self.run_dir, **options)

    @property
    def worker_dir(self) -> Path:
        return self.run_dir / "workers" / "w0"

    def record(self, run: RunRecorder, steps: int, *, versions=None, rewards=None, points=None,
               **step_flags) -> dict:
        points = body() if points is None else points
        with run.episode(COURSE, seed=1, initial_points=points) as ep:
            for i in range(steps):
                flags = step_flags if i == steps - 1 else {}
                version = versions[i] if versions is not None else 0
                reward = rewards[i] if rewards is not None else 0.5
                ep.step(points + 0.001 * i, np.zeros(3), reward, policy_version=version, **flags)
            ep.end(info=INFO)
        return ep.summary

    def rows(self, event: str | None = None) -> list[dict]:
        rows = read_ledger(self.worker_dir / "episodes.jsonl")[0]
        return rows if event is None else [r for r in rows if r["event"] == event]

    def episode_json(self, episode_id: str) -> dict:
        return json.loads((self.worker_dir / "episodes" / episode_id / "episode.json").read_text())


class LedgerChainTest(RecorderCase):
    def make_ledger(self) -> Path:
        with self.recorder() as run:
            for _ in range(3):
                self.record(run, 5)
        path = self.worker_dir / "episodes.jsonl"
        self.assertEqual([r["event"] for r in read_ledger(path)[0]],
                         ["worker_opened"] + ["episode_started", "episode_finished"] * 3)
        return path

    def test_tampered_interior_row_rejected(self):
        path = self.make_ledger()
        lines = path.read_text().splitlines(keepends=True)
        row = json.loads(lines[1])
        row["steps"] = 999
        lines[1] = json.dumps(row, sort_keys=True) + "\n"
        path.write_text("".join(lines))
        with self.assertRaisesRegex(RecordingError, "hash chain broken"):
            read_ledger(path)
        with self.assertRaises(RecordingError):
            self.recorder()

    def test_deleted_interior_row_rejected(self):
        path = self.make_ledger()
        lines = path.read_text().splitlines(keepends=True)
        del lines[2]
        path.write_text("".join(lines))
        with self.assertRaisesRegex(RecordingError, "hash chain broken"):
            read_ledger(path)

    def test_reordered_rows_rejected(self):
        path = self.make_ledger()
        lines = path.read_text().splitlines(keepends=True)
        lines[2], lines[3] = lines[3], lines[2]
        path.write_text("".join(lines))
        with self.assertRaises(RecordingError):
            read_ledger(path)

    def test_partial_trailing_line_ignored_and_truncated(self):
        path = self.make_ledger()
        intact = path.read_bytes()
        rows, head = read_ledger(path)
        with path.open("ab") as stream:
            stream.write(b'{"event": "episode_started", "episode_')
        self.assertEqual(read_ledger(path), (rows, head))
        with self.recorder() as run:
            self.assertTrue(path.read_bytes().startswith(intact))   # partial tail dropped, then worker_opened
            self.assertEqual(path.read_bytes()[len(intact):].count(b"\n"), 1)
            self.record(run, 3)
        rows_after, _ = read_ledger(path)
        self.assertEqual(len(rows_after), 10)
        self.assertEqual((rows_after[7]["event"], rows_after[7]["prev_sha256"]), ("worker_opened", head))


class LineageTest(RecorderCase):
    def test_policy_version_per_step(self):
        versions = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 5, 5, 8, 8, 8, 8, 9, 9, 9, 13, 13, 21, 21]
        with self.recorder() as run:
            summary = self.record(run, len(versions), versions=versions)
        trace = load_trace(self.worker_dir / "episodes" / summary["episode_id"])
        np.testing.assert_array_equal(trace["policy_version"], versions)
        np.testing.assert_array_equal(trace["step"], np.arange(1, len(versions) + 1))
        self.assertEqual(trace["policy_version"].dtype, np.int64)
        saved = self.episode_json(summary["episode_id"])
        self.assertEqual(saved["policy_version_first"], 0)
        self.assertEqual(saved["policy_version_last"], 21)
        self.assertEqual(saved["policy_versions_distinct"], len(set(versions)))
        self.assertEqual(saved["steps"], len(versions))
        self.assertEqual(sum(c["steps"] for c in saved["trace_chunks"]), len(versions))


class StatusTest(RecorderCase):
    def finished_status(self, episode_id: str) -> str:
        rows, _ = read_ledger(self.worker_dir / "episodes.jsonl")
        return next(r["status"] for r in rows if r["event"] == "episode_finished" and r["episode_id"] == episode_id)

    def test_terminated_truncated_stopped(self):
        with self.recorder() as run:
            for flags, expected in (({"terminated": True}, "terminated"), ({"truncated": True}, "truncated"),
                                    ({}, "stopped")):
                summary = self.record(run, 4, **flags)
                self.assertEqual(self.episode_json(summary["episode_id"])["status"], expected)
                self.assertEqual(self.finished_status(summary["episode_id"]), expected)

    def test_exception_inside_episode_is_interrupted(self):
        points = body()
        with self.recorder() as run:
            with self.assertRaises(ZeroDivisionError):
                with run.episode(COURSE, initial_points=points) as ep:
                    for i in range(15):
                        ep.step(points, np.zeros(3), 1.0, policy_version=0)
                    1 / 0
            self.record(run, 2)
        saved = self.episode_json(ep.episode_id)
        self.assertEqual(saved["status"], "interrupted")
        self.assertIn("ZeroDivisionError", saved["error"])
        self.assertEqual(saved["steps"], 15)
        self.assertEqual(sum(c["steps"] for c in saved["trace_chunks"]), 15)
        self.assertEqual(self.finished_status(ep.episode_id), "interrupted")

    def test_recording_failure_is_failed_and_propagates(self):
        points = body()
        with self.recorder() as run:
            with self.assertRaisesRegex(RecordingError, "disk full"):
                with run.episode(COURSE, initial_points=points) as ep:
                    with mock.patch("ariel.recording.recorder.atomic_write_bytes",
                                    side_effect=RecordingError("disk full")):
                        for i in range(20):
                            ep.step(points, np.zeros(3), 1.0, policy_version=0)
            self.assertEqual(ep.steps, 10)
        saved = self.episode_json(ep.episode_id)
        self.assertEqual(saved["status"], "failed")
        self.assertIn("disk full", saved["error"])
        self.assertEqual(self.finished_status(ep.episode_id), "failed")

    def test_run_close_interrupts_open_episode(self):
        run = self.recorder()
        ep = run.episode(COURSE, initial_points=body())
        ep.step(body(), np.zeros(3), 1.0, policy_version=0)
        run.close()
        self.assertEqual(self.episode_json(ep.episode_id)["status"], "interrupted")

    def test_step_after_end_rejected(self):
        with self.recorder() as run:
            ep = run.episode(COURSE, initial_points=body())
            ep.end()
            with self.assertRaises(RecordingError):
                ep.step(body(), np.zeros(3), 0.0, policy_version=0)


CHILD = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    import numpy as np
    from ariel.recording import RunRecorder, View

    run_dir, sentinel = sys.argv[1], Path(sys.argv[2])
    run = RunRecorder(run_dir, run_id="crash", chunk_steps=100, view=View(64, 64))
    points = np.random.default_rng(0).random((300, 2)) * 0.5 + 0.25
    ep = run.episode({"id": "flat", "goal_x": 0.9, "ground_y": 0.1}, seed=0, training_step=0,
                     initial_points=points)
    for i in range(1, 251):
        ep.step(points + 0.001 * i, np.zeros(3), 1.0, policy_version=-1 if i <= 20 else i // 50)
    sentinel.write_text(str(ep.steps))
    while True:
        time.sleep(1)
""")


class CrashRecoveryTest(RecorderCase):
    def test_sigkill_mid_episode_recovered(self):
        sentinel = self.root / "ready"
        child = subprocess.Popen([sys.executable, "-c", CHILD, str(self.run_dir), str(sentinel)],
                                 env={**os.environ, "PYTHONPATH": str(SRC)}, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 30
            while not sentinel.exists():
                if child.poll() is not None:
                    self.fail(f"child exited early: {child.stderr.read().decode()}")
                if time.monotonic() > deadline:
                    self.fail("child never reached step 250")
                time.sleep(0.05)
            time.sleep(0.5)
        finally:
            os.killpg(child.pid, signal.SIGKILL)       # child and its ffmpeg encoder, like a lost container
            child.wait()
            child.stderr.close()
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertEqual(sentinel.read_text(), "250")

        with RunRecorder(self.run_dir, run_id="crash", chunk_steps=100, view=SMALL) as run:
            self.assertEqual(run.recovered, ["crash-w0-e000000"])
            saved = json.loads((run.worker_dir / "episodes" / "crash-w0-e000000" / "episode.json").read_text())
            self.assertEqual(saved["status"], "interrupted")
            self.assertTrue(saved["recovered"])
            self.assertTrue(saved["missing"])
            self.assertIsNone(saved["steps"])                  # true length (250) unknown after SIGKILL
            self.assertEqual(saved["missing"][:2], [
                "steps after the last complete trace chunk (up to chunk_steps - 1); true length unknown",
                "end of the last video segment (about one second)"])
            self.assertEqual([c["path"] for c in saved["trace_chunks"]], ["trace-0000.npz", "trace-0001.npz"])
            directory = run.worker_dir / "episodes" / "crash-w0-e000000"
            trace = load_trace(directory)
            np.testing.assert_array_equal(trace["step"], np.arange(1, 201))
            expected = np.where(np.arange(1, 201) <= 20, EXPLORATION, np.arange(1, 201) // 50)
            np.testing.assert_array_equal(trace["policy_version"], expected)
            np.testing.assert_allclose(points_of(trace)[-1], body() + 0.2, atol=1e-3)
            self.assertEqual((saved["steps_recorded"], saved["last_step"], saved["trace_exploration_steps"]),
                             (200, 200, 20))
            self.assertEqual((saved["trace_policy_version_first"], saved["trace_policy_version_last"]), (0, 4))
            self.assertAlmostEqual(saved["trace_return"], 200.0)
            for key in ("return", "exploration_steps", "policy_version_first", "policy_version_last",
                        "policy_versions_distinct", "video_frames", "training_step_end"):
                self.assertIsNone(saved[key], key)              # partial values only under trace_* keys
            self.assertEqual(saved["trace_meta_sha256"],
                             hashlib.sha256((directory / "trace-meta.npz").read_bytes()).hexdigest())
            self.assertEqual(saved["video_segments"][0]["frames"], None)
            for chunk in saved["trace_chunks"]:
                self.assertEqual(chunk["sha256"], hashlib.sha256((directory / chunk["path"]).read_bytes()).hexdigest())
            self.assertEqual([s["path"] for s in saved["video_segments"]], ["video.seg000.mp4"])
            frames = count_frames(directory / "video.seg000.mp4")
            self.assertGreater(frames, 0)
            self.assertGreaterEqual(frames, 251 - 2 * 30)      # at most about one GOP lost
            rows, _ = read_ledger(run.worker_dir / "episodes.jsonl")
            self.assertEqual([r["event"] for r in rows[-2:]], ["episode_finished", "worker_opened"])
            finish = rows[-2]
            self.assertEqual(finish, {**finish, "status": "interrupted", "recovered": True, "steps": None,
                                      "return": None})
            self.assertEqual(finish["episode_json_sha256"],
                             hashlib.sha256((directory / "episode.json").read_bytes()).hexdigest())
            self.assertEqual(rows[-1]["recovered_episodes"], ["crash-w0-e000000"])
            nxt = run.episode(COURSE, initial_points=body())
            self.assertEqual(nxt.episode_id, "crash-w0-e000001")
            nxt.end(info=INFO)
        with RunRecorder(self.run_dir, run_id="crash", chunk_steps=100, view=SMALL) as run:
            self.assertEqual(run.recovered, [])
        with mock.patch("ariel.recording.verify._decoded_frames", count_frames):
            report = verify_run(self.run_dir, decode=True)
        self.assertTrue(report["ok"], report)


class WriterLockTest(RecorderCase):
    def test_second_writer_same_worker_rejected(self):
        with self.recorder():
            with self.assertRaisesRegex(RecordingError, "already being recorded"):
                self.recorder()
            with self.recorder(worker_id="w1") as other:
                self.record(other, 2)
        with self.recorder() as reopened:
            self.record(reopened, 1)

    def test_run_id_mismatch_rejected(self):
        self.recorder().close()
        with self.assertRaises(RecordingError):
            self.recorder(run_id="r2", worker_id="w1")

    def test_unsafe_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.recorder(worker_id="../w0")


class LabelTest(RecorderCase):
    def test_fixture_run_only_records_fixture_episodes(self):
        with self.recorder(fixture=True) as run:
            for kind in ("training", "evaluation"):
                with self.assertRaises(ValueError):
                    run.episode(COURSE, kind=kind)
            with run.episode(COURSE, kind="fixture", initial_points=body()) as ep:
                ep.step(body(), np.zeros(3), 0.0, policy_version=0)
        saved = self.episode_json(ep.episode_id)
        self.assertEqual((saved["kind"], saved["fixture"]), ("fixture", True))
        self.assertTrue(json.loads((self.run_dir / "run.json").read_text())["fixture"])
        rows, _ = read_ledger(self.worker_dir / "episodes.jsonl")
        started = next(r for r in rows if r["event"] == "episode_started")
        self.assertEqual((started["kind"], started["fixture"]), ("fixture", True))

    def test_real_run_rejects_fixture_and_unknown_kinds(self):
        with self.recorder() as run:
            for kind in ("fixture", "demo"):
                with self.assertRaises(ValueError):
                    run.episode(COURSE, kind=kind)
            with run.episode(COURSE, kind="evaluation", initial_points=body()) as ep:
                ep.step(body(), np.zeros(3), 0.0, policy_version=0)
        saved = self.episode_json(ep.episode_id)
        self.assertEqual((saved["kind"], saved["fixture"], saved["video_source"]), ("evaluation", False, "recorded"))


class InputValidationTest(RecorderCase):
    def assert_failed(self, bad_points) -> None:
        points = body()
        with self.recorder() as run:
            with self.assertRaises(RecordingError):
                with run.episode(COURSE, initial_points=points) as ep:
                    ep.step(points, np.zeros(3), 0.0, policy_version=0)
                    ep.step(bad_points, np.zeros(3), 0.0, policy_version=0)
        self.assertEqual(self.episode_json(ep.episode_id)["status"], "failed")
        self.assertEqual(ep.steps, 1)

    def test_point_count_change(self):
        self.assert_failed(body()[:-1])

    def test_nan_points(self):
        points = body()
        points[7, 1] = np.nan
        self.assert_failed(points)

    def test_inf_points(self):
        points = body()
        points[0, 0] = np.inf
        self.assert_failed(points)


class VideoTest(RecorderCase):
    def test_frame_zero_is_initial_state(self):
        with self.recorder(video_every=1) as run:
            summary = self.record(run, 12)
        self.assertEqual(summary["video_frames"], 13)
        directory = self.worker_dir / "episodes" / summary["episode_id"]
        segment = summary["video_segments"][0]
        self.assertEqual(segment["sha256"], hashlib.sha256((directory / segment["path"]).read_bytes()).hexdigest())
        self.assertEqual(count_frames(directory / segment["path"]), 13)


class CheckpointTest(RecorderCase):
    def test_checkpoint_registry(self):
        weights = self.root / "policy.pth"
        weights.write_bytes(os.urandom(4096))
        with self.recorder() as run:
            row = run.checkpoint(weights, policy_version=42, training_step=1000)
            with self.assertRaises(RecordingError):
                run.checkpoint(self.root / "missing.pth", policy_version=43, training_step=1100)
        rows, _ = read_ledger(self.worker_dir / "checkpoints.jsonl")
        self.assertEqual(rows, [row])
        self.assertEqual(row["sha256"], hashlib.sha256(weights.read_bytes()).hexdigest())
        self.assertEqual((row["policy_version"], row["training_step"], row["bytes"]), (42, 1000, 4096))
        self.assertIsNone(row["prev_sha256"])


class LedgerWriteFailureTest(RecorderCase):
    def make(self) -> Ledger:
        ledger = Ledger(self.root / "ledger.jsonl")
        ledger.append({"event": "a"})
        return ledger

    def test_failed_fsync_rolls_back_and_chain_continues(self):
        ledger = self.make()
        size, head = ledger.path.stat().st_size, ledger.head
        real = os.fsync
        calls = []

        def fsync_once(fd):
            calls.append(fd)
            if len(calls) == 1:
                raise OSError(28, "No space left on device")
            return real(fd)

        with mock.patch("ariel.recording.ledger.os.fsync", side_effect=fsync_once):
            with self.assertRaisesRegex(RecordingError, "ledger write failed"):
                ledger.append({"event": "b"})
        self.assertEqual(len(calls), 2)                    # failed write + rollback fsync
        self.assertEqual(ledger.path.stat().st_size, size)
        self.assertEqual((ledger.head, len(ledger.rows), ledger.poisoned), (head, 1, False))
        ledger.append({"event": "c"})
        rows, _ = read_ledger(ledger.path)
        self.assertEqual([r["event"] for r in rows], ["a", "c"])
        self.assertEqual(rows[1]["prev_sha256"], head)

    def test_failed_rollback_poisons_ledger(self):
        ledger = self.make()
        with mock.patch("ariel.recording.ledger.os.fsync", side_effect=OSError(5, "I/O error")):
            with self.assertRaises(RecordingError):
                ledger.append({"event": "b"})
        self.assertTrue(ledger.poisoned)
        with self.assertRaisesRegex(RecordingError, "unusable"):
            ledger.append({"event": "c"})
        self.assertEqual([r["event"] for r in read_ledger(ledger.path)[0]], ["a"])

    def test_unserializable_row_writes_nothing(self):
        ledger = self.make()
        size = ledger.path.stat().st_size
        with self.assertRaisesRegex(RecordingError, "cannot be recorded"):
            ledger.append({"event": "b", "value": float("nan")})
        self.assertEqual(ledger.path.stat().st_size, size)
        ledger.append({"event": "c"})
        self.assertEqual(len(read_ledger(ledger.path)[0]), 2)


class ValueRecordingTest(RecorderCase):
    def test_numpy_seed_course_and_info_recorded(self):
        course = {"id": "np", "goal_x": np.float32(0.75), "ground_y": np.float64(0.1),
                  "obstacles": np.array([[0.6, 0.1, 0.7, 0.2]])}
        with self.recorder() as run:
            with run.episode(course, seed=np.int64(7), training_step=np.int64(3), initial_points=body()) as ep:
                ep.step(body(), np.zeros(3), np.float32(0.25), policy_version=np.int64(2))
                ep.end(info={"success": np.bool_(True), "failure_mode": None, "distance": np.float64(0.5)})
        (started,) = self.rows("episode_started")
        self.assertEqual((started["seed"], started["course"]["goal_x"], started["schema_version"]),
                         (7, 0.75, SCHEMA_VERSION))
        self.assertEqual(started["course"]["obstacles"], [[0.6, 0.1, 0.7, 0.2]])
        saved = self.episode_json(ep.episode_id)
        self.assertEqual(saved["info"], {"success": True, "failure_mode": None, "distance": 0.5})
        self.assertEqual((saved["course"]["id"], saved["return"], saved["policy_version_first"]), ("np", 0.25, 2))

    def test_unserializable_info_fails_episode_but_worker_continues(self):
        with self.recorder() as run:
            with self.assertRaisesRegex(RecordingError, "cannot be recorded"):
                with run.episode(COURSE, initial_points=body()) as ep:
                    ep.step(body(), np.zeros(3), 1.0, policy_version=0)
                    ep.end(info={"success": object()})
            saved = self.episode_json(ep.episode_id)
            self.assertEqual(saved["status"], "failed")
            self.assertIn("cannot be recorded", saved["error"])
            self.assertIn("unrecordable_info", saved["info"])
            self.assertEqual(saved["steps"], 1)
            nxt = self.record(run, 3)
        self.assertEqual(nxt["episode_index"], 1)
        self.assertEqual(nxt["status"], "stopped")
        finished = self.rows("episode_finished")
        self.assertEqual([r["status"] for r in finished], ["failed", "stopped"])
        self.assertTrue(verify_run(self.run_dir)["ok"])

    def test_nonfinite_reward_gives_null_return(self):
        with self.recorder() as run:
            summary = self.record(run, 4, rewards=[1.0, float("nan"), 2.0, float("inf")])
        saved = self.episode_json(summary["episode_id"])
        self.assertIsNone(saved["return"])
        self.assertEqual(saved["nonfinite_rewards"], 2)
        self.assertIsNone(self.rows("episode_finished")[-1]["return"])
        reward = load_trace(self.worker_dir / "episodes" / summary["episode_id"])["reward"]
        self.assertTrue(np.isnan(reward[1]) and np.isinf(reward[3]))
        self.assertEqual(reward[[0, 2]].tolist(), [1.0, 2.0])

    def test_finite_rewards_give_return(self):
        with self.recorder() as run:
            saved = self.record(run, 4, rewards=[1.0, -0.5, 2.0, 0.25])
        self.assertEqual((saved["return"], saved["nonfinite_rewards"]), (2.75, 0))


class ExplorationLineageTest(RecorderCase):
    def test_exploration_steps_excluded_from_lineage(self):
        versions = [EXPLORATION] * 4 + [0, 0, 3, EXPLORATION, 5]
        with self.recorder() as run:
            summary = self.record(run, len(versions), versions=versions)
        saved = self.episode_json(summary["episode_id"])
        self.assertEqual(saved["exploration_steps"], 5)
        self.assertEqual((saved["policy_version_first"], saved["policy_version_last"]), (0, 5))
        self.assertEqual(saved["policy_versions_distinct"], 3)
        trace = load_trace(self.worker_dir / "episodes" / summary["episode_id"])
        np.testing.assert_array_equal(trace["policy_version"], versions)

    def test_all_exploration_has_no_policy_versions(self):
        with self.recorder() as run:
            saved = self.record(run, 6, versions=[EXPLORATION] * 6)
        self.assertEqual(saved["exploration_steps"], 6)
        self.assertEqual((saved["policy_version_first"], saved["policy_version_last"],
                          saved["policy_versions_distinct"]), (None, None, 0))

    def test_versions_below_exploration_rejected(self):
        with self.recorder() as run:
            with self.assertRaises(RecordingError):
                with run.episode(COURSE, initial_points=body()) as ep:
                    ep.step(body(), np.zeros(3), 0.0, policy_version=-2)
        self.assertEqual(self.episode_json(ep.episode_id)["status"], "failed")


class RecoveryTest(RecorderCase):
    def drop_last_ledger_row(self) -> dict:
        path = self.worker_dir / "episodes.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:-1]))
        return json.loads(lines[-1])

    def test_intact_episode_json_kept_when_finish_row_lost(self):
        with self.recorder(video_every=1) as run:
            summary = self.record(run, 25, terminated=True)
        original = self.episode_json(summary["episode_id"])
        dropped = self.drop_last_ledger_row()               # killed after end() wrote episode.json
        self.assertEqual(dropped["event"], "episode_finished")
        with self.recorder(video_every=1) as run:
            self.assertEqual(run.recovered, [summary["episode_id"]])
            self.assertEqual(self.record(run, 2)["episode_index"], 1)
        saved = self.episode_json(summary["episode_id"])
        self.assertEqual(saved, {**original, "recovered": True, "missing": [
            "none: only the ledger finish row was written on recovery (summary intact)"]})
        self.assertEqual((saved["status"], saved["steps"], saved["info"]), ("terminated", 25, INFO))
        finish = self.rows("episode_finished")[0]
        self.assertEqual((finish["status"], finish["steps"], finish["recovered"]), ("terminated", 25, True))
        self.assertEqual(finish["episode_json_sha256"], hashlib.sha256(
            (self.worker_dir / "episodes" / summary["episode_id"] / "episode.json").read_bytes()).hexdigest())
        self.assertTrue(verify_run(self.run_dir)["ok"])

    def test_damaged_episode_json_is_salvaged(self):
        with self.recorder() as run:
            summary = self.record(run, 25)
        directory = self.worker_dir / "episodes" / summary["episode_id"]
        self.drop_last_ledger_row()
        (directory / "trace-0000.npz").write_bytes(b"torn")  # unreadable, so episode.json hashes fail too
        with self.recorder() as run:
            self.assertEqual(run.recovered, [summary["episode_id"]])
        saved = self.episode_json(summary["episode_id"])
        self.assertEqual((saved["status"], saved["recovered"]), ("interrupted", True))
        self.assertEqual(saved["missing"], [
            "steps after the last complete trace chunk (up to chunk_steps - 1); true length unknown",
            "discarded partial file trace-0000.npz"])           # video_every=0: no video note
        self.assertEqual([c["path"] for c in saved["trace_chunks"]], ["trace-0001.npz", "trace-0002.npz"])
        self.assertEqual((saved["steps"], saved["steps_recorded"], saved["last_step"]), (None, 15, 25))

    def test_started_row_without_directory(self):
        with self.recorder() as run:
            self.record(run, 2)
        Ledger(self.worker_dir / "episodes.jsonl").append(
            {"event": "episode_started", "episode_id": "r1-w0-e000001", "episode_index": 1, "kind": "training",
             "fixture": False, "course": COURSE, "seed": None, "training_step_start": None})
        with self.recorder() as run:                        # killed after the row, before mkdir
            self.assertEqual(run.recovered, ["r1-w0-e000001"])
            self.assertEqual(self.record(run, 2)["episode_index"], 2)
        finish = [r for r in self.rows("episode_finished") if r["episode_id"] == "r1-w0-e000001"][-1]
        self.assertEqual((finish["status"], finish["steps"], finish["recovered"]), ("interrupted", None, True))
        self.assertNotIn("episode_json_sha256", finish)     # no directory, so no episode.json to hash

    def test_old_orphan_dir_reserves_index(self):
        """An unlisted episode directory still reserves its index."""
        with self.recorder() as run:
            self.record(run, 2)
        orphan = self.worker_dir / "episodes" / "r1-w0-e000001"
        orphan.mkdir()
        (orphan / "trace-meta.npz").write_bytes(b"left by a crash")
        with self.recorder() as run:
            self.assertEqual(run.recovered, [])
            self.assertEqual(self.record(run, 2)["episode_index"], 2)
        self.assertEqual((orphan / "trace-meta.npz").read_bytes(), b"left by a crash")
        self.assertEqual([r["episode_index"] for r in self.rows("episode_started")], [0, 2])

    def test_empty_orphan_dir_does_not_wedge_worker(self):
        """A directory created after opening the worker is preserved on collision."""
        with self.recorder() as run:
            self.record(run, 2)
        (self.worker_dir / "episodes" / "r1-w0-e000001").mkdir()
        with self.recorder() as run:
            self.assertEqual(self.record(run, 2)["episode_index"], 2)
            self.assertEqual(self.record(run, 2)["episode_index"], 3)

    def test_colliding_dir_moved_aside(self):
        with self.recorder() as run:
            self.record(run, 2)
            stale = self.worker_dir / "episodes" / "r1-w0-e000001"
            stale.mkdir()                                   # appears after open (e.g. a copied-in backup)
            (stale / "note").write_text("stale")
            summary = self.record(run, 2)
        self.assertEqual(summary["episode_index"], 1)
        (aside,) = summary["orphaned_dirs"]
        self.assertTrue(aside.startswith("r1-w0-e000001.orphan-"))
        self.assertEqual((self.worker_dir / "episodes" / aside / "note").read_text(), "stale")
        self.assertFalse((self.worker_dir / "episodes" / "r1-w0-e000001" / "note").exists())
        with self.recorder() as run:                        # aside dir does not reserve an index
            self.assertEqual(self.record(run, 1)["episode_index"], 2)


class WorkerOpenedTest(RecorderCase):
    def test_each_open_writes_worker_opened_row(self):
        resumed = {"checkpoint_sha256": "ab" * 32, "training_step": 1000, "policy_version": 42}
        with self.recorder() as run:
            self.record(run, 2)
        with self.recorder(resumed_from=resumed) as run:
            self.record(run, 2)
        opened = self.rows("worker_opened")
        self.assertEqual([r["resumed_from"] for r in opened], [None, resumed])
        self.assertEqual([r["recovered_episodes"] for r in opened], [[], []])
        for row in opened:
            self.assertEqual(row["pid"], os.getpid())
            self.assertIsInstance(row["opened_at"], str)
        self.assertEqual([r["event"] for r in self.rows()],
                         ["worker_opened", "episode_started", "episode_finished"] * 2)
        self.assertTrue(verify_run(self.run_dir)["ok"])

    def test_started_row_carries_settings_and_initial_state(self):
        with self.recorder(video_every=2, trace_max_points=64, fps=24) as run:
            with run.episode(COURSE, initial_points=body()) as ep:
                ep.end(info=INFO)
            with run.episode(COURSE) as ep:
                ep.step(body(), np.zeros(3), 0.0, policy_version=0)
        started = self.rows("episode_started")
        self.assertEqual(started[0]["settings"], {"video_every": 2, "chunk_steps": 10, "trace_max_points": 64,
                                                  "fps": 24})
        self.assertEqual([r["initial_state"] for r in started], [True, False])


class RecoveredSummaryTest(RecorderCase):
    def crash_mid_episode(self, steps: int, **options) -> str:
        """Leave an episode open as a SIGKILL would: chunks and video on disk, no episode.json or finish row."""
        run = self.recorder(**options)
        ep = run.episode(COURSE, initial_points=body())
        for i in range(steps):
            ep.step(body() + 0.001 * i, np.zeros(3), 1.0, policy_version=0)
        ep.video.close()                                # encoder would die with the process
        run._lock.close()
        return ep.episode_id

    def test_salvage_uses_video_every_from_started_row(self):
        episode_id = self.crash_mid_episode(25, video_every=5)
        with self.recorder(video_every=0) as run:       # reopened with different settings
            self.assertEqual(run.recovered, [episode_id])
        saved = self.episode_json(episode_id)
        self.assertEqual(saved["video_every"], 5)
        self.assertEqual([s["path"] for s in saved["video_segments"]], ["video.seg000.mp4"])
        self.assertIn("end of the last video segment (about one second)", saved["missing"])
        self.assertEqual((saved["steps"], saved["steps_recorded"]), (None, 20))
        self.assertTrue(verify_run(self.run_dir)["ok"])

    def test_edited_recovered_episode_json_detected(self):
        episode_id = self.crash_mid_episode(25)
        with self.recorder() as run:
            self.assertEqual(run.recovered, [episode_id])
        path = self.worker_dir / "episodes" / episode_id / "episode.json"
        finish = self.rows("episode_finished")[-1]
        self.assertEqual((finish["episode_id"], finish["recovered"]), (episode_id, True))
        self.assertEqual(finish["episode_json_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(verify_run(self.run_dir)["ok"])
        saved = json.loads(path.read_text())
        path.write_text(json.dumps({**saved, "missing": ["nothing much"]}))
        report = verify_run(self.run_dir)
        self.assertFalse(report["ok"])
        self.assertEqual(report["workers"]["w0"]["failures"],
                         {episode_id: ["episode.json does not match its ledger hash"]})


class OffsetPrecisionTest(RecorderCase):
    def test_points_far_from_origin_keep_precision(self):
        points = body(400, seed=3) + np.array([100.0, 3.0])
        with self.recorder(trace_max_points=128) as run:
            summary = self.record(run, 12, points=points)
        directory = self.worker_dir / "episodes" / summary["episode_id"]
        trace = load_trace(directory)
        self.assertEqual((trace["offsets"].dtype, trace["com"].dtype), (np.float16, np.float64))
        with np.load(directory / "trace-meta.npz") as meta:
            indices = meta["point_indices"]
            initial = meta["initial_com"][None, :] + meta["initial_offsets"].astype(np.float64)
            self.assertNotIn("initial_points", meta.files)
        self.assertEqual(len(indices), 128)
        np.testing.assert_allclose(initial, points[indices], rtol=0, atol=1e-3)
        expected = np.stack([points[indices] + 0.001 * i for i in range(12)])
        self.assertLess(np.abs(points_of(trace) - expected).max(), 1e-3)
        np.testing.assert_allclose(trace["com"], np.stack([points.mean(0) + 0.001 * i for i in range(12)]),
                                   rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
