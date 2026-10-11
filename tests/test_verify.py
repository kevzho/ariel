"""Recording verification tests with synthetic bodies; no learned behavior."""

from __future__ import annotations

import contextlib
import gc
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings

import imageio_ffmpeg
import numpy as np

from ariel.recording import RunRecorder, View, read_ledger
from ariel.recording.ledger import Ledger
from ariel.recording import verify as verify_module
from ariel.recording.verify import main, verify_run

REAL_DECODED_FRAMES = verify_module._decoded_frames

COURSE = {"id": "flat", "goal_x": 0.9, "ground_y": 0.1, "obstacles": []}
INFO = {"success": False, "failure_mode": "synthetic"}
SMALL = View(64, 64)


def decoded_frames(path: Path) -> int:
    """Full decode without imageio_ffmpeg.read_frames, which leaks pipes (ResourceWarning)."""
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return len(out) // (SMALL.width_px * SMALL.height_px)

class VerifyCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"
        self.worker = self.run_dir / "workers" / "w0"
        patcher = mock.patch("ariel.recording.verify._decoded_frames", decoded_frames)
        patcher.start()
        self.addCleanup(patcher.stop)
        weights = self.root / "policy.pth"
        weights.write_bytes(b"synthetic weights")
        points = np.random.default_rng(0).random((200, 2)) * 0.5 + 0.25
        with RunRecorder(self.run_dir, run_id="r1", chunk_steps=10, view=SMALL, video_every=1,
                         segment_frames=16) as run:
            for steps in (25, 12):
                with run.episode(COURSE, seed=steps, initial_points=points) as ep:
                    for i in range(steps):
                        ep.step(points + 0.001 * i, np.zeros(2), 1.0, policy_version=-1 if i < 5 else 0,
                                terminated=i == steps - 1)
                    ep.end(info=INFO)
            run.checkpoint(weights, policy_version=0, training_step=37)
        self.first = self.worker / "episodes" / "r1-w0-e000000"

    def failures(self, decode: bool = False) -> dict:
        report = verify_run(self.run_dir, decode=decode)
        self.assertFalse(report["ok"])
        return report["workers"]["w0"]["failures"]

    def drop_last_ledger_row(self) -> None:
        path = self.worker / "episodes.jsonl"
        path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))


class VerifyTest(VerifyCase):
    def test_good_run_passes(self):
        report = verify_run(self.run_dir, decode=True)
        self.assertTrue(report["ok"], report)
        worker = report["workers"]["w0"]
        self.assertEqual((worker["episodes"], worker["by_status"], worker["checkpoints"]),
                         (2, {"terminated": 2}, 1))
        self.assertEqual((worker["failures"], worker["problems"]), ({}, []))
        self.assertGreater(len(json.loads((self.first / "episode.json").read_text())["video_segments"]), 1)

    def test_cli_exit_status_and_json(self):
        out = self.root / "verify.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.run_dir), "--json", str(out)]), 0)
            (self.first / "trace-0001.npz").unlink()
            self.assertEqual(main([str(self.run_dir)]), 1)
        self.assertTrue(json.loads(out.read_text())["ok"])

    def test_tampered_chunk(self):
        path = self.first / "trace-0001.npz"
        path.write_bytes(path.read_bytes() + b"x")
        self.assertEqual(self.failures(), {"r1-w0-e000000": ["trace-0001.npz hash mismatch"]})

    def test_missing_segment(self):
        (self.first / "video.seg001.mp4").unlink()
        self.assertEqual(self.failures(), {"r1-w0-e000000": ["video.seg001.mp4 missing"]})

    def test_wrong_decoded_frame_count(self):
        with mock.patch("ariel.recording.verify._decoded_frames", lambda path: 3):
            problems = self.failures(decode=True)["r1-w0-e000000"]
        self.assertIn("video.seg000.mp4 decodes 3 frames, recorded 16", problems)

    def test_gap_in_steps(self):
        self.drop_last_ledger_row()                          # second episode left open by a "crash"
        second = self.worker / "episodes" / "r1-w0-e000001"
        (second / "episode.json").unlink()
        (second / "trace-0000.npz").write_bytes(b"torn")    # earlier chunk lost, later chunk intact
        with RunRecorder(self.run_dir, run_id="r1", chunk_steps=10, view=SMALL) as run:
            self.assertEqual(run.recovered, ["r1-w0-e000001"])
        self.assertIn("trace gap before step 11 (expected 1)", self.failures()["r1-w0-e000001"])

    def test_duplicated_finish_row(self):
        rows, _ = read_ledger(self.worker / "episodes.jsonl")
        finish = next(r for r in rows if r["event"] == "episode_finished")
        Ledger(self.worker / "episodes.jsonl").append({k: v for k, v in finish.items() if k != "prev_sha256"})
        self.assertEqual(self.failures(), {"r1-w0-e000000": ["2 finish rows (expected 1; 0 = still open)"]})

    def test_open_episode(self):
        self.drop_last_ledger_row()
        self.assertEqual(self.failures(), {"r1-w0-e000001": ["0 finish rows (expected 1; 0 = still open)"]})

    def test_edited_episode_json(self):
        path = self.first / "episode.json"
        summary = json.loads(path.read_text())
        path.write_text(json.dumps({**summary, "steps": 99}))
        problems = self.failures()["r1-w0-e000000"]
        self.assertIn("episode.json does not match its ledger hash", problems)
        self.assertIn("trace covers 25 steps, summary says 99", problems)

    def test_tampered_ledger(self):
        path = self.worker / "episodes.jsonl"
        path.write_text(path.read_text().replace('"seed": 25', '"seed": 26'))
        report = verify_run(self.run_dir)
        self.assertFalse(report["ok"])
        self.assertIn("hash chain broken", report["workers"]["w0"]["problems"][0])

    def test_changed_checkpoint(self):
        (self.root / "policy.pth").write_bytes(b"other weights")
        report = verify_run(self.run_dir)
        self.assertFalse(report["ok"])
        self.assertIn("hash mismatch", report["workers"]["w0"]["problems"][0])

    def test_missing_run(self):
        self.assertEqual(verify_run(self.root / "nope"), {"run_dir": str(self.root / "nope"), "workers": {},
                                                          "ok": False, "error": "run.json missing"})


class RecoveryVerifyTest(VerifyCase):
    """Regression tests for interrupted writes and recovery."""

    def test_recovered_intact_episode_passes(self):
        self.drop_last_ledger_row()                          # killed after end() wrote episode.json
        with RunRecorder(self.run_dir, run_id="r1", chunk_steps=10, view=SMALL) as run:
            self.assertEqual(run.recovered, ["r1-w0-e000001"])
        summary = json.loads((self.worker / "episodes" / "r1-w0-e000001" / "episode.json").read_text())
        self.assertEqual(summary["missing"],
                         ["none: only the ledger finish row was written on recovery (summary intact)"])
        self.assertTrue(verify_run(self.run_dir)["ok"])

    def test_salvaged_episode_without_trace_meta_passes_when_disclosed(self):
        self.drop_last_ledger_row()
        second = self.worker / "episodes" / "r1-w0-e000001"
        (second / "episode.json").unlink()
        (second / "trace-meta.npz").unlink()
        with RunRecorder(self.run_dir, run_id="r1", chunk_steps=10, view=SMALL) as run:
            self.assertEqual(run.recovered, ["r1-w0-e000001"])
        summary = json.loads((second / "episode.json").read_text())
        self.assertIn("trace-meta.npz (initial state and point indices)", summary["missing"])
        self.assertEqual((summary["steps"], summary["steps_recorded"], summary["trace_meta_sha256"]), (None, 12, None))
        report = verify_run(self.run_dir, decode=True)
        self.assertTrue(report["ok"], report)

    def test_salvaged_steps_recorded_mismatch(self):
        self.drop_last_ledger_row()
        second = self.worker / "episodes" / "r1-w0-e000001"
        (second / "episode.json").unlink()
        with RunRecorder(self.run_dir, run_id="r1", chunk_steps=10, view=SMALL) as run:
            self.assertEqual(run.recovered, ["r1-w0-e000001"])
        self.assertTrue(verify_run(self.run_dir)["ok"])
        path = second / "episode.json"
        summary = json.loads(path.read_text())
        path.write_text(json.dumps({**summary, "steps_recorded": 99}))
        problems = self.failures()["r1-w0-e000001"]
        self.assertIn("trace covers 12 steps, summary says steps_recorded=99", problems)

    def test_reserved_orphan_index_passes(self):
        (self.worker / "episodes" / "r1-w0-e000002").mkdir()
        with RunRecorder(self.run_dir, run_id="r1", chunk_steps=10, view=SMALL, video_every=0) as run:
            with run.episode(COURSE, initial_points=np.zeros((4, 2))) as ep:
                ep.end(info=INFO)
        self.assertEqual(ep.index, 3)
        report = verify_run(self.run_dir)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["workers"]["w0"]["orphan_indices"], [2])

    def test_unexplained_index_gap_flagged(self):
        orphan = self.worker / "episodes" / "r1-w0-e000002"
        orphan.mkdir()
        with RunRecorder(self.run_dir, run_id="r1", chunk_steps=10, view=SMALL, video_every=0) as run:
            with run.episode(COURSE, initial_points=np.zeros((4, 2))) as ep:
                ep.end(info=INFO)
        orphan.rmdir()                                       # gap no longer explained by a reserved directory
        report = verify_run(self.run_dir)
        self.assertFalse(report["ok"])
        self.assertEqual(report["workers"]["w0"]["problems"], ["episode indices not contiguous: missing [2]"])
        self.assertNotIn("orphan_indices", report["workers"]["w0"])


class IntegrityVerifyTest(VerifyCase):
    def test_trace_meta_hash_checked(self):
        summary = json.loads((self.first / "episode.json").read_text())
        self.assertEqual(summary["trace_meta_sha256"], verify_module.sha256_file(self.first / "trace-meta.npz"))
        path = self.first / "trace-meta.npz"
        path.write_bytes(path.read_bytes() + b"x")
        self.assertEqual(self.failures(), {"r1-w0-e000000": ["trace-meta.npz hash mismatch"]})

    def test_undisclosed_missing_trace_meta(self):
        (self.first / "trace-meta.npz").unlink()
        self.assertEqual(self.failures(), {"r1-w0-e000000": ["trace-meta.npz missing"]})

    def test_newer_schema_version_rejected(self):
        self.drop_last_ledger_row()
        path = self.worker / "episodes" / "r1-w0-e000001" / "episode.json"
        summary = json.loads(path.read_text())
        path.write_text(json.dumps({**summary, "schema_version": 99}))
        Ledger(self.worker / "episodes.jsonl").append(
            {"event": "episode_finished", "episode_id": "r1-w0-e000001", "status": summary["status"],
             "steps": summary["steps"], "return": summary["return"]})
        self.assertEqual(self.failures()["r1-w0-e000001"],
                         ["schema_version 99 is newer than this verifier (2)"])

    def test_real_decode_has_no_resource_warnings(self):
        with mock.patch("ariel.recording.verify._decoded_frames", REAL_DECODED_FRAMES), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = verify_run(self.run_dir, decode=True)
            gc.collect()
        self.assertTrue(report["ok"], report)
        self.assertEqual([w for w in caught if issubclass(w.category, ResourceWarning)], [])


if __name__ == "__main__":
    unittest.main()
