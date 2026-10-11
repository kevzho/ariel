"""Record episode video, state traces, actions, rewards and policy lineage.

Each worker has one writer process:

    <run_dir>/run.json                              run identity (first writer creates it)
    <run_dir>/workers/<worker>/worker.json          worker identity/settings
    <run_dir>/workers/<worker>/episodes.jsonl       hash-chained index: episode_started / episode_finished
    <run_dir>/workers/<worker>/checkpoints.jsonl    hash-chained checkpoint lineage
    <run_dir>/workers/<worker>/episodes/<episode_id>/
        trace-meta.npz      initial state, recorded point indices (written at episode start)
        trace-NNNN.npz      per-step chunks (atomic; complete or absent)
        video.segNNN.mp4    fragmented H.264 segments (decodable if killed)
        episode.json        summary written at episode end

Training-loop integration:

    run = RunRecorder("runs/r1", run_id="r1", meta={...})
    with run.episode(course, seed=s, initial_points=pts, training_step=n) as ep:
        ep.step(points, action, reward, policy_version=updates, terminated=..., truncated=...)
    run.checkpoint("ckpt.pth", policy_version=updates, training_step=n)
    run.close()

RecordingError must stop training. On reopening a worker, crash recovery closes
unfinished episodes as `interrupted` and retains complete chunks and segments.
"""

from __future__ import annotations

import fcntl
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time

import numpy as np

from .ledger import Ledger, RecordingError, atomic_write_bytes, atomic_write_json, dumps, sha256_file
from .render import View, render_frame
from .video import SegmentedVideoWriter

SCHEMA_VERSION = 2  # Point offsets relative to the centre of mass.
EXPLORATION = -1  # Action was not chosen by a policy.
STATUSES = ("terminated", "truncated", "stopped", "interrupted", "failed")
KINDS = ("training", "evaluation", "fixture")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _npz_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def shape_descriptors(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre of mass [2], bounds [x0, y0, x1, y1] and moments [xx, xy, yy] over all points."""
    com = points.mean(axis=0)
    bbox = np.concatenate([points.min(axis=0), points.max(axis=0)])
    centred = points - com
    cov = centred.T @ centred / max(len(points), 1)
    return com, bbox, np.array([cov[0, 0], cov[0, 1], cov[1, 1]])


class EpisodeRecorder:
    def __init__(self, run: "RunRecorder", index: int, course: dict, seed, kind: str,
                 training_step: int | None, initial_points, initial_frame):
        self.run = run
        self.index = index
        self.episode_id = f"{run.run_id}-{run.worker_id}-e{index:06d}"
        self.dir = run.worker_dir / "episodes" / self.episode_id
        self.course = course
        self.kind = kind
        self.steps = 0
        self.chunk = 0
        self.closed = False
        self.status = None
        self.terminated = self.truncated = False
        self.return_sum = 0.0
        self.policy_versions: list[int] = []
        self.nonfinite_rewards = 0
        self.training_step_start = training_step
        self.training_step_end = training_step
        self._buffer: dict[str, list] = {k: [] for k in ("step", "offsets", "com", "bbox", "cov", "action",
                                                          "reward", "policy_version", "wall_time")}
        self._indices = None
        self._point_count = None
        self.chunks: list[dict] = []
        self.orphaned: list[str] = []
        start = {"event": "episode_started", "schema_version": SCHEMA_VERSION,
                 "episode_id": self.episode_id, "episode_index": index, "worker": run.worker_id,
                 "run_id": run.run_id, "kind": kind, "fixture": run.fixture, "course": course,
                 "seed": seed, "training_step_start": training_step, "started_at": _now(),
                 "settings": {"video_every": run.video_every, "chunk_steps": run.chunk_steps,
                              "trace_max_points": run.trace_max_points, "fps": run.fps},
                 "initial_state": initial_points is not None}
        dumps(start)                                   # validate before anything touches disk
        if self.dir.exists() and any(self.dir.iterdir()):
            # Older recorders could leave a directory without a ledger row.
            aside = self.dir.with_name(f"{self.dir.name}.orphan-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}")
            self.dir.rename(aside)
            self.orphaned.append(aside.name)
        run.index.append(start)                        # durable first: recovery sees every started episode
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RecordingError(f"cannot create episode directory {self.dir}: {exc}") from exc
        self.video = SegmentedVideoWriter(self.dir, "video", fps=run.fps, segment_frames=run.segment_frames)
        if initial_points is not None:
            self._write_meta(np.asarray(initial_points, dtype=np.float64))
            self._frame(np.asarray(initial_points, dtype=np.float64), initial_frame, force=True)

    def _choose_indices(self, count: int) -> np.ndarray:
        limit = self.run.trace_max_points
        if count <= limit:
            return np.arange(count)
        return np.linspace(0, count - 1, limit).round().astype(np.int64)

    def _write_meta(self, points: np.ndarray) -> None:
        self._point_count = len(points)
        self._indices = self._choose_indices(len(points))
        com, bbox, cov = shape_descriptors(points)
        atomic_write_bytes(self.dir / "trace-meta.npz", _npz_bytes(
            point_indices=self._indices, point_count=np.int64(len(points)),
            initial_offsets=(points[self._indices] - com).astype(np.float16), initial_com=com.astype(np.float64),
            initial_bbox=bbox.astype(np.float32), initial_cov=cov.astype(np.float32)))

    def _frame(self, points, frame, force=False) -> None:
        every = self.run.video_every
        if not every or (not force and self.steps % every):
            return
        if frame is None:
            frame = self.run.renderer(points, self.course, self.run.view)
        self.video.write(frame)

    def _flush(self) -> None:
        if not self._buffer["step"]:
            return
        b = self._buffer
        name = f"trace-{self.chunk:04d}.npz"
        atomic_write_bytes(self.dir / name, _npz_bytes(
            step=np.asarray(b["step"], dtype=np.int32),
            offsets=np.stack(b["offsets"]).astype(np.float16),
            com=np.stack(b["com"]).astype(np.float64), bbox=np.stack(b["bbox"]).astype(np.float32),
            cov=np.stack(b["cov"]).astype(np.float32), action=np.stack(b["action"]).astype(np.float16),
            reward=np.asarray(b["reward"], dtype=np.float32),
            policy_version=np.asarray(b["policy_version"], dtype=np.int64),
            wall_time=np.asarray(b["wall_time"], dtype=np.float64)))
        self.chunks.append({"path": name, "first_step": int(b["step"][0]), "steps": len(b["step"]),
                            "sha256": sha256_file(self.dir / name)})
        self.chunk += 1
        for values in b.values():
            values.clear()

    def step(self, points, action, reward: float, *, policy_version: int, terminated: bool = False,
             truncated: bool = False, frame: np.ndarray | None = None, training_step: int | None = None) -> None:
        """Record the state after one env step and the action/reward that produced it.

        policy_version is the learner's update counter when the action was chosen;
        use EXPLORATION (-1) for random warm-up or other non-policy actions.
        frame accepts an HxWx3 simulator render; otherwise `points` are rendered.
        """
        if self.closed:
            raise RecordingError("step() after episode end")
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self._indices is None:
            self._write_meta(points)                  # no initial state was given: first state stands in
        if len(points) != self._point_count:
            raise RecordingError(f"point count changed {self._point_count} -> {len(points)}")
        if not np.all(np.isfinite(points)):
            raise RecordingError("non-finite body positions")
        self.steps += 1
        com, bbox, cov = shape_descriptors(points)
        b = self._buffer
        b["step"].append(self.steps)
        b["offsets"].append(points[self._indices] - com)  # Centre offsets to reduce float16 error.
        b["com"].append(com)
        b["bbox"].append(bbox)
        b["cov"].append(cov)
        b["action"].append(np.asarray(action, dtype=np.float64).ravel())
        if int(policy_version) < EXPLORATION:
            raise RecordingError(f"policy_version must be >= {EXPLORATION}")
        reward = float(reward)
        b["reward"].append(reward)
        b["policy_version"].append(int(policy_version))
        b["wall_time"].append(time.time())
        if np.isfinite(reward):
            self.return_sum += reward
        else:
            self.nonfinite_rewards += 1
        self.policy_versions.append(int(policy_version))
        if training_step is not None:
            self.training_step_end = training_step
        elif self.training_step_end is not None:
            self.training_step_end += 1
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self._frame(points, frame)
        if len(b["step"]) >= self.run.chunk_steps:
            self._flush()

    def end(self, status: str | None = None, *, info: dict | None = None, error: str | None = None) -> dict:
        if self.closed:
            return self.summary
        if status is None:
            status = "terminated" if self.terminated else "truncated" if self.truncated else "stopped"
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        self.closed = True
        problems = []
        for action in (self._flush, self.video.close):
            try:
                action()
            except RecordingError as exc:             # keep closing; report and re-raise below
                problems.append(str(exc))
        segments = [{**seg, "sha256": sha256_file(self.dir / seg["path"])}
                    for seg in self.video.segments if (self.dir / seg["path"]).exists()]
        versions = [v for v in self.policy_versions if v != EXPLORATION]
        self.summary = {
            "schema_version": SCHEMA_VERSION, "episode_id": self.episode_id, "run_id": self.run.run_id,
            "worker": self.run.worker_id, "episode_index": self.index, "kind": self.kind,
            "fixture": self.run.fixture, "status": "failed" if problems else status, "error": error or (
                "; ".join(problems) or None),
            "steps": self.steps, "return": None if self.nonfinite_rewards else self.return_sum,
            "nonfinite_rewards": self.nonfinite_rewards,
            "exploration_steps": self.policy_versions.count(EXPLORATION), "course": self.course,
            "training_step_start": self.training_step_start, "training_step_end": self.training_step_end,
            "policy_version_first": versions[0] if versions else None,
            "policy_version_last": versions[-1] if versions else None,
            "policy_versions_distinct": len(set(versions)),
            "trace_chunks": self.chunks, "trace_point_count": self._point_count,
            "trace_meta_sha256": sha256_file(self.dir / "trace-meta.npz") if (self.dir / "trace-meta.npz").exists() else None,
            "trace_points_recorded": None if self._indices is None else int(len(self._indices)),
            "video_segments": segments, "video_frames": self.video.frames, "video_every": self.run.video_every,
            "video_source": "recorded", "missing": [], "orphaned_dirs": self.orphaned,
            "info": info or {}, "ended_at": _now(),
        }
        try:
            dumps(self.summary)
        except RecordingError as exc:                  # e.g. unserializable info: keep the episode, flag it
            problems.append(str(exc))
            self.summary.update(info={"unrecordable_info": repr(info)[:500]}, status="failed",
                                error="; ".join(problems))
        self.run._current = None
        atomic_write_json(self.dir / "episode.json", self.summary)
        self.run.index.append({"event": "episode_finished", "episode_id": self.episode_id,
                               "status": self.summary["status"], "steps": self.steps,
                               "return": self.summary["return"],
                               "episode_json_sha256": sha256_file(self.dir / "episode.json"),
                               "finished_at": _now()})
        if problems:
            raise RecordingError("; ".join(problems))
        return self.summary

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.closed:
            return False
        if exc_type is None:
            self.end()
        elif issubclass(exc_type, RecordingError):
            try:
                self.end("failed", error=str(exc))
            except RecordingError:
                pass
        else:
            self.end("interrupted", error=f"{exc_type.__name__}: {exc}")
        return False


class RunRecorder:
    def __init__(self, run_dir, *, run_id: str, worker_id: str = "w0", meta: dict | None = None,
                 fixture: bool = False, trace_max_points: int = 2048, chunk_steps: int = 250,
                 video_every: int = 1, fps: int = 30, segment_frames: int = 900, view: View = View(),
                 renderer=render_frame, resumed_from: dict | None = None):
        for label, value in (("run_id", run_id), ("worker_id", worker_id)):
            if not _SAFE_ID.match(value):
                raise ValueError(f"{label} must match {_SAFE_ID.pattern}")
        self.run_dir = Path(run_dir)
        self.run_id, self.worker_id, self.fixture = run_id, worker_id, bool(fixture)
        self.trace_max_points, self.chunk_steps = trace_max_points, chunk_steps
        self.video_every, self.fps, self.segment_frames = video_every, fps, segment_frames
        self.view, self.renderer = view, renderer
        self.resumed_from = resumed_from
        self.worker_dir = self.run_dir / "workers" / worker_id
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self._lock = (self.worker_dir / ".lock").open("w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RecordingError(f"worker {worker_id} of run {run_id} is already being recorded") from None
        try:
            self._open(meta, view)
        except BaseException:
            self._lock.close()
            raise

    def _open(self, meta: dict | None, view: View) -> None:
        run_id, worker_id = self.run_id, self.worker_id
        trace_max_points, chunk_steps, video_every = self.trace_max_points, self.chunk_steps, self.video_every
        fps, segment_frames = self.fps, self.segment_frames
        run_json = self.run_dir / "run.json"
        if run_json.exists():
            existing = json.loads(run_json.read_text())
            if existing["run_id"] != run_id:
                raise RecordingError(f"{run_json} belongs to run {existing['run_id']}")
        else:
            atomic_write_json(run_json, {"schema_version": SCHEMA_VERSION, "run_id": run_id, "fixture": self.fixture,
                                         "created_at": _now(), "git_commit": _git_commit(), "meta": meta or {}})
        atomic_write_json(self.worker_dir / "worker.json", {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "worker": worker_id, "opened_at": _now(),
            "host": platform.node(), "platform": platform.platform(), "pid": os.getpid(),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"), "settings": {
                "trace_max_points": trace_max_points, "chunk_steps": chunk_steps, "video_every": video_every,
                "fps": fps, "segment_frames": segment_frames,
                "view": {"width_px": view.width_px, "height_px": view.height_px,
                         "world_width": view.world_width, "world_height": view.world_height}}})
        self.index = Ledger(self.worker_dir / "episodes.jsonl")
        self.checkpoints = Ledger(self.worker_dir / "checkpoints.jsonl")
        self._current: EpisodeRecorder | None = None
        self.recovered = self._recover()
        # Preserve resume boundaries so reused version numbers can be distinguished.
        self.index.append({"event": "worker_opened", "pid": os.getpid(), "opened_at": _now(),
                           "resumed_from": self.resumed_from, "recovered_episodes": self.recovered})
        indices = [r["episode_index"] for r in self.index.rows if r["event"] == "episode_started"]
        for path in (self.worker_dir / "episodes").glob(f"{run_id}-{worker_id}-e*"):
            match = re.search(r"-e(\d{6})$", path.name)
            if match:                                  # directories without a ledger row still reserve an index
                indices.append(int(match.group(1)))
        self._next_index = max(indices) + 1 if indices else 0

    def _recover(self) -> list[str]:
        """Close episodes a crash left open, keeping every complete chunk and video segment."""
        finished = {r["episode_id"] for r in self.index.rows if r["event"] == "episode_finished"}
        recovered = []
        for row in [r for r in self.index.rows if r["event"] == "episode_started"]:
            if row["episode_id"] in finished:
                continue
            directory = self.worker_dir / "episodes" / row["episode_id"]
            summary = self._intact_summary(directory)
            if summary is None:
                summary = self._salvage(row, directory)
            else:                                      # killed after episode.json, before its ledger row
                summary = {**summary, "recovered": True,
                           "missing": summary.get("missing", []) + ["none: only the ledger finish row was "
                                                                     "written on recovery (summary intact)"]}
            row_out = {"event": "episode_finished", "episode_id": row["episode_id"], "status": summary["status"],
                       "steps": summary["steps"], "return": summary["return"], "recovered": True,
                       "finished_at": _now()}
            if directory.exists():
                atomic_write_json(directory / "episode.json", summary)
                row_out["episode_json_sha256"] = sha256_file(directory / "episode.json")
            self.index.append(row_out)
            recovered.append(row["episode_id"])
        return recovered

    @staticmethod
    def _intact_summary(directory: Path) -> dict | None:
        """A summary written by end() whose chunk and segment hashes all verify, else None."""
        path = directory / "episode.json"
        try:
            summary = json.loads(path.read_text())
            files = summary["trace_chunks"] + summary["video_segments"]
            if summary.get("trace_meta_sha256"):
                files = files + [{"path": "trace-meta.npz", "sha256": summary["trace_meta_sha256"]}]
            if all(sha256_file(directory / f["path"]) == f["sha256"] for f in files):
                return summary
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _salvage(self, row: dict, directory: Path) -> dict:
        def number(path: Path) -> int:
            match = re.search(r"(\d+)\.(?:npz|mp4)$", path.name)
            return int(match.group(1)) if match else -1

        chunks, steps_recorded, last_step, discarded = [], 0, 0, []
        rewards, versions = [], []
        for path in sorted(directory.glob("trace-[0-9]*.npz"), key=number):
            try:
                with np.load(path) as data:
                    step, reward, version = data["step"], data["reward"], data["policy_version"]
            except Exception:                         # noqa: BLE001 - unreadable chunk is disclosed
                discarded.append(path.name)
                continue
            chunks.append({"path": path.name, "first_step": int(step[0]), "steps": int(len(step)),
                           "sha256": sha256_file(path)})
            steps_recorded += int(len(step))
            last_step = max(last_step, int(step[-1]))
            rewards.append(reward)
            versions.append(version)
        discarded += [p.name for p in directory.glob("*.tmp")]
        segments = [{"path": p.name, "frames": None, "sha256": sha256_file(p)}
                    for p in sorted(directory.glob("video.seg*.mp4"), key=number)]
        settings = row.get("settings", {})
        video_every = settings.get("video_every", self.video_every)
        missing = ["steps after the last complete trace chunk (up to chunk_steps - 1); true length unknown"]
        if segments:
            missing.append("end of the last video segment (about one second)")
        if not (directory / "trace-meta.npz").exists():
            missing.append("trace-meta.npz (initial state and point indices)")
        if not directory.exists():
            missing.append("episode directory (crash before it was created)")
        missing += [f"discarded partial file {name}" for name in discarded]
        reward = np.concatenate(rewards) if rewards else np.zeros(0)
        version = np.concatenate(versions) if versions else np.zeros(0, dtype=np.int64)
        policy = version[version != EXPLORATION]
        return {
            "schema_version": SCHEMA_VERSION, "episode_id": row["episode_id"], "run_id": self.run_id,
            "worker": self.worker_id, "episode_index": row["episode_index"], "kind": row["kind"],
            "fixture": row.get("fixture", False), "status": "interrupted",
            "error": "process ended before the episode was closed; recovered on next open",
            "recovered": True, "steps": None, "steps_recorded": steps_recorded, "last_step": last_step,
            "return": None, "trace_return": float(reward[np.isfinite(reward)].sum()) if len(reward) else None,
            "course": row["course"], "training_step_start": row.get("training_step_start"),
            "training_step_end": None,
            "policy_version_first": None, "policy_version_last": None, "policy_versions_distinct": None,
            "exploration_steps": None,
            "trace_policy_version_first": int(policy[0]) if len(policy) else None,
            "trace_policy_version_last": int(policy[-1]) if len(policy) else None,
            "trace_exploration_steps": int((version == EXPLORATION).sum()),
            "trace_meta_sha256": sha256_file(directory / "trace-meta.npz") if (directory / "trace-meta.npz").exists() else None,
            "trace_chunks": chunks, "video_segments": segments, "video_frames": None,
            "video_every": video_every, "video_source": "recorded", "missing": missing,
            "info": {}, "ended_at": _now(),
        }

    def episode(self, course: dict, *, seed=None, kind: str = "training", training_step: int | None = None,
                initial_points=None, initial_frame=None) -> EpisodeRecorder:
        if self._current is not None and not self._current.closed:
            raise RecordingError("previous episode not ended")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.fixture and kind != "fixture":
            raise ValueError("a fixture run may only record kind='fixture' episodes")
        if kind == "fixture" and not self.fixture:
            raise ValueError("kind='fixture' requires RunRecorder(fixture=True)")
        episode = EpisodeRecorder(self, self._next_index, course, seed, kind, training_step,
                                  initial_points, initial_frame)
        self._next_index += 1
        self._current = episode
        return episode

    def checkpoint(self, path, *, policy_version: int, training_step: int, note: str | None = None) -> dict:
        path = Path(path)
        if not path.exists():
            raise RecordingError(f"checkpoint file missing: {path}")
        return self.checkpoints.append({"event": "checkpoint", "path": str(path), "sha256": sha256_file(path),
                                        "bytes": path.stat().st_size, "policy_version": int(policy_version),
                                        "training_step": int(training_step), "note": note, "saved_at": _now()})

    def close(self) -> None:
        if self._current is not None and not self._current.closed:
            self._current.end("interrupted", error="run closed while episode open")
        self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def install_termination_handler() -> None:
    """Convert SIGTERM to KeyboardInterrupt so open episodes close as interrupted.

    SIGKILL requires recovery on the next worker open.
    """
    def handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, handler)
