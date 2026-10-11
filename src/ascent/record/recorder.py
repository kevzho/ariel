"""Record every training episode, from every worker, from the first random step.

Each worker owns one ``WorkerRecorder``. Episodes are buffered in memory and written
in chunk files (``.npz``) with one JSON line per episode in the worker's index. A
chunk holds the replayable trace of each episode:

* ``qpos`` at the playback cadence (robot and loose debris; enough to render any
  frame with forward kinematics, no physics re-run),
* every applied action, reward and reward-term vector at the control rate,
* the policy version that chose each action.

Writes are synchronous and checked. Any persistence error raises, which stops the
worker and therefore training: no episode is silently dropped. Interrupted episodes
(worker shutdown) are flushed with status ``interrupted``.
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TERM_KEYS = ("progress", "track", "orient", "height", "air_time", "feet_dist", "default_pose", "fallen",
             "lateral", "heading",
             "energy", "action_rate", "impact", "self_contact", "joint_limit", "wobble", "slip", "joint_acc",
             "success", "terminal")
SCHEMA = "ascent.recording.v1"


@dataclass
class _Episode:
    episode_id: str
    meta: dict
    t_start: float
    step_start: int
    qpos: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terms: list = field(default_factory=list)
    versions: list = field(default_factory=list)
    x: list = field(default_factory=list)


class WorkerRecorder:
    def __init__(self, root: str | Path, run_id: str, worker: int, playback_every: int = 2,
                 chunk_episodes: int = 64, chunk_bytes: int = 48 << 20, term_keys: tuple = TERM_KEYS,
                 control_hz: float = 50.0):
        self.term_keys, self.control_hz = tuple(term_keys), float(control_hz)
        self.dir = Path(root) / run_id / "episodes" / f"w{worker:03d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.worker = run_id, worker
        self.every = playback_every
        self.chunk_episodes, self.chunk_bytes = chunk_episodes, chunk_bytes
        self.index_path = self.dir / "index.jsonl"
        self.chunk_no = len(list(self.dir.glob("chunk_*.npz")))
        self.pending: list[tuple[_Episode, dict]] = []
        self.pending_bytes = 0
        self.open: dict[int, _Episode] = {}
        # continue numbering after a resume so episode ids stay unique
        self.count = sum(1 for _ in open(self.index_path, encoding="utf-8")) if self.index_path.exists() else 0

    def start(self, env_slot: int, meta: dict, global_step: int, qpos: np.ndarray) -> None:
        eid = f"{self.run_id}-w{self.worker:03d}-e{self.count:07d}"
        self.count += 1
        ep = _Episode(eid, meta, time.time(), global_step)
        ep.qpos.append(qpos.astype(np.float32).copy())
        self.open[env_slot] = ep

    def step(self, env_slot: int, action, reward: float, terms: dict, version: int, x: float,
             qpos: np.ndarray) -> None:
        ep = self.open[env_slot]
        ep.actions.append(np.asarray(action, np.float16))
        ep.rewards.append(np.float32(reward))
        ep.terms.append(np.array([terms.get(k, 0.0) for k in self.term_keys], np.float32))
        ep.versions.append(np.int32(version))
        ep.x.append(np.float32(x))
        if len(ep.actions) % self.every == 0:
            ep.qpos.append(qpos.astype(np.float32).copy())

    def finish(self, env_slot: int, status: str, reason: str, global_step: int, info: dict) -> dict:
        ep = self.open.pop(env_slot)
        n = len(ep.actions)
        row = {
            "schema": SCHEMA, "episode_id": ep.episode_id, "run_id": self.run_id, "worker": self.worker,
            "status": status, "reason": reason, "steps": n, "seconds": n / self.control_hz,
            "return": float(np.sum(ep.rewards)) if n else 0.0,
            "return_terms": dict(zip(self.term_keys, (np.sum(ep.terms, axis=0) if n else np.zeros(len(self.term_keys))).round(3).tolist())),
            "start_time": ep.t_start, "end_time": time.time(),
            "global_step_start": ep.step_start, "global_step_end": global_step,
            "policy_version_min": int(min(ep.versions)) if n else -1,
            "policy_version_max": int(max(ep.versions)) if n else -1,
            "max_x": float(max(ep.x)) if n else 0.0,
            "success": reason == "success",
            "passed": info.get("passed", {}),
            "max_penetration": float(info.get("max_penetration_episode", 0.0)),
            "playback_hz": self.control_hz / self.every, "control_hz": self.control_hz,
            "attempted": info.get("attempted", {}), **info.get("extra", {}),
            **ep.meta,
        }
        nb = sum(a.nbytes for a in ep.qpos) + n * (ep.actions[0].nbytes + 4 * (3 + len(self.term_keys))) if n else 0
        self.pending.append((ep, row))
        self.pending_bytes += nb
        if len(self.pending) >= self.chunk_episodes or self.pending_bytes >= self.chunk_bytes:
            self.flush()
        return row

    def flush(self, attempts: int = 3) -> None:
        """Write pending episodes; transient storage errors are retried before training is stopped."""
        for k in range(attempts):
            try:
                return self._flush_once()
            except OSError:
                if k == attempts - 1:
                    raise
                time.sleep(2.0 * (k + 1))

    def _flush_once(self) -> None:
        if not self.pending:
            return
        name = f"chunk_{self.chunk_no:05d}.npz"
        arrays = {}
        for i, (ep, row) in enumerate(self.pending):
            k = f"e{i:03d}"
            arrays[f"{k}_qpos"] = np.stack(ep.qpos)
            if ep.actions:
                arrays[f"{k}_actions"] = np.stack(ep.actions)
                arrays[f"{k}_rewards"] = np.array(ep.rewards, np.float32)
                arrays[f"{k}_terms"] = np.stack(ep.terms)
                arrays[f"{k}_versions"] = np.array(ep.versions, np.int32)
            row["chunk"] = name
            row["key"] = k
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        tmp = self.dir / (name + ".tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, self.dir / name)
        with open(self.index_path, "a", encoding="utf-8") as f:
            for _, row in self.pending:
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        # verify the chunk is readable before forgetting the episodes
        with np.load(self.dir / name) as z:
            assert all(f"{row['key']}_qpos" in z.files for _, row in self.pending), "chunk verification failed"
        self.chunk_no += 1
        self.pending.clear()
        self.pending_bytes = 0

    def close(self, global_step: int) -> None:
        """Flush open (unfinished) episodes as interrupted, then everything pending."""
        for slot in list(self.open):
            self.finish(slot, "interrupted", "worker_shutdown", global_step, {})
        self.flush()


def load_episode(run_dir: str | Path, row: dict) -> dict:
    """Read one recorded episode back (for the viewer, analysis or rendering)."""
    p = Path(run_dir) / "episodes" / f"w{row['worker']:03d}" / row["chunk"]
    k = row["key"]
    with np.load(p) as z:
        out = {name.split("_", 1)[1]: z[name] for name in z.files if name.startswith(k + "_")}
    return out
