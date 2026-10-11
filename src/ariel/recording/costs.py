"""Measure recording cost per step and per episode on this host.

    python -m ariel.recording.costs [--points 14400 --steps 800 --episodes 2 --trace-max-points 2048 --out DIR]

Records scripted motion twice: trace only (`video_every=0`), then trace with
rendering and H.264 encoding (`video_every=1`). Episodes are fixtures with
policy_version -1, not learned behavior. The JSON report includes recorder
settings, timings, bytes per step and projected sizes for an 800-step episode.
It is printed and saved to DIR/recording-costs-<host>-<time>.json.
Recordings are deleted unless --keep is given.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

from .recorder import EXPLORATION, SCHEMA_VERSION, RunRecorder
from .render import View

COURSE = {"id": "costs-synthetic", "goal_x": 2.0, "ground_y": 0.05, "obstacles": [[0.6, 0.05, 0.68, 0.12], [1.1, 0.05, 1.2, 0.2], [1.7, 0.05, 1.75, 0.3]]}
ACTION_DIM = 8


class SyntheticBody:
    """Square block of points that crawls right, squashes/stretches and ripples smoothly."""

    def __init__(self, count: int, size: float = 0.16):
        side = int(np.ceil(np.sqrt(count)))
        grid = (np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)[:count]
                / max(side - 1, 1) - 0.5)
        self.rest = grid * size
        self.size = size
        self.phase = (grid[:, 0] + grid[:, 1]) * 2 * np.pi

    def points(self, t: int) -> np.ndarray:
        w = 2 * np.pi * t / 60
        stretch = 1 + 0.2 * np.sin(w)
        local = self.rest * np.array([stretch, 1 / stretch])
        local = local + 0.004 * np.stack([np.sin(self.phase + w), np.cos(self.phase + w)], axis=1)
        centre = np.array([0.3 + 0.0015 * t, 0.05 + self.size / 2 / stretch + 0.02 * max(np.sin(w / 2), 0)])
        return local + centre

    @staticmethod
    def action(t: int) -> np.ndarray:
        return np.sin(2 * np.pi * t / 60 + np.arange(ACTION_DIM))


def _dir_bytes(directory: Path, pattern: str) -> int:
    return sum(p.stat().st_size for p in directory.glob(pattern))


def _stats(values: list[float]) -> dict:
    array = np.asarray(values) * 1000
    return {"mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)), "max_ms": float(array.max())}


def measure(mode: str, video_every: int, args, root: Path) -> dict:
    body = SyntheticBody(args.points)
    run_id = f"costs-{mode}"
    view = View(width_px=args.size, height_px=args.size, world_width=1.0, world_height=1.0)
    options = {"trace_max_points": args.trace_max_points} if args.trace_max_points else {}
    run = RunRecorder(root / run_id, run_id=run_id, fixture=True, video_every=video_every, view=view,
                      meta={"purpose": "recording cost measurement", "synthetic": True}, **options)
    settings = {"trace_max_points": run.trace_max_points, "chunk_steps": run.chunk_steps,
                "video_every": run.video_every, "trace_points_recorded": min(args.points, run.trace_max_points)}
    step_times, end_times, episodes = [], [], []
    try:
        for e in range(args.episodes):
            ep = run.episode(COURSE, seed=e, kind="fixture", training_step=0, initial_points=body.points(0))
            for t in range(1, args.steps + 1):
                points, action = body.points(t), body.action(t)
                start = time.perf_counter()
                ep.step(points, action, 0.0015, policy_version=EXPLORATION, truncated=t == args.steps)
                step_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            summary = ep.end(info={"success": None, "failure_mode": None, "synthetic": True})
            end_times.append(time.perf_counter() - start)
            directory = run.worker_dir / "episodes" / summary["episode_id"]
            episodes.append({"bytes": _dir_bytes(directory, "*"), "video_bytes": _dir_bytes(directory, "video.seg*.mp4"),
                             "trace_bytes": _dir_bytes(directory, "trace-*.npz"),
                             "chunk_bytes": [(directory / c["path"]).stat().st_size for c in summary["trace_chunks"]],
                             "video_frames": summary["video_frames"], "status": summary["status"]})
    finally:
        run.close()
    chunks = [b for ep in episodes for b in ep["chunk_bytes"]]
    total_steps = args.steps * args.episodes
    per_step = {k: float(np.mean([e[k] for e in episodes])) / args.steps
                for k in ("bytes", "trace_bytes", "video_bytes")}
    return {"settings": settings, "per_step": _stats(step_times),
            "per_step_amortized_ms": (sum(step_times) + sum(end_times)) * 1000 / total_steps,
            "episode_end_ms": [t * 1000 for t in end_times],
            "bytes_per_episode": float(np.mean([e["bytes"] for e in episodes])),
            "video_bytes_per_episode": float(np.mean([e["video_bytes"] for e in episodes])),
            "trace_bytes_per_episode": float(np.mean([e["trace_bytes"] for e in episodes])),
            "chunk_bytes_mean": float(np.mean(chunks)) if chunks else 0.0,
            "bytes_per_step": per_step["bytes"], "trace_bytes_per_step": per_step["trace_bytes"],
            "video_bytes_per_step": per_step["video_bytes"],
            "bytes_per_800_step_episode": {k: round(v * 800) for k, v in per_step.items()},
            "episodes": [{k: v for k, v in e.items() if k != "chunk_bytes"} for e in episodes]}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--points", type=int, default=14400)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--trace-max-points", type=int, default=None,
                        help="recorded trace points per step (default: the RunRecorder default)")
    parser.add_argument("--size", type=int, default=512, help="frame side in pixels (multiple of 16)")
    parser.add_argument("--out", type=Path, default=Path("runs/recording-costs"))
    parser.add_argument("--keep", action="store_true", help="keep the recorded fixture episodes")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out / f"recordings-{stamp}"
    try:
        modes = {"trace_only": measure("trace", 0, args, root),
                 "trace_render_encode": measure("video", 1, args, root)}
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    host = platform.node()
    report = {
        "kind": "recording_cost_measurement", "schema_version": SCHEMA_VERSION, "synthetic_body": True,
        "measured_at": stamp,
        "host": host, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
        "numpy": np.__version__, "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "settings": {"points": args.points, "steps": args.steps, "episodes": args.episodes, "frame_px": args.size,
                     **{k: v for k, v in modes["trace_only"]["settings"].items() if k != "video_every"}},
        "modes": modes,
        "render_encode_ms_per_step": modes["trace_render_encode"]["per_step"]["mean_ms"]
        - modes["trace_only"]["per_step"]["mean_ms"],
        "recordings": str(root) if args.keep else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    safe_host = "".join(c if c.isalnum() or c in "-_" else "_" for c in host) or "host"
    path = args.out / f"recording-costs-{safe_host}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    report["results_path"] = str(path)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
