"""Learning curves across workers, grouped by episode kind.

Curves default to training episodes; other kinds are counted in
`excluded_by_kind`. `exploration_share` tracks actions with policy_version -1,
including random warm-up.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from .metrics import episode_metrics, load_run

CURVE_METRICS = ("return", "progress_x", "completed", "exploration_share")


def _rolling(values: list, end: int, window: int) -> dict:
    present = [float(v) for v in values[max(0, end - window + 1):end + 1] if v is not None]
    if not present:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": float(np.mean(present)), "median": float(np.median(present)), "n": len(present)}

def _order_key(row: dict):
    return (row["training_step_start"], row.get("started_at") or "", row["episode_id"])

def curve_from_rows(rows: list[dict], *, window: int = 20, kind: str = "training") -> dict:
    """Return rolling means, medians and sample counts for episode metrics."""
    if window < 1:
        raise ValueError("window must be >= 1")
    selected = [r for r in rows if r["kind"] == kind]
    excluded: dict[str, int] = {}
    for r in rows:
        if r["kind"] != kind:
            excluded[r["kind"]] = excluded.get(r["kind"], 0) + 1
    unordered = [r["episode_id"] for r in selected if r["training_step_start"] is None]
    ordered = sorted((r for r in selected if r["training_step_start"] is not None), key=_order_key)
    series = {m: [None if r.get(m) is None else float(r[m]) for r in ordered] for m in CURVE_METRICS}
    points = []
    for i, r in enumerate(ordered):
        points.append({"index": i, "episode_id": r["episode_id"], "worker": r["worker"], "status": r["status"],
                       "training_step_start": r["training_step_start"],
                       **{m: r.get(m) for m in CURVE_METRICS},
                       "rolling": {m: _rolling(series[m], i, window) for m in CURVE_METRICS}})
    shares = [r.get("exploration_share") for r in ordered]
    exploration = {"episodes_with_exploration": sum(1 for v in shares if v),
                   "episodes_all_exploration": sum(1 for v in shares if v == 1),
                   "episodes_share_unknown": sum(1 for v in shares if v is None),
                   "exploration_steps": sum(r.get("exploration_steps") or 0 for r in ordered)}
    return {"kind": kind, "window": window, "episodes": len(ordered), "points": points,
            "excluded_by_kind": excluded, "unordered_episode_ids": unordered, "exploration": exploration}


def learning_curve(run_dir, *, window: int = 20, kind: str = "training") -> dict:
    """Build a curve for `kind` episodes, ordered by training_step_start."""
    run_dirs = [run_dir] if isinstance(run_dir, (str, Path)) else list(run_dir)
    rows = [episode_metrics(ep) for d in run_dirs for ep in load_run(d)]
    curve = curve_from_rows(rows, window=window, kind=kind)
    curve["run_dirs"] = [str(d) for d in run_dirs]
    return curve
