"""Export a recorded run as static files for the web viewer.

    out_dir/index.json                      run meta, runtime identity, worker sessions, episodes, checkpoints
    out_dir/episodes/<id>/trace.json        downsampled trace (<= MAX_FRAMES frames, <= MAX_POINTS points)
    out_dir/episodes/<id>/video.segNNN.mp4  recorded video segments (hardlinked when possible)

Values lost in a crash stay null. Partial trace values use `steps_recorded`,
`last_step` and `trace_*`. Files with mismatched hashes are excluded and listed
in `missing`. Newer run schemas are rejected; newer episode schemas are exported
without trace or video, with an explanation.

`workers` in index.json includes each worker's latest runtime identity and all
`worker_opened` sessions. `resumed_from` identifies the checkpoint used to
restart training; earlier interrupted episodes may belong to a discarded branch.

Runs with meta.eval_job.course.final_only are exported with `withheld: true`
and no trace, video or results unless the evaluation plan allows the checkpoint.

    python -m ariel.recording.export RUN_DIR OUT_DIR [--plan eval-plan.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np

from .ledger import RecordingError, read_ledger, sha256_file

MAX_FRAMES = 600
MAX_POINTS = 256
MAX_ACTION = 64
EXPORT_VERSION = 2
SUPPORTED_SCHEMA = 2                 # newest recorder schema_version this exporter understands
EXPLORATION = -1
TRACE_KEYS = ("step", "com", "bbox", "cov", "action", "reward", "policy_version")
# Retain only these fields when withholding held-out results.
WITHHELD_KEEP = ("id", "run_id", "worker", "index", "kind", "fixture", "label", "seed", "course",
                 "training_step_start", "policy_version_first", "policy_version_last", "started_at")
FIXTURE_LABEL = "scripted fixture - not learned behavior"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _numbered(directory: Path, pattern: str) -> list[Path]:
    """Files matching `pattern`, sorted by their trailing number (trace-10 after trace-9)."""
    def number(path: Path) -> int:
        match = re.search(r"(\d+)\.(?:npz|mp4)$", path.name)
        return int(match.group(1)) if match else -1
    return sorted(directory.glob(pattern), key=lambda p: (number(p), p.name))


def _newer_schema(*records: dict | None) -> int | None:
    """The first schema_version above SUPPORTED_SCHEMA among `records`, else None."""
    for record in records:
        version = (record or {}).get("schema_version")
        if isinstance(version, int) and version > SUPPORTED_SCHEMA:
            return version
    return None


def _rounded(values, digits: int):
    """Rounded nested list; non-finite entries become None (JSON has no NaN/inf)."""
    array = np.round(np.asarray(values, dtype=np.float64), digits)
    if np.all(np.isfinite(array)):
        return array.tolist()
    return np.where(np.isfinite(array), array, None).tolist()


def _aspect(cov) -> float | None:
    """sqrt(largest / smallest eigenvalue) of [xx, xy, yy]; None when degenerate (same as analysis.metrics)."""
    xx, xy, yy = (float(v) for v in cov)
    mean, spread = (xx + yy) / 2, math.hypot((xx - yy) / 2, xy)
    high, low = mean + spread, mean - spread
    ratio = math.sqrt(high / low) if low > 1e-12 * max(high, 1e-300) else None
    return None if ratio is None or not math.isfinite(ratio) else round(ratio, 4)


def _action(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {"action_mean_abs": round(float(np.abs(finite).mean()), 4) if len(finite) else None,
            "action": _rounded(values[:MAX_ACTION], 3)}


def _load_trace(directory: Path, summary: dict | None) -> tuple[dict | None, dict, list[str]]:
    """Return (meta arrays, concatenated chunk arrays, missing notes)."""
    missing = []
    meta = None
    meta_hash = (summary or {}).get("trace_meta_sha256")
    if (directory / "trace-meta.npz").exists() and meta_hash and sha256_file(directory / "trace-meta.npz") != meta_hash:
        missing.append("trace-meta.npz does not match its recorded sha256; initial state excluded")
    elif (directory / "trace-meta.npz").exists():
        with np.load(directory / "trace-meta.npz") as data:
            meta = {key: data[key] for key in data.files}
        if "initial_offsets" in meta and "initial_com" in meta:   # schema 2: offsets from the f64 centre of mass
            meta["initial_points"] = meta["initial_com"][None, :] + meta["initial_offsets"].astype(np.float64)
    listed = (summary or {}).get("trace_chunks")
    if listed is None:
        listed = [{"path": p.name, "sha256": None} for p in _numbered(directory, "trace-[0-9]*.npz")]
    parts: dict[str, list] = {}
    for chunk in listed:
        path = directory / chunk["path"]
        if not path.exists():
            missing.append(f"trace chunk {chunk['path']} not found")
            continue
        if chunk.get("sha256") and sha256_file(path) != chunk["sha256"]:
            missing.append(f"trace chunk {chunk['path']} does not match its recorded sha256; excluded")
            continue
        with np.load(path) as data:
            for key in TRACE_KEYS:
                parts.setdefault(key, []).append(data[key])
            points = data["points"] if "points" in data.files else \
                data["com"][:, None, :] + data["offsets"].astype(np.float64)       # schema 1 stored points
            parts.setdefault("points", []).append(points)
    arrays = {key: np.concatenate(values) for key, values in parts.items()}
    return meta, arrays, missing


INITIAL_KEYS = ("initial_com", "initial_bbox", "initial_cov", "initial_points")


def _trace_json(episode: dict, meta: dict | None, arrays: dict, initial_state: bool) -> dict:
    count = len(arrays.get("step", []))
    stride = max(1, math.ceil(count / (MAX_FRAMES - 1)))
    rows = list(range(stride - 1, count, stride))
    if count and rows[-1:] != [count - 1]:
        rows.append(count - 1)
    recorded = arrays["points"].shape[1] if count else (len(meta.get("point_indices", ())) if meta else 0)
    keep = np.unique(np.linspace(0, recorded - 1, min(recorded, MAX_POINTS)).round().astype(np.int64)) \
        if recorded else np.zeros(0, dtype=np.int64)
    frames = []
    # Without an initial state, trace-meta contains step 1. Do not label it step 0.
    # Older or damaged metadata may also lack some initial arrays.
    has_initial = initial_state and meta is not None and all(key in meta for key in INITIAL_KEYS)
    if has_initial:
        frames.append({"step": 0, "com": _rounded(meta["initial_com"], 4), "bbox": _rounded(meta["initial_bbox"], 4),
                       "aspect": _aspect(meta["initial_cov"]), "points": _rounded(meta["initial_points"][keep], 3),
                       "reward": 0.0, "policy_version": None, "exploration": None,
                       "action_mean_abs": None, "action": None})
    if count:
        rewards = arrays["reward"].astype(np.float64)
        cumulative = np.cumsum(np.where(np.isfinite(rewards), rewards, np.nan))   # null from a non-finite reward on
        for row in rows:
            version = int(arrays["policy_version"][row])
            frames.append({"step": int(arrays["step"][row]), "com": _rounded(arrays["com"][row], 4),
                           "bbox": _rounded(arrays["bbox"][row], 4), "aspect": _aspect(arrays["cov"][row]),
                           "points": _rounded(arrays["points"][row][keep], 3),
                           "reward": _rounded(cumulative[row], 5), "policy_version": version,
                           "exploration": version == EXPLORATION, **_action(arrays["action"][row])})
    if frames:
        complete = count > 0 and episode.get("steps") is not None and frames[-1]["step"] == episode["steps"]
        frames[-1]["end"] = {"status": episode["status"], "terminated": episode["status"] == "terminated",
                             "truncated": episode["status"] == "truncated", "final_step": complete}
    return {"export_version": EXPORT_VERSION, "id": episode["id"], "run_id": episode["run_id"],
            "kind": episode["kind"], "fixture": episode["fixture"], "label": episode.get("label"),
            "status": episode["status"], "recovered": episode["recovered"], "course": episode["course"],
            "steps_in_trace": count, "initial_state": has_initial, "stride": stride, "point_count": episode.get("point_count"),
            "points_recorded": int(recorded), "points_exported": int(len(keep)),
            "action_dims": int(arrays["action"].shape[1]) if count else None,
            "fields": {"step": "1-based env step; 0 is the initial state (absent when initial_state is false)", "com": "[x, y] of all points",
                       "bbox": "[x0, y0, x1, y1] of all points",
                       "aspect": "sqrt(major / minor eigenvalue) of the all-point covariance; null if degenerate",
                       "points": "[[x, y], ...] fixed subset (reconstructed as com + recorded f16 offsets)",
                       "reward": "cumulative reward up to this step; null from the first non-finite reward on",
                       "policy_version": "learner update counter that chose this step's action; -1 = none",
                       "exploration": "true when no policy chose the action (policy_version -1: random "
                                      "warm-up or scripted); null for the initial state",
                       "action_mean_abs": "mean |a| over every action dimension (finite values)",
                       "action": f"applied action, first {MAX_ACTION} of action_dims values, rounded",
                       "end": "final frame only: episode status, terminated/truncated flags, and final_step "
                              "= this frame is the episode's last step (false when the trace stops early)"},
            "frames": frames}


def _episode(run_dir: Path, out_dir: Path, worker: str, started: dict, finished: dict | None,
             withhold: str | None) -> dict:
    episode_id = started["episode_id"]
    directory = run_dir / "workers" / worker / "episodes" / episode_id
    if withhold:
        shutil.rmtree(out_dir / "episodes" / episode_id, ignore_errors=True)   # no stale trace/video from before
    summary = _read_json(directory / "episode.json")
    newer = _newer_schema(started, summary)
    missing = list((summary or {}).get("missing") or [])
    if newer is not None:
        missing.append(f"schema_version {newer} is newer than this exporter ({SUPPORTED_SCHEMA}); "
                       "trace and video not exported")
    if finished is None:
        missing.append("episode not closed: no episode_finished row (writer still running, or crashed "
                       "and the worker was not reopened)")
    elif summary is None:
        missing.append("episode.json not found")
    elif finished.get("episode_json_sha256") and sha256_file(directory / "episode.json") != \
            finished["episode_json_sha256"]:
        missing.append("episode.json does not match the hash recorded in the ledger")
    if newer is None:
        meta, arrays, trace_missing = _load_trace(directory, summary)
        missing += trace_missing
    else:
        meta, arrays = None, {}
    s = summary or {}
    fixture = bool(started.get("fixture") or s.get("fixture"))
    status = (finished or {}).get("status") or s.get("status") or "open"
    versions = arrays.get("policy_version")
    chosen = versions[versions != EXPLORATION] if versions is not None else np.zeros(0, dtype=np.int64)
    rewards = arrays["reward"].astype(np.float64) if versions is not None else None
    # Prefer recorded trace_* values, falling back to exported chunks.
    # Keep partial values separate from full-episode totals and lineage.
    computed = {
        "trace_return": float(rewards[np.isfinite(rewards)].sum()) if versions is not None else None,
        "trace_nonfinite_rewards": int((~np.isfinite(rewards)).sum()) if versions is not None else None,
        "trace_exploration_steps": int((versions == EXPLORATION).sum()) if versions is not None else None,
        "trace_policy_version_first": int(chosen[0]) if len(chosen) else None,
        "trace_policy_version_last": int(chosen[-1]) if len(chosen) else None,
    }
    trace_values = {key: s[key] if key in s else value for key, value in computed.items()}
    episode = {
        "id": episode_id, "run_id": started.get("run_id"), "worker": worker, "index": started["episode_index"],
        "kind": started["kind"], "fixture": fixture, "label": FIXTURE_LABEL if fixture else None,
        "status": status, "recovered": bool((finished or {}).get("recovered") or s.get("recovered")),
        "error": s.get("error"), "steps": s["steps"] if "steps" in s else (finished or {}).get("steps"),
        "return": s.get("return"), "nonfinite_rewards": s.get("nonfinite_rewards"), "seed": started.get("seed"),
        "training_step_start": s.get("training_step_start", started.get("training_step_start")),
        "training_step_end": s.get("training_step_end"),
        "policy_version_first": s.get("policy_version_first"), "policy_version_last": s.get("policy_version_last"),
        "policy_versions_distinct": s.get("policy_versions_distinct"),
        "exploration_steps": s.get("exploration_steps"),
        "steps_recorded": s.get("steps_recorded"), "last_step": s.get("last_step"),
        "trace_steps": int(len(versions)) if versions is not None else 0, **trace_values,
        "course": s.get("course", started.get("course")), "point_count": s.get("trace_point_count"),
        "info": s.get("info") or {}, "started_at": started.get("started_at"), "ended_at": s.get("ended_at"),
        "missing": missing, "video_source": s.get("video_source", "recorded"),
        "video_frames": s.get("video_frames"), "video_every": s.get("video_every"),
        "settings": started.get("settings"), "initial_state": started.get("initial_state"),
        "schema_version": s.get("schema_version", started.get("schema_version")),
        "video": [], "trace": f"episodes/{episode_id}/trace.json", "withheld": False,
    }
    if withhold:
        return {**{k: episode[k] for k in WITHHELD_KEEP}, "withheld": True, "withheld_reason": withhold,
                "trace": None, "video": []}
    if newer is not None:
        shutil.rmtree(out_dir / "episodes" / episode_id, ignore_errors=True)
        return {**episode, "trace": None}
    if episode["point_count"] is None and meta is not None and "point_count" in meta:
        episode["point_count"] = int(meta["point_count"])
    segments = s.get("video_segments")
    if segments is None:
        segments = [{"path": p.name} for p in _numbered(directory, "video.seg*.mp4")]
    for segment in segments:
        source = directory / segment["path"]
        if not source.exists():
            missing.append(f"video segment {segment['path']} not found")
            continue
        if segment.get("sha256") and sha256_file(source) != segment["sha256"]:
            missing.append(f"video segment {segment['path']} does not match its recorded sha256; excluded")
            (out_dir / "episodes" / episode_id / segment["path"]).unlink(missing_ok=True)
            continue
        relative = f"episodes/{episode_id}/{segment['path']}"
        _link_or_copy(source, out_dir / relative)
        episode["video"].append(relative)
    initial_state = started.get("initial_state", True)       # rows before the flag always had one
    _write_json(out_dir / episode["trace"], _trace_json(episode, meta, arrays, bool(initial_state)))
    return episode


def _heldout_guard(run: dict, plan) -> dict | None:
    """None for ordinary runs; else {final_only, reportable, reason} for a held-out (final_only) evaluation run."""
    job = (run.get("meta") or {}).get("eval_job") or {}
    if not (job.get("course") or {}).get("final_only"):
        return None
    if plan is None:
        return {"final_only": True, "reportable": False,
                "reason": "held-out (final_only) course: pass the evaluation plan to export results"}
    from ariel.analysis.eval_plan import heldout_reportable      # lazy: recording does not need analysis

    plan_data = plan if isinstance(plan, dict) else _read_json(Path(plan))
    if plan_data is None:
        raise RecordingError(f"evaluation plan {plan} missing or unreadable")
    ok, reason = heldout_reportable(plan_data, (job.get("checkpoint") or {}).get("sha256"))
    return {"final_only": True, "reportable": bool(ok), "reason": reason, "plan_id": plan_data.get("plan_id")}


def export_viewer(run_dir, out_dir, plan=None) -> dict:
    """Export every episode and checkpoint of `run_dir` into `out_dir`; return the index.

    plan: evaluation plan (path or dict); needed to export a held-out (final_only) evaluation run.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    run = _read_json(run_dir / "run.json")
    if run is None:
        raise RecordingError(f"{run_dir / 'run.json'} missing or unreadable")
    if _newer_schema(run) is not None:
        raise RecordingError(f"{run_dir / 'run.json'} has schema_version {run['schema_version']}; this exporter "
                             f"supports up to {SUPPORTED_SCHEMA}")
    heldout = _heldout_guard(run, plan)
    withhold = None if heldout is None or heldout["reportable"] else f"withheld: {heldout['reason']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes, checkpoints, workers = [], [], []
    for worker_dir in sorted(p for p in (run_dir / "workers").glob("*") if p.is_dir()):
        worker = worker_dir.name
        rows, _ = read_ledger(worker_dir / "episodes.jsonl")
        identity = _read_json(worker_dir / "worker.json") or {}
        workers.append({
            "worker": worker, "host": identity.get("host"), "platform": identity.get("platform"),
            "modal_task_id": identity.get("modal_task_id"), "pid": identity.get("pid"),
            "opened_at": identity.get("opened_at"), "settings": identity.get("settings"),
            "sessions": [{"pid": r.get("pid"), "opened_at": r.get("opened_at"),
                          "resumed_from": r.get("resumed_from"),
                          "recovered_episodes": r.get("recovered_episodes") or []}
                         for r in rows if r.get("event") == "worker_opened"]})
        finished = {r["episode_id"]: r for r in rows if r.get("event") == "episode_finished"}
        for started in (r for r in rows if r.get("event") == "episode_started"):
            episodes.append(_episode(run_dir, out_dir, worker, started, finished.get(started["episode_id"]),
                                     withhold))
        for row in (r for r in read_ledger(worker_dir / "checkpoints.jsonl")[0] if r.get("event") == "checkpoint"):
            checkpoints.append({"worker": worker, "file": Path(row["path"]).name, "path": row["path"],
                                "sha256": row["sha256"], "bytes": row["bytes"],
                                "policy_version": row["policy_version"], "training_step": row["training_step"],
                                "note": row.get("note"), "saved_at": row.get("saved_at"),
                                "fixture": bool(run.get("fixture"))})
    episodes.sort(key=lambda e: (e["training_step_start"] is None, e["training_step_start"] or 0,
                                 e["worker"], e["index"]))
    checkpoints.sort(key=lambda c: (c["training_step"], c["worker"]))
    fixture = bool(run.get("fixture"))
    index = {"export_version": EXPORT_VERSION, "run_id": run["run_id"], "fixture": fixture,
             "label": FIXTURE_LABEL if fixture else None, "run": run,
             "git_commit": run.get("git_commit"), "workers": workers,
             "heldout": heldout, "withheld_episodes": sum(e["withheld"] for e in episodes),
             "limits": {"max_frames": MAX_FRAMES, "max_points": MAX_POINTS, "max_action": MAX_ACTION},
             "episodes": episodes, "checkpoints": checkpoints}
    _write_json(out_dir / "index.json", index)
    return index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a recorded run for the web viewer.")
    parser.add_argument("run_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--plan", help="evaluation plan JSON; required to export held-out (final_only) results")
    args = parser.parse_args(argv)
    index = export_viewer(args.run_dir, args.out_dir, plan=args.plan)
    statuses: dict[str, int] = {}
    for episode in index["episodes"]:
        status = "withheld" if episode["withheld"] else episode["status"]
        statuses[status] = statuses.get(status, 0) + 1
    print(json.dumps({"run_id": index["run_id"], "fixture": index["fixture"], "episodes": len(index["episodes"]),
                      "withheld": index["withheld_episodes"], "statuses": statuses,
                      "checkpoints": len(index["checkpoints"]), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
