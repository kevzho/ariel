"""Per-episode metrics from Ariel recordings. See docs/recording-contract.md.

Reads episode summaries, worker ledgers and trace chunks. Supports schema v1
points and v2 offsets from the centre of mass; newer schemas are rejected.
Missing or unverifiable metrics are None. Recovered episodes count as
incomplete even when their summaries are intact. Worker resume records identify
policy versions from discarded learner branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from ariel.recording import RecordingError, read_ledger
from ariel.recording.ledger import sha256_file

TRACE_KEYS = ("step", "offsets", "points", "com", "bbox", "cov", "action", "reward", "policy_version", "wall_time")
EXPLORATION = -1  # Matches recorder.EXPLORATION.
SUPPORTED_SCHEMA = 2


class UnsupportedSchemaError(ValueError):
    """The recording uses a schema_version this analysis does not understand."""


def _check_schema(record: dict | None, where) -> None:
    version = (record or {}).get("schema_version")
    if isinstance(version, int) and version > SUPPORTED_SCHEMA:
        raise UnsupportedSchemaError(f"{where}: schema_version {version} is newer than this analysis "
                                     f"supports ({SUPPORTED_SCHEMA}); update ariel.analysis")


@dataclass
class EpisodeData:
    """One recorded episode: summary, ledger start row, run identity and the recorded trace."""

    path: Path | None
    summary: dict
    start: dict | None = None
    run: dict = field(default_factory=dict)
    initial: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    lineage_discarded: str | None = None             # why policy lineage is excluded (discarded learner branch)

    @property
    def episode_id(self) -> str:
        return self.summary["episode_id"]

    @property
    def kind(self) -> str:
        return self.summary.get("kind") or (self.start or {}).get("kind") or "unknown"

    @property
    def status(self) -> str:
        return self.summary.get("status") or "open"

    @property
    def recovered(self) -> bool:
        return bool(self.summary.get("recovered"))

    @property
    def steps_recorded(self) -> int:
        return int(len(self.trace["step"])) if "step" in self.trace else 0

    @property
    def trace_complete(self) -> bool:
        """True only when every step the episode reports is present in readable, verified chunks."""
        return (self.status != "open" and not self.recovered and not self.problems
                and self.summary.get("steps") is not None and self.steps_recorded == self.summary["steps"])

    @property
    def course(self) -> dict:
        return self.summary.get("course") or (self.start or {}).get("course") or {}


def course_hash(course: dict | None) -> str:
    """Hash the full course config; comparisons require matching ids and hashes."""
    text = json.dumps(course or {}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def course_id(course: dict | None) -> str:
    """The course's declared `id` (or `name`), else a short hash of its configuration."""
    course = course or {}
    for key in ("id", "name"):
        if course.get(key) is not None:
            return str(course[key])
    return "sha-" + course_hash(course)[:10]


def _ledger_rows(worker_dir: Path) -> tuple[dict, dict, list[dict]]:
    """(episode_started by id, episode_finished by id, every row in ledger order). Other events are ignored."""
    rows, _ = read_ledger(worker_dir / "episodes.jsonl")
    started = {r["episode_id"]: r for r in rows if r.get("event") == "episode_started"}
    finished = {r["episode_id"]: r for r in rows if r.get("event") == "episode_finished"}
    for row in started.values():
        _check_schema(row, worker_dir / "episodes.jsonl")
    return started, finished, rows


def _max_policy_version(ep: "EpisodeData") -> int | None:
    """Highest learner version seen in the episode (trace first, then any summary field); None if unknown."""
    pv = ep.trace.get("policy_version")
    if pv is not None and len(pv) and (pv != EXPLORATION).any():
        return int(pv[pv != EXPLORATION].max())
    values = [ep.summary.get(k) for k in ("policy_version_last", "policy_version_first",
                                          "trace_policy_version_last", "trace_policy_version_first")]
    values = [int(v) for v in values if v is not None]
    return max(values) if values else None


def _mark_discarded_lineage(rows: list[dict], episodes: dict[str, "EpisodeData"], notes: list[str]) -> None:
    """Mark pre-resume episodes with versions beyond resumed_from.policy_version.

    These versions belong to a discarded learner branch. Keep them separate from
    versions produced after resuming from the earlier checkpoint.
    """
    for position, row in enumerate(rows):
        if row.get("event") != "worker_opened" or not row.get("resumed_from"):
            continue
        resumed = row["resumed_from"]
        version = resumed.get("policy_version") if isinstance(resumed, dict) else None
        if version is None:
            notes.append(f"worker_opened at {row.get('opened_at')} has resumed_from without policy_version; "
                         "earlier lineage cannot be checked against it")
            continue
        for earlier in rows[:position]:
            ep = episodes.get(earlier.get("episode_id")) if earlier.get("event") == "episode_started" else None
            if ep is None or ep.lineage_discarded or ep.kind != "training":
                continue
            highest = _max_policy_version(ep)
            if highest is not None and highest > int(version):
                ep.lineage_discarded = (f"recorded with policy_version up to {highest}, beyond resumed_from "
                                        f"policy_version {version} (resume at {row.get('opened_at')}); "
                                        "that learner branch was discarded")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _chunk_number(path: Path) -> int:
    digits = "".join(c for c in path.stem.split("-")[-1] if c.isdigit())
    return int(digits) if digits else -1


def _read_trace(directory: Path, listed: list[dict] | None, problems: list[str]) -> tuple[dict, dict]:
    """(initial arrays, concatenated trace). Chunks without a listed sha256 (open episode) are unverified."""
    initial = {}
    meta_path = directory / "trace-meta.npz"
    if meta_path.exists():
        try:
            with np.load(meta_path) as data:
                initial = {k: np.asarray(data[k]) for k in data.files}
        except Exception as exc:                      # noqa: BLE001 - disclosed, never hidden
            problems.append(f"trace-meta.npz unreadable: {exc}")
    if "initial_offsets" in initial and "initial_com" in initial:
        initial["initial_points"] = (initial["initial_com"].astype(np.float64)[None, :]
                                     + initial["initial_offsets"].astype(np.float64))
    if listed is None:
        listed = [{"path": p.name} for p in sorted(directory.glob("trace-[0-9]*.npz"), key=_chunk_number)]
        if listed:
            problems.append(f"{len(listed)} trace chunk{'s' if len(listed) != 1 else ''} read without a summary "
                            "to verify sha256 against")
    parts: dict[str, list] = {k: [] for k in TRACE_KEYS}
    expected_step = 1
    for chunk in listed:
        chunk_path = directory / chunk["path"]
        if not chunk_path.exists():
            problems.append(f"missing trace chunk {chunk['path']}")
            continue
        if chunk.get("sha256") and sha256_file(chunk_path) != chunk["sha256"]:
            problems.append(f"trace chunk {chunk['path']} does not match its recorded sha256")
            continue
        try:
            with np.load(chunk_path) as data:
                arrays = {k: np.asarray(data[k]) for k in TRACE_KEYS if k in data.files}
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"trace chunk {chunk['path']} unreadable: {exc}")
            continue
        if int(arrays["step"][0]) != expected_step:
            problems.append(f"trace gap before step {int(arrays['step'][0])}")
        expected_step = int(arrays["step"][-1]) + 1
        for key, value in arrays.items():
            parts[key].append(value)
    trace = {k: np.concatenate(v) for k, v in parts.items() if v}
    if "offsets" in trace and "com" in trace:               # schema v2: points = com + offsets
        trace["points"] = trace["com"].astype(np.float64)[:, None, :] + trace["offsets"].astype(np.float64)
    return initial, trace


def load_episode(path, *, start: dict | None = None, run: dict | None = None,
                 finish: dict | None = None) -> EpisodeData:
    """Load an episode directory (or its episode.json) and every verified trace chunk.

    finish: the ledger's episode_finished row, when known; episode.json is checked against its hash.
    """
    path = Path(path)
    directory = path.parent if path.name == "episode.json" else path
    problems: list[str] = []
    summary = _read_json(directory / "episode.json")
    if not summary:
        raise FileNotFoundError(f"no readable episode.json in {directory}")
    _check_schema(summary, directory / "episode.json")
    if start is None:
        try:
            start = _ledger_rows(directory.parents[1])[0].get(summary["episode_id"])
        except (RecordingError, IndexError) as exc:
            problems.append(f"ledger unreadable: {exc}")
    if run is None:
        run = _read_json(directory.parents[3] / "run.json") if len(directory.parents) > 3 else {}
    _check_schema(run, "run.json")
    if finish is not None and finish.get("episode_json_sha256") and \
            sha256_file(directory / "episode.json") != finish["episode_json_sha256"]:
        problems.append("episode.json does not match the hash recorded in the ledger")
    meta_sha = summary.get("trace_meta_sha256")
    if meta_sha and (directory / "trace-meta.npz").exists() and sha256_file(directory / "trace-meta.npz") != meta_sha:
        problems.append("trace-meta.npz does not match its recorded sha256")
    initial, trace = _read_trace(directory, summary.get("trace_chunks"), problems)
    return EpisodeData(directory, summary, start, run, initial, trace, problems)


def load_run(run_dir, *, problems: list[str] | None = None) -> list[EpisodeData]:
    """Load all workers' episodes, including open and recovered recordings.

    Missing directories or summaries are reported in the episode's problems.
    If `problems` is supplied, unreadable ledgers are added there instead of
    raising. Newer schemas always raise UnsupportedSchemaError. Episodes from
    discarded learner branches are marked with `lineage_discarded`.
    """
    run_dir = Path(run_dir)
    run = _read_json(run_dir / "run.json")
    _check_schema(run, run_dir / "run.json")
    episodes = []
    for worker_dir in sorted((run_dir / "workers").glob("*")):
        if not (worker_dir / "episodes.jsonl").exists():
            continue
        _check_schema(_read_json(worker_dir / "worker.json"), worker_dir / "worker.json")
        try:
            started, finished, ledger_rows = _ledger_rows(worker_dir)
        except RecordingError as exc:
            if problems is None:
                raise
            problems.append(f"{worker_dir}: {exc}")
            continue
        worker_episodes: dict[str, EpisodeData] = {}
        for episode_id, start in started.items():
            directory = worker_dir / "episodes" / episode_id
            end = finished.get(episode_id)
            if (directory / "episode.json").exists():
                ep = load_episode(directory, start=start, run=run, finish=end)
                if end is None:                          # killed after episode.json, before its ledger row
                    ep.summary = {**ep.summary, "status": None}
                    ep.problems.append("episode.json present but no episode_finished row (not yet recovered)")
                worker_episodes[episode_id] = ep
                continue
            summary = {"episode_id": episode_id, "run_id": start.get("run_id"), "worker": start.get("worker"),
                       "episode_index": start.get("episode_index"), "kind": start.get("kind"),
                       "fixture": start.get("fixture", False), "status": end["status"] if end else None,
                       "steps": end.get("steps") if end else None, "return": end.get("return") if end else None,
                       "course": start.get("course"), "training_step_start": start.get("training_step_start"),
                       "recovered": bool(end and end.get("recovered"))}
            notes = ["episode.json missing" if end else "episode still open (no end row)"]
            initial, trace = _read_trace(directory, None, notes) if directory.exists() else ({}, {})
            worker_episodes[episode_id] = EpisodeData(directory if directory.exists() else None, summary, start,
                                                      run, initial, trace, notes)
        _mark_discarded_lineage(ledger_rows, worker_episodes, problems if problems is not None else [])
        episodes += worker_episodes.values()
    return episodes


def aspect_series(cov: np.ndarray) -> np.ndarray:
    """sqrt(largest / smallest eigenvalue) of each [xx, xy, yy] second-moment row; NaN when degenerate."""
    cov = np.asarray(cov, dtype=np.float64).reshape(-1, 3)
    xx, xy, yy = cov[:, 0], cov[:, 1], cov[:, 2]
    mean, spread = (xx + yy) / 2, np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
    high, low = mean + spread, mean - spread
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(low > 1e-12 * np.maximum(high, 1e-300), np.sqrt(high / low), np.nan)
    return ratio


def _f(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _stats(values: np.ndarray, prefix: str) -> dict:
    values = values[np.isfinite(values)] if len(values) else values
    if not len(values):
        return {f"{prefix}_{k}": None for k in ("min", "max", "range", "std", "mean")}
    return {f"{prefix}_min": float(values.min()), f"{prefix}_max": float(values.max()),
            f"{prefix}_range": float(values.max() - values.min()), f"{prefix}_std": float(values.std()),
            f"{prefix}_mean": float(values.mean())}


def _failure_mode(ep: EpisodeData, completed, progress) -> str | None:
    info = ep.summary.get("info") or {}
    status = ep.status
    if completed:
        return None
    if status == "open":
        return "open"
    if status == "interrupted":
        return "interrupted"
    if status == "failed":
        return "recording_failed"
    for key in ("failure_mode", "failure", "reason"):
        if isinstance(info.get(key), str) and info[key]:
            return info[key]
    for flag in ("split", "fell", "stuck", "collision"):
        if info.get(flag) is True:
            return flag
    if isinstance(info.get("scripted_ending"), str) and info["scripted_ending"]:
        return "scripted_" + info["scripted_ending"].replace(" ", "_")     # fixture script, not a learned outcome
    if info.get("scripted") or (ep.course or {}).get("scripted"):
        return "scripted"
    if progress is not None and progress < 0:
        return "backward"
    if status == "truncated":
        return "timeout"
    if status == "stopped":
        return "stopped"
    return "terminated_unexplained" if completed is False else "unknown"


def episode_metrics(ep: EpisodeData, course: dict | None = None) -> dict:
    """Flat metric row for one episode. Missing or unverifiable values are None."""
    course = ep.course if course is None else course
    summary, trace, info = ep.summary, ep.trace, ep.summary.get("info") or {}
    complete = ep.trace_complete
    com = trace.get("com")
    x = np.asarray(com[:, 0], dtype=np.float64) if com is not None and len(com) else np.empty(0)
    start_x = None
    if "initial_com" in ep.initial:
        start_x = float(ep.initial["initial_com"][0])
    elif len(x):
        start_x = float(x[0])                                # no initial state given: first recorded state
    goal_x = course.get("goal_x")
    reached = np.flatnonzero(x >= goal_x) if goal_x is not None and len(x) else np.empty(0, dtype=int)
    if isinstance(info.get("success"), bool):
        completed = info["success"]
    elif len(reached):
        completed = True
    elif goal_x is not None and complete and ep.status in ("terminated", "truncated", "stopped"):
        completed = False
    else:
        completed = None
    steps_to_goal = int(trace["step"][reached[0]]) if len(reached) else None
    if steps_to_goal is None and completed and info.get("steps_to_goal") is not None:
        steps_to_goal = int(info["steps_to_goal"])
    progress = _f(x[-1] - start_x) if complete and len(x) and start_x is not None else None
    max_progress = _f(x.max() - start_x) if complete and len(x) and start_x is not None else None
    episode_return = summary.get("return")                   # recorder's in-memory sum; None when salvaged
    rewards = trace.get("reward")
    trace_return = summary.get("trace_return")               # recovered: partial, from surviving chunks only
    if trace_return is None and rewards is not None and len(rewards):
        finite = rewards.astype(np.float64)
        trace_return = float(finite[np.isfinite(finite)].sum())
    last_step = summary.get("last_step")
    if last_step is None and "step" in trace and len(trace["step"]):
        last_step = int(trace["step"][-1])
    steps = None if "steps_recorded" in summary else summary.get("steps")   # salvaged: true length unknown
    cov, bbox = trace.get("cov"), trace.get("bbox")
    shape: dict = {}
    if cov is not None and len(cov):
        series = np.asarray(cov, dtype=np.float64)
        if "initial_cov" in ep.initial:
            series = np.vstack([ep.initial["initial_cov"].astype(np.float64), series])
        shape.update(_stats(aspect_series(series), "aspect"))
    else:
        shape.update(_stats(np.empty(0), "aspect"))
    if bbox is not None and len(bbox):
        boxes = np.asarray(bbox, dtype=np.float64)
        if "initial_bbox" in ep.initial:
            boxes = np.vstack([ep.initial["initial_bbox"].astype(np.float64), boxes])
        height, width = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
        shape.update(height_min=float(height.min()), height_max=float(height.max()),
                     width_min=float(width.min()), width_max=float(width.max()))
    else:
        shape.update(height_min=None, height_max=None, width_min=None, width_max=None)
    action = trace.get("action")
    mean_abs_action = float(np.abs(action.astype(np.float64)).mean()) if action is not None and action.size else None
    pv = trace.get("policy_version")
    discarded = ep.lineage_discarded is not None
    versions = None if ep.recovered else summary.get("policy_versions_distinct")
    exploration = None if ep.recovered else summary.get("exploration_steps")
    if pv is not None and complete:
        if versions is None:
            versions = int(len(np.unique(pv[pv != EXPLORATION])))
        if exploration is None:
            exploration = int((pv == EXPLORATION).sum())
    share = exploration / steps if exploration is not None and steps else None
    partial_exploration = int((pv == EXPLORATION).sum()) if pv is not None and not complete else None
    eval_job = (ep.run.get("meta") or {}).get("eval_job") or {}
    point_count = ep.initial.get("point_count")
    point_count = int(point_count) if point_count is not None else summary.get("trace_point_count")
    return {
        "episode_id": ep.episode_id, "run_id": summary.get("run_id"), "worker": summary.get("worker"),
        "kind": ep.kind, "fixture": bool(summary.get("fixture")), "status": ep.status,
        "recovered": ep.recovered, "trace_complete": complete, "steps": steps,
        "steps_recorded": ep.steps_recorded, "last_step": last_step, "course_id": course_id(course),
        "course_hash": course_hash(course), "point_count": point_count,
        "seed": (ep.start or {}).get("seed"), "training_step_start": summary.get("training_step_start"),
        "started_at": (ep.start or {}).get("started_at"),
        "policy_version_first": None if discarded else summary.get("policy_version_first"),
        "policy_version_last": None if discarded else summary.get("policy_version_last"),
        "policy_versions_distinct": None if discarded else versions,
        "trace_policy_version_first": None if discarded else summary.get("trace_policy_version_first"),
        "trace_policy_version_last": None if discarded else summary.get("trace_policy_version_last"),
        "lineage_discarded": ep.lineage_discarded,
        "exploration_steps": exploration, "exploration_share": share,
        "exploration_steps_in_partial_trace": partial_exploration,
        "checkpoint_label": eval_job.get("checkpoint", {}).get("label"),
        "checkpoint_sha256": eval_job.get("checkpoint", {}).get("sha256"),
        "arm": eval_job.get("arm"), "course_role": eval_job.get("course", {}).get("role"),
        "final_only": eval_job.get("course", {}).get("final_only"), "deterministic": eval_job.get("deterministic"),
        "return": None if episode_return is None else float(episode_return), "trace_return": trace_return,
        "progress_x": progress, "max_progress": max_progress, "completed": completed,
        "steps_to_goal": steps_to_goal, "failure_mode": _failure_mode(ep, completed, progress),
        "mean_abs_action": mean_abs_action, **shape,
        "problems": "; ".join(ep.problems) or None,
    }
