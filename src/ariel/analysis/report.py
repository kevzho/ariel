"""Write Markdown, SVG, CSV and JSON reports from recorded runs.

    python -m ariel.analysis.report runs/r1 runs/eval-* --out reports/r1 [--plan eval-plan.json]

The inventory includes failed, interrupted and open episodes. Curves and
comparisons exclude fixtures. Held-out results require an evaluation plan that
allows reporting them (see eval_plan.py).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from ariel.recording import RecordingError, read_ledger

from .curves import curve_from_rows
from .eval_plan import heldout_reportable, mark_heldout_viewed
from .metrics import episode_metrics, load_run
from .svg import bar_chart, line_chart

STATUS_ORDER = ("terminated", "truncated", "stopped", "interrupted", "failed", "open")
KIND_ORDER = ("training", "evaluation", "fixture")
SHAPE_METRICS = (("aspect_range", "aspect-ratio range"), ("aspect_std", "aspect-ratio std"),
                 ("height_span", "height max - min"), ("width_span", "width max - min"))


def bootstrap_ci(values, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean with a percentile-bootstrap (1 - alpha) interval. None values are dropped; n is reported."""
    data = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=np.float64)
    n = int(len(data))
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if n == 1:
        return {"mean": float(data[0]), "lo": None, "hi": None, "n": 1}
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(data.mean()), "lo": float(lo), "hi": float(hi), "n": n}


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def _ci_text(ci: dict, small_n: int) -> str:
    if ci["n"] == 0:
        return "no data (n=0)"
    text = f"{_num(ci['mean'])}"
    text += f" [{_num(ci['lo'])}, {_num(ci['hi'])}]" if ci["lo"] is not None else " [no interval]"
    text += f" (n={ci['n']}{', small n' if ci['n'] < small_n else ''})"
    return text


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return lines


def _n(count: int, word: str, plural: str | None = None) -> str:
    return f"{count} {word if count == 1 else plural or word + 's'}"


IDENTITY = ("episode_id", "run_id", "worker", "kind", "fixture", "status", "recovered", "trace_complete",
            "course_id", "course_hash", "seed", "started_at", "checkpoint_label", "checkpoint_sha256", "arm",
            "course_role", "final_only", "deterministic", "lineage_discarded", "problems")


def _withhold(row: dict, reason: str) -> dict:
    """Strip metrics from a withheld episode, retaining identity and status."""
    return {**{k: row.get(k) for k in IDENTITY}, "withheld": True, "withheld_reason": reason}


def _heldout_courses(evaluation: list[dict], plan: dict | None) -> dict[str, tuple[bool, str | None]]:
    """Map course ids to (held out, ambiguity reason). Ambiguous roles are withheld.

    The plan determines each course's role. Missing ids or conflicting recorded
    roles are ambiguous. Without a plan, any run marking a course as held out
    makes that course held out across all runs.
    """
    roles: dict[str, set] = {}
    if plan:
        for course in plan.get("courses", []):
            roles.setdefault(str(course["id"]), set()).add("heldout" if course.get("final_only")
                                                           or course.get("role") == "heldout" else "showcase")
    result = {}
    for cid in sorted({r["course_id"] for r in evaluation}):
        members = [r for r in evaluation if r["course_id"] == cid]
        claimed = {"heldout" if r["final_only"] or r["course_role"] == "heldout" else r["course_role"]
                   for r in members} - {None}
        if not plan:
            result[cid] = ("heldout" in claimed, None)
        elif cid not in roles:
            result[cid] = (True, "course id is not in the evaluation plan, so its role is ambiguous")
        elif len(roles[cid]) > 1:
            result[cid] = (True, "the evaluation plan lists this course id with more than one role")
        elif claimed - roles[cid]:
            result[cid] = (True, f"recorded course role {sorted(claimed)} conflicts with the plan "
                                 f"({next(iter(roles[cid]))})")
        else:
            result[cid] = (roles[cid] == {"heldout"}, None)
    return result


def _inventory(rows: list[dict]) -> dict:
    kinds = sorted({r["kind"] for r in rows}, key=lambda k: (KIND_ORDER + (k,)).index(k))
    statuses = list(STATUS_ORDER) + sorted({r["status"] for r in rows} - set(STATUS_ORDER))
    counts = {k: {s: sum(1 for r in rows if r["kind"] == k and r["status"] == s) for s in statuses} for k in kinds}
    return {"total": len(rows), "by_kind_status": counts,
            "recovered": sum(1 for r in rows if r["recovered"]),
            "incomplete_trace": sum(1 for r in rows if not r["trace_complete"]),
            "with_problems": sum(1 for r in rows if r["problems"]),
            "fixture": sum(1 for r in rows if r["fixture"]),
            "withheld": sum(1 for r in rows if r.get("withheld"))}


def _checkpoint_key(row: dict) -> tuple:
    if row["checkpoint_sha256"]:
        return (row["policy_version_first"] if row["policy_version_first"] is not None else -1,
                f"{row['checkpoint_label']} (pv {row['policy_version_first']}, {row['checkpoint_sha256'][:8]})")
    return (row["policy_version_first"] if row["policy_version_first"] is not None else -1,
            f"pv {row['policy_version_first']} (no plan metadata)")


def _shape_spans(row: dict) -> dict:
    span = lambda a, b: None if row[a] is None or row[b] is None else row[b] - row[a]   # noqa: E731
    return {**row, "height_span": span("height_min", "height_max"), "width_span": span("width_min", "width_max")}


def _runtime_identity(run_dir: Path) -> dict:
    """Read the run's commit, worker hosts, Modal tasks and resume history.

    worker.json describes the latest session. The ledger supplies the total
    session count and each session's `resumed_from` record.
    """
    try:
        run = json.loads((run_dir / "run.json").read_text())
    except (OSError, json.JSONDecodeError):
        run = {}
    workers = []
    for worker_dir in sorted((run_dir / "workers").glob("*")):
        try:
            info = json.loads((worker_dir / "worker.json").read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
        try:
            opened = [r for r in read_ledger(worker_dir / "episodes.jsonl")[0] if r.get("event") == "worker_opened"]
        except (RecordingError, OSError):
            opened = []
        workers.append({"worker": worker_dir.name, "host": info.get("host"), "modal_task_id": info.get("modal_task_id"),
                        "platform": info.get("platform"), "last_opened_at": info.get("opened_at"),
                        "sessions": len(opened), "pids": [r.get("pid") for r in opened],
                        "resumed_from": [r["resumed_from"] for r in opened if r.get("resumed_from")]})
    return {"run_dir": str(run_dir), "run_id": run.get("run_id"), "git_commit": run.get("git_commit"),
            "created_at": run.get("created_at"), "workers": workers}


def _values(members: list[dict], key: str) -> list[str]:
    return sorted({str(r[key]) for r in members})


def build_report(run_dirs, out_dir, *, plan=None, window: int = 20, n_boot: int = 2000, seed: int = 0,
                 small_n: int = 10) -> dict:
    """Write report.md, episodes.csv, report.json and SVG plots to out_dir; return their paths and data.

    plan: path to the evaluation plan JSON. It decides which course ids are held out. Without it,
    held-out course results are withheld. Withheld episodes keep only identity and status in every
    output (rows, CSV, JSON, failure table).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(d) for d in run_dirs]
    problems: list[str] = []
    episodes = [ep for d in run_dirs for ep in load_run(d, problems=problems)]
    rows = [_shape_spans(episode_metrics(ep)) for ep in episodes]
    plan_data = json.loads(Path(plan).read_text()) if plan else None
    identities = [_runtime_identity(d) for d in run_dirs]
    # Lineage is not an outcome, so it is checked before held-out rows lose their metrics.
    policy_warnings = [{"episode_id": r["episode_id"], "run_id": r["run_id"],
                        "policy_versions_distinct": r["policy_versions_distinct"],
                        "policy_version_first": r["policy_version_first"],
                        "policy_version_last": r["policy_version_last"]}
                       for r in rows if r["kind"] == "evaluation" and r["policy_versions_distinct"] != 1]
    ci = lambda values: bootstrap_ci(values, n_boot=n_boot, seed=seed)       # noqa: E731

    # Strip withheld metrics before any report section reads the rows.
    heldout = _heldout_courses([r for r in rows if r["kind"] == "evaluation"], plan_data)
    withheld = []
    for i, r in enumerate(rows):
        is_heldout, ambiguity = heldout.get(r["course_id"], (False, None)) if r["kind"] == "evaluation" else (
            False, None)
        if not is_heldout:
            rows[i] = {**r, "withheld": False, "withheld_reason": None}
            continue
        if ambiguity:
            ok, reason = False, ambiguity
        elif plan_data:
            ok, reason = heldout_reportable(plan_data, r["checkpoint_sha256"])
        else:
            ok, reason = False, "no evaluation plan given, so the held-out guard cannot be checked"
        if ok:
            rows[i] = {**r, "withheld": False, "withheld_reason": None}
        else:
            rows[i] = _withhold(r, reason)
            withheld.append({"episode_id": r["episode_id"], "course_id": r["course_id"], "reason": reason})

    files = {"report": out_dir / "report.md", "csv": out_dir / "episodes.csv", "json": out_dir / "report.json"}
    md: list[str] = ["# Ariel learning-progress report", ""]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md += [f"Generated {generated} from {_n(len(run_dirs), 'run directory', 'run directories')}: "
           + ", ".join(f"`{d}`" for d in run_dirs) + ".", ""]
    real = [r for r in rows if not r["fixture"]]
    if rows and not real:
        md += ["> **FIXTURE DATA ONLY.** Every episode here is synthetic fixture data. Nothing in this report "
               "shows learned behavior.", ""]
    elif len(real) != len(rows):
        fixtures = len(rows) - len(real)
        md += [f"> {_n(fixtures, 'fixture episode')} {'is' if fixtures == 1 else 'are'} counted in the inventory "
               "and excluded from every curve and comparison.", ""]
    md += ["This report states measured values with counts and 95% bootstrap intervals. It makes no claim "
           f"of improvement by itself. Groups with n < {small_n} are marked *small n*: their intervals are "
           "wide and differences between them are not conclusive.", ""]

    inventory = _inventory(rows)
    md += ["## Data inventory", ""]
    statuses = list(next(iter(inventory["by_kind_status"].values()), {s: 0 for s in STATUS_ORDER}))
    md += _table(["kind", *statuses, "total"],
                 [[k, *[c[s] for s in statuses], sum(c.values())] for k, c in inventory["by_kind_status"].items()])
    md += ["", f"Total episodes: {inventory['total']}. Recovered after a crash: {inventory['recovered']}. "
           f"Incomplete or unverified trace: {inventory['incomplete_trace']}. "
           f"Episodes with disclosed problems: {inventory['with_problems']}. "
           f"Held-out episodes withheld (status only): {inventory['withheld']}.", ""]
    if problems:
        md += ["Unreadable ledgers (their episodes are absent from every count above):", ""]
        md += [f"- {p}" for p in problems] + [""]

    curve = curve_from_rows(rows, window=window, kind="training")
    md += ["## Learning curve (training episodes only)", ""]
    plots = {}
    if curve["episodes"]:
        xs = [p["training_step_start"] for p in curve["points"]]
        for metric, label in (("return", "return"), ("progress_x", "progress in x"),
                              ("completed", "completion rate"), ("exploration_share", "exploration share")):
            name = f"curve_{metric}.svg"
            svg = line_chart([{"label": f"rolling mean ({window})", "x": xs,
                               "y": [p["rolling"][metric]["mean"] for p in curve["points"]]},
                              {"label": f"rolling median ({window})", "x": xs,
                               "y": [p["rolling"][metric]["median"] for p in curve["points"]]}],
                             title=f"Training {label}, rolling window of {window} episodes",
                             xlabel="training step at episode start", ylabel=label,
                             y_range=(0.0, 1.0) if metric in ("completed", "exploration_share") else None)
            (out_dir / name).write_text(svg)
            plots[name] = out_dir / name
            md += [f"![training {label}]({name})", ""]
        cell = lambda r: f"{_num(r['mean'])} / {_num(r['median'])} (n={r['n']})"       # noqa: E731
        metrics = ("return", "progress_x", "completed", "exploration_share")
        if curve["episodes"] <= window:
            last = curve["points"][-1]
            md += _table(["metric", f"all {curve['episodes']} episodes mean / median"],
                         [[m, cell(last["rolling"][m])] for m in metrics])
            md += ["", f"Only {_n(curve['episodes'], 'training episode')} (window {window}), so first and last "
                   "windows would be the same episodes; one all-episodes row is shown and no early/late change "
                   "can be read from it.", ""]
        else:
            first, last = curve["points"][window - 1], curve["points"][-1]
            md += _table(["metric", "first window mean / median", "last window mean / median"],
                         [[m, cell(first["rolling"][m]), cell(last["rolling"][m])] for m in metrics])
        md += ["", f"{_n(curve['episodes'], 'training episode')} from all workers, ordered by training step. "
               "Each window counts only episodes where the value exists (n); interrupted episodes without a "
               "recorded return or complete trace leave gaps instead of estimates.", ""]
        x = curve["exploration"]
        md += [f"Exploration (actions no policy chose, `policy_version` -1, e.g. random warm-up): "
               f"{_n(x['exploration_steps'], 'step')} in {_n(x['episodes_with_exploration'], 'episode')}, of which "
               f"{x['episodes_all_exploration']} had no policy-chosen step; the share is unknown for "
               f"{_n(x['episodes_share_unknown'], 'episode')}. Curve values from exploration steps measure the "
               "warm-up, not a learned policy.", ""]
        discarded = [r for r in rows if r["kind"] == "training" and r.get("lineage_discarded")]
        if discarded:
            md += [f"{_n(len(discarded), 'training episode')} {'was' if len(discarded) == 1 else 'were'} recorded "
                   "before a resume by a learner branch that was then discarded (policy versions beyond the "
                   "checkpoint training resumed from). They stay in the inventory and curve as experience, but "
                   "their policy versions are left empty and no lineage claim uses them: "
                   + ", ".join(f"`{r['episode_id']}`" for r in discarded) + ".", ""]
        configs = sorted({(r["course_id"], r["course_hash"]) for r in rows if r["kind"] == "training"})
        if len(configs) > 1:
            md += [f"The curve mixes {_n(len(configs), 'course config')}: "
                   + ", ".join(f"`{c}` ({h})" for c, h in configs) + ". Values are not comparable across them.", ""]
    else:
        md += ["No training episodes were found, so there is no learning curve.", ""]
    if curve["excluded_by_kind"]:
        md += ["Excluded from the curve by kind: " + ", ".join(
            f"{k} {_n(v, 'episode')}" for k, v in sorted(curve["excluded_by_kind"].items())) + ".", ""]
    if curve["unordered_episode_ids"]:
        count = len(curve["unordered_episode_ids"])
        md += [f"{_n(count, 'training episode')} {'has' if count == 1 else 'have'} no training step and "
               f"{'is' if count == 1 else 'are'} left out of the curve order.", ""]

    md += ["## Checkpoint comparison on the same course (evaluation episodes)", ""]
    evaluation = [r for r in rows if r["kind"] == "evaluation"]
    viewing_marked, comparison, conflicts = False, [], []
    for course in sorted({r["course_id"] for r in evaluation}):
        is_heldout = heldout[course][0]
        course_rows = [r for r in evaluation if r["course_id"] == course and not r["withheld"]]
        if not course_rows:
            continue
        hashes = {h: sum(1 for r in course_rows if r["course_hash"] == h) for h in _values(course_rows, "course_hash")}
        if len(hashes) > 1:
            conflicts.append({"course_id": course, "configs": hashes})
            md += [f"### Course `{course}`: not compared", "",
                   f"Episodes recorded under this id use {_n(len(hashes), 'different course config')} ("
                   + ", ".join(f"{h}: {_n(n, 'episode')}" for h, n in hashes.items())
                   + "). Give each layout or physics setting its own `id`.", ""]
            continue
        if is_heldout and plan_data and not plan_data.get("heldout_first_viewed_at"):
            plan_data["heldout_first_viewed_at"] = mark_heldout_viewed(plan)
            viewing_marked = True
        md += [f"### Course `{course}`" + (" (held-out, final evaluation only)" if is_heldout else "")
               + f", config {next(iter(hashes))}", ""]
        groups: dict[tuple, list] = {}
        for r in course_rows:
            groups.setdefault((r["arm"] or "full",) + _checkpoint_key(r), []).append(r)
        table, bars = [], []
        for (arm, _, label), members in sorted(groups.items(), key=lambda kv: (kv[0][0] != "full", kv[0][1])):
            completed = [None if r["completed"] is None else float(r["completed"]) for r in members]
            stats = {m: ci([r[m] for r in members]) for m in ("progress_x", "return")}
            stats["completed"] = ci(completed)
            unknown = sum(1 for c in completed if c is None)
            seeds, deterministic, points = (_values(members, k) for k in ("seed", "deterministic", "point_count"))
            table.append([arm, label, len(members), _ci_text(stats["completed"], small_n), unknown,
                          _ci_text(stats["progress_x"], small_n), _ci_text(stats["return"], small_n),
                          ", ".join(seeds), ", ".join(deterministic), ", ".join(points)])
            bars.append({"label": label if arm == "full" else f"{label} [{arm}]", **stats["progress_x"]})
            comparison.append({"course_id": course, "course_hash": next(iter(hashes)), "heldout": is_heldout,
                               "arm": arm, "checkpoint": label, "episodes": len(members), "seeds": seeds,
                               "deterministic": deterministic, "point_counts": points,
                               **{f"{m}_ci": s for m, s in stats.items()}})
        md += _table(["arm", "checkpoint", "episodes", "completion rate", "completion unknown",
                      "progress in x", "return", "seeds", "deterministic", "initial points"], table)
        shown = [c for c in comparison if c["course_id"] == course]
        notes = ["seed sets differ between groups"] if len({tuple(c["seeds"]) for c in shown}) > 1 else []
        for key, label in (("deterministic", "deterministic flags"), ("point_counts", "initial point counts")):
            values = sorted({v for c in shown for v in c[key]})
            if len(values) > 1:
                notes.append(f"{label} are mixed ({', '.join(values)})")
        if notes:
            md += ["", "**Caution:** " + "; ".join(notes) + ". Differences may come from these, not the checkpoint."]
        name = f"compare_{''.join(c if c.isalnum() else '_' for c in course)}.svg"
        (out_dir / name).write_text(bar_chart(bars, title=f"Mean progress in x on {course} (95% bootstrap CI)",
                                              ylabel="progress in x"))
        plots[name] = out_dir / name
        md += ["", f"![progress on {course}]({name})", ""]
    if not evaluation:
        md += ["No evaluation episodes were found. Frozen-checkpoint comparisons need the jobs in the "
               "evaluation plan to be run and recorded with kind `evaluation`.", ""]
    eval_runs = {r["run_id"] for r in evaluation}
    eval_identities = [i for i in identities if i["run_id"] in eval_runs]
    if eval_identities:
        md += ["### Runtime identity of evaluation runs", ""]
        md += _table(["run", "git commit", "worker", "host", "Modal task", "sessions", "resumed from"],
                     [[i["run_id"], (i["git_commit"] or "unknown")[:12], w["worker"], w["host"] or "unknown",
                       w["modal_task_id"] or "-", w["sessions"], len(w["resumed_from"]) or "-"]
                      for i in eval_identities for w in i["workers"]])
        md += ["", "Host and Modal task come from worker.json, which names the worker's last session.", ""]
        known = {i["git_commit"] for i in eval_identities} - {None}
        unknown_commit = sum(1 for i in eval_identities if i["git_commit"] is None)
        if len(known) > 1:
            md += [f"**Caution:** evaluation runs were recorded from {len(known)} different git commits; "
                   "differences between checkpoints may come from code changes.", ""]
        if unknown_commit:
            md += [f"**Caution:** {_n(unknown_commit, 'evaluation run')} recorded no git commit, so the code "
                   "that produced them cannot be confirmed.", ""]
    if policy_warnings:
        changed = [w for w in policy_warnings if w["policy_versions_distinct"] is not None]
        unknown = len(policy_warnings) - len(changed)
        md += ["**Caution (policy lineage):** an evaluation episode must be played by one frozen checkpoint "
               "(`policy_versions_distinct` = 1)."]
        if changed:
            md += [f"{_n(len(changed), 'evaluation episode')} did not: "
                   + ", ".join(f"`{w['episode_id']}` ({w['policy_versions_distinct']} versions, "
                               f"{w['policy_version_first']}-{w['policy_version_last']})" for w in changed)
                   + ". The policy changed during evaluation, or no policy chose the actions."]
        if unknown:
            md += [f"For {_n(unknown, 'evaluation episode')} the count is unknown (recovered or unverified "
                   "trace), so a frozen policy cannot be confirmed."]
        md += [""]
    if viewing_marked:
        md += [f"Held-out results were viewed for the first time in this report; the plan now records "
               f"`heldout_first_viewed_at = {plan_data['heldout_first_viewed_at']}`.", ""]

    md += ["## Reshaping usage, early versus late training", ""]
    ordered = [r for r in rows if r["kind"] == "training" and r["training_step_start"] is not None]
    ordered.sort(key=lambda r: (r["training_step_start"], r["started_at"] or "", r["episode_id"]))
    shape = {}
    if len(ordered) >= 2:
        third = max(len(ordered) // 3, 1)
        early, late = ordered[:third], ordered[-third:]
        table = []
        for metric, label in SHAPE_METRICS:
            shape[metric] = {"early": ci([r[metric] for r in early]), "late": ci([r[metric] for r in late])}
            table.append([label, _ci_text(shape[metric]["early"], small_n), _ci_text(shape[metric]["late"], small_n)])
        md += [f"Early = first {_n(third, 'training episode')} (steps {early[0]['training_step_start']}-"
               f"{early[-1]['training_step_start']}); late = last {third} (steps {late[0]['training_step_start']}-"
               f"{late[-1]['training_step_start']}). Values are per-episode statistics of the whole-body "
               "second moments and bounding box.", ""]
        md += _table(["per-episode statistic", "early", "late"], table)
        md += ["", "A change here describes how much the body changed shape. It does not show that reshaping "
               "helped; that needs a restricted baseline with reshaping locked.", ""]
    else:
        md += ["Fewer than two ordered training episodes; no early/late comparison.", ""]

    md += ["## Failures and incomplete episodes", ""]
    modes: dict[tuple, int] = {}
    for r in rows:
        mode = "withheld (held-out)" if r["withheld"] else r["failure_mode"] or "completed"
        mode += " (scripted fixture, not learned)" if r["fixture"] else ""
        modes[(r["kind"], mode)] = modes.get((r["kind"], mode), 0) + 1
    md += _table(["kind", "outcome / failure mode", "episodes"], [[k, m, n] for (k, m), n in sorted(modes.items())])
    md += [""]
    if any(m.startswith("unknown") for _, m in modes):
        md += ["`unknown` means the recording does not establish success or failure (no `info.success` and no "
               "`goal_x` in the course).", ""]
    flagged = [r for r in rows if r["status"] in ("interrupted", "failed", "open") or r["problems"]]
    if flagged:
        md += [f"{_n(len(flagged), 'episode')} {'was' if len(flagged) == 1 else 'were'} interrupted, failed, "
               "still open or had disclosed problems (all listed; see episodes.csv for every field):", ""]
        md += _table(["episode", "kind", "status", "recovered", "steps recorded / reported", "problems"],
                     [[r["episode_id"], r["kind"], r["status"], r["recovered"],
                       "withheld" if r["withheld"] else f"{r['steps_recorded']} / {_num(r['steps'])}",
                       r["problems"] or "-"] for r in flagged])
        md += [""]

    md += ["## Not shown or untested", ""]
    gaps = []
    if not any(r["arm"] not in (None, "full") for r in evaluation):
        gaps.append("No restricted (no-reshape) baseline was evaluated, so this report cannot say whether "
                    "reshaping adds value.")
    if not any(h for h, _ in heldout.values()):
        gaps.append("No held-out course results are present; generalisation beyond the showcase course is untested.")
    for w in withheld:
        gaps.append(f"Held-out episode `{w['episode_id']}` (course `{w['course_id']}`) withheld: {w['reason']}.")
    if policy_warnings:
        gaps.append(f"{_n(len(policy_warnings), 'evaluation episode')} could not be confirmed as played by a "
                    "single frozen policy version (see the policy-lineage caution).")
    lineage_gone = sum(1 for r in rows if r.get("lineage_discarded"))
    if lineage_gone:
        gaps.append(f"{_n(lineage_gone, 'training episode')} from a discarded learner branch "
                    f"{'is' if lineage_gone == 1 else 'are'} excluded from every policy-lineage claim.")
    for c in conflicts:
        gaps.append(f"Course `{c['course_id']}` was not compared: its episodes use different course configs.")
    if plan_data:
        plan_heldout = {str(c["id"]) for c in plan_data.get("courses", []) if c.get("final_only")}
        leaked = sorted({r["course_id"] for r in rows if r["kind"] == "training" and r["course_id"] in plan_heldout})
        if leaked:
            gaps.append(f"Training episodes ran on held-out course ids {leaked}; those courses are not held out.")
    if inventory["fixture"]:
        gaps.append(f"{_n(inventory['fixture'], 'fixture episode')} {'is' if inventory['fixture'] == 1 else 'are'} "
                    "synthetic and excluded from curves and comparisons.")
    if inventory["incomplete_trace"]:
        gaps.append(f"{_n(inventory['incomplete_trace'], 'episode')} "
                    f"{'has' if inventory['incomplete_trace'] == 1 else 'have'} incomplete or unverified traces; "
                    "their progress values are left empty rather than estimated.")
    gaps += ["Video was not inspected by this analysis; every number comes from the recorded trace and summaries.",
             "Bootstrap intervals assume episodes are independent; episodes sharing a seed or a worker may not be."]
    md += [f"- {g}" for g in gaps] + [""]

    files["report"].write_text("\n".join(md))
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with files["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    data = {"generated_at": generated, "run_dirs": [str(d) for d in run_dirs], "inventory": inventory,
            "ledger_problems": problems, "comparison": comparison, "course_config_conflicts": conflicts,
            "heldout_courses": {c: {"heldout": h, "ambiguity": a} for c, (h, a) in heldout.items()},
            "reshaping_early_late": shape, "withheld_heldout": withheld, "runtime_identity": identities,
            "evaluation_policy_warnings": policy_warnings,
            "lineage_discarded": [r["episode_id"] for r in rows if r.get("lineage_discarded")],
            "curve": {k: v for k, v in curve.items() if k != "points"}}
    files["json"].write_text(json.dumps(data, indent=2, default=str) + "\n")
    return {"files": {**files, **plots}, **data, "rows": rows, "curve_points": curve["points"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build the learning-progress report from recorded runs.")
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan", help="evaluation plan JSON (required to show held-out results)")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = build_report(args.run_dirs, args.out, plan=args.plan, window=args.window, n_boot=args.bootstrap,
                          seed=args.seed)
    print(f"{result['files']['report']}: {result['inventory']['total']} episodes")


if __name__ == "__main__":
    main()
