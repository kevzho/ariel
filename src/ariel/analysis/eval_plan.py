"""Build frozen-checkpoint evaluation jobs for Guhan to run on Modal.

Each job specifies a checkpoint, course and seed, and records as `evaluation`.
Held-out courses are `final_only`. Checkpoints selected after
`heldout_first_viewed_at` cannot be reported on those courses; carry that
timestamp forward between plans.

    python -m ariel.analysis.eval_plan --checkpoints-from runs/r1 --showcase showcase \\
        --heldout layout-b layout-c --seeds 0 1 2 --episodes-per-seed 5 --out eval-plan.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from ariel.recording import read_ledger
from ariel.recording.ledger import atomic_write_json

SCHEMA = "ariel-eval-plan/1"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class HeldoutGuardError(ValueError):
    """Held-out evaluation or reporting would let held-out results influence checkpoint choice."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _slug(text: str) -> str:
    return _SAFE.sub("-", str(text)).strip("-") or "x"


def _run_id(*parts) -> str:
    """A recorder-safe run id (<= 80 chars of [A-Za-z0-9._-]); long ids keep a hash suffix."""
    text = "-".join(_slug(p) for p in parts if p not in (None, ""))
    if len(text) <= 80:
        return text
    return text[:69] + "-" + hashlib.sha256(text.encode()).hexdigest()[:10]


def _course(value, role: str) -> dict:
    course = {"id": value} if isinstance(value, str) else dict(value)
    if not course.get("id"):
        raise ValueError("each course needs an 'id' (also passed in the recorder's course dict)")
    return {"id": str(course["id"]), "role": role, "final_only": role == "heldout", "config": course}


def _checkpoint(value: dict) -> dict:
    missing = [k for k in ("path", "sha256", "policy_version", "selected_at") if value.get(k) is None]
    if missing:
        raise ValueError(f"checkpoint {value.get('path')} lacks {missing}")
    if not _SHA.match(str(value["sha256"])):
        raise ValueError(f"checkpoint {value['path']}: sha256 must be 64 lowercase hex characters")
    _time(value["selected_at"])
    return {"label": value.get("label") or f"pv{int(value['policy_version'])}", "path": str(value["path"]),
            "sha256": value["sha256"], "policy_version": int(value["policy_version"]),
            "training_step": value.get("training_step"), "source_run": value.get("source_run"),
            "selected_at": value["selected_at"],
            "selection_basis": value.get("selection_basis", "not stated")}


def select_checkpoints(run_dir, labels=("early", "mid", "late"), *, selected_at: str | None = None) -> list[dict]:
    """Pick checkpoints spread evenly by training_step from every worker's checkpoints.jsonl.

    The basis is the training schedule only, so no evaluation result enters the choice.
    """
    run_dir = Path(run_dir)
    rows = []
    for ledger in sorted(run_dir.glob("workers/*/checkpoints.jsonl")):
        rows += [r for r in read_ledger(ledger)[0] if r.get("event") == "checkpoint"]
    rows.sort(key=lambda r: (r["training_step"], r["policy_version"]))
    if len(rows) < len(labels):
        raise ValueError(f"{run_dir} has {len(rows)} saved checkpoint{'s' if len(rows) != 1 else ''}; "
                         f"{len(labels)} requested ({' '.join(labels)}). Pass fewer labels with --labels "
                         "(e.g. --labels early late) or save more checkpoints")
    count = len(labels)
    picks = [rows[round(i * (len(rows) - 1) / max(count - 1, 1))] for i in range(count)]
    if len({p["sha256"] for p in picks}) < count:
        raise ValueError("selected checkpoints are not distinct (the same file was saved more than once); "
                         "pass fewer labels with --labels")
    when = selected_at or _now()
    return [{"label": label, "path": r["path"], "sha256": r["sha256"], "policy_version": r["policy_version"],
             "training_step": r["training_step"], "source_run": run_dir.name, "selected_at": when,
             "selection_basis": f"evenly spaced by training_step among {len(rows)} saved checkpoints"}
            for label, r in zip(labels, picks)]


def build_eval_plan(checkpoints: list[dict], showcase_course, heldout_courses, seeds, episodes_per_seed: int, *,
                    deterministic: bool = True, plan_id: str | None = None, baseline: dict | None = None,
                    heldout_viewed_at: str | None = None, runs_root: str = "runs",
                    created_at: str | None = None) -> dict:
    """Return the JSON-serialisable job spec.

    baseline: optional restricted arm, e.g. {"name": "no_reshape", "mechanism": "<Guhan's restriction>",
    "params": {...}, "checkpoints": [...]}. Without its own checkpoints the restriction is applied to
    the same checkpoints at evaluation time (tests reliance on reshaping, not its training value).
    heldout_viewed_at: when held-out results were first viewed (carry forward from earlier plans).
    """
    checkpoints = [_checkpoint(c) for c in checkpoints]
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    if len({c["label"] for c in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint labels must be unique")
    courses = [_course(showcase_course, "showcase")] + [_course(c, "heldout") for c in heldout_courses]
    if len({c["id"] for c in courses}) != len(courses):
        raise ValueError("course ids must be unique and the showcase course cannot also be held out")
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct non-negative integers")
    if not 1 <= int(episodes_per_seed) <= 1000:
        raise ValueError("episodes_per_seed must be between 1 and 1000")
    if heldout_viewed_at and len(courses) > 1:
        viewed = _time(heldout_viewed_at)
        late = [c["label"] for c in checkpoints if _time(c["selected_at"]) >= viewed]
        if late:
            raise HeldoutGuardError(
                f"checkpoints {late} were selected after held-out results were viewed at {heldout_viewed_at}; "
                "they may be evaluated on the showcase course only (drop the held-out courses)")
    arms = [{"name": "full", "restriction": None, "design": None}]
    if baseline:
        if not baseline.get("mechanism"):
            raise ValueError("baseline needs a 'mechanism' string defined with Guhan")
        own = [_checkpoint(c) for c in baseline.get("checkpoints") or []]
        arms.append({"name": _slug(baseline.get("name", "no_reshape")),
                     "restriction": {"mechanism": str(baseline["mechanism"]), "params": baseline.get("params", {})},
                     "design": "trained_with_restriction" if own else "restricted_at_evaluation",
                     "checkpoints": own or None})
    created_at = created_at or _now()
    plan_id = _slug(plan_id or "p" + created_at[:19].replace(":", "").replace("-", ""))
    jobs = []
    for arm in arms:
        for checkpoint in arm.get("checkpoints") or checkpoints:
            for course in courses:
                for seed in seeds:
                    run_id = _run_id("eval", plan_id, checkpoint["label"], course["id"], f"s{seed}",
                                     None if arm["name"] == "full" else arm["name"])
                    job = {"job_id": run_id, "run_id": run_id, "kind": "evaluation",
                           "deterministic": bool(deterministic), "arm": arm["name"],
                           "restriction": arm["restriction"], "checkpoint": checkpoint,
                           "course": {k: course[k] for k in ("id", "role", "final_only")}, "seed": seed,
                           "episode_seeds": [seed * 1000 + i for i in range(int(episodes_per_seed))]}
                    job["recorder"] = {
                        "run_dir": f"{runs_root}/{run_id}", "run_id": run_id, "worker_id": "w0",
                        "meta": {"eval_job": {**{k: job[k] for k in ("job_id", "arm", "restriction", "seed",
                                                                      "deterministic", "checkpoint", "course")},
                                              "plan_id": plan_id}},
                        "episode": {"kind": "evaluation", "course": course["config"], "training_step": None}}
                    jobs.append(job)
    if len({j["run_id"] for j in jobs}) != len(jobs):
        raise ValueError("run ids collide; shorten labels or course ids")
    return {
        "schema": SCHEMA, "plan_id": plan_id, "created_at": created_at, "kind": "evaluation",
        "deterministic": bool(deterministic), "seeds": seeds, "episodes_per_seed": int(episodes_per_seed),
        "courses": courses, "checkpoints": checkpoints, "arms": arms,
        "heldout_first_viewed_at": heldout_viewed_at,
        "rules": [
            "Record every evaluation episode with RunRecorder(run_dir, run_id=..., worker_id=..., meta=recorder.meta)"
            " and run.episode(course, seed=episode_seed, kind='evaluation').",
            "Load each checkpoint by path and refuse to run if its sha256 differs; pass its policy_version"
            " as policy_version for every step.",
            "Held-out (final_only) courses are never used to choose, tune or re-select checkpoints.",
            "Report every job's episodes, including failures and interruptions.",
        ],
        "jobs": jobs,
    }


def mark_heldout_viewed(plan_path, when: str | None = None) -> str:
    """Record the first time held-out results were viewed; an existing timestamp is never moved."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if not plan.get("heldout_first_viewed_at"):
        plan["heldout_first_viewed_at"] = when or _now()
        atomic_write_json(plan_path, plan)
    return plan["heldout_first_viewed_at"]


def heldout_reportable(plan: dict, checkpoint_sha256: str | None) -> tuple[bool, str]:
    """Whether held-out results for this checkpoint may be reported under the plan's guard."""
    matches = [c for c in plan.get("checkpoints", []) + [c for a in plan.get("arms", [])
                                                          for c in a.get("checkpoints") or []]
               if c["sha256"] == checkpoint_sha256]
    if not matches:
        return False, "checkpoint is not in the evaluation plan"
    viewed = plan.get("heldout_first_viewed_at")
    if viewed is None:
        return True, "held-out results not yet viewed; this report is the first viewing"
    if all(_time(c["selected_at"]) < _time(viewed) for c in matches):
        return True, f"selected before held-out results were first viewed ({viewed})"
    return False, f"selected after held-out results were first viewed ({viewed})"


def _load_course(value: str):
    path = Path(value)
    return json.loads(path.read_text()) if path.suffix == ".json" and path.exists() else value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoints-from", metavar="RUN_DIR", help="pick evenly spaced checkpoints from a run")
    source.add_argument("--checkpoints", metavar="JSON", help="file with a list of checkpoint dicts")
    parser.add_argument("--labels", nargs="+", default=["early", "mid", "late"])
    parser.add_argument("--showcase", required=True, help="course id or course JSON file (with 'id')")
    parser.add_argument("--heldout", nargs="*", default=[], help="held-out course ids or JSON files")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the policy mean")
    parser.add_argument("--baseline-mechanism", help="restriction mechanism string agreed with Guhan")
    parser.add_argument("--baseline-name", default="no_reshape")
    parser.add_argument("--baseline-params", default="{}", help="JSON object of restriction parameters")
    parser.add_argument("--prior-plan", help="earlier plan whose heldout_first_viewed_at is carried forward")
    parser.add_argument("--plan-id")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    checkpoints = (select_checkpoints(args.checkpoints_from, args.labels) if args.checkpoints_from
                   else json.loads(Path(args.checkpoints).read_text()))
    baseline = ({"name": args.baseline_name, "mechanism": args.baseline_mechanism,
                 "params": json.loads(args.baseline_params)} if args.baseline_mechanism else None)
    viewed = json.loads(Path(args.prior_plan).read_text()).get("heldout_first_viewed_at") if args.prior_plan else None
    plan = build_eval_plan(checkpoints, _load_course(args.showcase), [_load_course(c) for c in args.heldout],
                           args.seeds, args.episodes_per_seed, deterministic=not args.stochastic,
                           plan_id=args.plan_id, baseline=baseline, heldout_viewed_at=viewed,
                           runs_root=args.runs_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(args.out), plan)
    print(f"{args.out}: {len(plan['jobs'])} jobs, {len(plan['checkpoints'])} checkpoints, "
          f"{len(plan['courses'])} courses")


if __name__ == "__main__":
    main()
