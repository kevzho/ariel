"""Check episode coverage and artifact integrity for a recorded run.

    python -m ariel.recording.verify runs/<run_id> [--decode] [--json out.json]

Checks ledger chains, contiguous episode indices, one finish per start, artifact
hashes and trace coverage. Recovered episodes may stop early but must disclose
missing steps. --decode also checks video frame counts. Exits with 1 on failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .ledger import RecordingError, read_ledger, sha256_file


SUPPORTED_SCHEMA = 2


def _decoded_frames(path: Path) -> int:
    import imageio_ffmpeg

    return int(imageio_ffmpeg.count_frames_and_secs(str(path))[0])


def verify_episode(directory: Path, finish: dict, decode: bool) -> list[str]:
    problems = []
    path = directory / "episode.json"
    if not path.exists():
        return ["episode.json missing"]
    if finish.get("episode_json_sha256") and sha256_file(path) != finish["episode_json_sha256"]:
        problems.append("episode.json does not match its ledger hash")
    summary = json.loads(path.read_text())
    if summary.get("schema_version", 0) > SUPPORTED_SCHEMA:
        return [f"schema_version {summary.get('schema_version')} is newer than this verifier ({SUPPORTED_SCHEMA})"]
    if finish.get("recovered") and not finish.get("episode_json_sha256"):
        problems.append("recovered finish row lacks episode_json_sha256 (written by an older recorder)")
    if summary.get("status") != finish.get("status"):
        problems.append("status differs between episode.json and ledger")
    expected = 1
    for chunk in summary.get("trace_chunks", []):
        file = directory / chunk["path"]
        if not file.exists():
            problems.append(f"{chunk['path']} missing")
        elif sha256_file(file) != chunk["sha256"]:
            problems.append(f"{chunk['path']} hash mismatch")
        if chunk["first_step"] != expected:
            problems.append(f"trace gap before step {chunk['first_step']} (expected {expected})")
        expected = chunk["first_step"] + chunk["steps"]
    recorded = expected - 1
    salvaged = summary.get("recovered") and summary.get("steps") is None
    if salvaged:
        if recorded != summary.get("steps_recorded"):
            problems.append(f"trace covers {recorded} steps, summary says steps_recorded={summary.get('steps_recorded')}")
    elif recorded != summary.get("steps"):
        problems.append(f"trace covers {recorded} steps, summary says {summary.get('steps')}")
    if summary.get("recovered") and not summary.get("missing"):
        problems.append("recovered episode does not disclose what was lost")
    meta = directory / "trace-meta.npz"
    disclosed = any("trace-meta" in note for note in summary.get("missing", []))
    if recorded and not meta.exists() and not disclosed:
        problems.append("trace-meta.npz missing")
    if meta.exists() and summary.get("trace_meta_sha256") and sha256_file(meta) != summary["trace_meta_sha256"]:
        problems.append("trace-meta.npz hash mismatch")
    frames = 0
    for segment in summary.get("video_segments", []):
        file = directory / segment["path"]
        if not file.exists():
            problems.append(f"{segment['path']} missing")
            continue
        if sha256_file(file) != segment["sha256"]:
            problems.append(f"{segment['path']} hash mismatch")
        if decode:
            count = _decoded_frames(file)
            frames += count
            if segment.get("frames") is not None and count != segment["frames"]:
                problems.append(f"{segment['path']} decodes {count} frames, recorded {segment['frames']}")
    every = summary.get("video_every") or 0
    if every and not summary.get("video_segments") and recorded >= every:
        problems.append("no video for an episode with steps")
    if decode and summary.get("video_frames") is not None and frames != summary["video_frames"]:
        problems.append(f"video decodes {frames} frames, summary says {summary['video_frames']}")
    return problems


def verify_run(run_dir: Path, decode: bool = False) -> dict:
    run_dir = Path(run_dir)
    report = {"run_dir": str(run_dir), "workers": {}, "ok": True}
    if not (run_dir / "run.json").exists():
        return {**report, "ok": False, "error": "run.json missing"}
    for worker_dir in sorted((run_dir / "workers").glob("*")):
        result = {"episodes": 0, "by_status": {}, "failures": {}, "checkpoints": 0, "problems": []}
        try:
            rows, _ = read_ledger(worker_dir / "episodes.jsonl")
            checkpoints, _ = read_ledger(worker_dir / "checkpoints.jsonl")
        except RecordingError as exc:
            result["problems"].append(str(exc))
            report["workers"][worker_dir.name] = result
            report["ok"] = False
            continue
        started = [r for r in rows if r["event"] == "episode_started"]
        finishes: dict[str, list[dict]] = {}
        for r in rows:
            if r["event"] == "episode_finished":
                finishes.setdefault(r["episode_id"], []).append(r)
        indices = [r["episode_index"] for r in started]
        expected_indices = list(range(max(indices) + 1)) if indices else []
        gaps = sorted(set(expected_indices) - set(indices))
        reserved = {int(p.name.rsplit("-e", 1)[1][:6]) for p in (worker_dir / "episodes").glob("*-e[0-9]*")
                    if p.name.rsplit("-e", 1)[1][:6].isdigit()}
        unexplained = [i for i in gaps if i not in reserved]
        if indices != sorted(indices) or len(set(indices)) != len(indices) or unexplained:
            result["problems"].append(f"episode indices not contiguous: missing {unexplained[:10]}")
        if gaps and not unexplained:
            result["orphan_indices"] = gaps          # directories from a crash with no ledger row
        for r in started:
            episode_id = r["episode_id"]
            done = finishes.get(episode_id, [])
            if len(done) != 1:
                result["failures"][episode_id] = [f"{len(done)} finish rows (expected 1; 0 = still open)"]
                continue
            status = done[0]["status"]
            result["by_status"][status] = result["by_status"].get(status, 0) + 1
            problems = verify_episode(worker_dir / "episodes" / episode_id, done[0], decode)
            if problems:
                result["failures"][episode_id] = problems
        result["episodes"] = len(started)
        for c in checkpoints:
            path = Path(c["path"])
            if not path.exists():
                result["problems"].append(f"checkpoint {path} missing (moved? verify from the training host)")
            elif sha256_file(path) != c["sha256"]:
                result["problems"].append(f"checkpoint {path} hash mismatch")
        result["checkpoints"] = len(checkpoints)
        if result["problems"] or result["failures"]:
            report["ok"] = False
        report["workers"][worker_dir.name] = result
    if not report["workers"]:
        report["ok"] = False
        report["error"] = "no workers recorded"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--decode", action="store_true", help="decode every video segment (slower)")
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    report = verify_run(args.run_dir, decode=args.decode)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
