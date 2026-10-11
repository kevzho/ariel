"""2x2 panel for the recorded demo: candidate B's policy on four re-ordered courses (ffmpeg only).

    uv run modal run infra/scenario_app.py::scenarios      # samples the runs -> runs/scenarios-<stamp>/
    uv run python infra/demo_scenarios.py                  # labelled and clean panels -> runs/demo-clips/

Each cell is one checkpoint evaluation (plain simulator render). A run that ended in success plays to its
end (the stand); a run that reached the astronaut and fell afterwards is cut at ``cut_frame``, the last
frame of its stand. Counts per order are written to runs/demo-clips/scenarios.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/scenarios-20261011-081302")
OUT = Path("runs/demo-clips")
FONT = "runs/demo-clips/work/font.ttf"
# (run name, frame to cut at or None for the whole episode), in panel order: top row, then bottom row
CELLS = [("DCFS_s226", None), ("CFDS_s205", None), ("DFCS_s239", None), ("DFSC_s221", None)]
HOLD = 1.0


def text(s: str, x: str, y: str, size: int, enable: str = "", color: str = "white") -> str:
    e = f":enable='{enable}'" if enable else ""
    return (f"drawtext=fontfile={FONT}:text='{s}':x={x}:y={y}:fontsize={size}:fontcolor={color}"
            f":box=1:boxcolor=black@0.55:boxborderw={max(4, size // 5)}{e}")


def main() -> None:
    (OUT / "work").mkdir(parents=True, exist_ok=True)
    shutil.copy("C:/Windows/Fonts/arialbd.ttf", FONT)
    meta = [json.loads((SRC / f"{n}.json").read_text()) for n, _ in CELLS]
    ends = [(cut if cut is not None else len(m["track"]) - 1) / m["fps"] for (_, cut), m in zip(CELLS, meta)]
    total = max(ends) + HOLD
    for labels in (True, False):
        parts, ins = [], []
        for i, ((name, cut), m, end) in enumerate(zip(CELLS, meta, ends)):
            ins += ["-i", str(SRC / f"{name}.mp4")]
            c = (f"[{i}:v]trim=end={end:.3f},setpts=PTS-STARTPTS,fps=30,scale=960:540:flags=lanczos,"
                 f"tpad=stop_mode=clone:stop_duration=30,trim=duration={total:.3f},setpts=PTS-STARTPTS")
            if labels:
                c += "," + text(", ".join(m["sections"]), "18", "16", 28)
                done = "stopped at the astronaut" if m["reason"] == "success" else "reached the astronaut"
                c += "," + text(done, "18", "62", 24, f"gte(t,{end:.3f})", "0x7be08a")
            parts.append(c + f",drawbox=x=0:y=0:w=iw:h=ih:color=black:t=3[c{i}]")
        parts.append("[c0][c1][c2][c3]xstack=inputs=4:layout=0_0|960_0|0_540|960_540[v]")
        out = OUT / f"scenarios_2x2_{'labelled' if labels else 'clean'}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", ";".join(parts), "-map", "[v]",
                        "-an", "-r", "30", "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", str(out)], check=True)
        print(out, f"{total:.1f} s")
    rows = json.loads((SRC / "scenarios.json").read_text())
    per_order = {}
    for o in sorted({r["order"] for r in rows}):
        g = [r for r in rows if r["order"] == o]
        per_order[o] = {"sections": g[0]["sections"], "episodes": len(g),
                        "stopped_and_stood": sum(r["reason"] == "success" for r in g),
                        "passed_all_four_and_reached_the_astronaut": sum(r["usable"] for r in g),
                        "passed": {k: sum(bool(r["passed"].get(k)) for r in g) for k in ("blocks", "low", "fissure", "scarp")}}
    (OUT / "scenarios.json").write_text(json.dumps({
        "note": "Checkpoint evaluations of candidate B's policy on the showcase course with its four sections placed "
                "in other orders (same obstacle sizes; the policy trained on crates, deck, fissure, slope only). "
                "Each cell is the best of the sampled seeds for its order, not a typical run; see per_order.",
        "policy": {k: meta[0][k] for k in ("run_id", "checkpoint", "policy_version", "level", "gravity")},
        "cells": [{"name": n, "sections": m["sections"], "seed": m["seed"], "reason": m["reason"],
                   "cut_frame": cut, "shown_seconds": round(e, 2), "episode_seconds": m["seconds"],
                   "stand_steps_max": m["stand_steps_max"], "layout": m["layout"]}
                  for (n, cut), m, e in zip(CELLS, meta, ends)],
        "per_order": per_order, "episodes_total": len(rows), "source": str(SRC)}, indent=1))
    print(json.dumps(per_order, indent=1))


if __name__ == "__main__":
    main()
