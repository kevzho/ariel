"""Progress montage for the recorded demo from clips that were rendered with the lunar scene (ffmpeg only).

    uv run python infra/demo_montage_rendered.py      # labelled and clean copies -> runs/demo-clips/

The first three shots are the first controller (own body, trained from scratch, Oct 10 evening; MuJoCo
renders with the visual library). The last two are the submitted approach (learned layer over the frozen
MoCapAct prior; Blender renders). They are two different controllers, so this shows the project's
progress, not one training run. All shots play at real speed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

R = Path("runs")
OUT = R / "demo-clips"
FONT = "runs/demo-clips/work/font.ttf"
XF = 0.25
# (file, start s, end s, headline, second line, origin)
SHOTS = [
    ("audit-smoke-20261010-185728/mujoco_cinematic.mp4", 0.2, 3.7, "First controller, trained from scratch",
     "first minutes of training", "training_recording smoke-20261010-185728-w002-e0000017, policy v56-103"),
    ("audit-reshape-20261010-234522/mujoco_cinematic.mp4", 9.0, 12.5, "First controller, about 13M steps",
     "scrambles along the ground", "training_recording reshape-20261010-234522-w003-e0000303, policy v1224-1271"),
    ("audit-reshape-20261010-234522/mujoco_cinematic.mp4", 46.0, 49.5, "First controller, about 13M steps",
     "14 m in a minute, stops at the fissure", "same episode"),
    ("final/candidateA_side.mp4", 3.5, 7.5, "Rebuilt on a pretrained motion model",
     "11.6M steps, finishes the course", "checkpoint_evaluation reshape-s1-20261011-030933 v1003 seed 108"),
    ("final/candidateB_side.mp4", 5.0, 9.0, "30M steps",
     "sprints and reshapes", "checkpoint_evaluation athletic-s0-20261011-031906 v1300 seed 153"),
    ("final/candidateB_side.mp4", 15.4, 18.39, "30M steps",
     "stops at the astronaut", "same episode, cut at the end of its 1.3 s stand"),
]


def text(s: str, y: str, size: int, color: str = "white") -> str:
    return (f"drawtext=fontfile={FONT}:text='{s}':x=60:y={y}:fontsize={size}:fontcolor={color}"
            f":box=1:boxcolor=black@0.55:boxborderw={size // 5}")


def build(labels: bool) -> dict:
    ins, parts, t = [], [], 0.0
    for i, (f, a, b, head, sub, _) in enumerate(SHOTS):
        ins += ["-i", str(R / f)]
        c = (f"[{i}:v]trim={a}:{b},setpts=PTS-STARTPTS,fps=30,scale=1920:1080:flags=lanczos,format=yuv420p,settb=AVTB")
        if labels:
            c += "," + text(head, "50", 50) + "," + text(sub, "124", 38, "0xffd166")
        parts.append(c + f"[s{i}]")
    cur = "[s0]"
    t = SHOTS[0][2] - SHOTS[0][1]
    for i in range(1, len(SHOTS)):
        parts.append(f"{cur}[s{i}]xfade=transition=fade:duration={XF}:offset={t - XF:.3f}[x{i}]")
        cur = f"[x{i}]"
        t += SHOTS[i][2] - SHOTS[i][1] - XF
    out = OUT / f"montage_rendered_{'labelled' if labels else 'clean'}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", ";".join(parts), "-map", cur, "-an",
                    "-r", "30", "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", str(out)], check=True)
    return {"file": str(out), "seconds": round(t, 2)}


def main() -> None:
    (OUT / "work").mkdir(parents=True, exist_ok=True)
    shutil.copy("C:/Windows/Fonts/arialbd.ttf", FONT)
    outs = [build(True), build(False)]
    (OUT / "montage_rendered.json").write_text(json.dumps({
        "outputs": outs,
        "note": "Shots 1-3 are the first controller (own body, trained from scratch, Oct 10; MuJoCo renders); shots "
                "4-6 are the submitted approach (learned layer over the frozen MoCapAct prior; Blender renders). Two "
                "different controllers: the project's progress, not one training run. Real speed throughout.",
        "shots": [{"file": f, "from_s": a, "to_s": b, "label": f"{h} / {s}", "origin": o} for f, a, b, h, s, o in SHOTS]},
        indent=1))
    print(json.dumps(outs))


if __name__ == "__main__":
    main()
