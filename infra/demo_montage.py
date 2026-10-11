"""Montage of training progress for the recorded demo, cut from clips already in runs/ (ffmpeg only).

Two early attempts alone, then a 3x3 grid of attempts playing at once: each tile dims when its robot
falls, the last one standing (candidate B) expands and dissolves into the Blender render of the same run.

    uv run python infra/demo_montage.py            # labelled and clean copies -> runs/demo-clips/

Every tile is a real recording (origin per clip in runs/demo-clips/montage.json); the clips come from
four training runs whose course and reward changed over the night, so they are not one controlled series.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

R = Path("runs")
OUT = R / "demo-clips"
WORK = OUT / "work"
FONT = "runs/demo-clips/work/font.ttf"
P_S1 = "progression-reshape-s1-20261011-030933"
P_INS = "progression-ins-reshape-20261011-030244"
B_PREVIEW = "sample-20261011-071107-atK/s153/preview.mp4"
B_FILM = "final/candidateB_side.mp4"

# (file, label, metres reached, origin, run, training steps)
SOLOS = [
    ("eval-rsmoke-course-20261011-023659/policy_v000037_learned_spawn0_e0.mp4", "0.4M", 9.8,
     "checkpoint_evaluation", "rsmoke-course-20261011-023659 v37 (6-minute smoke run)", 426240),
    (f"{P_S1}/stage2_step8432933.mp4", "8.4M", 14.1, "training_recording", "reshape-s1-20261011-030933", 8432933),
]
TILES = [
    (f"{P_S1}/stage0_step783667.mp4", "0.8M", 10.3, "training_recording", "reshape-s1-20261011-030933", 783667),
    (f"{P_INS}/stage2_step1693793.mp4", "1.7M", 20.9, "training_recording", "ins-reshape-20261011-030244 (older course)", 1693793),
    (f"{P_INS}/stage3_step2788166.mp4", "2.8M", 18.7, "training_recording", "ins-reshape-20261011-030244 (older course)", 2788166),
    (f"{P_S1}/stage1_step5690982.mp4", "5.7M", 10.4, "training_recording", "reshape-s1-20261011-030933", 5690982),
    (B_PREVIEW, "30M", 42.1, "checkpoint_evaluation", "athletic-s0-20261011-031906 v1300 seed 153 (candidate B)", 29952000),
    ("eval-athletic-s0-20261011-031906/policy_v000250_learned_spawn0_e0.mp4", "5.8M", 12.9,
     "checkpoint_evaluation", "athletic-s0-20261011-031906 v250", 5760000),
    ("eval-reshape-s0-20261011-030959/policy_v000350_learned_spawn0_e0.mp4", "8.1M", 5.9,
     "checkpoint_evaluation", "reshape-s0-20261011-030959 v350", 8064000),
    (f"{P_S1}/stage3_step15633008.mp4", "15.6M", 10.5, "training_recording", "reshape-s1-20261011-030933", 15633008),
    (f"{P_S1}/stage4_step19480351.mp4", "19.5M", 13.1, "training_recording", "reshape-s1-20261011-030933", 19480351),
]
SOLO_LEN, SOLO_PLAY = 3.6, 3.2
HOLD, ZOOM = 0.7, 0.8            # grid: pause after the last fall, then the zoom into the survivor
FILM_CUTS = ((None, 1.5), (15.4, 18.39))   # candidate B film: from the end of the zoom, then the walk-up and stand
XF = 0.3


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def dur(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def text(s: str, x: str, y: str, size: int, enable: str = "", color: str = "white") -> str:
    e = f":enable='{enable}'" if enable else ""
    return (f"drawtext=fontfile={FONT}:text='{s}':x={x}:y={y}:fontsize={size}:fontcolor={color}"
            f":box=1:boxcolor=black@0.55:boxborderw={max(4, size // 5)}{e}")


def knock(t: float) -> str:
    """Dim the picture from time t on, with a short red flash."""
    return (f"eq=saturation=0.1:brightness=-0.22:enable='gte(t,{t:.3f})',"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.35:t=fill:enable='between(t,{t:.3f},{t + 0.2:.3f})'")


def enc(out: Path) -> list[str]:
    return ["-map", "[v]", "-an", "-r", "30", "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)]


def solo(clip, out: Path, labels: bool) -> None:
    f, lab, metres = clip[0], clip[1], clip[2]
    start = max(0.0, dur(R / f) - SOLO_PLAY)
    chain = (f"[0:v]trim=start={start:.3f},setpts=PTS-STARTPTS,fps=30,scale=1920:1080:flags=lanczos,"
             f"tpad=stop_mode=clone:stop_duration=5,trim=duration={SOLO_LEN},setpts=PTS-STARTPTS,{knock(SOLO_PLAY)}")
    if labels:
        chain += "," + text(f"{lab} training steps", "60", "50", 52)
        chain += "," + text(f"fell at {metres:.0f} m", "60", "130", 44, f"gte(t,{SOLO_PLAY})", "0xff6b5e")
    ff("-i", str(R / f), "-filter_complex", chain + "[v]", *enc(out))


def grid(out: Path, labels: bool) -> float:
    falls = [dur(R / t[0]) for t in TILES]
    last = max(d for i, d in enumerate(falls) if i != 4)
    total = last + HOLD + ZOOM
    parts, ins = [], []
    for i, t in enumerate(TILES):
        ins += ["-i", str(R / t[0])]
        c = (f"[{i}:v]fps=30,scale=640:360:flags=lanczos,tpad=stop_mode=clone:stop_duration=20,"
             f"trim=duration={total:.3f},setpts=PTS-STARTPTS")
        if i != 4:
            c += "," + knock(falls[i])
        if labels:
            c += "," + text(f"{t[1]} steps", "14", "12", 22)
            if i != 4:
                c += "," + text(f"fell at {t[2]:.0f} m", "14", "48", 20, f"gte(t,{falls[i]:.3f})", "0xff6b5e")
        c += f",drawbox=x=0:y=0:w=iw:h=ih:color=black:t=3[t{i}]"
        parts.append(c)
    n0 = round((total - ZOOM) * 30)
    nz = round(ZOOM * 30)
    k = f"min(1,max(0,(in-{n0})/{nz}))"
    parts.append("".join(f"[t{i}]" for i in range(9))
                 + "xstack=inputs=9:layout=0_0|640_0|1280_0|0_360|640_360|1280_360|0_720|640_720|1280_720[g]")
    parts.append(f"[g]zoompan=z='1+2*{k}*{k}*(3-2*{k})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                 f":d=1:s=1920x1080:fps=30[v]")
    ff(*ins, "-filter_complex", ";".join(parts), *enc(out))
    return total


def film(out: Path, t0: float, labels: bool) -> float:
    a0, a1 = t0, t0 + FILM_CUTS[0][1]
    b0, b1 = FILM_CUTS[1]
    g = (f"[0:v]trim={a0:.3f}:{a1:.3f},setpts=PTS-STARTPTS,fps=30,settb=AVTB[a];"
         f"[0:v]trim={b0:.3f}:{b1:.3f},setpts=PTS-STARTPTS,fps=30,settb=AVTB[b];"
         f"[a][b]xfade=transition=fade:duration={XF}:offset={a1 - a0 - XF:.3f}")
    if labels:
        g += "," + text("30M training steps", "60", "50", 52)
    ff("-i", str(R / B_FILM), "-filter_complex", g + "[v]", *enc(out))
    return (a1 - a0) + (b1 - b0) - XF


def build(labels: bool) -> dict:
    tag = "labelled" if labels else "clean"
    s = [WORK / f"{tag}_solo{i}.mp4" for i in range(2)]
    for clip, o in zip(SOLOS, s):
        solo(clip, o, labels)
    g, f = WORK / f"{tag}_grid.mp4", WORK / f"{tag}_film.mp4"
    g_len = grid(g, labels)
    f_len = film(f, g_len, labels)       # the grid starts at t = 0 of every run, so its end is the film's start time
    off = 2 * SOLO_LEN + g_len - XF
    out = OUT / f"montage_{tag}.mp4"
    ff("-i", str(s[0]), "-i", str(s[1]), "-i", str(g), "-i", str(f), "-filter_complex",
       f"[0:v][1:v][2:v]concat=n=3:v=1,fps=30,settb=AVTB[abc];[3:v]fps=30,settb=AVTB[d];"
       f"[abc][d]xfade=transition=fade:duration={XF}:offset={off:.3f}[v]", *enc(out))
    return {"file": str(out), "seconds": round(off + f_len, 2), "grid_seconds": round(g_len, 2)}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    shutil.copy("C:/Windows/Fonts/arialbd.ttf", FONT)
    outs = [build(True), build(False)]
    row = lambda c: {"file": c[0], "label_steps": c[1], "reached_m": c[2], "origin": c[3], "run": c[4], "training_steps": c[5]}  # noqa: E731
    (OUT / "montage.json").write_text(json.dumps({
        "outputs": outs,
        "note": "Plain simulator renders of real recordings from four training runs; the course and reward changed "
                "between and during these runs, so this is not one controlled series. Labels are each policy's own "
                "training steps and the distance its attempt reached. The last shot is the Blender render of the "
                "same candidate B run shown in the centre tile, cut from 15.4 s to its 1.3 s stand.",
        "solos": [row(c) for c in SOLOS], "grid_tiles_row_major": [row(c) for c in TILES],
        "final_shot": {"file": B_FILM, "origin": "checkpoint_evaluation, Blender render of recorded states"}}, indent=1))
    print(json.dumps(outs, indent=1))


if __name__ == "__main__":
    main()
