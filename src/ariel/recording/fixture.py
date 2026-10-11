"""Scripted recordings for viewer development; no physics or learned behavior.

An ellipse flattens under an overhang and recovers. Later episodes travel
further by construction. All episodes use `kind="fixture"` and `fixture: true`;
normal endings include `info.scripted: true`. The first episode starts with
policy_version -1 to cover exploration. Early, mid and late placeholder
checkpoints contain no weights and support testing checkpoint selection.

    python -m ariel.recording.fixture OUT_DIR [--export VIEWER_DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .recorder import RunRecorder

LABEL = "scripted fixture - not learned behavior"
GROUND_Y = 0.1
START_X = 0.25
GOAL_X = 2.0
COURSE = {
    "id": "fixture-lunar-overhang",
    "scripted": True,
    "ground_y": GROUND_Y,
    "goal_x": GOAL_X,
    "gravity": 1.62,
    "obstacles": [[0.55, 0.10, 0.65, 0.15],      # rock
                  [1.00, 0.19, 1.30, 0.60],      # low overhang: 0.09 gap above the ground
                  [1.60, 0.10, 1.75, 0.20]],     # ledge
}
SEMI_A, SEMI_B = 0.07, 0.05                       # resting ellipse semi-axes
FLAT_B = 0.035                                    # semi-height needed under the overhang
EXPLORATION_FRACTION = 0.25                       # share of episode 0 recorded as policy_version -1
CHECKPOINT_LABELS = ("early", "mid", "late")


class FixtureInterruption(RuntimeError):
    """Scripted interruption used to test partial recordings."""


def _unit_disk(count: int, rng: np.random.Generator) -> np.ndarray:
    radius = np.sqrt(rng.random(count))
    angle = rng.random(count) * 2 * np.pi
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def _ramp(x: float, x0: float, x1: float, width: float) -> float:
    """1 inside [x0, x1], falling linearly to 0 over `width` outside."""
    return float(np.clip(min(x - (x0 - width), (x1 + width) - x) / width, 0.0, 1.0))


def _floor(x: float, a: float) -> float:
    floor = GROUND_Y
    for x0, y0, x1, y1 in COURSE["obstacles"]:
        if y0 <= GROUND_Y + 1e-9:
            floor = max(floor, GROUND_Y + (y1 - GROUND_Y) * _ramp(x, x0, x1, a))
    return floor


def _semi_axes(x: float, t: int) -> tuple[float, float]:
    x0, _, x1, _ = COURSE["obstacles"][1]
    squash = _ramp(x, x0, x1, 0.12)
    b = SEMI_B + (FLAT_B - SEMI_B) * squash
    a = SEMI_A * SEMI_B / b                       # keep area constant
    wobble = 0.04 * np.sin(0.35 * t)
    return a * (1 + wobble), b * (1 - wobble)


def _body(disk: np.ndarray, x: float, t: int, velocity: float) -> np.ndarray:
    a, b = _semi_axes(x, t)
    y = _floor(x, a) + b
    shear = np.clip(velocity * 4, -0.3, 0.3)
    local = disk * [a, b]
    local[:, 0] += shear * local[:, 1]
    return local + [x, y]


def _plan(index: int, episodes: int, steps: int) -> dict:
    """Scripted reach and ending for one episode (later episodes go further)."""
    interrupted = max(1, episodes // 3)
    stuck = interrupted + 1
    fraction = (index + 1) / episodes
    reach = START_X + (GOAL_X + 0.05 - START_X) * fraction ** 1.3
    plan = {"reach": reach, "stop_at": steps, "ending": "truncated"}
    if index == interrupted:
        plan.update(stop_at=int(steps * 0.4), ending="interrupted")
    elif index == stuck:
        plan.update(reach=COURSE["obstacles"][1][0] - 0.02, stop_at=int(steps * 0.6), ending="terminated")
    elif reach >= GOAL_X:
        plan["ending"] = "goal"
    return plan


def make_fixture(run_dir, episodes: int = 6, steps: int = 240, seed: int = 0, points: int = 3000,
                 **recorder_kwargs) -> Path:
    """Write a scripted fixture run and return its directory."""
    if episodes < 3:
        raise ValueError("episodes must be at least 3 (one interrupted, one terminated early)")
    run_dir = Path(run_dir)
    if (run_dir / "run.json").exists():
        raise FileExistsError(f"{run_dir} already holds a run")
    rng = np.random.default_rng(seed)
    disk = _unit_disk(points, rng)
    meta = {"label": LABEL, "description": LABEL, "generator": "ariel.recording.fixture",
            "seed": seed, "episodes": episodes, "steps": steps, "body": "synthetic ellipse blob, scripted motion"}
    run = RunRecorder(run_dir, run_id=f"fixture-seed{seed}", fixture=True, meta=meta, **recorder_kwargs)
    total_steps = 0
    version = 0
    # Place the three checkpoint placeholders at distinct episode boundaries.
    saves = {round(k * (episodes - 1) / (len(CHECKPOINT_LABELS) - 1)): label
             for k, label in enumerate(CHECKPOINT_LABELS)}
    with run:
        for index in range(episodes):
            plan = _plan(index, episodes, steps)
            jitter = 0.004 * (1 - index / episodes)
            x = START_X
            body = _body(disk, x, 0, 0.0)
            try:
                with run.episode(COURSE, seed=seed * 1000 + index, kind="fixture", training_step=total_steps,
                                 initial_points=body) as ep:
                    for t in range(1, plan["stop_at"] + 1):
                        if plan["ending"] == "interrupted" and t == plan["stop_at"]:
                            raise FixtureInterruption(f"scripted interruption at step {t}")
                        progress = min(t / (0.85 * steps), 1.0)
                        target = START_X + (plan["reach"] - START_X) * (3 * progress ** 2 - 2 * progress ** 3)
                        step_x = target + jitter * np.sin(1.7 * t + index)
                        velocity = step_x - x
                        x = step_x
                        body = _body(disk, x, t, velocity)
                        reached = x >= GOAL_X
                        terminated = reached or (plan["ending"] == "terminated" and t == plan["stop_at"])
                        truncated = not terminated and t == steps
                        action = [np.clip(velocity * 50, -1, 1), 1 - _semi_axes(x, t)[1] / SEMI_B]
                        reward = 10 * velocity - 0.001 + (1.0 if reached else 0.0)
                        total_steps += 1
                        explore = index == 0 and t <= EXPLORATION_FRACTION * steps
                        if not explore:
                            version = index * 50 + t // 10
                        ep.step(body, action, reward, policy_version=-1 if explore else version,
                                terminated=terminated, truncated=truncated, training_step=total_steps)
                        if terminated or truncated:
                            break
                    success = bool(x >= GOAL_X)
                    failure = None if success else "stuck" if terminated else "timeout"
                    ep.end(info={"success": success, "failure_mode": failure, "scripted": True, "label": LABEL})
            except FixtureInterruption:
                pass
            if index in saves:
                label = saves[index]
                checkpoint = run_dir / "checkpoints" / f"fixture-{label}.ckpt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(f"{LABEL}: {label} placeholder at training_step {total_steps}, "
                                      "contains no model weights\n")
                run.checkpoint(checkpoint, policy_version=version, training_step=total_steps,
                               note=f"{label}: {LABEL}: dummy checkpoint")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Write a {LABEL} run.")
    parser.add_argument("out_dir")
    parser.add_argument("--export", metavar="VIEWER_DIR", help="also export the run for the web viewer")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_dir = make_fixture(args.out_dir, episodes=args.episodes, steps=args.steps, seed=args.seed)
    print(f"fixture run ({LABEL}): {run_dir}")
    if args.export:
        from .export import export_viewer
        index = export_viewer(run_dir, args.export)
        print(json.dumps({"viewer_dir": args.export, "episodes": len(index["episodes"])}))


if __name__ == "__main__":
    main()
