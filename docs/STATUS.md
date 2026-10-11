# Current state

Updated October 11, 2026, 07:50 EDT. All training is stopped. The video to submit is
**candidate B** (`runs/final/candidateB_side.mp4`). See
[SUBMISSION_NOTES.md](SUBMISSION_NOTES.md) for what may be claimed.

## What the system is

- **Body:** the CMU humanoid used by MoCapAct (56 position-controlled joints) with nine
  telescoping segments added: thighs, shins, upper arms, forearms and torso
  ([walker.py](../src/ariel/mocap/walker.py)). Left and right move together, so the policy
  commands five lengths. Segments change at up to 0.25 m/s while the robot keeps moving.
- **Pretrained part (not ours, never updated):** MoCapAct's velocity-control policy and
  low-level motor decoder (Wagener et al., NeurIPS 2022; MIT code, CDLA-2.0 models), trained on
  CMU motion-capture clips. It turns a speed and heading into human joint targets, walking or running.
- **Learned part (ours):** a PPO policy above the frozen prior
  ([env.py](../src/ariel/mocap/env.py), [train.py](../src/ariel/mocap/train.py)). Every control
  step (33 Hz) it outputs speed, a heading adjustment, a correction to the prior's 60-d motor
  intention, small joint corrections on legs and spine, and the five segment lengths. It sees its
  own joints and motion, a terrain height scan (13 points ahead by 5 across) and an overhead
  clearance scan.
- **Physics:** MuJoCo 2.2.2 through dm_control, **Earth gravity (9.81 m/s²)**. The lunar base is the
  visual setting only. The pretrained prior falls at lunar gravity
  (`mocapact-probe-20261011-014450`); we do not claim lunar-gravity locomotion.
- **Course** ([course.py](../src/ariel/world/course.py), a new random layout every training
  episode): landing pad, two cargo crates near the walking line, a deck hanging under the
  collapsed module, a fissure, a rocky constant-gradient slope, a flat approach, the astronaut.
- **Training aids:**
  - Shape teacher: a hand-written rule suggests a shape for the deck and the fissure through an
    imitation reward whose weight fell linearly to zero.
  - Steering and pace hint around the crates: same form, kept at 30% weight to the end.
  - Braking zone: over the last 8 m the commanded speed is capped at 0.2 m/s per metre left to
    the stop point; speed above the cap costs reward; standing still at the astronaut earns reward.
  - Curriculum: each obstacle's difficulty rises when most attempts pass it and falls when few do.
  - Evaluation uses none of the teachers; the braking cap is part of the environment.
- **Comparison:** the same setup with every segment locked at nominal length.

## The run in the video (candidate B)

`runs/final/candidateB_side.mp4` (1080p, 613 frames, 18.4 s; Blender render of recorded states).

- Policy: run `athletic-s0-20261011-031906`, checkpoint `policy_v001300.pt`, deterministic,
  evaluation seed 153, origin `checkpoint_evaluation`. Export: `runs/candidateB/`.
- Course in this run: crates 0.42 m off the centre line, deck 1.82 m, fissure 0.90 m, slope
  1.46 m rise over 5 m (16°), astronaut 10 m beyond the top of the slope.
- What happens: sprint off the pad (median 3.4 m/s), swerve past both crates, pass under the deck
  with thighs, torso and arms shortened, leap the fissure with shins lengthened, climb the slope,
  slow across the plateau, walk up to the astronaut and stand still for 1.3 s about 1.8 m from it.
- Shape use (length ratios): thigh 0.71-1.33, shin 0.83-1.22, upper arm 0.70-1.13,
  forearm 0.71-1.08, torso 0.73-1.42.
- **Trim:** the video ends at frame 612, the end of the stand. In the full episode (645 frames)
  the robot loses balance about half a second later and the episode ends as a fall.
  The untrimmed plain render is `runs/sample-20261011-071107-atK/s153/preview.mp4`.
- How typical: best of 240 sampled full-course runs across three policies at 07:13. Ten runs
  stood for 0.6-1.3 s at the astronaut; none stood for 2.4 s.

Candidate A (`runs/final/candidateA_*.mp4`, policy `reshape-s1` v1003, seed 108) is an earlier
complete run without a stop.

## Results

Checkpoint evaluations (deterministic policy, one attempt per layout):

| Test | Learned policy | Frozen prior alone | Source |
|---|---|---|---|
| 24 unseen layouts, crates 1.08 m off centre, 10° slope: reached the astronaut | 15 of 24 | 0 of 24 | `runs/heldout-*easy`, policy `reshape-s1` v1003 |
| 24 unseen layouts, crates 0.42 m off centre, 15° slope: reached the astronaut | 8 of 24 | 0 of 24 | `runs/heldout-*hard`, same policy |
| 28 m of cratered ground as in training (9 craters to 10 cm deep), 12 unseen layouts: crossed | 11 of 12 | 4 of 12 | `runs/roughtest-20261011-071109`, same policy |
| Craters twice as deep (to 21 cm), 9 craters | 9 of 12 | 5 of 12 | same |
| Twice as deep, 30 craters | 3 of 12 | 0 of 12 | same |
| Four and six times as deep, 30 craters | 0 of 12 | 0 of 12 | same |

In the first two rows "reached the astronaut" is the earlier finish line (no stop required).

Reshaping versus locked body, **training statistics** (share of the last 400 training attempts;
not checkpoint evaluations):

| Course | Measure | Reshaping | Locked | Runs |
|---|---|---|---|---|
| Older (1.82 m deck, 0.5 m fissure, low rise) | steps to 90% pass under the deck | 0.81M | 7.37M | `ins-reshape-20261011-030244`, `ins-locked-20261011-030244` |
| Older | steps to 90% pass over the fissure | 3.4M | 11.35M | same |
| Older | reached the astronaut at the end (15.9M / 15.6M steps) | 76% | 65% | same |
| Current, 1.70 m deck, 0.65-0.8 m fissure | pass under the deck at the end | 92% | 78% | `reshape-s0-20261011-030959` (38.7M steps), `locked-s0-20261011-030959` (36.3M) |
| Current | pass over the fissure at the end | 90% | 75% | same |
| Current | pass the slope (18°) at the end | 75% | 60% | same |

A deterministic evaluation of that pair at 05:42 (24 full-course episodes each, crates off to the
side, 12° slope) gave: past the deck 11 vs 5, past the fissure 6 vs 3, reached the astronaut 1 vs 0.

Stop and stand (training counts 1.2 s of standing as success): in its last 400 training attempts
the athletic run stood at the astronaut in 83 (21%), `reshape-s0` in 34, `locked-s0` in 31,
`reshape-s1` in 2.

Re-ordered courses (added 08:20; `runs/scenarios-20261011-081302`, 280 checkpoint evaluations, seeds
200-239 per order). Candidate B's policy (`athletic-s0` v1300, same obstacle sizes as candidate B) on the
showcase course with its four sections placed in other orders
([scenario_app.py](../infra/scenario_app.py)); it trained on crates, deck, fissure, slope only.
"Reached" means past all four sections and within 2 m of the stop point; "stood" is the training
success rule (1.2 s standing at the astronaut).

| Order | Reached | Stood |
|---|---|---|
| deck, crates, fissure, slope | 2 of 40 | 2 |
| crates, fissure, deck, slope | 3 of 40 | 1 |
| fissure, crates, deck, slope | 0 of 40 | 0 |
| fissure, deck, crates, slope | 8 of 40 | 0 |
| deck, fissure, crates, slope | 8 of 40 | 5 |
| crates, fissure, slope, deck | 3 of 40 | 0 |
| deck, fissure, slope, crates | 1 of 40 | 1 |
| All seven | 25 of 280 | 9 |

For scale: in the original order at the same difficulty the same policy reached the astronaut in 4 of 60
(`sample-20261011-071107-atK`). No order with the fissure after the slope was tried.

No learning involved (frozen prior, or scripted rules on top of it):

| What | Result | Run |
|---|---|---|
| Frozen prior, flat floor | walks; runs at up to 3.4 m/s | `mocapact-tolerance-20261011-021258`, `mocapact-runcheck-20261011-031526` |
| Frozen prior, one segment group changed | keeps walking with thighs 0.7x, shins 1.4x, arms 0.7-1.4x, torso 0.75-1.4x | `mocapact-tolerance-20261011-021258` |
| Frozen prior, both leg segments changed | falls below about 0.8x and above about 1.25x | same, `mocapact-tolerance-20261011-021618` |
| Frozen prior on raised ground | topples within 2 s: it is given absolute body height and was trained at z = 0 | `mocapact-variants-20261011-0610*` |
| Frozen prior with a zero speed command | does not stand still; shuffles, overshoots or falls (4 of 4) | `mocapact-variants-20261011-061340` |

## Training runs

| Run | Body | Steps | Wall time | Notes |
|---|---|---|---|---|
| `athletic-s0-20261011-031906` | reshaping, speed to 3.4 m/s, fissure to 1.1 m | 37.3M | 4.4 h, 64 cores | source of candidate B |
| `reshape-s1-20261011-030933` | reshaping | 24.9M | 4.1 h, 32 cores | source of candidate A and the held-out tests |
| `reshape-s0-20261011-030959` | reshaping, 1.70 m deck | 38.7M | 4.1 h, 64 cores | comparison pair |
| `locked-s0-20261011-030959` | locked, 1.70 m deck | 36.3M | 4.1 h, 64 cores | comparison pair |
| `ins-reshape-20261011-030244`, `ins-locked-20261011-030244` | reshaping / locked, older course | 15.9M / 15.6M | 2.2 h, 32 cores each | code at tag `insurance-env` |
| `shapegym-s0-20261011-030933` | reshaping, flat floor, wider shape ranges | 19M | 1.8 h | diverged; stopped, unused |

The four current-course runs were stopped and resumed from their latest checkpoint 12-15 times
while the course and reward changed (crates moved toward the walking line, slope made a constant
gradient and steeper, arms added to the deck shape, more runway before the crates, stop-and-stand
task, solar panels removed from the route, body-height fix beyond the slope). Each resume loses
the episodes in progress; the exact history is in each run's `train_log.jsonl` (`resumed` events)
and `config.json`. Final checkpoints, configs and logs are in [`models/`](../models/README.md).

Every training episode is recorded on the Modal volume `ariel-runs` under `<run>/episodes/`
(state every second control step, every action, reward term and policy version); for example
`reshape-s1` had 107,703 recorded episodes and 19.9M recorded steps at 06:50.
`runs/progression-reshape-s1-20261011-030933/` holds five clips, one typical attempt from each
stage of that run, re-rendered from the recordings.

## Commands (repository root, Modal profile `tejas-16842`)

```
uv run modal run infra/mocapact_app.py::env_check          # frozen prior and scripted rules on the course
uv run modal run infra/mocapact_app.py::reshape_smoke      # 6-minute end-to-end training check
uv run modal run --detach infra/mocapact_app.py::reshape_launch --spec configs/reshape_runs.json
uv run python infra/status.py RUN [RUN ...]                # latest training row of each run
uv run modal run infra/mocapact_app.py::reshape_eval --run-id RUN        # checkpoint evaluation with video
uv run modal run infra/mocapact_app.py::heldout_eval --run-id RUN        # unseen layouts, learned vs prior
uv run modal run infra/mocapact_app.py::rough_test --run-id RUN          # cratered ground
uv run modal run infra/mocapact_app.py::sample_runs --run-id RUN --levels JSON   # many runs, previews on the volume
uv run modal run infra/mocapact_app.py::fetch_export --export SAMPLE/sN  # download one for rendering
uv run modal run infra/mocapact_app.py::progression --run-id RUN         # replay training recordings
uv run modal run infra/modal_app.py::cmu_film --src DIR --camera side    # Blender video (also: track, flythrough)
uv run modal run infra/modal_app.py::cmu_stills --src DIR --xs 6,20      # Blender stills
```

## Cost

Estimated from Modal's price table, not from the billing page: about $70 of CPU training, about
$25-30 of GPU rendering and about $5 of evaluations, so roughly $100-105 in total before any
renders started by other sessions. Check the Modal usage page for the real figure.

## Not done

- A same-difficulty checkpoint evaluation of the final reshaping and locked policies
  (`reshape_compare`); the comparison above rests on training statistics.
- The orbit-camera render of candidate B.
- Blender renders of the demo clips. The clips themselves are in `runs/demo-clips/` in the plain
  simulator look: `montage_*.mp4` ([demo_montage.py](../infra/demo_montage.py), sources in
  `montage.json`) and `scenarios_2x2_*.mp4` ([demo_scenarios.py](../infra/demo_scenarios.py), counts in
  `scenarios.json`). Exports for Blender are on the volume under `scenarios-20261011-081302/<name>/`.
- The progression viewer (A5) is not connected to these recordings; the format differs from the
  2D recording contract on `main`.
- Merging this branch into `main`, whose docs still describe a 2D soft body.
