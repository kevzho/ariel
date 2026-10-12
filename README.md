# Ariel

**A rescue humanoid that learns, with reinforcement learning, to change the length of its own
limbs while it moves.**

Ariel crosses a damaged lunar base after a moonquake to reach a stranded astronaut. Along the way it
swerves past cargo crates, shortens its thighs, torso and arms to pass under a collapsed deck,
lengthens its shins to leap a fissure, and climbs a rocky slope. At the end it slows down and
stands next to the astronaut. The body changes shape during one continuous attempt, and the
physics simulator decides what each change actually achieves. Every training attempt is recorded
and can be replayed.

## How it works

```
 observations ──► learned PPO layer (ours) ──► speed, heading, motor corrections, 5 segment lengths
 (joints, motion,          │
  terrain + overhead       ▼
  clearance scans)   frozen MoCapAct prior ──► 56 joint targets ──► MuJoCo physics (telescoping CMU humanoid)
```

| Part | What it is |
|---|---|
| **Body** | The CMU humanoid from dm_control/MoCapAct (56 position-controlled joints), with **nine telescoping segments** added: thighs, shins, upper arms, forearms and torso ([walker.py](src/ariel/mocap/walker.py)). Left and right sides move together, so the policy sets five lengths. Limbs range from 0.70x to 1.40x their normal length and the torso from 0.75x to 1.45x (defaults in `walker.py`). Each segment changes at no more than 0.25 m/s, so reshaping takes visible time and can't be used as a catapult. |
| **Pretrained prior (not ours, frozen)** | MoCapAct's velocity-control policy and its low-level motor decoder (Wagener et al., NeurIPS 2022), trained on CMU motion capture. Given a speed and heading, it produces human-like walking or running. |
| **Learned layer (ours)** | A PPO policy running on top of the frozen prior at 33 Hz ([env.py](src/ariel/mocap/env.py), [train.py](src/ariel/mocap/train.py)). At each step it outputs a speed, a heading adjustment, a correction to the prior's 60-d motor intention, small joint corrections on the legs and spine, and the five segment lengths. Its inputs are its own joints and motion, a terrain height scan (13 points ahead by 5 across) and an overhead clearance scan. |
| **Physics** | MuJoCo 2.2.2 through dm_control, at **Earth gravity (9.81 m/s²)**. The Moon is the visual setting only, because the pretrained prior falls over at lunar gravity. |
| **Course** | [course.py](src/ariel/world/course.py) generates a new random layout every training episode: landing pad, two cargo crates near the walking line, a deck hanging under a collapsed module, a fissure, a rocky slope, a flat approach, then the astronaut. Every obstacle spans the whole corridor, and leaving the corridor ends the episode. |
| **Rendering** | MuJoCo previews of the recorded states, plus Blender (Cycles) cinematic renders ([src/ariel/art/](src/ariel/art/)). Both are rendered from the saved states without re-running the physics. |

### Reward and training aids

- **Progress reward.** Ariel is paid only when its pelvis moves past its previous farthest point,
  so lunging back and forth earns nothing:
  `r_progress = w · max(0, x_t − max_{k<t} x_k)`
- **Small costs** for energy, jerky actions, impacts and wobble. Shape itself is never rewarded,
  so the policy only reshapes when it helps.
- **Shape teacher.** A hand-written rule suggests a shape for the deck and the fissure. Its
  reward weight falls linearly to zero during training: `w_teacher(t) = max(0, 1 − t/T)`.
- **Steering and pace hint** around the crates. It works the same way, but its weight stays at
  30% until the end of training.
- **Braking zone.** Over the last 8 m, the commanded speed is capped in proportion to the
  distance left. Standing still at the astronaut is rewarded.
- **Curriculum.** An obstacle gets harder when most attempts pass it and easier when few do.
- **Evaluation** uses none of the teachers. The braking cap is part of the environment, so it
  stays on.
- **Baseline:** the same setup with every segment locked at normal length.

## Results

All numbers come from [docs/STATUS.md](docs/STATUS.md), which lists the run behind each one.

**Submission video:** run `athletic-s0-20261011-031906`, checkpoint `policy_v001300.pt`,
deterministic, evaluation seed 153. It is 18.4 s long, rendered in Blender from the recorded
states. It is the **best of 240 sampled runs**, not a typical one. The clip ends after a 1.3 s
stand; in the full simulation the robot loses balance about half a second later.

**Unseen layouts** (checkpoint evaluations, one attempt per layout):

| Test | Learned policy | Frozen prior alone |
|---|---|---|
| 24 layouts, easier setting: reached the astronaut | 15 / 24 | 0 / 24 |
| 24 layouts, harder setting: reached the astronaut | 8 / 24 | 0 / 24 |
| Cratered ground twice as deep as in training: crossed | 9 / 12 | 5 / 12 |
| Same four obstacles in seven new orders: reached the astronaut | 25 / 280 | — |

**Reshaping vs locked body** (training statistics, not checkpoint evaluations):

| Measure | Reshaping | Locked |
|---|---|---|
| Steps to reach a 90% pass rate under the deck (older course) | 0.81M | 7.37M |
| Steps to reach a 90% pass rate over the fissure (older course) | 3.4M | 11.35M |
| Passing under a 1.70 m deck, end of training | 92% | 78% |
| Passing over the fissure, end of training | 90% | 75% |

**Limits:**

- Physics runs at Earth gravity, not lunar gravity.
- The arms only shorten under the deck; they never push, grab or climb.
- The crates are fixed in place.
- Deep or dense craters still defeat the policy.
- Stopping is unreliable: the best policy stood at the astronaut in 21% of its last 400 training
  attempts.

[docs/SUBMISSION_NOTES.md](docs/SUBMISSION_NOTES.md) lists what can and cannot be claimed.

**Scale:** about 190M training steps across six kept runs, and every episode was recorded. The
video's policy trained for 37.3M steps, which took 4.4 h on one 64-core Modal container. Total
compute was roughly $100–105, estimated from Modal's price table.

## Every attempt is recorded

Every episode from every worker is recorded from its first step. This includes random
exploration, falls, and episodes cut short by shutdown. Each recording holds:

- the full physical state at every second control step
- every action and reward term
- the policy version behind each action
- the course seed and parameters

If a write fails, training stops instead of continuing unrecorded. The format is
`ariel.recording.v1` ([docs/RECORDING.md](docs/RECORDING.md),
[recorder.py](src/ariel/record/recorder.py)). Recordings live on the Modal volume `ariel-runs`
and are not stored in Git.

## Repository layout

| Path | Contents |
|---|---|
| `src/ariel/mocap/` | **Current system:** telescoping walker, reshaping environment, PPO trainer, export |
| `src/ariel/world/` | Seeded course generation and scene description |
| `src/ariel/record/` | Episode recorder (`ariel.recording.v1`) |
| `src/ariel/art/`, `src/ariel/render/` | Blender scene, materials, robot meshes; MuJoCo render rig |
| `src/ariel/robot/`, `env/`, `train/`, `verify/` | Earlier from-scratch humanoid stack. The current trainer reuses its structure. |
| `src/ariel/recording/`, `src/ariel/analysis/` | Recorder, playback export and analysis (learning curves, metrics, reports) for the earlier recording contract. The training runs use `src/ariel/record/`. |
| `infra/` | Modal jobs: training, evaluation, sampling, Blender rendering, demo montages |
| `configs/` | Run specs for launches and resumes |
| `models/` | Checkpoints, config and training log for the run behind the video ([models/README.md](models/README.md)) |
| `docs/` | [PROJECT](docs/PROJECT.md) (requirements), [DESIGN](docs/DESIGN.md) (original design decisions), [STATUS](docs/STATUS.md) (measured state), [RECORDING](docs/RECORDING.md), [SUBMISSION_NOTES](docs/SUBMISSION_NOTES.md), [TASKS](docs/TASKS.md) |
| `tests/` | Tests for `src/ariel/` |

`docs/DESIGN.md` describes the first design, which used lunar gravity and a custom 23-joint
robot. Where it disagrees with `docs/STATUS.md`, STATUS describes what was actually built.

## Running it

Simulation, training, evaluation and rendering all run on [Modal](https://modal.com). The pinned
MoCapAct stack (Python 3.10, MuJoCo 2.2.2, dm_control 1.0.7) is defined in
[infra/mocapact_app.py](infra/mocapact_app.py), and that image downloads the pretrained models.
Run these from the repository root:

```bash
uv run modal run infra/mocapact_app.py::env_check                 # frozen prior + scripted rules on the course
uv run modal run infra/mocapact_app.py::reshape_smoke             # ~6-minute end-to-end training check
uv run modal run --detach infra/mocapact_app.py::reshape_launch --spec configs/reshape_runs.json
uv run python infra/status.py RUN                                 # latest training row of a run
uv run modal run infra/mocapact_app.py::reshape_eval --run-id RUN # checkpoint evaluation with video
uv run modal run infra/mocapact_app.py::heldout_eval --run-id RUN # unseen layouts, learned vs prior
uv run modal run infra/modal_app.py::cmu_film --src DIR --camera side   # Blender video
```

To load a checkpoint, call `ariel.mocap.train.load_policy("models/athletic-s0-20261011-031906/policy_v001300.pt")`.
It returns a deterministic policy function and needs the pinned stack above.

Local tests (recording/analysis only):

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests
```

Built for HackPHS 2026. Contribution guidelines are in [AGENTS.md](AGENTS.md).

## Credits

- [MoCapAct](https://github.com/microsoft/MoCapAct) (Wagener, Kolobov, Frujeri, Loynd, Cheng,
  Hausknecht; NeurIPS 2022). The code is MIT-licensed and the models are CDLA-Permissive-2.0.
- [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/), used through dm_control's
  CMU humanoid.
- [MuJoCo](https://mujoco.org) and [dm_control](https://github.com/google-deepmind/dm_control)
  (Google DeepMind, Apache-2.0).
- Stable-Baselines3 (MIT), used only to load MoCapAct's policy. The PPO implementation is our own.
- Blender 4.5 (GPL) for rendering, and NASA Visible Earth imagery (public domain) for the Earth.
- Reward-shaping references: legged_gym (Rudin et al., 2022) and Humanoid-Gym (Gu et al., 2024).
- Compute: Modal.

## License

Our code is released under the [MIT License](LICENSE). Third-party models, data and tools
keep their own licences, listed above.
