# Submission notes (facts for Devpost, the demo script and judge Q&A)

Updated October 11, 2026, 07:55 EDT. Every number here has its run in [STATUS.md](STATUS.md).

## One-paragraph description

Ariel is a rescue humanoid that changes the length of its own limbs while it moves. A
pretrained human motion model (MoCapAct) gives it a natural walk and run; our reinforcement
learning layer sits on top and decides, step by step, how fast to go, where to steer, how to
correct the motion, and how long each body segment should be. On a damaged lunar base course it
sprints off the landing pad, swerves past cargo crates, pulls its thighs, torso and arms in to
pass under a low deck, extends its shins to leap a fissure, climbs a rocky 16° slope, then slows
down, walks up to a stranded astronaut and stands. Every training attempt is recorded and can
be replayed.

## The video

`runs/final/candidateB_side.mp4`: one complete evaluation run of the trained policy
(run `athletic-s0-20261011-031906`, checkpoint v1300, seed 153), 18.4 s, rendered in Blender from
the recorded simulation states. Rough timings: crates 3 s, deck 6 s, fissure 8 s, slope 9-10 s,
standing from 17 s.

- It is the best of 240 sampled runs, not a typical one. Say "a successful run", not "it does this
  every time".
- It is cut at the end of a 1.3 s stand. In the simulation the robot lost balance about half a
  second after the cut. If asked, say so.

## What to say, and what not to

| Topic | Say | Do not say |
|---|---|---|
| Gravity | Physics runs at Earth gravity; the Moon is the visual setting. | "Lunar-gravity locomotion", "trained on the Moon" |
| Pretrained model | The walking and running come from MoCapAct (Microsoft, NeurIPS 2022), used frozen. | "We trained it to walk from scratch" |
| Our contribution | The telescoping body, the course, the learned layer that makes a fixed-body motion model work on a body that reshapes, the reshape-versus-locked comparison, the full training record. | "The robot learned everything itself" |
| Teachers | Early in training a hand-written rule suggested shapes; its weight faded to zero. A steering and pace hint around the crates stayed at 30% weight. Evaluation uses neither. | "Shape was never guided" |
| Stopping | The environment limits the commanded speed over the last 8 m; the policy learned to stay balanced, slow down and stand. In training the best policy stood at the astronaut in 21% of recent attempts. | "It always stops", "it stands indefinitely" |
| Reshaping versus locked | The reshaping body learned the deck about nine times faster (0.81M vs 7.37M steps to 90%) and the fissure about three times faster. On the older course the locked body later caught up to within 11 points (76% vs 65%, training statistics). On a harder 1.70 m deck and wider fissure the reshaping body passes more often (92% vs 78%, 90% vs 75%, training statistics). | "The locked body cannot do it" |
| Unseen layouts | On 24 layouts never used in training the policy reached the astronaut in 15 (easier setting) and 8 (harder setting); the pretrained walker alone in 0. | "It generalises to any terrain" |
| Re-ordered course | Without retraining, the policy from the video also finished courses with the same four obstacles in other orders: it reached the astronaut in 25 of 280 attempts across seven orders and stopped and stood in 9; one order (fissure first, then crates) it never finished. The panel shows the best run of four orders. | "It handles any order", "it survives all of them" |
| Rough ground | Without retraining it crossed cratered ground twice as rough as its training terrain in 9 of 12 attempts (pretrained walker 5 of 12); it fails on dense or deep craters. | "Handles arbitrary terrain" |
| Arms | The arms telescope in under the deck. They are not used to push, grab or climb. | "Uses its hands" |
| Crates | Fixed in place, 0.42 m off the centre line in the video; the robot swerves past them. | "Pushes debris aside", "crates block the path" |
| Slope | A slope of 16° in the video (1.46 m rise over 5 m); policies trained up to 18°. | "Climbs a cliff", "hands-assisted climb" |
| Dimensionality | 3D rigid-body physics (MuJoCo). | "2D soft body" (the earlier plan on `main`) |
| Scripted clips | Clips from `mocapact-check-*` or exports with `origin: scripted_rules` are scripted. | Presenting them as learned |

## Numbers that are safe to quote

- 37.3M training steps for the policy in the video; 4.4 hours on one 64-core Modal container.
- About 190M training steps across all runs, every episode recorded.
- Six training runs kept; up to 1,800 simulated robots stepping in parallel (seven runs at once).
- Unseen layouts: 15 of 24 and 8 of 24 (pretrained walker 0 of 24 in both).
- Body shape range used in the video: thighs 0.71x to 1.33x, shins 0.83x to 1.22x, arms down to
  0.70x, torso 0.73x to 1.42x.
- Top speed in the video about 4.7 m/s, median 3.4 m/s while running.

## Credits and licences

- MoCapAct (Wagener, Kolobov, Frujeri, Loynd, Cheng, Hausknecht; NeurIPS 2022): code MIT,
  models CDLA-Permissive-2.0. https://github.com/microsoft/MoCapAct
- CMU Graphics Lab Motion Capture Database (source clips), via dm_control's CMU humanoid.
- dm_control and MuJoCo (Google DeepMind, Apache-2.0).
- Stable-Baselines3 (MIT) is used only to load MoCapAct's pretrained policy; our PPO is our own code.
- Blender 4.5 (GPL) for rendering; NASA Visible Earth imagery (public domain) for the Earth.
- Reward shaping references from the first stack: legged_gym (Rudin et al. 2022), Humanoid-Gym (Gu et al. 2024).
- Compute: Modal.

## For Kevin and Sparsh (what changed since `main`)

- The body is a 3D humanoid with telescoping segments, not a 2D soft body; claims P1, P2, P7 and
  P10 in `docs/demo/claims.md` on `main` need rewriting from this file.
- Recordings are `ariel.recording.v1` (per-worker `index.jsonl` plus `chunk_*.npz` holding full
  joint states), not the 2D point format in `docs/recording-contract.md`. A viewer can consume
  the rendered clips in `runs/progression-*` and `runs/final/` and the JSON summaries next to them.
- Code, docs and checkpoints are on branch `feat/guhan-a1-sim` (pushed). Videos are not in Git;
  Guhan sends `runs/final/` directly.
