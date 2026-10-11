# Checkpoints

Policies of the learned layer (PyTorch, CPU), copied from the Modal volume `ascent-runs` on
October 11, 2026 at 07:43 EDT, after all runs were stopped.
The folder is the training run behind the submission video and holds its checkpoints, the
run's `config.json` and its `train_log.jsonl`. The pretrained MoCapAct models are not here; the Modal image downloads them.

| Run | Body | Course | Checkpoint | Notes |
|---|---|---|---|---|
| `athletic-s0-20261011-031906` | reshaping, may run to 3.4 m/s | current, fissure to 1.1 m | `policy_v001300.pt` | The policy in the candidate B video (evaluation seed 153). |
| `athletic-s0-20261011-031906` | reshaping, may run to 3.4 m/s | current, fissure to 1.1 m | `policy_v001623.pt` | Final. |

A checkpoint is a dict: `model` (actor and critic weights), `norm_actor` / `norm_critic`
(observation normalisers), `version`, `global_step`, `level` (course difficulty when saved),
`config` (the full training configuration, including the environment overrides) and `dims`.
`ascent.mocap.train.load_policy(path)` returns a deterministic policy function.

Running a policy needs the pinned MoCapAct stack (Python 3.10, MuJoCo 2.2.2, dm_control 1.0.7),
which is defined in `infra/mocapact_app.py` and runs on Modal.

`policy_version` in a training log counts PPO updates; `global_step` counts environment steps.
Every training episode of every run is recorded on the volume under `<run>/episodes/`
(see `docs/RECORDING.md`); those recordings are not in Git.
