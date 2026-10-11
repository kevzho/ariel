# Recording format (`ariel.recording.v1`)

Every training episode from every worker is recorded from its first step, including
random exploration, failures and episodes cut short by shutdown. Checkpoint evaluations
are recorded the same way and labelled `origin: checkpoint_evaluation`.

Implementation: [`src/ariel/record/recorder.py`](../src/ariel/record/recorder.py).
Audit: `uv run modal run infra/modal_app.py::audit --run-id <run>` (Modal profile `tejas-16842`).

## Layout (Modal volume `ariel-runs`)

```
/<run_id>/
  config.json                  training configuration
  train_log.jsonl              one row per logged PPO update
  checkpoints/policy_vNNNNNN.pt
  episodes/wNNN/index.jsonl    one JSON row per episode (worker NNN)
  episodes/wNNN/chunk_NNNNN.npz
```

## Index row (one per episode)

| Field | Meaning |
|---|---|
| `episode_id` | `<run>-w<worker>-e<counter>` |
| `status` | `complete` (terminated), `truncated` (timeout or stall), `interrupted` (shutdown) |
| `reason` | `success`, `out_of_corridor`, `fell_into_fissure`, `fell_off_course`, `penetration_exploit`, `physics_unstable`, `timeout`, `stalled`, `worker_shutdown` |
| `steps`, `seconds` | control steps (50 Hz) and duration |
| `return`, `return_terms` | total reward and its parts (progress, energy, action_rate, impact, self_contact, joint_limit, wobble, slip, joint_acc, success, terminal) |
| `global_step_start/end` | cumulative training steps when the episode began and ended |
| `policy_version_min/max` | PPO update numbers that chose its actions (an episode can span updates) |
| `max_x`, `passed`, `success` | furthest pelvis position, obstacles passed, goal reached |
| `course_id`, `course_params`, `level` | exact course layout (regenerable) and curriculum difficulty |
| `spawn_index`, `spawn_x` | start point (0 = landing pad) |
| `max_penetration` | physics audit for exploit detection |
| `chunk`, `key` | where the arrays live |

## Chunk arrays (per episode key `eNNN`)

| Array | Shape | Content |
|---|---|---|
| `eNNN_qpos` | (frames, nq) float32 | full physical state at 25 Hz: robot (first 39 values: root position, root quaternion, 32 joints/segments) then loose debris |
| `eNNN_actions` | (steps, 32) float16 | applied actions in [-1, 1] (23 rotary joints, then 9 telescoping segments) |
| `eNNN_rewards` | (steps,) float32 | per-step reward |
| `eNNN_terms` | (steps, 11) float32 | reward terms in `return_terms` order |
| `eNNN_versions` | (steps,) int32 | policy version for each action |

Any frame renders without re-running physics: regenerate the course from
`course_params`, set `qpos`, run forward kinematics. `infra/modal_app.py::mj_render_trace`
renders MuJoCo video with the detailed meshes; `render_video_chunk` renders Blender
showcase frames.

Videos are rendered for sampled training episodes and every checkpoint evaluation.
Every other episode stays renderable on demand from its trace.

## Persistence guarantees

Chunks are written atomically (temporary file, rename), then re-opened and checked
before their episodes are released; index rows are fsynced. The trainer commits the
volume every minute. A write or verification error raises inside the worker, which
stops training rather than continuing with unrecorded attempts.
