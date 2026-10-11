# Active tasks

This is the single index for the corrected project. GitHub issues hold current
progress and evidence. The old D/S/E/M/N task tree is archived.

| Order | Task | Owner | Completion evidence |
|---|---|---|---|
| A1 / #25 | Define body controls and prove reshaping within an episode | Guhan | Selected mechanism/runtime, actual shape-change recording, and launch command. |
| A2 / #26 | Build the rendered obstacle course and RL environment | Guhan | Movement/reshaping controls, observations, reward, contacts, and success condition in one continuing episode. |
| A3 / #27 | Record every training episode and checkpoint lineage | Kevin | Recorder code and artifact interface; Guhan verifies complete recordings on Modal. |
| A4 / #28 | Train movement and body adaptation with RL | Guhan | Real reward-driven learning, checkpoints, measured runtime, and complete episode recordings. |
| A5 / #29 | Build the progression viewer and lunar presentation | Sparsh | Browse actual episodes and checkpoint comparisons with visible training progress. |
| A6 / #30 | Prepare comparisons and report learning progress | Kevin | Comparison/analysis code and evidence report using runs executed by Guhan. |
| A7 / #31 | Write and package the training-progression demo | Kevin + Sparsh | Kevin owns the script and claims; Sparsh owns presentation assets and submission packaging. |

## Responsibility and compute

Guhan builds the physical body, world, reward/control interface, and RL training.
All simulation, training, evaluation, analysis, and bulk rendering/encoding jobs
run through Guhan's Modal access. Teammates prepare code and job configurations;
Guhan launches them and returns artifacts. Modal credentials stay with Guhan.

Kevin builds the recorder, run index, and recording metadata contract. Guhan
provides physics state/frames and integrates the recorder into the Modal workers.
Kevin also prepares evaluation/analysis scripts and writes the demo narrative,
results explanation, source credits, and judge Q&A from actual exported evidence.

Sparsh builds the progression viewer: browse attempts, scrub playback, compare
early/later behavior, and show training progress and outcomes. He owns lunar
visual presentation, the demo interaction path, screenshots, and submission
packaging. Media-processing jobs are handed to Guhan for execution.

## Immediate handoffs

- Guhan and Kevin agree a minimal recording interface before the first training
  run: episode/run IDs, video or frame chunks, replay state, actions/rewards,
  outcome/interruption status, training steps, and policy lineage.
- Kevin supplies a clearly labeled fixture recording package to Sparsh so the
  viewer can develop before real training output exists. Fixtures are never
  presented as learned behavior.
- Guhan supplies the first real body/world recording and then early/intermediate/
  later training artifacts. Kevin and Sparsh replace fixtures with these outputs.
- Kevin drafts the demo structure now and fills in result claims after evidence
  arrives. Sparsh prepares the playback path and presentation assets in parallel.

## Working order

A1 comes first for physics. A3's artifact interface, A5's fixture-based viewer,
A6's comparison plan, and A7's script outline can start immediately. A2 and A3
integrate once the body/state interface is known. A4 needs A2 and A3 working.
A6's final report needs saved policies and Guhan's recorded comparisons. A7's
final claims and playback use the viewer and verified results.

Owners identify the workstream; record the implementation branch, actual status,
and evidence in the issue. A proposed component is not complete until its
acceptance evidence exists.

## Next action

Run a small body/simulator pilot for A1. Do not launch substantial training before
the recorder captures every attempt. Keep the exact mechanism open until the
pilot produces an actual in-episode shape change.
