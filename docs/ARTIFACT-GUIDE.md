# Understanding the saved artifacts

Reviewed October 10, 2026. The evidence is preserved in
[archive/previous-direction/artifacts/](../archive/previous-direction/artifacts/).
All six collections belong to the previous sample-return/repair-search project.
The active body-reshaping RL project is described in
[REPOSITORY-GUIDE.md](REPOSITORY-GUIDE.md).

This guide interprets existing receipts without rerunning physics or changing
archived evidence. Results below describe their original task and runtime.

## File formats and common fields

| Format or field | Everyday meaning | Interpretation |
|---|---|---|
| `.json` | Structured report or index. | Read fields such as runtime, results, and file references. |
| `.jsonl` | One structured record per line. | A line can be an event, not a distinct completed experiment. |
| `.npz` | Compressed numerical arrays. | Physical positions/actions, rather than a video. |
| `.png` | Saved rendered image. | Step number identifies the recorded state. |
| `.csv` | Table. | Headers define the actual meaning of rows and columns. |
| IDs | Labels for a robot, episode, or individual trial. | Keep distinct experiments separate. |
| SHA-256/hash | Content fingerprint. | Checks identity/integrity; does not prove the scientific conclusion. |
| `runtime` | Software versions and machine identity. | Physics can differ across machines with identical policy bytes. |
| `null` | Absent or unmeasured value. | Not a measured zero or failure. |
| COM | Center of mass; these summaries use averaged simulator point positions. | Compact location for robot or package. |

Positions generally have shape `[T+1, 2, points]`: initial state plus one state
after each of T steps, x/y coordinates, and the body's simulated points.
Actions use `[T, actuators]`; rewards use `[T]`. Positions are EvoGym units,
not calibrated lunar meters.

`raw_actions` holds policy outputs. `applied_actions` holds commands after the
offset/adapter/fault override/clipping used by that run, rather than independent
actuator sensor readings. When present, `phase` is locomotion (0) or neutral
hold (1); `sim_time` is the simulator step counter, not wall-clock seconds.

## S01: replaying an existing trained robot

Start with [s01/sweep.md](../archive/previous-direction/artifacts/s01/sweep.md).
This checks whether public, already-trained Carrier-v0 controllers reproduce
useful movement in the pinned local simulator.

| File | What it explains |
|---|---|
| `sweep-top40.jsonl` | Each of 40 high-source-reward records: source/replay rewards, steps, goal and package position. |
| `sweep.md` | Interpretation of the sweep. |
| `official_check.py` | Independent official-loop check; historical replay code, not training. |
| `<uid>/manifest.json` | Body/policy/source identity, action/normalization contract, runtime, results, checksum. |
| `<uid>/trajectory.npz` | Robot/package positions, actions, and rewards. |

Thirty of 40 records reached the robot goal; 29 did so with the package above
the height threshold throughout; 27 reproduced source reward within 0.1.
Historical source reward alone did not establish useful local replay.

Chosen robot `5119cbca-9d59-4a0c-8605-b6b522227c63` finished in **460 steps** with
reward **10.5737194818**. Two local repeats have identical trajectory digests.
Robot positions are `[461, 2, 36]`, package positions `[461, 2, 12]`, and actions
`[460, 9]`.

The controller uses saved observation normalization, reset hidden state, and
a `+1.0` offset in float32. The sweep shows that changing numerical precision
can change chaotic outcomes. This establishes imported-controller replay in
the recorded local runtime, not new RL training or lunar physics.

## S02: permanent faults and legal edits

Read the selected robot's
[receipts.json](../archive/previous-direction/artifacts/s02/5119cbca-9d59-4a0c-8605-b6b522227c63/receipts.json).
This checks whether a motor stays disabled while small material and controller
changes are tested.

An actuator is a powered block. A failed actuator is forced to neutral command
`1.0` after all other adjustments. The adapter contains horizontal gain/bias
and vertical gain/bias; it adjusts surviving commands without reviving the fault.

Body codes: 0 empty, 1 rigid, 2 soft, 3 horizontal actuator, 4 vertical actuator.
Cell IDs start at zero, left to right then top to bottom. In a 5×5 body, cell 15
is row 3, column 0. Actuator indices count only powered cells and are different.

| File | Experiment | Recorded EvoGym reward |
|---|---|---:|
| `healthy.npz` | Original healthy robot. | 10.5737 |
| `identity_transform.npz` | New action pipeline with settings that change nothing. | 10.5737 |
| `damaged.npz` | Permanently disable cell 15. | 3.0845 |
| `edited_1.npz` | Keep the fault; change cell 5 horizontal→vertical. | 2.3226 |
| `edited_2_adapter.npz` | Also change cell 0 soft→rigid; apply an adapter. | 1.7337 |

The identity trajectory exactly matches healthy replay. All nine single faults
are summarized in the receipt, without a separate NPZ for each fault. The
lowest-reward fault was chosen for these examples. Both example edits made
reward worse; these are integration checks, not successful learned repairs.

`illegal_candidates` reports six deliberately invalid proposals and rejection
reasons. `reset_and_mapping` checks compatible initial positions and controller
dimensions. `override` records neutral failed commands on every checked step.
`intervention_records` links each file to original/edited body, fault, adapter,
and checksum. The legal pool contains 172 possible candidates, not 172 executed
experiments.

These fixed-outline edits run in fresh simulations. They do not establish the
new continuous in-episode reshaping requirement.

## S03: checking actual sample delivery

Read the selected robot's
[receipts.json](../archive/previous-direction/artifacts/s03/5119cbca-9d59-4a0c-8605-b6b522227c63/receipts.json).
S03 checks that both robot and canister arrive and remain settled, beyond the
stock environment's robot-only goal.

The development delivery region is x = 8.5–9.8. Both centers must stay there
while the package remains above y = 0.3 throughout a 50-step neutral hold.
The horizon is 500 steps, requiring arrival by step 450. The healthy reference
enters at step 391 and finishes its hold at step 441.

Score **1.0** means delivery. Otherwise score is half the package's clipped
progress, up to **0.5**. Instability makes the run invalid with score zero.
Therefore score 0.5 and progress 1.0 do not mean delivery. `measured_gain` is
the score improvement over this episode's damaged baseline.

### Files and accounting

| File/directory | Meaning |
|---|---|
| `receipts.json` | Protocol, healthy reference, fault screen, final trial summaries, accounting/reset checks. |
| `registry.jsonl` | Episode index for this robot. |
| `reference/healthy.npz` and `reference/healthy/` | Healthy physical trace and rendered frames. |
| `ep-68ad5afe66cb56c3/episode.json` | Identity, fault, used budget, best trial, runtime, ledger hash. |
| `trials.jsonl` | Reservation before physics and final result afterward: 24 events for 12 trials here. |
| `rejected.jsonl` | Illegal failed-cell edit rejected before physics, consuming no physics trial. |
| `trial-01.npz` through `trial-12.npz` | Actual per-trial trajectories. |
| `trial-NN/step-0000.png` | Candidate-design image before that trial. |
| `trial-NN/step-XXXX.png` | Later recorded states, including hold-entry/final frames where applicable. |
| `.lock` | Writer coordination; existence alone does not establish an active process. |

Healthy reference and nine-fault screening are labeled uncharged work outside
this twelve-trial demonstration. They still consumed physics compute and must
remain visible in total experimental cost.

The receipt reports 12 slots used, rejection of a thirteenth attempt, refusal
to reset the budget by reopening, refusal of a concurrent writer, and zero
slots after resume. These check this historical runner's accounting behavior.

### All twelve trials

Cell **11**, actuator index **1**, fails in this episode; S02 used cell 15.
Candidates are a seeded sample to exercise the runner. No trained designer
checkpoint selected them.

| Trial | Body change | Score | Gain | Interpretation |
|---|---|---:|---:|---|
| 1 | None; neutral adapter. | 0.5000 | 0.0000 | Damaged baseline arrived too late for the hold. |
| 2 | Cell 14 rigid→soft; cell 19 vertical→horizontal. | 0.2494 | −0.2506 | Horizon reached with less package progress. |
| 3 | Cell 14 rigid→soft; cell 21 horizontal→vertical. | 0.3896 | −0.1104 | Horizon reached; gain adapter. |
| 4 | Cells 7 and 23 rigid→soft. | 0.4784 | −0.0216 | Horizon reached; bias adapter. |
| 5 | Cell 6 rigid→soft; cell 13 horizontal→vertical. | 0.5000 | 0.0000 | Progress cap reached, delivery requirements unmet. |
| 6 | Cell 1 rigid→soft; cell 18 horizontal→vertical. | 0.4814 | −0.0186 | Horizon reached; gain adapter. |
| 7 | Cell 9 rigid→soft. | 0.5000 | 0.0000 | Arrived too late; bias adapter. |
| 8 | Cell 4 rigid→soft; neutral adapter. | **1.0000** | **+0.5000** | Arrived at 441, held 50 steps, delivered at 491. |
| 9 | Cell 4 rigid→soft; cell 5 horizontal→vertical. | 0.5000 | 0.0000 | Arrived too late; gain adapter. |
| 10 | Cell 0 soft→rigid; cell 19 vertical→horizontal. | 0.2181 | −0.2819 | Horizon reached; bias adapter. |
| 11 | Cell 19 vertical→horizontal. | 0.2995 | −0.2005 | Horizon reached with less package progress. |
| 12 | Adapter only; no body edit. | 0.0234 | −0.4766 | Very little package progress. |

Full-precision scores, adapters, and validity fields are in the final events.
Trial 8 has 492 physical states and 491 action/reward entries. The fault remains
active. This is one successful sampled repair under a development protocol,
not learned improvement across unseen bodies.

Step-0 renders are candidate-design inputs. Later outcome images reveal what
happened; feeding them to a model predicting that same trial's result would
leak the answer. The before image does not identify the failed actuator, so the
old predictor also needs the damage mask.

## E01: finding robots that replay successfully

[e01/summary.md](../archive/previous-direction/artifacts/e01/summary.md) explains
the sweep; `replays.jsonl` stores per-record results. The
[split manifest](../archive/previous-direction/configs/splits/carrier-v0-e01.json)
and its `.sha256` companion are under archived `configs/`.

All 203 Carrier-v0 records were replayed twice, totaling 406 replays. There
were 203 distinct bodies; 115 met healthy-carry requirements and 88 were
excluded. Primary exclusions were 72 missing the goal and 16 dropping the
package. The report's `unstable` category means mismatched repeat digests;
simulator-instability messages for two bodies were categorized as `no_goal`.
Do not confuse that category with the simulator's physical-instability flag.

Near-related bodies were grouped to keep almost identical bodies out of
different splits. Forty groups combine exact identities with transitive
material-grid distance of at most two cells. Healthy bodies split into 84
training, 16 validation, and 15 test bodies. The 15 healthy test bodies occupy
five groups; they are not 30 independent healthy test bodies.

This is the old local macOS Carrier cohort. Its measured 168.2 seconds for 406
replays does not establish capacity for the new body mechanism or training.
The grouping utilities may be reusable; this cohort is not the new curriculum.

## I01: actual Modal execution evidence

[i01/README.md](../archive/previous-direction/artifacts/i01/README.md) describes
two real parity replays and an eight-trial pilot. These updated artifacts go
beyond the earlier planning-only discussion of I01.

| File/directory | Explanation |
|---|---|
| `i01-parity-20261010/request.json` | Parity invocation input. |
| `i01-parity-20261010/parity.json` | Detailed Mac-versus-Modal differences. |
| `i01-pilot-20261010/request.json` | Eight explicit trial specifications. |
| `<batch>/batch.json` | Trial statuses, runtime, durable batch duration. |
| `<batch>/client-receipt.json` | Invocation identity, source/build provenance, timings, allocation estimates. |
| `<batch>/<trial>/attempt-started.json` | Durable marker written before completion. |
| `<batch>/<trial>/manifest.json` | Completed result, checksums, runtime, measured outcomes. |
| `<batch>/<trial>/trajectory.npz` | Actual physical states and actions. |
| `cache-check/client-receipt.json` | Identical completed request reused evidence, with zero new physics. |
| `failed-start/client-receipt.json` | Earlier Windows encoding failure preserved explicitly. |
| `verification.json` and `tests.log` | Recorded audit and validation results. |
| `billing.json` | Authenticated app-metering snapshot for the recorded I01 scope. |

Mac healthy replay took **460 steps**, reward **10.5737**; Modal Linux took
**465 steps**, reward **10.5511**. Trajectory hashes differ. Four healthy Modal
replays agreed across recorded calls, and the body met the old healthy-carry
criterion on Modal. Cross-platform bitwise parity did not pass.

The eight-trial batch has two healthy repeats and six faulted interventions.
The faulted commands stayed neutral, but none of those six reached the Carrier
goal. `delivery_success` is null because this batch did not apply S03 scoring.
There is no Modal delivery-success, trained-model, or held-out recovery claim.

The durable batch took **49.263 seconds**, about **9.74 saved trials/minute**.
Average physics was about 1.33 seconds per trial; persistence adds overhead.
Build/startup/download are excluded from that throughput denominator. The
October 10 billing snapshot reports **$0.00469498** in its specified app scope.
It is a reporting snapshot, not a final invoice, current balance, or new-task
training estimate.

Start markers, explicit failures, checksums, cached-request checks, and cost
receipts are useful persistence patterns. The old image and repair-episode
contract need review before reuse for body reshaping and full training recording.

## Lunar feasibility: NASA source-discovery files

[source-check.json](../archive/previous-direction/artifacts/lunar-feasibility/source-check.json)
records OSD-476 discovery: 457 manifest entries, 64 dated plate-TIFF filenames,
selected URLs, and checksums/byte counts for two downloaded tables.

| File | Content | What it tells us |
|---|---|---|
| `OSD-476-sample-table.csv` | Twenty sample IDs and treatment conditions. | Association with Apollo 11/12/17 regolith or JSC-1A simulant. |
| `OSD-476-contrasts.csv` | Twelve directed condition comparisons, including reversed pairs. | Which treatment was compared against which. |
| `source-check.json` | Discovery metadata, selected files, checksums, limitations. | What was located/downloaded at that time. |

The tables contain conditions and comparison definitions, not plant-growth
measurements, crop-yield labels, or robot-repair outcomes. Growth images and
a differential-expression table were listed but not downloaded in this receipt.
A remote URL does not mean its content exists locally.

The receipt's early “no robot replay” and failed Carrier-query statements
describe that source check. Later S01/E01/I01 results establish subsequent
robot replay; the historical receipt remains unchanged. Plant modeling and
lunar agriculture are outside the corrected active scope.

## Relationship to the current project

The archive demonstrates earlier controller replay, real physical trajectories,
local sample delivery, and cloud artifact persistence. It does not establish
continuous body-reshaping RL training or complete recordings of every training
attempt for the corrected task.

Future recordings/checkpoints belong in ignored `runs/` or external storage,
using the new recorder contract and model lineage. Preserve historical results
and identities rather than rewriting them to fit the current direction.
