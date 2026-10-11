# Project definition

## The idea

**ASCENT trains an RL agent to control and visibly reshape its own physical body
while navigating a rendered obstacle course.**

The course is inspired by search-and-rescue situations. The lunar setting connects
the demo to the hackathon theme. The technical question is whether an agent can
learn useful body adaptation as part of completing a task.

The model learns across training attempts. During each attempt, it chooses how to
move and change its body in response to its current state and the obstacles.

## Agreed requirements

| Requirement | Meaning |
|---|---|
| A physical body | Mass, contacts, movement, and shape changes are handled by the simulator. |
| Visible reshaping | The body can change its geometry or arrangement, beyond swapping material labels on a fixed outline. |
| Changes within an attempt | The agent continues through the course as its body changes; it keeps its position and progress. |
| Learned decisions | An RL policy chooses movement and body-changing actions from experience. |
| A rendered course | The body and obstacles are visible, with playback based on actual simulated states. |
| Complete training recordings | Every training episode is recorded from the beginning, including exploration, failures, and interrupted attempts. |
| Visible progression | The demo lets us inspect earlier and later behavior, with the associated training progress and model identity. |

Real-world usefulness can be investigated after we establish this capability.

## The RL loop

1. Observe the current body, its motion, and relevant surrounding obstacles.
2. Choose available movement and body-changing commands.
3. Advance physics; retain the resulting body and course state.
4. Measure progress, task completion, and the chosen costs.
5. Learn from the collected transitions and repeat.

The environment defines the physical capabilities; the policy learns how to use
them. Reward should favor accomplishing the course. A particular shape is useful
when it helps accomplish that objective.

A narrow passage, ledge, and uneven surface are candidate obstacles. They should
create reasons to use different shapes. Specific strategies such as flattening
or elongating are possible learned behaviors, not scripted solutions.

## Choices we still need to make

| Choice | Candidate starting point | Evidence needed |
|---|---|---|
| Body mechanism | A deformable soft body; a modular body with physically implemented reconfiguration is another option. | A visible, stable shape change during one physics episode. |
| Simulator | Investigate DittoGym and alternatives supporting the chosen mechanism. | Working controls, contacts, rendering, runtime cost, and checked reuse terms. |
| Dimensionality | Start with 2D if it lets us establish the capability quickly. | Enough room for visibly different obstacle strategies. |
| Body limits | Fixed available material/parts, bounded actuator strength and shape-change rate. | A useful action space with meaningful physical tradeoffs. |
| Observations | Body/terrain state or an occupancy/velocity representation. | Sufficient information for control; rendered output does not itself require RGB input. |
| RL method | Choose an implementation compatible with the action space. | A genuine reward-driven update, reloadable checkpoint, and measured throughput. |
| Course and reward | Begin with one obstacle, then a varied course; progress/completion plus measured costs. | Learnable initial behavior and a clear success condition. |
| Moon presentation | Lunar visuals; gravity is a separate explicit setting. | A working course and accurately described physics settings. |

The accepted direction does not settle these choices. Record the first pilot's
decision and actual commands in [STATUS.md](STATUS.md). Do not inherit a stack,
budget, or deadline gate from the archived plan.

## Team ownership

Guhan owns the physical body, world, RL training, and all compute runs through
his Modal access. Kevin develops recording and results/reporting support;
Sparsh develops the progression viewer and presentation. Teammates hand compute
code/configurations to Guhan and work with returned artifacts. See
[TASKS.md](TASKS.md) for assignments and immediate handoffs.

## Record every attempt

Recording is part of the environment/training interface and must work before
substantial training begins.

Each episode needs:

- A session/run ID, episode ID, course configuration, seed, timestamps, and cumulative training steps.
- Actual body geometry/positions and other visible world state at the configured
  playback cadence, plus applied actions, rewards, termination, and shape changes.
- The policy update IDs or version ranges that produced the behavior. Training
  can update weights during an episode; record that lineage rather than assuming
  one unchanged checkpoint controlled the whole attempt.
- A complete rendered recording, stored incrementally as video or durable frame
  chunks that can be encoded without rerunning physics, plus a replayable trace.
- A final status, including failure, cancellation, or interruption. Retain partial
  recordings and disclose missing segments.

Record all episodes from all training workers, including initial random
exploration. Saving selected evaluation clips alone does not meet this requirement.
If persistence fails, stop affected training rather than silently continuing with
unrecorded attempts. Measure encoding and storage costs in the first pilot.

Save model checkpoints at known training intervals. Also run earlier/later
checkpoints on the same showcase course so the demo can compare behavior under
consistent conditions. Label original training recordings and later checkpoint
evaluation recordings by their actual origin.

## First working version

The first milestone is a body that responds to model-compatible commands and
visibly reshapes during one rendered physics attempt, with that attempt saved
and replayed. Then establish genuine RL learning on a small obstacle task,
with every attempt recorded.

Next, build the progression viewer and evaluate frozen earlier/later policies
on additional course layouts. Show successful strategies, failures, and what
has or has not improved. Tests of whether reshaping adds value should use a
clearly defined restriction/baseline suited to the selected physics.

## Scope removed from the active project

Lunar agriculture scoring, OSDR plant modeling, regolith sample return,
fixed-outline two-edit repair experiments, twelve-trial repair search,
supervised gain-prediction ensembles, and the optional-repair-RL gate are
historical work. They are preserved under [archive/](../archive/README.md).

## References to investigate

- [DittoGym project and demonstrations](https://suninghuang19.github.io/dittogym_page/).
- [DittoGym paper](https://arxiv.org/abs/2401.13231).
- [Official environment implementation](https://github.com/suninghuang19/dittogym).
- [Official training implementation](https://github.com/suninghuang19/cfp).

These are prior work and candidates for reuse. No dependency selection,
permission to reuse code, or successful reproduction is asserted by this list.
