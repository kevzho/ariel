# Ariel

**An RL agent learns to move and visibly reshape its own physical body while navigating a rendered obstacle course.**

The course is inspired by search and rescue. The Moon is the setting for the demo.
The agent changes its body during the same attempt, and physics determines what
those changes accomplish. Every training attempt is recorded so we can watch
the behavior develop.

**Current state:** recording, playback export, and analysis code live in
`src/ariel/`. The body mechanism and simulator are being built separately.

## Working in this repo

Start a feature branch from the updated `main`. Pick a task from the active
backlog, keep the change within that task, and attach the actual run, recording,
or checkpoint that demonstrates completion.

The environment and trainer still need integration in `src/ariel/`; `app/` is the intended
progression viewer location, and `configs/` is for selected runtime/course
settings. Neither has a tracked active implementation yet. Generated recordings and weights go under ignored
`runs/` or external artifact storage.

Agent instructions are in [AGENTS.md](AGENTS.md).

## Research starting point

[DittoGym](https://suninghuang19.github.io/dittogym_page/) studies RL control of
bodies that change shape within a task. It is a candidate to investigate.
The simulator, body controls, learning algorithm, and reuse terms are still
part of the first task.
