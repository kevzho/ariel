# Working on Ariel

## Project requirements

- The core is an RL policy controlling movement and visible body reshaping.
- Body changes occur during one continuous obstacle-course attempt. Preserve
  progress and physical state when the shape changes.
- The world is a real physics simulation with rendered playback.
- Record every training episode, including initial exploration and failures.
  Recording must work before substantial training starts.
- The Moon provides the presentation setting. Search and rescue inspires the
  obstacles; real-world transfer remains future investigation.
- The precise body mechanism, simulator, observation representation, algorithm,
  and gravity setting are open until the first pilot records a decision.
  Soft-body deformation is a candidate, not a settled implementation.

## Scope and truth

Use the project requirements above for decisions. Ignore the old sample-return,
NASA plant-data, permanent-fault repair search, gain-regression, two-edit,
twelve-trial, and conditional-repair-RL plans.

Distinguish implemented code, actual physics runs, trained checkpoints, recorded
training attempts, checkpoint evaluation runs, and proposed work. Report measured
behavior with its run/checkpoint/course identity. Improvement is an experimental
outcome; preserve failures and null results.

## Layout and collaboration

- `src/ariel/`: environment, policy/training, recording and playback support.
- `app/`: progression viewer and lunar presentation.
- `configs/`: selected runtime and course settings.
- `runs/`: ignored generated artifacts; persist large output externally.

Keep one active backlog in GitHub issues. Start branches
from current `main`, preserve unrelated work, and use normal merges/fast-forwards.
Add verified launch/test commands when the selected runtime actually works.
A repository cleanup does not authorize an unbounded compute run.

Guhan owns the physical body, world, training, and all compute execution through
his Modal access. Simulation, training, evaluation, analysis, bulk rendering/
encoding, and job validation are handed to Guhan; teammates prepare code,
configurations, fixtures, viewer work, and presentation material. Do not launch
these jobs on a teammate's machine or request/copy Guhan's Modal credentials.
Use fixtures with explicit labels until Guhan supplies actual run artifacts.

Machine and Bridge instructions supplied by the user or parent workspace still
apply. Use the desktop for Git/GitHub operations. Keep Bridge and unrelated
sessions alive. Select and size compute jobs before Guhan launches them on Modal.
