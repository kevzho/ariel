# Demo package (A7 #31)

Rewritten Oct 11, 2026, about 07:15 EDT. It replaces the abandoned 2D
DittoGym plan and its recorder/verify/report pipeline.

## What we are showing

Ariel is a dm_control CMU humanoid with nine telescoping segments, set on a
lunar base after a moonquake (B1, B4). A frozen, pretrained MoCapAct motor
prior turns a speed and heading into human-like joint targets (L1). Our own
PPO layer sits on top of it. Every control step it sets speed, heading,
corrections to the prior, and segment lengths, so the body reshapes
mid-episode (L2, B2). The course asks it to walk around crates, under a
collapsed habitat deck, across a fissure and up a rocky slope to a stranded
astronaut (B4); how often it gets there is pending evaluation (L4, L5).
Physics uses Earth gravity (9.81 m/s²). The Moon is the visual setting only
(B3). Hand-written teachers shape the reward during training; the shape
teacher fades to zero, but the steering hint keeps 30% of its weight for the
whole run on the current course (L6). Near the astronaut a hand-written braking
rule caps the commanded speed so the body slows to a stand, also at
evaluation (L7). Every
training episode is recorded by Guhan's recorder (R1).

Source of truth for the system: Guhan's branch `origin/feat/guhan-a1-sim`
(docs/STATUS.md, docs/SUBMISSION_NOTES.md, models/*/train_log.jsonl). It is
not merged into main yet.

## Files

| File | What it is | Owner |
|---|---|---|
| [claims.md](claims.md) | The ledger. Claim IDs (B, L, R, S, C), evidence, status (`pending`/`verified`) and safe wording. The other files use only `verified` rows or safe wording. | Kevin |
| [script.md](script.md) | The 3:00 pitch: skit cold open, real system, results, future uses, honesty section, close. | Kevin (words), Sparsh (playback), Guhan (system beat) |
| [qa.md](qa.md) | Judge Q&A with honest answers for weak spots. | Kevin; Guhan answers the physics and RL questions |
| [credits.md](credits.md) | Citations, licences, AI-use table, built-during-event record, Devpost text draft. | Kevin (text), Sparsh (submit) |

## Placeholders

| Placeholder | Means | Filled from |
|---|---|---|
| `[RESULT: <file> <field> · <ID>]` | A measured number | Only the named file (train_log.jsonl, compare.json, eval output), once claim `<ID>` is checked |
| `[CHECK: <name>]` | Needs a teammate to confirm | That person, in chat or the issue |
| `[TBD]` | Admin item | Owner named next to it |

Rules that matter most:

- Never claim lunar-gravity physics (B3).
- Training success rates are training statistics with mixed spawn points,
  not evaluations (L3, L4). Reshaping 76% vs locked 65% is a training
  statistic on the older course, one seed each.
- Do not say the stop at the astronaut or the steering was learned without
  the braking and teacher caveats (L6, L7).
- Kevin's A3 recorder and A6 analysis were not used on the real runs.
- The concept clip is scripted animation. Show it only in the cold open or
  the future-use slide, and never narrate it as the trained robot (S2).

## Filling a number from a file

1. Find the claim ID in claims.md. No ID? Add a row first.
2. Open the named file (on Guhan's branch or the Modal volume `ariel-runs`)
   and check every condition in the row, for example: pad start, no
   teachers, held-out seeds 90001-90003.
3. All hold: set the row to `verified`, add an evidence-log line (claim,
   path, run id, checkpoint, exact value), then paste the exact value.
4. Some fail: leave it `pending` and use the safe wording.
5. Nothing by 08:30: cut the line. Do not paraphrase around it.

Copy numbers from files, never from chat. Report completion as "k of n".

## Order of work (EDT, Oct 11)

Treat 09:00 as the deadline unless the organisers confirm 10:00.

| By | Work | Who |
|---|---|---|
| 07:15 | Demo docs rewritten for the real system (this package). | Kevin |
| 07:45 | Merge `feat/guhan-a1-sim` into main (rename to Ariel or note the old name). Tag `insurance-env`. | Guhan |
| 08:00 | `reshape_compare` (reshape vs locked, showcase + held-out seeds) and `reshape_eval` done; `progression` clips rendered. | Guhan |
| 08:15 | Fill claims L3-L5, L8, R2 from compare.json and the logs. Cut what is still missing. | Kevin |
| 08:30 | Freeze the script and slides. Viewer clips exported. Rehearse once with a stopwatch. | Sparsh, all |
| 08:45 | Devpost text from verified claims only. AI-use table and credits done. Repo and video public `[CHECK: claim ID in claims.md]`. | Kevin, Sparsh |
| 09:00 | Team sign-off, Sparsh submits on Devpost. | All, Sparsh |
| 10:00 | Live judging starts. Rehearse again before it. | All |

## Top 5 open items

| # | Item | Owner |
|---|---|---|
| 1 | Run `reshape_compare` and `reshape_eval` on Modal; post compare.json numbers (L4, L5). | Guhan |
| 2 | Run `progression` for the "watch it learn" footage (R2). Confirm the candidate A clip layout and spawn (L8). | Guhan |
| 3 | Merge `feat/guhan-a1-sim` into main so the repo matches the demo. | Guhan |
| 4 | Fill every claim in claims.md from files, or cut it; then sync script, Q&A and credits. | Kevin |
| 5 | Slides, viewer clips, and the Devpost submission. | Sparsh |
