"""Print the latest training-log row of each run on the ``ascent-runs`` volume.

    uv run python infra/status.py reshape-s0-20261011-030959 locked-s0-20261011-030959
"""

import json
import os
import subprocess
import sys
import tempfile

KEYS = ("global_step", "elapsed_s", "steps_per_s", "teach", "mean_steps", "mean_progress_m", "best_max_x",
        "pass_rates", "attempts", "success_rate", "reasons", "level", "shape_span", "action_std")


def main(run_ids):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for run in run_ids:
        tmp = os.path.join(tempfile.gettempdir(), f"status_{run}.jsonl")
        r = subprocess.run(["modal", "volume", "get", "--force", "ascent-runs", f"{run}/train_log.jsonl", tmp],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0 or not os.path.exists(tmp):
            print(f"{run}: no log yet ({(r.stderr or r.stdout).strip()[-120:]})")
            continue
        rows = [json.loads(line) for line in open(tmp, encoding="utf-8") if line.strip()]
        rows = [x for x in rows if "version" in x and "event" not in x]
        if not rows:
            print(f"{run}: log has no training rows yet")
            continue
        last = rows[-1]
        print(f"{run}: " + json.dumps({k: last.get(k) for k in KEYS}))


if __name__ == "__main__":
    main(sys.argv[1:])
