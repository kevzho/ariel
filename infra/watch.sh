#!/usr/bin/env bash
# One-line status of every training run (reads train_log.jsonl from the Modal volume).
export MODAL_PROFILE=tejas-16842 PYTHONIOENCODING=utf-8
uv run python infra/status.py ins-reshape-20261011-030244 ins-locked-20261011-030244 reshape-s0-20261011-030959 locked-s0-20261011-030959 reshape-s1-20261011-030933 shapegym-s0-20261011-030933 athletic-s0-20261011-031906 2>&1 | uv run python -c "
import sys,json
for l in sys.stdin:
    try:
        name, js = l.split(': ',1); r=json.loads(js)
        print(f\"{name[:14]:14} {r['global_step']/1e6:5.2f}M {r['steps_per_s']:5.0f}/s teach {r['teach']:.2f} prog {r['mean_progress_m']:4.1f}m succ {r['success_rate']:.2f} pass {r['pass_rates']} lvl { {k: round(v,2) for k,v in r['level'].items()} } span {[round(v,2) for v in r['shape_span']]}\")
    except Exception as e: print(l[:200])
"
date
