"""Evaluation plans and reports from Ariel recordings.

CLI modules load lazily to support `python -m`.
"""

from importlib import import_module

from .curves import curve_from_rows, learning_curve
from .metrics import (SUPPORTED_SCHEMA, EpisodeData, UnsupportedSchemaError, aspect_series, course_hash, course_id,
                      episode_metrics, load_episode, load_run)

_LAZY = {"HeldoutGuardError": "eval_plan", "build_eval_plan": "eval_plan", "heldout_reportable": "eval_plan",
         "mark_heldout_viewed": "eval_plan", "select_checkpoints": "eval_plan", "bootstrap_ci": "report",
         "build_report": "report"}

__all__ = ["SUPPORTED_SCHEMA", "EpisodeData", "UnsupportedSchemaError", "aspect_series", "course_hash", "course_id", "curve_from_rows", "episode_metrics",
           "learning_curve", "load_episode", "load_run", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
