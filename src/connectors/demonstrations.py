"""
Demonstration file manager for robot imitation learning.

Demonstrations are JSON files written by NavAgent._save_demos() to
data/demonstrations/demo_<timestamp>.json.  Each file has the shape:

  {
    "task": "open the fridge door",
    "steps": 4,
    "demos": [
      {"action": {"action": "click", "x": 640, "y": 400, ...},
       "app": "Finder",
       "timestamp": 1722000000.0},
      ...
    ]
  }

Functions here read, list, and aggregate those files without touching
the nav agent itself.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_DEMO_DIR = Path("data/demonstrations")


def _demo_files() -> list[Path]:
    if not _DEMO_DIR.exists():
        return []
    return sorted(_DEMO_DIR.glob("demo_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def list_demos(limit: int = 50) -> list[dict]:
    """
    Return metadata for the most recent demonstration files.
    Each entry: id, task, steps, timestamp_s, path.
    """
    results = []
    for path in _demo_files()[:limit]:
        try:
            data = json.loads(path.read_text())
            results.append({
                "id":          path.stem,          # "demo_1722000000"
                "task":        data.get("task", ""),
                "steps":       data.get("steps", 0),
                "timestamp_s": _stem_ts(path.stem),
                "path":        str(path),
            })
        except Exception:
            continue
    return results


def get_demo(demo_id: str) -> Optional[dict]:
    """
    Load a specific demonstration by id (the file stem, e.g. 'demo_1722000000').
    Returns None if not found.
    """
    path = _DEMO_DIR / f"{demo_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def summarize_policy() -> dict:
    """
    Aggregate all saved demonstrations into a simple action-frequency policy.

    Returns:
      {
        "total_demos": int,
        "total_steps": int,
        "action_counts": {"click": 42, "type_text": 18, ...},
        "top_apps": [["Finder", 12], ...],
        "tasks": ["task description", ...],   # most recent 20
      }
    """
    action_counts: Counter = Counter()
    app_counts: Counter = Counter()
    tasks: list[str] = []
    total_steps = 0

    for path in _demo_files():
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        tasks.append(data.get("task", ""))
        for step in data.get("demos", []):
            action = step.get("action", {})
            action_counts[action.get("action", "unknown")] += 1
            app_counts[step.get("app", "unknown")] += 1
            total_steps += 1

    return {
        "total_demos":   len(_demo_files()),
        "total_steps":   total_steps,
        "action_counts": dict(action_counts.most_common()),
        "top_apps":      action_counts.most_common(10),    # type: ignore[assignment]
        "tasks":         tasks[:20],
    }


def _stem_ts(stem: str) -> Optional[float]:
    try:
        return float(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
