"""
Loads agent run records into tidy tables for the observability dashboard.

Sources (all optional; whatever exists is loaded):
- data/runs/*.json              live runs (hand-written agent, LangGraph, webhook-triggered)
- evals/reports/*/runs/*.json   eval sweeps (git-ignored, regenerable)
- evals/sample_runs/*.json      a committed sample so a fresh clone has something to show

Older records predate the per-model usage breakdown; for those, cost is
attributed by phase (plan -> plan_model, execute -> exec_model), which is
exact for single-model phases.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from music_man.paths import DATA_DIR, PROJECT_ROOT

SOURCES = {
    "live": DATA_DIR / "runs",
    "eval": PROJECT_ROOT / "evals" / "reports",
    "sample": PROJECT_ROOT / "evals" / "sample_runs",
}


def _record_paths(sources: dict[str, Path]) -> list[tuple[str, Path]]:
    found = []
    for label, root in sources.items():
        if root.exists():
            found += [(label, p) for p in sorted(root.rglob("*.json")) if p.parent.name in ("runs", "sample_runs")]
    return found


def _error_kind(record: dict) -> str | None:
    """Failure taxonomy for the dashboard: what an on-call person needs to know first."""
    outcome = record.get("outcome")
    if outcome == "completed":
        return None
    if outcome in ("budget_exhausted", "refusal"):
        return outcome.replace("_", " ")
    error = (record.get("error") or "").lower()
    status = re.search(r"error code: (\d{3})", error)
    code = int(status.group(1)) if status else None
    if "credit balance" in error:
        return "billing"
    if code == 429:
        return "rate limit"
    if code is not None and code >= 500:
        return "provider error"
    if "timeout" in error:
        return "timeout"
    return "application error"


def _model_rows(record: dict) -> list[dict]:
    usage = record.get("usage") or {}
    if usage.get("by_model"):
        return [{"model": model, **stats} for model, stats in usage["by_model"].items()]
    rows = []
    for phase, cost in (usage.get("cost_usd_by_phase") or {}).items():
        model = record.get("plan_model") if phase == "plan" else record.get("exec_model")
        rows.append({"model": model, "cost_usd": cost})
    return rows


def load(sources: dict[str, Path] = SOURCES) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Returns (runs, model_usage, records_by_run_id). Duplicate run ids (a sample copied from a report) keep the first."""
    runs, model_rows, records = [], [], {}
    for source, path in _record_paths(sources):
        record = json.loads(path.read_text(encoding="utf-8"))
        run_id = record.get("run_id")
        if not run_id or run_id in records:
            continue
        records[run_id] = record
        usage = record.get("usage") or {}
        calls = record.get("tool_calls") or []
        runs.append(
            {
                "run_id": run_id,
                "source": source,
                "started_at": pd.to_datetime(record.get("started_at"), utc=True),
                "engine": record.get("engine", "custom"),
                "exec_model": record.get("exec_model"),
                "plan_model": record.get("plan_model"),
                "dry_run": record.get("dry_run", False),
                "outcome": record.get("outcome"),
                "error_kind": _error_kind(record),
                "error": record.get("error"),
                "duration_s": record.get("duration_s"),
                "iterations": record.get("iterations"),
                "tool_calls": len(calls),
                "tool_errors": sum(1 for c in calls if c.get("is_error")),
                "holds_proposed": sum(1 for c in calls if c.get("name") == "propose_hold" and not c.get("is_error")),
                "cost_usd": usage.get("cost_usd", 0.0),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "retries": usage.get("retries"),  # None where not measured (older records, LangGraph engine)
                "user_message": record.get("user_message"),
            }
        )
        for row in _model_rows(record):
            model_rows.append({"run_id": run_id, "started_at": runs[-1]["started_at"], "engine": runs[-1]["engine"], **row})

    runs_df = pd.DataFrame(runs)
    if not runs_df.empty:
        runs_df = runs_df.sort_values("started_at", ascending=False).reset_index(drop=True)
    return runs_df, pd.DataFrame(model_rows), records
