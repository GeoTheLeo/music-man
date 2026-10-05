"""
Observability loader and dashboard: failure taxonomy, per-model attribution
for old and new records, and a headless render of the Streamlit app.
"""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from music_man.observability import data


def _write(path, run_id, **overrides):
    record = {
        "run_id": run_id, "user_message": "m", "started_at": "2026-10-01T16:00:00+00:00",
        "plan_model": "claude-haiku-4-5", "exec_model": "claude-opus-5", "dry_run": False, "engine": "custom",
        "tool_calls": [], "outcome": "completed", "error": None, "iterations": 3, "duration_s": 12.0,
        "usage": {"api_calls": 2, "cost_usd": 0.05, "cost_usd_by_phase": {"plan": 0.001, "execute": 0.049}},
    }
    record.update(overrides)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{run_id}.json").write_text(json.dumps(record), encoding="utf-8")


def test_failure_taxonomy_and_model_attribution(tmp_path):
    runs_dir = tmp_path / "runs"
    _write(runs_dir, "ok_old")  # old record: cost by phase only
    _write(runs_dir, "ok_new", usage={"api_calls": 1, "cost_usd": 0.02, "retries": 1,
                                       "by_model": {"claude-opus-5": {"calls": 1, "input_tokens": 10,
                                                                      "output_tokens": 5, "cost_usd": 0.02}}})
    _write(runs_dir, "billing", outcome="error",
           error="AnthropicInvalidRequestError: Error code: 400 - Your credit balance is too low")
    _write(runs_dir, "overloaded", outcome="error", error="InternalServerError: Error code: 529 - overloaded")
    _write(runs_dir, "looped", outcome="budget_exhausted")

    runs, models, records = data.load({"live": runs_dir})

    kinds = dict(zip(runs["run_id"], runs["error_kind"]))
    assert kinds["billing"] == "billing"
    assert kinds["overloaded"] == "provider error"
    assert kinds["looped"] == "budget exhausted"
    assert kinds["ok_new"] is None
    old = models[models["run_id"] == "ok_old"].set_index("model")["cost_usd"].to_dict()
    assert old == {"claude-haiku-4-5": 0.001, "claude-opus-5": 0.049}
    assert runs.set_index("run_id").loc["ok_new", "retries"] == 1
    assert len(records) == 5


def test_dashboard_renders_without_errors():
    at = AppTest.from_file("src/music_man/observability/app.py", default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"Runs", "Success rate", "Failures", "Mean cost / run", "p95 latency", "Retries"} <= set(labels)
