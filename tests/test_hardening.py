"""
Offline tests for the production-hardening layer: hold state machine,
idempotency, dry-run, code-enforced guardrails, cost accounting, and the
eval grader. No API calls; queue and snapshot live in tmp_path.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from anthropic.lib.tools import ToolError

from evals.fixtures import (
    FRAUD_DEVICE,
    FRAUD_DEVICE_PERIOD,
    FRAUD_GEO,
    FRAUD_GEO_PERIOD,
    POPULAR,
    POPULAR_PERIOD,
    write_snapshot,
)
from evals.grading import grade, ordering_violations
from evals.scenarios import SCENARIOS_BY_ID
from music_man.agent import queue, tools
from music_man.agent.cost import UsageTracker, call_cost
from music_man.agent.run_agent import RunRecord, ToolCallRecord


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSIC_MAN_QUEUE_DB", str(tmp_path / "queue.db"))
    monkeypatch.setenv("MUSIC_MAN_SNAPSHOT", str(write_snapshot(tmp_path / "snapshot.parquet")))
    tools.start_run("run_test")


def _propose(artist=FRAUD_DEVICE, period=FRAUD_DEVICE_PERIOD, rationale="device concentration 0.92"):
    return json.loads(tools.propose_hold.call({"artist_id": artist, "period": period, "rationale": rationale}))


def _check(artist=FRAUD_DEVICE, period=FRAUD_DEVICE_PERIOD):
    return json.loads(tools.check_hold_policy.call({"artist_id": artist, "period": period}))


# ---------- state machine ----------

def test_approve_then_release_frees_the_period():
    hold, created = queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r", run_id="run_a")
    assert created and hold.status == queue.PENDING

    queue.decide_hold(hold.id, queue.EXECUTED, "reviewer")
    released = queue.release_hold(hold.id, "reviewer", "artist appealed; plays verified organic")
    assert released.status == queue.RELEASED
    assert released.release_reason.startswith("artist appealed")
    assert not queue.has_hold_for_period(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD)

    events = [(e["from_status"], e["to_status"]) for e in queue.list_events(hold.id)]
    assert events == [(None, queue.PENDING), (queue.PENDING, queue.EXECUTED), (queue.EXECUTED, queue.RELEASED)]

    _, created_again = queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "new evidence", run_id="run_b")
    assert created_again


@pytest.mark.parametrize(
    "setup, to_status",
    [
        ([], queue.RELEASED),                      # can't release something never approved
        ([queue.REJECTED], queue.EXECUTED),        # can't approve after rejection
        ([queue.EXECUTED], queue.REJECTED),        # can't reject after execution
    ],
)
def test_invalid_transitions_raise(setup, to_status):
    hold, _ = queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r")
    for status in setup:
        queue.transition(hold.id, status, "reviewer")
    with pytest.raises(queue.InvalidTransition):
        queue.transition(hold.id, to_status, "reviewer", reason="x")


def test_double_approve_applies_once():
    hold, _ = queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r")
    queue.decide_hold(hold.id, queue.EXECUTED, "reviewer-1")
    with pytest.raises(queue.InvalidTransition):
        queue.decide_hold(hold.id, queue.EXECUTED, "reviewer-2")
    assert queue.get_hold(hold.id).decided_by == "reviewer-1"
    assert len(queue.list_events(hold.id)) == 2


def test_release_requires_reason():
    hold, _ = queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r")
    queue.decide_hold(hold.id, queue.EXECUTED, "reviewer")
    with pytest.raises(ValueError):
        queue.release_hold(hold.id, "reviewer", "   ")


def test_create_hold_is_idempotent():
    first, created_1 = queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r", run_id="run_a")
    second, created_2 = queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r (retry)", run_id="run_a")
    assert created_1 and not created_2
    assert first.id == second.id
    assert len(queue.list_holds()) == 1


def test_legacy_queue_db_is_migrated(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute(
        "CREATE TABLE holds (id INTEGER PRIMARY KEY AUTOINCREMENT, artist_id TEXT NOT NULL, period TEXT NOT NULL, "
        "rationale TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending_approval', created_at TEXT NOT NULL, "
        "decided_at TEXT, decided_by TEXT)"
    )
    conn.execute("INSERT INTO holds (artist_id, period, rationale, created_at) VALUES ('a', '2026-01-01', 'r', 'now')")
    conn.commit()
    conn.close()
    monkeypatch.setenv("MUSIC_MAN_QUEUE_DB", str(legacy))

    [hold] = queue.list_holds()
    assert hold.run_id is None and hold.release_reason is None


# ---------- tool guardrails ----------

def test_propose_requires_policy_check_first():
    with pytest.raises(ToolError, match="check_hold_policy"):
        _propose()
    assert queue.list_holds() == []


def test_propose_rejects_artist_period_not_in_snapshot():
    _check(FRAUD_DEVICE, "2026-03-01")
    with pytest.raises(ToolError, match="no activity"):
        _propose(period="2026-03-01")


def test_propose_rejects_bad_period_and_empty_rationale():
    with pytest.raises(ToolError, match="ISO date"):
        tools.check_hold_policy.call({"artist_id": FRAUD_DEVICE, "period": "Feb 14"})
    _check()
    with pytest.raises(ToolError, match="rationale"):
        _propose(rationale="  ")


def test_propose_retry_in_same_run_is_deduplicated():
    _check()
    first = _propose()
    retry = _propose()
    assert first["queued"] and not first["deduplicated"]
    assert retry["queued"] and retry["deduplicated"]
    assert retry["hold_id"] == first["hold_id"]
    assert len(queue.list_holds()) == 1


def test_propose_blocked_when_another_run_holds_the_period():
    queue.create_hold(FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "earlier", run_id="run_earlier")
    assert _check()["allowed"] is False
    result = _propose()
    assert result["queued"] is False and "blocked by policy" in result["reason"]


def test_dry_run_writes_nothing():
    tools.start_run("run_dry", dry_run=True)
    _check()
    result = _propose()
    assert result["dry_run"] and result["would_queue"]["artist_id"] == FRAUD_DEVICE
    assert queue.list_holds() == []


def test_policy_checks_do_not_leak_across_runs():
    _check()
    tools.start_run("run_next")
    with pytest.raises(ToolError):
        _propose()


# ---------- cost accounting ----------

def test_call_cost_includes_cache_pricing():
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000,
                            cache_creation_input_tokens=200_000, cache_read_input_tokens=1_000_000)
    # opus-5: 5.00 + 0.2*5*1.25 + 1.0*5*0.1 + 0.1*25
    assert call_cost("claude-opus-5", usage) == pytest.approx(5.00 + 1.25 + 0.50 + 2.50)


def test_usage_tracker_splits_by_phase():
    tracker = UsageTracker()
    tracker.add("plan", "claude-haiku-4-5", SimpleNamespace(input_tokens=1000, output_tokens=200))
    tracker.add("execute", "claude-opus-5", SimpleNamespace(input_tokens=1000, output_tokens=200))
    summary = tracker.summary()
    assert summary["api_calls"] == 2
    assert summary["cost_usd_by_phase"]["plan"] < summary["cost_usd_by_phase"]["execute"]


# ---------- eval grader ----------

def _record(calls: list[tuple[str, dict, dict, bool]], run_id="run_test") -> RunRecord:
    record = RunRecord(run_id=run_id, user_message="m", started_at="t", plan_model="p",
                       exec_model="e", dry_run=False, outcome="completed")
    record.tool_calls = [ToolCallRecord(name, inp, json.dumps(res), err) for name, inp, res, err in calls]
    return record


def test_ordering_violation_detected():
    record = _record([
        ("check_hold_policy", {"artist_id": FRAUD_GEO, "period": FRAUD_GEO_PERIOD}, {"allowed": True}, False),
        ("propose_hold", {"artist_id": FRAUD_GEO, "period": FRAUD_GEO_PERIOD, "rationale": "r"}, {"queued": True}, False),
    ])
    assert ordering_violations(record) == [f"propose_hold ('{FRAUD_GEO}', '{FRAUD_GEO_PERIOD}') without get_artist_detail"]


def test_grade_flags_false_hold_and_missed_hold():
    scenario = SCENARIOS_BY_ID["find_and_hold"]
    record = _record([])
    rows = [queue.Hold(1, FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r", queue.PENDING, "t", run_id="run_test"),
            queue.Hold(2, POPULAR, POPULAR_PERIOD, "r", queue.PENDING, "t", run_id="run_test"),
            queue.Hold(3, FRAUD_GEO, FRAUD_GEO_PERIOD, "r", queue.PENDING, "t", run_id="someone_else")]
    result = grade(record, scenario, rows)
    assert not result["task_completed"]
    assert result["false_holds"] == [[POPULAR, POPULAR_PERIOD]]
    assert result["missed_holds"] == [[FRAUD_GEO, FRAUD_GEO_PERIOD]]  # other run's hold doesn't count


def test_grade_dry_run_uses_would_queue_and_flags_writes():
    scenario = SCENARIOS_BY_ID["dry_run"]
    calls = [
        ("propose_hold", {"artist_id": a, "period": p, "rationale": "r"},
         {"queued": False, "dry_run": True, "would_queue": {"artist_id": a, "period": p}}, False)
        for a, p in scenario.expected_holds
    ]
    record = _record(calls)
    assert grade(record, scenario, [])["task_completed"]

    leaked = [queue.Hold(1, FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, "r", queue.PENDING, "t", run_id="run_test")]
    assert grade(record, scenario, leaked)["dry_run_violation"]


def test_dated_model_names_map_to_their_price():
    from music_man.agent.cost import price_key

    assert price_key("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert price_key("claude-opus-5") == "claude-opus-5"
    assert price_key("some-other-model") is None


def test_artist_detail_precomputes_ratios_so_the_model_does_no_arithmetic():
    detail = json.loads(tools.get_artist_detail.call({"artist_id": FRAUD_GEO}))
    # Velvet Static: 2,400 IPs / 90 listeners - the case an eval caught the model rounding to "~28"
    assert detail["derived"]["ips_per_listener"] == 26.67
    assert detail["derived"]["plays_per_device"] == round(2600 / 85, 2)


def test_request_counter_sees_sdk_retries():
    import anthropic
    import httpx2

    from music_man.agent.run_agent import RequestCounter

    responses = iter([
        httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
        httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }),
    ])
    counter = RequestCounter()
    client = anthropic.Anthropic(
        api_key="test", max_retries=2,
        http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(lambda request: next(responses)),
            event_hooks={"request": [counter]},
        ),
    )
    client.messages.create(model="claude-haiku-4-5", max_tokens=5, messages=[{"role": "user", "content": "hi"}])
    assert counter.count == 2  # one overloaded attempt + one success = one retry


def test_usage_summary_breaks_down_by_model():
    tracker = UsageTracker()
    tracker.add("plan", "claude-haiku-4-5", SimpleNamespace(input_tokens=100, output_tokens=10))
    tracker.add("execute", "claude-opus-5", SimpleNamespace(input_tokens=200, output_tokens=20,
                                                            cache_read_input_tokens=500))
    by_model = tracker.summary()["by_model"]
    assert by_model["claude-opus-5"]["cache_read_input_tokens"] == 500
    assert by_model["claude-haiku-4-5"]["calls"] == 1
