"""
The LangGraph agent's control flow, offline: a scripted fake model drives
the real graph, real ToolNode, real guardrails, and real approval queue.
Covers the pause at human review, checkpointed resume, and that a guardrail
refusal comes back to the model as a tool error rather than crashing.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from evals.fixtures import FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, write_snapshot
from music_man.agent import queue
from music_man.agent.run_agent import Plan, PlanStep
from music_man.agent_graph import graph as graph_module
from music_man.agent_graph import run as run_module

TARGET = {"artist_id": FRAUD_DEVICE, "period": FRAUD_DEVICE_PERIOD}


class FakeChat:
    """Stands in for ChatAnthropic: replays scripted agent turns; returns a fixed plan."""

    def __init__(self, turns):
        self.turns = list(turns)

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages, config=None):
        if not self.turns:  # planning call
            return Plan(steps=[PlanStep(step="investigate and propose", rationale="test")])
        return self.turns.pop(0)


def _tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSIC_MAN_QUEUE_DB", str(tmp_path / "queue.db"))
    monkeypatch.setenv("MUSIC_MAN_SNAPSHOT", str(write_snapshot(tmp_path / "snapshot.parquet")))


def _script(monkeypatch, turns):
    agent = FakeChat(turns)
    planner = FakeChat([])
    monkeypatch.setattr(graph_module, "_chat", lambda model, max_tokens: agent if max_tokens == 8000 else planner)


def test_pauses_for_review_then_resumes_and_applies_decision(monkeypatch):
    _script(monkeypatch, [
        AIMessage("", tool_calls=[_tool_call("get_artist_detail", {"artist_id": FRAUD_DEVICE}, "c1"),
                                  _tool_call("check_hold_policy", TARGET, "c2")]),
        AIMessage("", tool_calls=[_tool_call("propose_hold", {**TARGET, "rationale": "device concentration 0.92"}, "c3")]),
        AIMessage("One hold is queued for human review; nothing has been withheld."),
    ])
    saver = InMemorySaver()
    record, paused = run_module.run_graph("find fraud", checkpointer=saver, verbose=False, save=False)

    assert record.outcome == "completed" and record.engine == "langgraph"
    assert [c.name for c in record.tool_calls] == ["get_artist_detail", "check_hold_policy", "propose_hold"]
    [hold] = queue.list_holds()
    assert hold.status == queue.PENDING and hold.run_id == record.run_id
    assert paused["pending_holds"][0]["hold_id"] == hold.id

    decisions = run_module.resume(record.run_id, {str(hold.id): "approve"}, saver)
    assert decisions == {str(hold.id): queue.EXECUTED}
    assert queue.get_hold(hold.id).decided_by == "langgraph-reviewer"


def test_guardrail_refusal_is_a_tool_error_not_a_crash(monkeypatch):
    _script(monkeypatch, [
        AIMessage("", tool_calls=[_tool_call("propose_hold", {**TARGET, "rationale": "r"}, "c1")]),
        AIMessage("I need to check policy first; stopping here."),
    ])
    record, paused = run_module.run_graph("find fraud", verbose=False, save=False)

    assert record.outcome == "completed" and paused is None
    [call] = record.tool_calls
    assert call.is_error and "check_hold_policy" in call.result
    assert queue.list_holds() == []


def test_dry_run_queues_nothing_and_never_pauses(monkeypatch):
    _script(monkeypatch, [
        AIMessage("", tool_calls=[_tool_call("check_hold_policy", TARGET, "c1")]),
        AIMessage("", tool_calls=[_tool_call("propose_hold", {**TARGET, "rationale": "r"}, "c2")]),
        AIMessage("Dry run complete."),
    ])
    record, paused = run_module.run_graph("find fraud", dry_run=True, verbose=False, save=False)

    assert paused is None and queue.list_holds() == []
    assert json.loads(record.tool_calls[-1].result)["dry_run"] is True


def test_iteration_budget_stops_a_looping_agent(monkeypatch):
    looping = [AIMessage("", tool_calls=[_tool_call("list_flagged_artists", {}, f"c{i}")]) for i in range(50)]
    _script(monkeypatch, looping)
    record, _ = run_module.run_graph("loop", max_iterations=3, verbose=False, save=False)
    assert record.outcome == "budget_exhausted"
