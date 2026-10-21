"""
The Agno agent's control flow, offline: a scripted fake Anthropic client
drives the real Agno agent, Agno's real Claude adapter, the real guardrails,
and the real approval queue. Covers queue-then-review, a guardrail refusal
coming back as a tool error, dry-run, the model-turn budget (which Agno's own
tool_call_limit doesn't enforce), and that Agno sends the same system prompt
as the other engines.
"""

from __future__ import annotations

import json

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from evals.fixtures import FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, write_snapshot
from music_man.agent import queue
from music_man.agent.run_agent import EXEC_MODEL, PLAN_MODEL, Plan, PlanStep, build_execute_system_prompt
from music_man.agent_agno import run as run_module

TARGET = {"artist_id": FRAUD_DEVICE, "period": FRAUD_DEVICE_PERIOD}
PLAN = Plan(steps=[PlanStep(step="investigate and propose", rationale="test")])


def _message(content, stop_reason, model=EXEC_MODEL):
    return Message(
        id="msg", type="message", role="assistant", model=model, content=content, stop_reason=stop_reason,
        stop_sequence=None, usage=Usage(input_tokens=100, output_tokens=20, cache_read_input_tokens=50),
    )


def _tools(*calls):
    blocks = [ToolUseBlock(id=f"c{i}", type="tool_use", name=name, input=args) for i, (name, args) in enumerate(calls)]
    return _message(blocks, "tool_use")


def _text(text):
    return _message([TextBlock(type="text", text=text)], "end_turn")


class _Messages:
    def __init__(self, respond):
        self.respond = respond

    def create(self, **request):
        return self.respond(request)


class FakeClient:
    """Stands in for anthropic.Anthropic: plans arrive on the structured-output (beta) path, agent turns on the plain one."""

    def __init__(self, turns, plan=PLAN):
        self.turns = list(turns)
        self.plan = plan
        self.requests = []
        self.messages = _Messages(self._turn)
        self.beta = type("Beta", (), {"messages": _Messages(self._plan)})()

    def is_closed(self):
        return False

    def _plan(self, request):
        self.requests.append(("plan", request))
        if isinstance(self.plan, Exception):
            raise self.plan
        return _message([TextBlock(type="text", text=self.plan.model_dump_json())], "end_turn", request["model"])

    def _turn(self, request):
        self.requests.append(("execute", request))
        return self.turns.pop(0)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSIC_MAN_QUEUE_DB", str(tmp_path / "queue.db"))
    monkeypatch.setenv("MUSIC_MAN_SNAPSHOT", str(write_snapshot(tmp_path / "snapshot.parquet")))


def _run(client, **kwargs):
    return run_module.run_agno("find fraud", client=client, verbose=False, save=False, **kwargs)


def test_queues_hold_then_review_applies_decision():
    client = FakeClient([
        _tools(("get_artist_detail", {"artist_id": FRAUD_DEVICE}), ("check_hold_policy", TARGET)),
        _tools(("propose_hold", {**TARGET, "rationale": "device concentration 0.92"})),
        _text("One hold is queued for human review; nothing has been withheld."),
    ])
    record = _run(client)

    assert record.outcome == "completed" and record.engine == "agno"
    assert [c.name for c in record.tool_calls] == ["get_artist_detail", "check_hold_policy", "propose_hold"]
    assert record.iterations == 3 and record.final_text.startswith("One hold is queued")
    assert record.usage["api_calls"] == 4  # one plan call + three agent turns
    [hold] = queue.list_holds()
    assert hold.status == queue.PENDING and hold.run_id == record.run_id

    assert run_module.review(record.run_id, {str(hold.id): "approve"}) == {str(hold.id): queue.EXECUTED}
    assert queue.get_hold(hold.id).decided_by == "agno-reviewer"


def test_guardrail_refusal_is_a_tool_error_not_a_crash():
    client = FakeClient([
        _tools(("propose_hold", {**TARGET, "rationale": "r"})),
        _text("I need to check policy first; stopping here."),
    ])
    record = _run(client)

    assert record.outcome == "completed"
    [call] = record.tool_calls
    assert call.is_error and "check_hold_policy" in call.result
    assert queue.list_holds() == []


def test_dry_run_queues_nothing():
    client = FakeClient([
        _tools(("check_hold_policy", TARGET)),
        _tools(("propose_hold", {**TARGET, "rationale": "r"})),
        _text("Dry run complete."),
    ])
    record = _run(client, dry_run=True)

    assert queue.list_holds() == []
    assert json.loads(record.tool_calls[-1].result)["dry_run"] is True


def test_turn_budget_stops_a_looping_agent():
    client = FakeClient([_tools(("list_flagged_artists", {})) for _ in range(50)])
    record = _run(client, max_iterations=3)

    assert record.outcome == "budget_exhausted"
    assert record.iterations == 3
    assert len(record.tool_calls) == 2  # the third turn's call was refused, not run
    assert sum(kind == "execute" for kind, _ in client.requests) == 3


def test_sends_the_same_system_prompt_as_the_other_engines():
    client = FakeClient([_text("Nothing suspicious.")])
    _run(client)

    [(_, request)] = [r for r in client.requests if r[0] == "execute"]
    assert request["system"][0]["text"] == build_execute_system_prompt(PLAN)
    assert request["cache_control"] == {"type": "ephemeral"}


def test_planning_falls_back_to_the_execution_model():
    client = FakeClient([_text("Nothing suspicious.")], plan=RuntimeError("planner down"))
    calls = []
    real_plan = client._plan

    def flaky(request):  # fail on the cheap planner only
        calls.append(request["model"])
        if request["model"] == PLAN_MODEL:
            raise RuntimeError("planner down")
        client.plan = PLAN
        return real_plan(request)

    client.beta.messages.respond = flaky
    record = _run(client)

    assert record.outcome == "completed" and record.plan_fallback_used
    assert calls[0] == PLAN_MODEL and calls[-1] == EXEC_MODEL
