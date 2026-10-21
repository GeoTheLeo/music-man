"""
Run the Agno investigation agent, or decide the holds a run queued.

    python -m music_man.agent_agno.run "Find the suspicious artists and propose holds"
    python -m music_man.agent_agno.run --dry-run "..."
    python -m music_man.agent_agno.run --review <run_id> --approve 3 --reject 4

A run that queues holds ends with them pending in the SQLite approval queue;
--review decides them later, from a new process, through the queue's
compare-and-set state machine (the Streamlit reviewer is an equivalent way to
decide the same holds). Each run writes the same JSON run record as the other
engines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import anthropic
from agno.run.base import RunStatus

from music_man.agent import operations, queue
from music_man.agent.cost import UsageTracker, price_key
from music_man.agent.run_agent import (
    EXEC_MODEL,
    MAX_ITERATIONS,
    PLAN_MODEL,
    Plan,
    RunRecord,
    ToolCallRecord,
    make_client,
)
from music_man.agent_agno.agent import BUDGET_EXHAUSTED, build_investigator, build_planner

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _track_usage(tracker: UsageTracker, phase: str, model: str, output) -> None:
    """One tracker entry per model turn, from Agno's per-message metrics (cache tokens already split out)."""
    for msg in output.messages or []:
        if msg.role != "assistant":
            continue
        m = msg.metrics
        tracker.add(
            phase,
            price_key(model) or model,
            SimpleNamespace(
                input_tokens=m.input_tokens or 0,
                output_tokens=m.output_tokens or 0,
                cache_creation_input_tokens=m.cache_write_tokens or 0,
                cache_read_input_tokens=m.cache_read_tokens or 0,
            ),
        )


def _plan(client, user_message: str, model: str, tracker: UsageTracker) -> Plan | None:
    output = build_planner(model, client).run(user_message)
    _track_usage(tracker, "plan", model, output)
    return output.content if isinstance(output.content, Plan) else None


def run_agno(
    user_message: str,
    *,
    dry_run: bool = False,
    plan_model: str = PLAN_MODEL,
    exec_model: str = EXEC_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    client: anthropic.Anthropic | None = None,
    verbose: bool = True,
    save: bool = True,
) -> RunRecord:
    client = client or make_client()
    record = RunRecord(
        run_id=f"agno_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}",
        user_message=user_message,
        started_at=datetime.now(timezone.utc).isoformat(),
        plan_model=plan_model,
        exec_model=exec_model,
        dry_run=dry_run,
        engine="agno",
    )
    operations.start_run(record.run_id, dry_run=dry_run)
    tracker = UsageTracker()
    counter = getattr(client, "request_counter", None)
    requests_before = counter.count if counter else 0
    say = print if verbose else (lambda *a, **k: None)
    started = time.monotonic()

    try:
        plan = _plan(client, user_message, plan_model, tracker)
        if plan is None and plan_model != exec_model:
            record.plan_fallback_used = True
            say(f"[warn] planning on {plan_model} failed; retrying on {exec_model}")
            plan = _plan(client, user_message, exec_model, tracker)
        if plan is None:
            raise RuntimeError("planning phase did not return a parseable plan")
        record.plan = [s.model_dump() for s in plan.steps]

        investigator = build_investigator(plan, exec_model, client, max_iterations)
        output = investigator.run(user_message)
        _track_usage(tracker, "execute", exec_model, output)
        record.iterations = investigator.model.turns

        budget_hit = False
        for tool in output.tools or []:
            if tool.result == BUDGET_EXHAUSTED:  # refused by the turn budget, never ran
                budget_hit = True
                continue
            record.tool_calls.append(
                ToolCallRecord(
                    name=tool.tool_name,
                    input=dict(tool.tool_args or {}),
                    result=tool.result if isinstance(tool.result, str) else json.dumps(tool.result),
                    is_error=bool(tool.tool_call_error),
                )
            )
        for msg in output.messages or []:
            if msg.role == "assistant" and isinstance(msg.content, str) and msg.content:
                record.final_text = msg.content

        if output.status == RunStatus.error:
            record.outcome = "error"
            record.error = str(output.content)
        else:
            record.outcome = "budget_exhausted" if budget_hit else "completed"
    except Exception as exc:  # the run record is most valuable exactly when something broke
        record.outcome = "error"
        record.error = f"{exc.__class__.__name__}: {exc}"
        say(f"[error] {record.error}")
    finally:
        record.duration_s = round(time.monotonic() - started, 2)
        record.usage = tracker.summary()
        if counter is not None:
            http_requests = counter.count - requests_before
            record.usage["http_requests"] = http_requests
            record.usage["retries"] = max(0, http_requests - record.usage["api_calls"])
        if save:
            record.save()

    say(record.final_text)
    pending = [h for h in queue.list_holds(queue.PENDING) if h.run_id == record.run_id]
    if pending:
        say("\n=== PENDING HUMAN REVIEW ===")
        for hold in pending:
            say(f"  hold {hold.id}: {hold.artist_id} / {hold.period}")
        say(f"Decide: python -m music_man.agent_agno.run --review {record.run_id} --approve <id> --reject <id>")
    say(f"\n[run] {record.outcome}, {record.iterations} model turns, ${record.usage['cost_usd']:.4f}")
    return record


def review(run_id: str, decisions: dict[str, str], reviewer: str = "agno-reviewer") -> dict:
    """Apply approve/reject decisions to the holds a run queued; anything undecided stays pending."""
    applied = {}
    for hold in [h for h in queue.list_holds(queue.PENDING) if h.run_id == run_id]:
        choice = str(decisions.get(str(hold.id), "")).lower()
        if choice not in ("approve", "reject"):
            applied[str(hold.id)] = "left pending"
            continue
        status = queue.EXECUTED if choice == "approve" else queue.REJECTED
        try:
            queue.decide_hold(hold.id, status, reviewer)
            applied[str(hold.id)] = status
        except queue.InvalidTransition as exc:  # decided elsewhere in the meantime
            applied[str(hold.id)] = f"not applied: {exc}"
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Man Agno agent")
    parser.add_argument("message", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-model", default=PLAN_MODEL)
    parser.add_argument("--exec-model", default=EXEC_MODEL)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--review", metavar="RUN_ID")
    parser.add_argument("--approve", type=int, action="append", default=[])
    parser.add_argument("--reject", type=int, action="append", default=[])
    args = parser.parse_args()

    if args.review:
        decisions = {str(i): "approve" for i in args.approve} | {str(i): "reject" for i in args.reject}
        print(json.dumps(review(args.review, decisions), indent=2))
        return
    message = " ".join(args.message) or "Find the most suspicious artist and propose a royalty hold if warranted."
    record = run_agno(
        message,
        dry_run=args.dry_run,
        plan_model=args.plan_model,
        exec_model=args.exec_model,
        max_iterations=args.max_iterations,
    )
    sys.exit(0 if record.outcome == "completed" else 1)


if __name__ == "__main__":
    main()
