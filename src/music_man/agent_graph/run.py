"""
Run the LangGraph investigation agent, or resume one paused at human review.

    python -m music_man.agent_graph.run "Find the suspicious artists and propose holds"
    python -m music_man.agent_graph.run --dry-run "..."
    python -m music_man.agent_graph.run --resume <thread_id> --approve 3 --reject 4

Runs are checkpointed to data/graph_checkpoints.db, keyed by thread_id (the
run_id), so a run paused at human review can be resumed from a new process.
Each run also writes the same JSON run record as the hand-written agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from music_man.agent import operations
from music_man.agent.cost import UsageTracker, price_key
from music_man.agent.run_agent import EXEC_MODEL, MAX_ITERATIONS, PLAN_MODEL, RunRecord, ToolCallRecord
from music_man.agent_graph.graph import build_graph
from music_man.paths import DATA_DIR

CHECKPOINT_DB = DATA_DIR / "graph_checkpoints.db"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def _track_usage(handler: UsageMetadataCallbackHandler, tracker: UsageTracker, plan_model: str, exec_model: str):
    """LangChain reports input_tokens including cache reads/writes; split them out so cache pricing applies."""
    for model, usage in handler.usage_metadata.items():
        details = usage.get("input_token_details", {}) or {}
        cache_read = details.get("cache_read", 0) or 0
        cache_write = details.get("cache_creation", 0) or 0
        priced_as = price_key(model) or exec_model
        phase = "plan" if priced_as == plan_model and plan_model != exec_model else "execute"
        tracker.add(
            phase,
            priced_as,
            SimpleNamespace(
                input_tokens=usage["input_tokens"] - cache_read - cache_write,
                output_tokens=usage["output_tokens"],
                cache_creation_input_tokens=cache_write,
                cache_read_input_tokens=cache_read,
            ),
        )


def _record_from_messages(record: RunRecord, messages) -> None:
    calls = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            record.iterations += 1
            for tc in msg.tool_calls:
                calls[tc["id"]] = tc
            if msg.text:
                record.final_text = msg.text
        elif isinstance(msg, ToolMessage) and msg.tool_call_id in calls:
            tc = calls[msg.tool_call_id]
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            record.tool_calls.append(
                ToolCallRecord(name=tc["name"], input=dict(tc["args"]), result=content, is_error=msg.status == "error")
            )


def run_graph(
    user_message: str,
    *,
    dry_run: bool = False,
    plan_model: str = PLAN_MODEL,
    exec_model: str = EXEC_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    checkpointer=None,
    verbose: bool = True,
    save: bool = True,
) -> tuple[RunRecord, dict | None]:
    """Runs until the graph ends or pauses at human review. Returns (record, interrupt payload or None)."""
    record = RunRecord(
        run_id=f"lg_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}",
        user_message=user_message,
        started_at=datetime.now(timezone.utc).isoformat(),
        plan_model=plan_model,
        exec_model=exec_model,
        dry_run=dry_run,
        engine="langgraph",
    )
    operations.start_run(record.run_id, dry_run=dry_run)
    # interrupt() needs a checkpointer; without a persistent one, keep the pause in memory.
    checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    graph = build_graph(plan_model, exec_model, checkpointer=checkpointer)
    usage_handler = UsageMetadataCallbackHandler()
    tracker = UsageTracker()
    config = {
        "configurable": {"thread_id": record.run_id},
        "callbacks": [usage_handler],
        # each agent->tools round trip is two steps, plus plan and review
        "recursion_limit": 2 * max_iterations + 4,
    }
    say = print if verbose else (lambda *a, **k: None)
    started = time.monotonic()
    paused = None
    state = None
    try:
        state = graph.invoke({"messages": [HumanMessage(user_message)], "run_id": record.run_id}, config)
        record.plan = state.get("plan", [])
        record.plan_fallback_used = state.get("plan_fallback_used", False)
        interrupts = state.get("__interrupt__") or []
        paused = interrupts[0].value if interrupts else None
        record.outcome = "completed"
    except GraphRecursionError:
        record.outcome = "budget_exhausted"
    except Exception as exc:
        record.outcome = "error"
        record.error = f"{exc.__class__.__name__}: {exc}"
        say(f"[error] {record.error}")
    finally:
        if state is None:
            snapshot = graph.get_state({"configurable": {"thread_id": record.run_id}})
            state = snapshot.values if snapshot else None
        _record_from_messages(record, (state or {}).get("messages", []))
        _track_usage(usage_handler, tracker, plan_model, exec_model)
        record.usage = tracker.summary()
        record.duration_s = round(time.monotonic() - started, 2)
        if save:
            record.save()

    say(record.final_text)
    if paused:
        say("\n=== PAUSED FOR HUMAN REVIEW ===")
        for hold in paused["pending_holds"]:
            say(f"  hold {hold['hold_id']}: {hold['artist_id']} / {hold['period']}")
        say(f"Resume: python -m music_man.agent_graph.run --resume {record.run_id} --approve <id> --reject <id>")
    say(f"\n[run] {record.outcome}, {record.iterations} model turns, ${record.usage['cost_usd']:.4f}")
    return record, paused


def resume(thread_id: str, decisions: dict[str, str], checkpointer) -> dict:
    graph = build_graph(checkpointer=checkpointer)
    state = graph.invoke(Command(resume=decisions), {"configurable": {"thread_id": thread_id}})
    return state.get("decisions", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Man LangGraph agent")
    parser.add_argument("message", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-model", default=PLAN_MODEL)
    parser.add_argument("--exec-model", default=EXEC_MODEL)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--resume", metavar="THREAD_ID")
    parser.add_argument("--approve", type=int, action="append", default=[])
    parser.add_argument("--reject", type=int, action="append", default=[])
    args = parser.parse_args()

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        if args.resume:
            decisions = {str(i): "approve" for i in args.approve} | {str(i): "reject" for i in args.reject}
            print(json.dumps(resume(args.resume, decisions, saver), indent=2))
            return
        message = " ".join(args.message) or "Find the most suspicious artist and propose a royalty hold if warranted."
        record, _ = run_graph(
            message,
            dry_run=args.dry_run,
            plan_model=args.plan_model,
            exec_model=args.exec_model,
            max_iterations=args.max_iterations,
            checkpointer=saver,
        )
    sys.exit(0 if record.outcome == "completed" else 1)


if __name__ == "__main__":
    main()
