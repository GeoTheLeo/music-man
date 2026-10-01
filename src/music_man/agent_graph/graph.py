"""
The fraud-investigation agent as a LangGraph state graph.

    START -> plan -> agent <-> tools
                       |
                       v
                  human_review -> END

- plan: a cheap model (Haiku 4.5 by default) writes a structured plan; if it
  fails, planning falls back to the execution model.
- agent / tools: the execution model works through the plan, calling the same
  four operations as every other surface. The guardrails live in
  agent/operations.py, so this graph can't bypass them either.
- human_review: if this run queued any holds, the graph calls interrupt() and
  stops. With a checkpointer the paused run survives a restart and resumes
  with Command(resume={hold_id: "approve" | "reject"}); each decision goes
  through the queue's compare-and-set state machine. The Streamlit reviewer
  remains an equivalent way to decide the same holds.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from music_man.agent import operations, queue
from music_man.agent.run_agent import (
    EXEC_MODEL,
    MAX_RETRIES,
    PLAN_MODEL,
    PLAN_SYSTEM_PROMPT,
    REQUEST_TIMEOUT_S,
    Plan,
    build_execute_system_prompt,
)

LC_TOOLS = [StructuredTool.from_function(op, parse_docstring=True) for op in operations.OPERATIONS]


class InvestigationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    run_id: str
    plan: list[dict]
    plan_fallback_used: bool
    decisions: dict[str, str]


def _chat(model: str, max_tokens: int) -> ChatAnthropic:
    return ChatAnthropic(model=model, max_tokens=max_tokens, max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT_S)


def build_graph(plan_model: str = PLAN_MODEL, exec_model: str = EXEC_MODEL, checkpointer=None):
    # Tools and system prompt are stable across the loop; top-level cache_control
    # lets each later iteration read the growing prefix from cache.
    agent_llm = _chat(exec_model, 8000).bind_tools(LC_TOOLS).bind(cache_control={"type": "ephemeral"})

    def plan(state: InvestigationState) -> dict:
        request = [SystemMessage(PLAN_SYSTEM_PROMPT), *state["messages"]]
        fallback = False
        try:
            result = _chat(plan_model, 2000).with_structured_output(Plan).invoke(request)
        except Exception:
            result = None
        if result is None and plan_model != exec_model:
            fallback = True
            result = _chat(exec_model, 2000).with_structured_output(Plan).invoke(request)
        if result is None:
            raise RuntimeError("planning phase did not return a parseable plan")
        return {"plan": [s.model_dump() for s in result.steps], "plan_fallback_used": fallback}

    def agent(state: InvestigationState) -> dict:
        system = build_execute_system_prompt(Plan(steps=state["plan"]))
        return {"messages": [agent_llm.invoke([SystemMessage(system), *state["messages"]])]}

    def route_after_agent(state: InvestigationState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "human_review"

    def human_review(state: InvestigationState) -> dict:
        pending = [h for h in queue.list_holds(queue.PENDING) if h.run_id == state["run_id"]]
        if not pending:
            return {}
        decisions = interrupt(
            {
                "run_id": state["run_id"],
                "pending_holds": [
                    {"hold_id": h.id, "artist_id": h.artist_id, "period": h.period, "rationale": h.rationale}
                    for h in pending
                ],
                "resume_with": {"<hold_id>": "approve | reject"},
            }
        )
        applied = {}
        for hold in pending:
            choice = str(decisions.get(str(hold.id), decisions.get(hold.id, ""))).lower()
            if choice not in ("approve", "reject"):
                applied[str(hold.id)] = "left pending"
                continue
            status = queue.EXECUTED if choice == "approve" else queue.REJECTED
            try:
                queue.decide_hold(hold.id, status, "langgraph-reviewer")
                applied[str(hold.id)] = status
            except queue.InvalidTransition as exc:  # already decided elsewhere (e.g. in Streamlit)
                applied[str(hold.id)] = f"not applied: {exc}"
        return {"decisions": applied}

    builder = StateGraph(InvestigationState)
    builder.add_node("plan", plan)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(LC_TOOLS, handle_tool_errors=True))
    builder.add_node("human_review", human_review)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "human_review"])
    builder.add_edge("tools", "agent")
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=checkpointer)
