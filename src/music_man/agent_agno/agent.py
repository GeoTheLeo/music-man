"""
The fraud-investigation agent on the Agno agent framework.

    planner (Agent, output_schema=Plan) -> investigator (Agent + tools) -> approval queue

- planner: a cheap model (Haiku 4.5 by default) returns a structured Plan;
  if it fails, planning falls back to the execution model.
- investigator: the execution model works through the plan, calling the same
  operations as every other surface. The guardrails live in
  agent/operations.py, so this agent can't bypass them either; a guardrail
  refusal comes back to the model as a tool error.
- human review: propose_hold only queues a pending hold. The SQLite approval
  queue is the durable pause point, so a decision can come from run.review(),
  the Streamlit reviewer, or any other surface, through the same
  compare-and-set state machine.

Two Agno behaviours are overridden so the engine comparison stays fair:
- Agno's Claude adapter appends its own sentence to the system prompt when
  tools are present; BudgetedClaude drops it, so every engine sends the same
  prompt.
- Agno's tool_call_limit answers further calls with an error but keeps
  calling the model, so a model that keeps asking for tools never stops.
  BudgetedClaude counts model turns and a tool hook raises StopAgentRun once
  the budget is spent - the same max-turns budget as the other engines.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from agno.agent import Agent
from agno.exceptions import StopAgentRun
from agno.models.anthropic import Claude

from music_man.agent import operations
from music_man.agent.run_agent import PLAN_SYSTEM_PROMPT, Plan, build_execute_system_prompt

BUDGET_EXHAUSTED = "budget_exhausted: model-turn budget reached, tool not run"


@dataclass
class BudgetedClaude(Claude):
    """Claude without Agno's system-prompt suffix, counting model turns."""

    max_turns: int | None = None
    turns: int = 0

    def get_system_message_for_model(self, tools=None):
        return None

    def invoke(self, *args, **kwargs):
        self.turns += 1
        return super().invoke(*args, **kwargs)

    @property
    def budget_spent(self) -> bool:
        return self.max_turns is not None and self.turns >= self.max_turns


def _claude(model: str, max_tokens: int, client: anthropic.Anthropic, **kwargs) -> BudgetedClaude:
    # The client already carries SDK retries, the request timeout, and the HTTP request counter.
    return BudgetedClaude(id=model, max_tokens=max_tokens, client=client, **kwargs)


def build_planner(model: str, client: anthropic.Anthropic) -> Agent:
    return Agent(
        model=_claude(model, 2000, client),
        instructions=PLAN_SYSTEM_PROMPT,
        output_schema=Plan,
        telemetry=False,
    )


def build_investigator(plan: Plan, model: str, client: anthropic.Anthropic, max_iterations: int) -> Agent:
    llm = _claude(
        model,
        8000,
        client,
        max_turns=max_iterations,
        # Tools and system prompt are stable across turns; top-level auto-caching
        # lets each later turn read the growing prefix from cache.
        request_params={"cache_control": {"type": "ephemeral"}},
    )

    def turn_budget(function_name, function_call, arguments):
        if llm.budget_spent:
            raise StopAgentRun(BUDGET_EXHAUSTED)
        return function_call(**arguments)

    return Agent(
        model=llm,
        tools=list(operations.OPERATIONS),
        tool_hooks=[turn_budget],
        instructions=build_execute_system_prompt(plan),
        telemetry=False,
    )
