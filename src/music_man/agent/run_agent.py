"""
CLI entry point for the Music Man fraud-investigation agent: plans first,
then investigates and proposes holds using the tool-runner, mirroring the
NorthStar Agentic Intervention Copilot's plan-then-execute pattern.
"""

from __future__ import annotations

import sys
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Windows consoles often default to a legacy codepage (e.g. cp1252) that
# can't encode characters like the model's own "->" arrows in its text -
# force UTF-8 stdout so print() never crashes on the model's output.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from music_man.agent.tools import TOOLS

# override=True: this shell environment can have an empty ANTHROPIC_API_KEY
# already set (seen previously in the NorthStar project's session), and
# load_dotenv() does not override existing env vars by default - which
# would silently keep the empty value instead of the real key from .env.
load_dotenv(override=True)

MODEL = "claude-opus-5"

TOOL_CATALOG = [
    ("list_flagged_artists", "List artists whose worst day's anomaly score is above a threshold."),
    ("get_artist_detail", "Get the full fraud-signal feature breakdown for an artist's worst day."),
    ("check_hold_policy", "Check whether a new hold is allowed for an artist/period."),
    ("propose_hold", "Queue a proposed royalty hold for human review - never executes anything."),
]


class PlanStep(BaseModel):
    step: str
    tool_hint: Optional[str] = None
    rationale: str


class Plan(BaseModel):
    steps: list[PlanStep] = Field(..., min_length=1, max_length=6)


PLAN_SYSTEM_PROMPT = (
    "You are the planning module for Music Man's royalty-fraud investigation agent.\n\n"
    "Available tools:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG)
    + "\n\nGiven the user's request, produce a short ordered plan (1-6 steps) of what you "
    "will do, before doing any of it. Each step should name the tool it will most likely "
    "use (or null for a pure reasoning/summary step). Keep steps concise."
)


def build_execute_system_prompt(plan: Plan) -> str:
    plan_text = "\n".join(
        f"{i + 1}. {s.step}" + (f" [{s.tool_hint}]" if s.tool_hint else "") + f" - {s.rationale}"
        for i, s in enumerate(plan.steps)
    )
    return (
        "You are Music Man's royalty-fraud investigation agent. You already produced this "
        f"plan for the user's request - follow it, adapting only if a tool result requires it:\n\n"
        f"{plan_text}\n\n"
        "Rules:\n"
        "- Never state an artist's metrics or fraud signals without first calling "
        "get_artist_detail (or list_flagged_artists) to look them up.\n"
        "- Always call check_hold_policy before propose_hold for a given artist/period.\n"
        "- propose_hold only queues a hold for a human to review in the Streamlit reviewer - "
        "it never withholds a real payment. Say this explicitly in your final summary.\n"
        "- Be concise. End with a short summary of what you found and what (if anything) is "
        "now pending human approval."
    )


def plan_phase(client: anthropic.Anthropic, user_message: str) -> Plan:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        system=PLAN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_format=Plan,
    )
    if response.parsed_output is None:
        raise RuntimeError("planning phase did not return a parseable plan")
    return response.parsed_output


def run_agent(user_message: str) -> None:
    client = anthropic.Anthropic()

    plan = plan_phase(client, user_message)
    print("=== PLAN ===")
    for i, step in enumerate(plan.steps, 1):
        hint = f" [{step.tool_hint}]" if step.tool_hint else ""
        print(f"{i}. {step.step}{hint}\n   {step.rationale}")
    print()

    system = build_execute_system_prompt(plan)
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=8000,
        system=system,
        tools=TOOLS,
        messages=[{"role": "user", "content": user_message}],
    )

    print("=== EXECUTION ===")
    for message in runner:
        for block in message.content:
            if block.type == "text":
                print(block.text, end="")
            elif block.type == "tool_use":
                print(f"\n[tool_call] {block.name}({block.input})")
        if message.stop_reason == "refusal":
            print("\n[error] the model declined to continue.")
            break
    print()


if __name__ == "__main__":
    user_message = " ".join(sys.argv[1:]) or (
        "Find the most suspicious artist and propose a royalty hold if warranted."
    )
    run_agent(user_message)
