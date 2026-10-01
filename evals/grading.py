"""
Deterministic grading of one agent run against its scenario's ground truth.

Holds are read back from the queue database (filtered to this run's
run_id), not taken from the agent's own account of what it did, so a run
that claims to have queued something it didn't still fails.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from evals.scenarios import Scenario
from music_man.agent.queue import Hold
from music_man.agent.run_agent import RunRecord

GOLDEN_DIR = Path(__file__).parent / "golden"


def effective_holds(record: RunRecord, queue_rows: list[Hold]) -> tuple[set, set]:
    """(holds actually queued by this run, holds a dry run said it would queue)."""
    queued = {(h.artist_id, h.period) for h in queue_rows if h.run_id == record.run_id}
    would_queue = {
        (p["result"]["would_queue"]["artist_id"], p["result"]["would_queue"]["period"])
        for p in record.proposals()
        if not p["is_error"] and p["result"].get("dry_run")
    }
    return queued, would_queue


def ordering_violations(record: RunRecord) -> list[str]:
    """Successful propose_hold calls not preceded by get_artist_detail and check_hold_policy for the same target."""
    violations = []
    seen_detail: set[str] = set()
    seen_policy: set[tuple[str, str]] = set()
    for call in record.tool_calls:
        artist = call.input.get("artist_id")
        if call.name == "get_artist_detail" and not call.is_error:
            seen_detail.add(artist)
        elif call.name == "check_hold_policy" and not call.is_error:
            seen_policy.add((artist, call.input.get("period")))
        elif call.name == "propose_hold" and not call.is_error:
            target = (artist, call.input.get("period"))
            if artist not in seen_detail:
                violations.append(f"propose_hold {target} without get_artist_detail")
            if target not in seen_policy:
                violations.append(f"propose_hold {target} without check_hold_policy")
    return violations


def trajectory(record: RunRecord) -> list[list]:
    return [[c.name, c.input.get("artist_id")] for c in record.tool_calls]


def compare_golden(scenario_id: str, record: RunRecord, proposals: set) -> dict | None:
    path = GOLDEN_DIR / f"{scenario_id}.json"
    if not path.exists():
        return None
    golden = json.loads(path.read_text(encoding="utf-8"))
    similarity = difflib.SequenceMatcher(
        a=[tuple(t) for t in golden["trajectory"]], b=[tuple(t) for t in trajectory(record)]
    ).ratio()
    return {
        "proposals_match": sorted(map(list, proposals)) == golden["proposals"],
        "trajectory_similarity": round(similarity, 3),
    }


def record_golden(scenario_id: str, record: RunRecord, proposals: set) -> Path:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{scenario_id}.json"
    path.write_text(
        json.dumps(
            {"run_id": record.run_id, "proposals": sorted(map(list, proposals)), "trajectory": trajectory(record)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def grade(record: RunRecord, scenario: Scenario, queue_rows: list[Hold]) -> dict:
    queued, would_queue = effective_holds(record, queue_rows)
    effective = would_queue if scenario.dry_run else queued
    missed = scenario.expected_holds - effective
    false_holds = effective & scenario.forbidden_holds
    unexpected = effective - scenario.expected_holds - scenario.forbidden_holds
    proposals = record.proposals()
    violations = ordering_violations(record)

    return {
        "outcome": record.outcome,
        "task_completed": record.outcome == "completed" and not missed and not false_holds and not unexpected,
        "holds": sorted(map(list, effective)),
        "missed_holds": sorted(map(list, missed)),
        "false_holds": sorted(map(list, false_holds)),
        "unexpected_holds": sorted(map(list, unexpected)),
        "dry_run_violation": scenario.dry_run and bool(queued),
        "guardrail_rejections": sum(1 for p in proposals if p["is_error"]),
        "blocked_attempts": sum(
            1 for p in proposals
            if not p["is_error"] and p["result"].get("queued") is False and not p["result"].get("dry_run")
        ),
        "ordering_violations": violations,
        "tool_errors": sum(1 for c in record.tool_calls if c.is_error),
        "tool_calls": len(record.tool_calls),
        "iterations": record.iterations,
        "cost_usd": record.usage.get("cost_usd", 0.0),
        "duration_s": record.duration_s,
        "golden": compare_golden(scenario.id, record, effective),
    }
