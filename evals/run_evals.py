"""
Runs the agent against every scenario x routing config x trial, grades
each run, and writes a JSON + Markdown report to evals/reports/.

Every run calls the Claude API and costs real money, so this prints the
plan and exits unless --yes is passed. --max-cost stops the sweep once
cumulative spend (agent + judge) crosses the cap.

    python -m evals.run_evals                       # show the plan only
    python -m evals.run_evals --yes --trials 2      # full sweep
    python -m evals.run_evals --yes --scenario dry_run --config routed --trials 1
    python -m evals.run_evals --yes --record-golden # save passing runs as golden transcripts
    python -m evals.run_evals --yes --config routed --engine custom --engine langgraph  # engine comparison
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from evals.fixtures import write_snapshot
from evals.grading import grade, record_golden
from evals.judge import judge_run
from evals.scenarios import SCENARIOS, SCENARIOS_BY_ID
from music_man.agent.cost import UsageTracker
from music_man.agent.queue import list_holds
from music_man.agent.run_agent import EXEC_MODEL, PLAN_MODEL, make_client, run_agent
from music_man.agent_graph.run import run_graph

CONFIGS = {
    "routed": {"plan_model": PLAN_MODEL, "exec_model": EXEC_MODEL},
    "opus-only": {"plan_model": EXEC_MODEL, "exec_model": EXEC_MODEL},
}
REPORTS_DIR = Path(__file__).parent / "reports"
ENGINES = ("custom", "langgraph")
EST_COST_PER_RUN_USD = 0.15  # rough planning figure; the report records the real cost


def _run_trial(client, scenario, engine, config_name, trial, workdir: Path, use_judge: bool, judge_tracker):
    os.environ["MUSIC_MAN_QUEUE_DB"] = str(workdir / "queue.db")
    os.environ["MUSIC_MAN_SNAPSHOT"] = str(write_snapshot(workdir / "snapshot.parquet"))
    if scenario.setup:
        scenario.setup()

    if engine == "langgraph":
        # Stops at the human-review interrupt; the grader reads the pending holds it queued.
        record, _ = run_graph(scenario.prompt, dry_run=scenario.dry_run, verbose=False, save=False,
                              **CONFIGS[config_name])
    else:
        record = run_agent(scenario.prompt, dry_run=scenario.dry_run, client=client, verbose=False,
                           save=False, **CONFIGS[config_name])
    result = grade(record, scenario, list_holds())
    if use_judge and record.outcome == "completed":
        try:
            verdict = judge_run(client, record, judge_tracker)
            result["judge"] = verdict.model_dump()
        except Exception as exc:  # a judge failure shouldn't discard the graded run
            result["judge_error"] = f"{exc.__class__.__name__}: {exc}"
    return record, result


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict]) -> dict:
    by_config: dict[str, dict] = {}
    for config in sorted({r["config"] for r in results}):
        rs = [r for r in results if r["config"] == config]
        rationales = [v for r in rs for v in (r.get("judge") or {}).get("rationales", [])]
        judged = [r for r in rs if r.get("judge")]
        by_config[config] = {
            "runs": len(rs),
            "task_completion_rate": _rate([r["task_completed"] for r in rs]),
            "false_hold_rate": _rate([bool(r["false_holds"]) for r in rs]),
            "hallucination_rate": _rate([not v["grounded"] for v in rationales]),
            "claimed_execution_rate": _rate([r["judge"]["claims_payment_already_withheld"] for r in judged]),
            "followed_injection_rate": _rate([r["judge"]["followed_injected_instructions"] for r in judged]),
            "dry_run_violations": sum(r["dry_run_violation"] for r in rs),
            "ordering_violations": sum(len(r["ordering_violations"]) for r in rs),
            "errors_or_budget_exhausted": sum(r["outcome"] != "completed" for r in rs),
            "mean_cost_usd": round(statistics.mean(r["cost_usd"] for r in rs), 4),
            "mean_duration_s": round(statistics.mean(r["duration_s"] for r in rs), 1),
            "mean_iterations": round(statistics.mean(r["iterations"] for r in rs), 1),
        }
    return by_config


def to_markdown(summary: dict, results: list[dict], judge_cost: float) -> str:
    metrics = list(next(iter(summary.values())).keys())
    configs = list(summary)
    lines = ["# Music Man agent eval report", "", "| metric | " + " | ".join(configs) + " |",
             "|---|" + "---|" * len(configs)]
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(str(summary[c][m]) for c in configs) + " |")
    lines += ["", f"Judge cost: ${judge_cost:.4f}", "", "## Runs", "",
              "| config | scenario | trial | completed | false holds | missed | cost | notes |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        notes = []
        if r["ordering_violations"]:
            notes.append(f"{len(r['ordering_violations'])} ordering violations")
        if r["dry_run_violation"]:
            notes.append("DRY-RUN VIOLATION")
        judge = r.get("judge") or {}
        ungrounded = [v["artist_id"] for v in judge.get("rationales", []) if not v["grounded"]]
        if ungrounded:
            notes.append(f"ungrounded: {', '.join(ungrounded)}")
        if judge.get("followed_injected_instructions"):
            notes.append("followed injection")
        if r["outcome"] != "completed":
            notes.append(r["outcome"])
        lines.append(
            f"| {r['config']} | {r['scenario']} | {r['trial']} | {'yes' if r['task_completed'] else 'NO'} "
            f"| {len(r['false_holds'])} | {len(r['missed_holds'])} | ${r['cost_usd']:.3f} | {'; '.join(notes)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Man agent evals")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS_BY_ID), help="default: all")
    parser.add_argument("--config", action="append", choices=sorted(CONFIGS), help="default: all")
    parser.add_argument("--engine", action="append", choices=ENGINES, help="default: custom")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--record-golden", action="store_true", help="save passing runs as golden transcripts")
    parser.add_argument("--max-cost", type=float, default=10.0, help="stop once spend (USD) exceeds this")
    parser.add_argument("--yes", action="store_true", help="actually run (calls the API, costs money)")
    args = parser.parse_args()

    scenarios = [SCENARIOS_BY_ID[s] for s in args.scenario] if args.scenario else SCENARIOS
    configs = args.config or list(CONFIGS)
    engines = args.engine or ["custom"]
    planned = len(engines) * len(scenarios) * len(configs) * args.trials
    print(f"Plan: {len(engines)} engines x {len(scenarios)} scenarios x {len(configs)} configs x "
          f"{args.trials} trials = {planned} agent runs")
    print(f"Rough estimate: ~${planned * EST_COST_PER_RUN_USD:.2f} (cap: ${args.max_cost:.2f})")
    if not args.yes:
        print("Nothing run. Re-run with --yes to call the API.")
        return

    client = make_client()
    judge_tracker = UsageTracker()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = REPORTS_DIR / stamp
    runs_dir = out_dir / "runs"
    results: list[dict] = []
    spent = 0.0

    for engine, config in [(e, c) for e in engines for c in configs]:
        for scenario in scenarios:
            for trial in range(1, args.trials + 1):
                if spent >= args.max_cost:
                    print(f"[stop] spend ${spent:.2f} reached the ${args.max_cost:.2f} cap")
                    break
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                    record, result = _run_trial(
                        client, scenario, engine, config, trial, Path(tmp), not args.no_judge, judge_tracker
                    )
                record.save(runs_dir)
                label = config if engine == "custom" else f"{engine}/{config}"
                result.update({"config": label, "engine": engine, "scenario": scenario.id, "trial": trial,
                               "run_id": record.run_id})
                results.append(result)
                spent = sum(r["cost_usd"] for r in results) + judge_tracker.total_cost
                status = "PASS" if result["task_completed"] else "FAIL"
                print(f"[{status}] {label:19s} {scenario.id:17s} #{trial}  ${result['cost_usd']:.3f}  "
                      f"{result['duration_s']:.0f}s  (total ${spent:.2f})")
                if args.record_golden and engine == "custom" and result["task_completed"] and not result.get("judge_error"):
                    effective = {tuple(h) for h in result["holds"]}
                    record_golden(scenario.id, record, effective)

    if not results:
        return
    summary = summarize(results)
    report = {"generated_at": stamp, "summary": summary, "judge_cost_usd": round(judge_tracker.total_cost, 6),
              "total_cost_usd": round(spent, 6), "results": results}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "report.md").write_text(to_markdown(summary, results, judge_tracker.total_cost), encoding="utf-8")
    print(f"\nReport: {out_dir / 'report.md'}  (total ${spent:.2f})")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if all(r["task_completed"] for r in results) else 1)


if __name__ == "__main__":
    main()
