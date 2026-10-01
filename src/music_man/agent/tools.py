"""
Claude tool definitions for the Music Man fraud-investigation agent.

Reads happen against a Parquet snapshot of the gold artist_daily_metrics
table (fast, no Spark/JVM needed for serving - see train_model.py's
write_scored_gold). propose_hold is the only tool with a side effect, and
that side effect is only queuing a pending row - it never holds a real
payout.

Guardrails are enforced here in code, not left to the prompt:
- propose_hold refuses unless check_hold_policy was called for the same
  artist/period earlier in this run, and the artist/period exists in the
  snapshot.
- In dry-run mode propose_hold reports what it would queue and writes
  nothing.
- Retried propose_hold calls are idempotent (see queue.create_hold).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from anthropic import beta_tool
from anthropic.lib.tools import ToolError

from music_man.agent.queue import create_hold, get_active_hold
from music_man.paths import GOLD_DIR

_DEFAULT_SNAPSHOT_PATH = GOLD_DIR / "artist_daily_metrics.snapshot.parquet"


def snapshot_path() -> Path:
    """Snapshot location; MUSIC_MAN_SNAPSHOT overrides it (evals point this at a fixture)."""
    override = os.environ.get("MUSIC_MAN_SNAPSHOT")
    return Path(override) if override else _DEFAULT_SNAPSHOT_PATH


@dataclass
class RunContext:
    """Per-run state the guardrails depend on. Reset by run_agent at the start of each run."""

    run_id: str = "adhoc"
    dry_run: bool = False
    policy_checked: set[tuple[str, str]] = field(default_factory=set)


_context = RunContext()


def start_run(run_id: str, dry_run: bool = False) -> RunContext:
    global _context
    _context = RunContext(run_id=run_id, dry_run=dry_run)
    return _context


# The snapshot has ~1M rows; reading it on every tool call dominated
# tool latency. Cache it per (path, mtime) so a retrained snapshot is
# still picked up.
_snapshot_cache: dict[tuple[str, float], pd.DataFrame] = {}


def _load_snapshot() -> pd.DataFrame:
    path = snapshot_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the pipeline and train_model.py first.")
    key = (str(path), path.stat().st_mtime)
    if key not in _snapshot_cache:
        _snapshot_cache.clear()
        df = pd.read_parquet(path)
        df["date"] = df["date"].astype(str)
        _snapshot_cache[key] = df
    return _snapshot_cache[key]


def _most_suspicious_day_per_artist(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses per-artist/day rows to each artist's single worst day."""
    idx = df.groupby("artist_id")["anomaly_score"].idxmax()
    return df.loc[idx].sort_values("anomaly_score", ascending=False)


def _validate_period(period: str) -> None:
    try:
        date.fromisoformat(period)
    except ValueError as exc:
        raise ToolError(f"period must be an ISO date like 2026-02-14, got {period!r}") from exc


def _require_artist_period(artist_id: str, period: str) -> None:
    df = _load_snapshot()
    if not ((df["artist_id"] == artist_id) & (df["date"] == period)).any():
        raise ToolError(
            f"no activity for artist {artist_id} on {period} in the gold snapshot; "
            "holds must reference an artist/day you have inspected"
        )


@beta_tool
def list_flagged_artists(min_anomaly_score: float = 0.5, limit: int = 20) -> str:
    """List artists whose worst day's anomaly score is at least min_anomaly_score, sorted highest first.

    Call this first to see who might be running fraudulent streaming. Each
    result is that artist's single most suspicious day, not every day
    they've ever had activity.

    Args:
        min_anomaly_score: Minimum anomaly score (0-1) to include. Defaults to 0.5.
        limit: Maximum number of artists to return. Defaults to 20.
    """
    df = _load_snapshot()
    worst = _most_suspicious_day_per_artist(df)
    flagged = worst[worst["anomaly_score"] >= min_anomaly_score].head(max(1, min(limit, 100)))
    columns = ["artist_id", "artist_name", "date", "anomaly_score", "total_plays", "royalty_accrued"]
    return flagged[columns].to_json(orient="records")


@beta_tool
def get_artist_detail(artist_id: str) -> str:
    """Get the full fraud-signal feature breakdown for an artist's single most suspicious day.

    Call this before proposing a hold so the rationale can cite the
    specific signals that fired (device concentration, burst velocity,
    duration variance, geo-impossible events).

    Args:
        artist_id: The artist's canonical ID (e.g. "artist_ab12cd34ef56").
    """
    df = _load_snapshot()
    rows = df[df["artist_id"] == artist_id]
    if rows.empty:
        raise ToolError(f"no data for artist_id {artist_id}")
    worst = rows.loc[rows["anomaly_score"].idxmax()]
    return worst.to_json()


@beta_tool
def check_hold_policy(artist_id: str, period: str) -> str:
    """Check whether a new royalty hold is allowed for an artist/period right now.

    Policy: at most one active (pending or executed) hold per artist per
    period. propose_hold will refuse unless this was called first for the
    same artist/period.

    Args:
        artist_id: The artist's canonical ID.
        period: The royalty period this hold would apply to, as an ISO date (e.g. "2026-02-14").
    """
    _validate_period(period)
    _context.policy_checked.add((artist_id, period))
    existing = get_active_hold(artist_id, period)
    return json.dumps(
        {
            "allowed": existing is None,
            "reason": (
                f"artist {artist_id} already has a {existing.status} hold (id {existing.id}) for period {period}"
                if existing
                else None
            ),
        }
    )


@beta_tool
def propose_hold(artist_id: str, period: str, rationale: str) -> str:
    """Queue a proposed royalty-payout hold for human review.

    This NEVER withholds a real payment - it only creates a pending record
    that a human must approve in the Streamlit reviewer before anything
    happens. check_hold_policy must be called first for the same
    artist/period. Safe to retry: repeating the same proposal returns the
    already-queued hold rather than creating a duplicate.

    Args:
        artist_id: The artist's canonical ID.
        period: The royalty period this hold applies to, as an ISO date (e.g. "2026-02-14").
        rationale: Why this artist/period, citing the specific fraud signals that fired.
    """
    _validate_period(period)
    if not rationale.strip():
        raise ToolError("rationale is required and must cite the fraud signals that fired")
    if (artist_id, period) not in _context.policy_checked:
        raise ToolError(
            f"call check_hold_policy for artist {artist_id} / period {period} before proposing a hold"
        )
    _require_artist_period(artist_id, period)

    existing = get_active_hold(artist_id, period)
    if existing is not None and existing.run_id != _context.run_id:
        return json.dumps(
            {
                "queued": False,
                "reason": f"blocked by policy: artist {artist_id} already has a {existing.status} hold for period {period}",
            }
        )

    if _context.dry_run:
        return json.dumps(
            {
                "queued": False,
                "dry_run": True,
                "would_queue": {"artist_id": artist_id, "period": period, "rationale": rationale},
                "note": "Dry run: nothing was written to the approval queue.",
            }
        )

    hold, created = create_hold(artist_id, period, rationale, run_id=_context.run_id)
    return json.dumps(
        {
            "queued": True,
            "hold_id": hold.id,
            "deduplicated": not created,
            "note": "Queued for human review only. No payout has been held yet.",
        }
    )


TOOLS = [list_flagged_artists, get_artist_detail, check_hold_policy, propose_hold]
