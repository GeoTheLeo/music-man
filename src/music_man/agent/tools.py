"""
Claude tool definitions for the Music Man fraud-investigation agent.

Reads happen against a Parquet snapshot of the gold artist_daily_metrics
table (fast, no Spark/JVM needed for serving - see train_model.py's
write_scored_gold). propose_hold is the only tool with a side effect, and
that side effect is only queuing a pending row - it never holds a real
payout.
"""

from __future__ import annotations

import json

import pandas as pd
from anthropic import beta_tool

from music_man.agent.queue import create_hold, has_hold_for_period
from music_man.paths import GOLD_DIR

_SNAPSHOT_PATH = GOLD_DIR / "artist_daily_metrics.snapshot.parquet"


def _load_snapshot() -> pd.DataFrame:
    if not _SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"{_SNAPSHOT_PATH} not found - run the pipeline and train_model.py first."
        )
    return pd.read_parquet(_SNAPSHOT_PATH)


def _most_suspicious_day_per_artist(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses per-artist/day rows to each artist's single worst day."""
    idx = df.groupby("artist_id")["anomaly_score"].idxmax()
    return df.loc[idx].sort_values("anomaly_score", ascending=False)


@beta_tool
def list_flagged_artists(min_anomaly_score: float = 0.5) -> str:
    """List artists whose worst day's anomaly score is at least min_anomaly_score, sorted highest first.

    Call this first to see who might be running fraudulent streaming. Each
    result is that artist's single most suspicious day, not every day
    they've ever had activity.

    Args:
        min_anomaly_score: Minimum anomaly score (0-1) to include. Defaults to 0.5.
    """
    df = _load_snapshot()
    worst = _most_suspicious_day_per_artist(df)
    flagged = worst[worst["anomaly_score"] >= min_anomaly_score]
    columns = ["artist_id", "artist_name", "date", "anomaly_score", "total_plays", "royalty_accrued"]
    return flagged[columns].to_json(orient="records", date_format="iso")


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
        return json.dumps({"error": f"no data for artist_id {artist_id}"})
    worst = rows.loc[rows["anomaly_score"].idxmax()]
    return worst.to_json(date_format="iso")


@beta_tool
def check_hold_policy(artist_id: str, period: str) -> str:
    """Check whether a new royalty hold is allowed for an artist/period right now.

    Policy: at most one active (pending or executed) hold per artist per
    period. Call this before propose_hold to avoid proposing a hold that
    will be rejected.

    Args:
        artist_id: The artist's canonical ID.
        period: The royalty period this hold would apply to, as an ISO date (e.g. "2026-02-14").
    """
    blocked = has_hold_for_period(artist_id, period)
    return json.dumps(
        {
            "allowed": not blocked,
            "reason": (
                f"artist {artist_id} already has a pending or executed hold for period {period}"
                if blocked
                else None
            ),
        }
    )


@beta_tool
def propose_hold(artist_id: str, period: str, rationale: str) -> str:
    """Queue a proposed royalty-payout hold for human review.

    This NEVER withholds a real payment - it only creates a pending record
    that a human must approve in the Streamlit reviewer before anything
    happens. Always call check_hold_policy first.

    Args:
        artist_id: The artist's canonical ID.
        period: The royalty period this hold applies to, as an ISO date (e.g. "2026-02-14").
        rationale: Why this artist/period, citing the specific fraud signals that fired.
    """
    if has_hold_for_period(artist_id, period):
        return json.dumps(
            {
                "queued": False,
                "reason": f"blocked by policy: artist {artist_id} already has a hold for period {period}",
            }
        )
    hold = create_hold(artist_id, period, rationale)
    return json.dumps(
        {
            "queued": True,
            "hold_id": hold.id,
            "note": "Queued for human review only. No payout has been held yet.",
        }
    )


TOOLS = [list_flagged_artists, get_artist_detail, check_hold_policy, propose_hold]
