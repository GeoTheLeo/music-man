"""
Synthetic play-event stream generator: organic listening behavior plus a
small number of deliberately injected royalty-fraud cases.
"""

from __future__ import annotations

import random
import uuid

import numpy as np
import pandas as pd

from music_man.generator.catalog import load_catalog
from music_man.generator.fraud_patterns import FRAUD_PATTERNS, PATTERN_GENERATORS
from music_man.generator.users import load_users

EVENT_COLUMNS = [
    "event_id",
    "user_id",
    "track_id",
    "artist_id",
    "played_at",
    "device_id",
    "ip_address",
    "duration_ms_played",
    "event_type",
]


def generate_organic_events(
    catalog: pd.DataFrame,
    users: pd.DataFrame,
    start_date: str,
    end_date: str,
    n_events: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Vectorized generation of ordinary (non-fraudulent) listening events."""
    rng = np.random.default_rng(seed)
    n_tracks, n_users = len(catalog), len(users)

    weights = catalog["popularity_score"].to_numpy(dtype=float) + 1.0
    weights = weights / weights.sum()
    track_idx = rng.choice(n_tracks, size=n_events, p=weights)
    user_idx = rng.integers(0, n_users, size=n_events)

    start_ts = pd.Timestamp(start_date).value // 10**9
    end_ts = pd.Timestamp(end_date).value // 10**9
    played_at = pd.to_datetime(rng.integers(start_ts, end_ts, size=n_events, endpoint=False), unit="s")

    track_ids = catalog["track_id"].to_numpy()[track_idx]
    artist_ids = catalog["artist_id"].to_numpy()[track_idx]
    durations_full = catalog["duration_ms"].to_numpy()[track_idx]
    user_ids = users["user_id"].to_numpy()[user_idx]

    # Each user has a small, stable set of (at most 2) devices.
    device_suffix = rng.integers(0, 2, size=n_events)
    device_ids = np.char.add(np.char.add(user_ids.astype(str), "-dev"), device_suffix.astype(str))

    # Organic listening skews toward full plays but with real variance -
    # the opposite of the near-zero-variance bot signature.
    play_fraction = np.clip(rng.beta(5, 2, size=n_events), 0.05, 1.0)
    duration_ms_played = (durations_full * play_fraction).astype(int)

    # A simple, stable-ish per-user IP block (organic users don't jump
    # continents mid-session, unlike the injected geo_impossible pattern).
    ip_block = (user_idx % 250) + 1
    ip_host = rng.integers(1, 254, size=n_events)
    ip_addresses = np.array([f"192.168.{b}.{h}" for b, h in zip(ip_block, ip_host)])

    event_ids = np.array([uuid.uuid4().hex for _ in range(n_events)])
    event_type = np.where(play_fraction < 0.2, "skip", "play")

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "user_id": user_ids,
            "track_id": track_ids,
            "artist_id": artist_ids,
            "played_at": played_at,
            "device_id": device_ids,
            "ip_address": ip_addresses,
            "duration_ms_played": duration_ms_played,
            "event_type": event_type,
        }
    )


def inject_fraud_events(
    catalog: pd.DataFrame,
    n_cases: int,
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Picks `n_cases` random (artist, track) targets, applies one fraud
    pattern to each, and returns (fraud_events, fraud_labels). fraud_labels
    is the ground truth used to train/evaluate the model - ["artist_id",
    "track_id", "period", "pattern"].
    """
    rng = random.Random(seed)
    targets = catalog.sample(n=n_cases, random_state=seed).reset_index(drop=True)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) - pd.Timedelta(days=2)
    span_seconds = int((end_ts - start_ts).total_seconds())

    all_events: list[dict] = []
    labels: list[dict] = []

    for _, row in targets.iterrows():
        pattern = rng.choice(FRAUD_PATTERNS)
        window_start = start_ts + pd.Timedelta(seconds=rng.randint(0, span_seconds))
        generator = PATTERN_GENERATORS[pattern]
        events = generator(row["track_id"], row["artist_id"], int(row["duration_ms"]), window_start, rng)
        all_events.extend(events)
        labels.append(
            {
                "artist_id": row["artist_id"],
                "track_id": row["track_id"],
                "period": window_start.date().isoformat(),
                "pattern": pattern,
            }
        )

    fraud_df = pd.DataFrame(all_events, columns=EVENT_COLUMNS)
    labels_df = pd.DataFrame(labels)
    return fraud_df, labels_df


def generate_dataset(
    n_organic_events: int = 2_000_000,
    n_fraud_cases: int = 40,
    start_date: str = "2026-01-01",
    end_date: str = "2026-03-31",
    seed: int = 42,
    catalog: pd.DataFrame | None = None,
    users: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (events_df, fraud_labels_df). Pass catalog/users explicitly in
    tests to avoid depending on the real Kaggle-backed catalog.
    """
    if catalog is None:
        catalog = load_catalog()
    if users is None:
        users = load_users()

    organic = generate_organic_events(catalog, users, start_date, end_date, n_organic_events, seed)
    fraud_events, fraud_labels = inject_fraud_events(catalog, n_fraud_cases, start_date, end_date, seed)

    events = pd.concat([organic, fraud_events], ignore_index=True)
    return events, fraud_labels


if __name__ == "__main__":
    events_df, labels_df = generate_dataset(n_organic_events=200_000, n_fraud_cases=20)
    print(f"Events: {len(events_df):,} ({labels_df.shape[0]} fraud cases injected)")
    print(events_df.head())
    print(labels_df.head())
