"""
Schema/row-count sanity checks for the bronze -> silver -> gold pipeline on
a small synthetic sample. Uses monkeypatched data directories (a tmp_path)
so this never touches the project's real data/{bronze,silver,gold} output.
"""

from __future__ import annotations

import pandas as pd
import pytest

from music_man.generator.events import generate_dataset
from music_man.generator.fraud_patterns import FRAUD_PATTERNS
from music_man.spark_session import get_spark


@pytest.fixture(scope="module")
def spark():
    session = get_spark("pytest")
    yield session
    session.stop()


@pytest.fixture
def tiny_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "artist_id": [f"artist_{i}" for i in range(5)],
            "artist_name": [f"Test Artist {i}" for i in range(5)],
            "track_id": [f"track_{i}" for i in range(5)],
            "track_name": [f"Test Track {i}" for i in range(5)],
            "album_name": ["Test Album"] * 5,
            "genre": ["test-genre"] * 5,
            "popularity_score": [50, 60, 70, 80, 90],
            "duration_ms": [200_000] * 5,
        }
    )


@pytest.fixture
def tiny_users() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(50)],
            "country": ["US"] * 50,
            "signup_date": ["2025-01-01"] * 50,
            "plan_tier": ["free"] * 50,
        }
    )


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr("music_man.pipeline.bronze.BRONZE_DIR", tmp_path / "bronze")
    monkeypatch.setattr("music_man.pipeline.silver.SILVER_DIR", tmp_path / "silver")
    monkeypatch.setattr("music_man.pipeline.gold.GOLD_DIR", tmp_path / "gold")


def test_bronze_to_gold_row_counts_and_schema(spark, tiny_catalog, tiny_users, monkeypatch, tmp_path):
    from music_man.pipeline.bronze import write_bronze
    from music_man.pipeline.gold import build_artist_daily_metrics, build_royalty_payouts
    from music_man.pipeline.silver import build_silver

    _patch_dirs(monkeypatch, tmp_path)

    events, fraud_labels = generate_dataset(
        n_organic_events=2000,
        n_fraud_cases=3,
        start_date="2026-01-01",
        end_date="2026-01-15",
        catalog=tiny_catalog,
        users=tiny_users,
    )
    assert set(fraud_labels["pattern"]).issubset(set(FRAUD_PATTERNS))

    write_bronze(spark, tiny_catalog, tiny_users, events, fraud_labels)

    clean = build_silver(spark)
    assert clean.count() > 0
    for col in ["session_id", "is_full_play", "geo_impossible_flag"]:
        assert col in clean.columns

    metrics = build_artist_daily_metrics(spark)
    assert metrics.count() > 0
    for col in [
        "device_concentration_ratio",
        "play_velocity_max",
        "duration_variance",
        "geo_impossible_count",
        "royalty_accrued",
    ]:
        assert col in metrics.columns

    payouts = build_royalty_payouts(spark)
    assert payouts.count() > 0
    assert {"artist_id", "period", "amount_due"}.issubset(set(payouts.columns))
