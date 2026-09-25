import random
from datetime import datetime

import pandas as pd
import pytest

from music_man.generator.fraud_patterns import PATTERN_GENERATORS

TRACK_ID = "track_test"
ARTIST_ID = "artist_test"
DURATION_MS = 200_000
WINDOW_START = datetime(2026, 1, 1)


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)


def _to_df(events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(events)


def test_burst_velocity_concentrates_in_a_short_window(rng):
    events = PATTERN_GENERATORS["burst_velocity"](TRACK_ID, ARTIST_ID, DURATION_MS, WINDOW_START, rng)
    df = _to_df(events)
    assert len(df) >= 1000
    span_minutes = (df["played_at"].max() - df["played_at"].min()).total_seconds() / 60
    assert span_minutes <= 6  # burst window is capped well under organic listening spread


def test_device_concentration_is_dominated_by_a_handful_of_devices(rng):
    events = PATTERN_GENERATORS["device_concentration"](TRACK_ID, ARTIST_ID, DURATION_MS, WINDOW_START, rng)
    df = _to_df(events)
    n_devices = df["device_id"].nunique()
    assert n_devices <= 4
    assert len(df) / n_devices > 100  # each device carries a large, unnatural share


def test_duration_regularity_has_near_zero_variance(rng):
    events = PATTERN_GENERATORS["duration_regularity"](TRACK_ID, ARTIST_ID, DURATION_MS, WINDOW_START, rng)
    df = _to_df(events)
    assert df["duration_ms_played"].std() < 50  # organic listening varies far more than this
    assert df["duration_ms_played"].mean() == pytest.approx(DURATION_MS * 0.98, rel=0.01)


def test_geo_impossible_pairs_share_identity_but_differ_in_location(rng):
    events = PATTERN_GENERATORS["geo_impossible"](TRACK_ID, ARTIST_ID, DURATION_MS, WINDOW_START, rng)
    df = _to_df(events)
    assert len(df) % 2 == 0  # generated in (t1, t2) pairs

    for user_id, group in df.groupby("user_id"):
        assert len(group) == 2
        prefixes = set(ip.rsplit(".", 2)[0] for ip in group["ip_address"])
        assert len(prefixes) == 2  # same user, two distinct network locations
        gap_minutes = (group["played_at"].max() - group["played_at"].min()).total_seconds() / 60
        assert gap_minutes < 10  # too fast to be real travel
