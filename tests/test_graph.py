"""
Fraud-ring graph: generator labels, graph features, the find_fraud_ring
guardrails, and Neo4j ring detection.

The Neo4j tests need a disposable database because they wipe it. They run
only when NEO4J_TEST_URI is set, e.g. against a throwaway container:

    docker run -d --name music-man-neo4j-test -p 7688:7687 \
        -e NEO4J_AUTH=neo4j/test-password -e 'NEO4J_PLUGINS=["graph-data-science"]' neo4j:5.26-community
    NEO4J_TEST_URI=bolt://localhost:7688 NEO4J_TEST_PASSWORD=test-password pytest tests/test_graph.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from music_man.agent import operations
from music_man.agent.operations import GuardrailError
from music_man.generator.events import inject_fraud_events

# --------------------------------------------------------------- labels


def test_fraud_labels_cover_every_day_an_attack_touches():
    catalog = pd.DataFrame(
        {
            "artist_id": [f"artist_{i}" for i in range(30)],
            "track_id": [f"track_{i}" for i in range(30)],
            "duration_ms": [200_000] * 30,
        }
    )
    events, labels = inject_fraud_events(catalog, n_cases=30, start_date="2026-01-01", end_date="2026-03-31", seed=7)

    event_days = set(zip(events["artist_id"], events["played_at"].dt.date.astype(str)))
    label_days = set(zip(labels["artist_id"], labels["period"]))

    assert event_days == label_days
    assert labels["case_id"].nunique() == 30
    assert len(labels) > 30  # some attacks start late and spill past midnight


# --------------------------------------------------------------- tool guardrails


def test_find_fraud_ring_degrades_gracefully_when_graph_is_down(monkeypatch):
    from music_man.graph import store

    def unavailable():
        raise ConnectionError("no neo4j")

    monkeypatch.setattr(store, "get_driver", unavailable)
    with pytest.raises(GuardrailError, match="fraud graph unavailable"):
        operations.find_fraud_ring("artist_x")


def test_find_fraud_ring_rejects_a_bad_period():
    with pytest.raises(GuardrailError, match="ISO date"):
        operations.find_fraud_ring("artist_x", "Feb 14")


def test_find_fraud_ring_is_an_operation_on_every_surface():
    from music_man.agent import tools

    assert operations.find_fraud_ring in operations.OPERATIONS
    assert tools.find_fraud_ring in tools.TOOLS


# --------------------------------------------------------------- graph features (Spark)


@pytest.fixture(scope="module")
def spark():
    from music_man.spark_session import get_spark

    session = get_spark("pytest-graph")
    yield session
    session.stop()


def test_graph_features_measure_ring_share_size_and_reach(spark):
    from music_man.graph.build_graph import build_graph_features

    t = pd.Timestamp("2026-02-01 10:00")
    events = pd.DataFrame(
        [
            # bot ring: two accounts sharing one device, streaming two artists
            ("bot_a", "dev_shared", "artist_target", t),
            ("bot_b", "dev_shared", "artist_target", t),
            ("bot_a", "dev_shared", "artist_other", t),
            # organic listener of the same artist
            ("fan", "fan-dev0", "artist_target", t),
            # an artist with only organic plays
            ("fan", "fan-dev0", "artist_clean", t),
        ],
        columns=["user_id", "device_id", "artist_id", "played_at"],
    )
    rings = pd.DataFrame(
        {
            "user_id": ["bot_a", "bot_b", "fan"],
            "ring_id": [1, 1, 2],
            "ring_accounts": [2, 2, 1],
            "ring_devices": [1, 1, 1],
        }
    )

    features = build_graph_features(spark, spark.createDataFrame(events), rings).toPandas().set_index("artist_id")

    assert features.loc["artist_target", "ring_play_share"] == pytest.approx(2 / 3)
    assert features.loc["artist_target", "max_ring_accounts"] == 2
    assert features.loc["artist_target", "max_ring_artist_reach"] == 2  # the ring streamed two artists
    assert features.loc["artist_clean", "ring_play_share"] == 0
    assert features.loc["artist_clean", "max_ring_artist_reach"] == 0


# --------------------------------------------------------------- Neo4j integration

NEO4J_TEST_URI = os.environ.get("NEO4J_TEST_URI")
needs_neo4j = pytest.mark.skipif(not NEO4J_TEST_URI, reason="set NEO4J_TEST_URI to a disposable Neo4j")


@pytest.fixture
def driver(monkeypatch):
    from music_man.graph import store

    monkeypatch.setenv("NEO4J_URI", NEO4J_TEST_URI)
    monkeypatch.setenv("NEO4J_PASSWORD", os.environ.get("NEO4J_TEST_PASSWORD", "test-password"))
    drv = store.get_driver()
    store.reset(drv)
    store.create_constraints(drv)
    yield drv
    store.reset(drv)
    drv.close()


@needs_neo4j
def test_rings_and_find_fraud_ring_against_neo4j(driver):
    from music_man.graph import store

    account_devices = pd.DataFrame(
        [
            ("bot_1", "farm_dev_1", 100),
            ("bot_2", "farm_dev_1", 100),
            ("bot_2", "farm_dev_2", 100),
            ("bot_3", "farm_dev_2", 100),
            ("fan", "fan-dev0", 10),
        ],
        columns=["user_id", "device_id", "plays"],
    )
    store.load_account_devices(driver, account_devices)
    store.detect_rings(driver)

    rings = store.account_rings(driver).set_index("user_id")
    assert rings.loc["bot_1", "ring_id"] == rings.loc["bot_3", "ring_id"]  # linked through bot_2
    assert rings.loc["bot_1", "ring_accounts"] == 3
    assert rings.loc["fan", "ring_accounts"] == 1

    streams = pd.DataFrame(
        [
            ("bot_1", "artist_target", "Target", "2026-02-01", 500),
            ("bot_3", "artist_target", "Target", "2026-02-01", 400),
            ("bot_2", "artist_other", "Other", "2026-02-02", 300),
        ],
        columns=["user_id", "artist_id", "artist_name", "date", "plays"],
    )
    store.load_streams(driver, streams)

    found = json.loads(operations.find_fraud_ring("artist_target", "2026-02-01"))
    ring = found["rings"][0]
    assert ring["ring_accounts"] == 3
    assert ring["plays_on_artist"] == 900
    assert ring["multi_account_devices"] == 2
    assert [a["artist_id"] for a in ring["top_other_artists"]] == ["artist_other"]

    clean = json.loads(operations.find_fraud_ring("artist_nobody"))
    assert clean["rings"] == []
