"""
Neo4j store for the fraud-ring graph.

Graph model:
    (:Account {id, ringId, ringAccounts, ringDevices})-[:USES {plays}]->(:Device {id})
    (:Account)-[:STREAMED {date, plays}]->(:Artist {id, name})

Accounts and devices form the ring graph: two accounts are in the same ring
when they share a device, directly or through a chain of shared devices.
Rings are found with Graph Data Science weakly connected components (WCC).
STREAMED edges are loaded only for accounts in suspicious rings, which is
what the agent's find_fraud_ring tool needs; organic listening stays in
the lakehouse.

Connection settings come from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
(defaults match docker-compose.neo4j.yml).
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
from neo4j import Driver, GraphDatabase

BATCH_SIZE = 5_000

# An account's ring is suspicious if it shares devices with other accounts
# or uses far more devices than a person plausibly owns.
SUSPICIOUS_MIN_ACCOUNTS = 2
SUSPICIOUS_MIN_DEVICES = 5

GDS_GRAPH_NAME = "account_device"


def get_driver() -> Driver:
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "musicman-dev"),
        ),
    )


def _batches(rows: list[dict], size: int = BATCH_SIZE) -> Iterable[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def reset(driver: Driver) -> None:
    """Drop all graph data and the GDS projection, keeping constraints."""
    with driver.session() as session:
        session.run("CALL gds.graph.drop($name, false)", name=GDS_GRAPH_NAME).consume()
        session.run(
            "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
        ).consume()


def create_constraints(driver: Driver) -> None:
    with driver.session() as session:
        for label in ("Account", "Device", "Artist"):
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            ).consume()
        # Ring lookups (find_rings_for_artist) match accounts by ringId.
        session.run("CREATE INDEX account_ring IF NOT EXISTS FOR (a:Account) ON (a.ringId)").consume()


def load_account_devices(driver: Driver, account_devices: pd.DataFrame) -> int:
    """account_devices: columns user_id, device_id, plays."""
    rows = account_devices.rename(columns={"user_id": "account"}).to_dict("records")
    with driver.session() as session:
        for batch in _batches(rows):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (a:Account {id: row.account})
                MERGE (d:Device {id: row.device_id})
                MERGE (a)-[u:USES]->(d)
                SET u.plays = row.plays
                """,
                rows=batch,
            ).consume()
    return len(rows)


def detect_rings(driver: Driver) -> dict:
    """Run GDS weakly connected components and write ring stats onto accounts."""
    with driver.session() as session:
        session.run("CALL gds.graph.drop($name, false)", name=GDS_GRAPH_NAME).consume()
        session.run(
            """
            CALL gds.graph.project($name, ['Account', 'Device'],
                                   {USES: {orientation: 'UNDIRECTED'}})
            """,
            name=GDS_GRAPH_NAME,
        ).consume()
        wcc = session.run(
            "CALL gds.wcc.write($name, {writeProperty: 'ringId'}) "
            "YIELD componentCount, nodePropertiesWritten",
            name=GDS_GRAPH_NAME,
        ).single()
        session.run(
            """
            MATCH (n) WHERE n:Account OR n:Device
            WITH n.ringId AS ring,
                 count(CASE WHEN n:Account THEN 1 END) AS accounts,
                 count(CASE WHEN n:Device THEN 1 END) AS devices
            MATCH (a:Account {ringId: ring})
            SET a.ringAccounts = accounts, a.ringDevices = devices
            """
        ).consume()
        session.run("CALL gds.graph.drop($name, false)", name=GDS_GRAPH_NAME).consume()
    return {"rings": wcc["componentCount"], "nodes_labelled": wcc["nodePropertiesWritten"]}


def account_rings(driver: Driver) -> pd.DataFrame:
    """Ring membership for every account: user_id, ring_id, ring_accounts, ring_devices."""
    with driver.session() as session:
        records = session.run(
            """
            MATCH (a:Account)
            RETURN a.id AS user_id, a.ringId AS ring_id,
                   a.ringAccounts AS ring_accounts, a.ringDevices AS ring_devices
            """
        ).data()
    return pd.DataFrame(records, columns=["user_id", "ring_id", "ring_accounts", "ring_devices"])


def is_suspicious(ring_accounts: pd.Series, ring_devices: pd.Series) -> pd.Series:
    return (ring_accounts >= SUSPICIOUS_MIN_ACCOUNTS) | (ring_devices >= SUSPICIOUS_MIN_DEVICES)


def load_streams(driver: Driver, streams: pd.DataFrame) -> int:
    """streams: columns user_id, artist_id, artist_name, date (ISO str), plays."""
    rows = streams.rename(columns={"user_id": "account"}).to_dict("records")
    with driver.session() as session:
        for batch in _batches(rows):
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Account {id: row.account})
                MERGE (ar:Artist {id: row.artist_id})
                  ON CREATE SET ar.name = row.artist_name
                MERGE (a)-[s:STREAMED {date: row.date}]->(ar)
                SET s.plays = row.plays
                """,
                rows=batch,
            ).consume()
    return len(rows)


def graph_counts(driver: Driver) -> dict:
    with driver.session() as session:
        return session.run(
            """
            CALL () { MATCH (a:Account) RETURN count(a) AS accounts }
            CALL () { MATCH (d:Device) RETURN count(d) AS devices }
            CALL () { MATCH (ar:Artist) RETURN count(ar) AS artists }
            CALL () { MATCH ()-[u:USES]->() RETURN count(u) AS uses }
            CALL () { MATCH ()-[s:STREAMED]->() RETURN count(s) AS streamed }
            CALL () {
              MATCH (a:Account)
              WHERE a.ringAccounts >= $min_accounts OR a.ringDevices >= $min_devices
              RETURN count(DISTINCT a.ringId) AS suspicious_rings
            }
            RETURN accounts, devices, artists, uses, streamed, suspicious_rings
            """,
            min_accounts=SUSPICIOUS_MIN_ACCOUNTS,
            min_devices=SUSPICIOUS_MIN_DEVICES,
        ).single().data()


def find_rings_for_artist(driver: Driver, artist_id: str, date: str | None = None) -> list[dict]:
    """
    Suspicious rings that streamed this artist (optionally on one date), with
    the evidence an investigator needs: how many accounts and devices the ring
    links, how many of this artist's plays it produced, and which other
    artists the same ring streamed.
    """
    with driver.session() as session:
        return session.run(
            """
            MATCH (a:Account)-[s:STREAMED]->(:Artist {id: $artist_id})
            WHERE $date IS NULL OR s.date = $date
            WITH a.ringId AS ring, a.ringAccounts AS ring_accounts, a.ringDevices AS ring_devices,
                 sum(s.plays) AS plays_on_artist, count(DISTINCT a) AS accounts_on_artist
            CALL (ring) {
              MATCH (m:Account {ringId: ring})-[:USES]->(d:Device)
              WHERE COUNT { (d)<-[:USES]-(:Account) } > 1
              RETURN count(DISTINCT d) AS multi_account_devices
            }
            CALL (ring) {
              MATCH (m:Account {ringId: ring})-[o:STREAMED]->(other:Artist)
              WHERE other.id <> $artist_id
              WITH other, sum(o.plays) AS plays
              ORDER BY plays DESC
              RETURN count(other) AS other_artist_count,
                     collect({artist_id: other.id, name: other.name, plays: plays})[..5] AS top_other_artists
            }
            RETURN ring AS ring_id, ring_accounts, ring_devices, accounts_on_artist,
                   plays_on_artist, multi_account_devices, other_artist_count, top_other_artists
            ORDER BY plays_on_artist DESC
            LIMIT 5
            """,
            artist_id=artist_id,
            date=date,
        ).data()
