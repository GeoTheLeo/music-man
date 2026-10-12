"""
Builds the fraud-ring graph in Neo4j from the silver layer, detects rings
with GDS, and writes per-artist/day graph features back to the gold layer.

    silver play_events_clean
        -> (:Account)-[:USES]->(:Device) in Neo4j
        -> GDS weakly connected components = rings
        -> STREAMED edges for accounts in suspicious rings
        -> gold artist_daily_graph_features (Delta + Parquet)

Run after gold.py and before train_model.py:
    python -m music_man.graph.build_graph
"""

from __future__ import annotations

import json

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from music_man.graph import store
from music_man.paths import GOLD_DIR
from music_man.pipeline.bronze import read_bronze
from music_man.pipeline.silver import read_silver

GRAPH_FEATURE_COLUMNS = [
    "ring_play_share",
    "max_ring_accounts",
    "max_ring_devices",
    "max_ring_artist_reach",
]

GRAPH_FEATURES_PARQUET = GOLD_DIR / "artist_daily_graph_features.parquet"
GRAPH_SUMMARY_JSON = GOLD_DIR / "graph_summary.json"


def account_devices(events: DataFrame) -> pd.DataFrame:
    return (
        events.groupBy("user_id", "device_id")
        .agg(F.count("*").cast("int").alias("plays"))
        .toPandas()
    )


def suspicious_streams(spark: SparkSession, events: DataFrame, rings: pd.DataFrame) -> pd.DataFrame:
    suspicious = rings[store.is_suspicious(rings["ring_accounts"], rings["ring_devices"])][["user_id"]]
    if suspicious.empty:
        return pd.DataFrame(columns=["user_id", "artist_id", "artist_name", "date", "plays"])
    names = read_bronze(spark, "catalog").select("artist_id", "artist_name").dropDuplicates(["artist_id"])
    return (
        events.join(spark.createDataFrame(suspicious), on="user_id")
        .withColumn("date", F.date_format("played_at", "yyyy-MM-dd"))
        .groupBy("user_id", "artist_id", "date")
        .agg(F.count("*").cast("int").alias("plays"))
        .join(names, on="artist_id", how="left")
        .toPandas()
    )


def build_graph_features(spark: SparkSession, events: DataFrame, rings: pd.DataFrame) -> DataFrame:
    """
    Per artist/day:
    - ring_play_share: share of plays from accounts in suspicious rings
    - max_ring_accounts / max_ring_devices: size of the largest ring among the day's listeners
    - max_ring_artist_reach: most distinct artists any suspicious ring listening that day streamed overall
    """
    ring_df = spark.createDataFrame(rings).withColumn(
        "suspicious",
        (F.col("ring_accounts") >= store.SUSPICIOUS_MIN_ACCOUNTS)
        | (F.col("ring_devices") >= store.SUSPICIOUS_MIN_DEVICES),
    )
    tagged = events.join(ring_df, on="user_id", how="left").withColumn("date", F.to_date("played_at"))

    reach = (
        tagged.filter("suspicious")
        .groupBy("ring_id")
        .agg(F.countDistinct("artist_id").alias("ring_artist_reach"))
    )
    tagged = tagged.join(reach, on="ring_id", how="left")

    return tagged.groupBy("artist_id", "date").agg(
        (F.sum(F.col("suspicious").cast("int")) / F.count("*")).alias("ring_play_share"),
        F.max("ring_accounts").alias("max_ring_accounts"),
        F.max("ring_devices").alias("max_ring_devices"),
        F.coalesce(F.max("ring_artist_reach"), F.lit(0)).alias("max_ring_artist_reach"),
    )


def build(spark: SparkSession, reset: bool = True) -> dict:
    events = read_silver(spark).select("user_id", "device_id", "artist_id", "played_at").cache()

    driver = store.get_driver()
    try:
        if reset:
            store.reset(driver)
        store.create_constraints(driver)

        n_uses = store.load_account_devices(driver, account_devices(events))
        ring_stats = store.detect_rings(driver)
        rings = store.account_rings(driver)

        n_streams = store.load_streams(driver, suspicious_streams(spark, events, rings))
        counts = store.graph_counts(driver)
    finally:
        driver.close()

    features = build_graph_features(spark, events, rings)
    features.write.format("delta").mode("overwrite").save(str(GOLD_DIR / "artist_daily_graph_features"))
    features_pd = features.toPandas()
    features_pd["date"] = features_pd["date"].astype(str)
    features_pd.to_parquet(GRAPH_FEATURES_PARQUET, index=False)

    summary = {
        **counts,
        "uses_loaded": n_uses,
        "streams_loaded": n_streams,
        "wcc_components": ring_stats["rings"],
        "artist_days_with_ring_plays": int((features_pd["ring_play_share"] > 0).sum()),
    }
    GRAPH_SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    from music_man.spark_session import get_spark

    spark = get_spark("build_graph")
    print(json.dumps(build(spark), indent=2))
    spark.stop()
