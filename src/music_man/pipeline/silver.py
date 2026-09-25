"""
Silver layer: dedupe, sessionize, join against catalog/users, and derive
per-event flags used by the gold-layer fraud features.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from music_man.paths import SILVER_DIR
from music_man.pipeline.bronze import read_bronze

SESSION_GAP_SECONDS = 1800  # 30 minutes of inactivity starts a new session
GEO_IMPOSSIBLE_GAP_SECONDS = 900  # 15 minutes is too fast for real travel


def build_silver(spark: SparkSession) -> DataFrame:
    events = read_bronze(spark, "play_events").dropDuplicates(["event_id"])
    catalog = read_bronze(spark, "catalog")
    users = read_bronze(spark, "users")

    events = events.join(catalog.select("track_id", "duration_ms"), on="track_id", how="left")
    events = events.join(users.select("user_id", "country"), on="user_id", how="left")

    user_window = Window.partitionBy("user_id").orderBy("played_at")
    events = events.withColumn("prev_played_at", F.lag("played_at").over(user_window))
    events = events.withColumn("prev_ip", F.lag("ip_address").over(user_window))
    events = events.withColumn(
        "gap_seconds",
        F.col("played_at").cast("long") - F.col("prev_played_at").cast("long"),
    )

    new_session = F.when(
        F.col("gap_seconds").isNull() | (F.col("gap_seconds") > SESSION_GAP_SECONDS), 1
    ).otherwise(0)
    running_session_window = Window.partitionBy("user_id").orderBy("played_at").rowsBetween(
        Window.unboundedPreceding, 0
    )
    events = events.withColumn("session_num", F.sum(new_session).over(running_session_window))
    events = events.withColumn(
        "session_id", F.concat_ws("-", F.col("user_id"), F.col("session_num").cast("string"))
    )

    events = events.withColumn(
        "is_full_play", F.col("duration_ms_played") >= (F.col("duration_ms") * 0.9)
    )

    ip_prefix = F.substring_index(F.col("ip_address"), ".", 2)
    prev_ip_prefix = F.substring_index(F.col("prev_ip"), ".", 2)
    events = events.withColumn(
        "geo_impossible_flag",
        (F.col("gap_seconds").isNotNull())
        & (F.col("gap_seconds") < GEO_IMPOSSIBLE_GAP_SECONDS)
        & (ip_prefix != prev_ip_prefix),
    )

    clean = events.drop("prev_played_at", "prev_ip", "gap_seconds", "session_num")

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    clean.write.format("delta").mode("overwrite").save(str(SILVER_DIR / "play_events_clean"))
    return clean


def read_silver(spark: SparkSession) -> DataFrame:
    return spark.read.format("delta").load(str(SILVER_DIR / "play_events_clean"))


if __name__ == "__main__":
    from music_man.spark_session import get_spark

    spark = get_spark("silver")
    clean = build_silver(spark)
    print(f"Silver play_events_clean: {clean.count():,} rows")
    clean.show(5)
    spark.stop()
