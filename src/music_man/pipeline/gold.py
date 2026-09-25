"""
Gold layer: per-artist/day fraud-signal features and monthly royalty
aggregates. anomaly_score is NOT computed here - it's added by
ml/train_model.py after training, which rewrites this table with the
model's predictions joined in.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from music_man.paths import GOLD_DIR
from music_man.pipeline.silver import read_silver

ROYALTY_RATE_PER_STREAM = 0.004
VELOCITY_BUCKET_SECONDS = 300  # 5-minute buckets for burst-velocity detection


def build_artist_daily_metrics(spark: SparkSession) -> DataFrame:
    from music_man.pipeline.bronze import read_bronze

    events = read_silver(spark).withColumn("date", F.to_date("played_at"))
    artist_names = (
        read_bronze(spark, "catalog")
        .select("artist_id", "artist_name")
        .dropDuplicates(["artist_id"])
    )

    base = events.groupBy("artist_id", "date").agg(
        F.count("*").alias("total_plays"),
        F.countDistinct("user_id").alias("unique_listeners"),
        F.countDistinct("device_id").alias("unique_devices"),
        F.countDistinct("ip_address").alias("unique_ips"),
        F.variance("duration_ms_played").alias("duration_variance"),
        F.sum(F.col("geo_impossible_flag").cast("int")).alias("geo_impossible_count"),
    )

    # Device concentration: a Herfindahl-style index - sum((device's share of
    # plays)^2). Close to 1 when a handful of devices dominate (bot farm
    # signature); close to 0 when plays are spread across many devices.
    device_counts = events.groupBy("artist_id", "date", "device_id").agg(F.count("*").alias("device_plays"))
    device_totals = device_counts.groupBy("artist_id", "date").agg(F.sum("device_plays").alias("total_for_share"))
    device_shares = device_counts.join(device_totals, on=["artist_id", "date"]).withColumn(
        "share_sq", F.pow(F.col("device_plays") / F.col("total_for_share"), 2)
    )
    concentration = device_shares.groupBy("artist_id", "date").agg(
        F.sum("share_sq").alias("device_concentration_ratio")
    )

    # Burst velocity: max plays observed in any 5-minute bucket that day.
    bucketed = events.withColumn(
        "bucket", (F.col("played_at").cast("long") / VELOCITY_BUCKET_SECONDS).cast("long")
    )
    bucket_counts = bucketed.groupBy("artist_id", "date", "bucket").agg(F.count("*").alias("bucket_plays"))
    velocity = bucket_counts.groupBy("artist_id", "date").agg(
        F.max("bucket_plays").alias("play_velocity_max")
    )

    metrics = (
        base.join(concentration, on=["artist_id", "date"], how="left")
        .join(velocity, on=["artist_id", "date"], how="left")
        .join(artist_names, on="artist_id", how="left")
        .withColumn("royalty_accrued", F.col("total_plays") * F.lit(ROYALTY_RATE_PER_STREAM))
    )

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    metrics.write.format("delta").mode("overwrite").save(str(GOLD_DIR / "artist_daily_metrics"))
    return metrics


def build_royalty_payouts(spark: SparkSession) -> DataFrame:
    metrics = read_gold(spark, "artist_daily_metrics")
    payouts = (
        metrics.withColumn("period", F.date_format("date", "yyyy-MM"))
        .groupBy("artist_id", "period")
        .agg(F.sum("royalty_accrued").alias("amount_due"))
    )
    payouts.write.format("delta").mode("overwrite").save(str(GOLD_DIR / "royalty_payouts"))
    return payouts


def read_gold(spark: SparkSession, name: str) -> DataFrame:
    return spark.read.format("delta").load(str(GOLD_DIR / name))


if __name__ == "__main__":
    from music_man.spark_session import get_spark

    spark = get_spark("gold")
    metrics = build_artist_daily_metrics(spark)
    payouts = build_royalty_payouts(spark)
    print(f"artist_daily_metrics: {metrics.count():,} rows")
    metrics.orderBy(F.desc("device_concentration_ratio")).show(10)
    print(f"royalty_payouts: {payouts.count():,} rows")
    spark.stop()
