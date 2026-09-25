"""
Bronze layer: writes the generated catalog/users/play-events as Delta
tables, unmodified except for the type conversion needed to hand pandas
DataFrames to Spark.
"""

from __future__ import annotations

import pandas as pd
from pyspark.sql import SparkSession

from music_man.paths import BRONZE_DIR


def write_bronze(
    spark: SparkSession,
    catalog: pd.DataFrame,
    users: pd.DataFrame,
    events: pd.DataFrame,
    fraud_labels: pd.DataFrame,
) -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    spark.createDataFrame(catalog).write.format("delta").mode("overwrite").save(
        str(BRONZE_DIR / "catalog")
    )
    spark.createDataFrame(users).write.format("delta").mode("overwrite").save(
        str(BRONZE_DIR / "users")
    )
    spark.createDataFrame(events).write.format("delta").mode("overwrite").save(
        str(BRONZE_DIR / "play_events")
    )
    spark.createDataFrame(fraud_labels).write.format("delta").mode("overwrite").save(
        str(BRONZE_DIR / "fraud_labels")
    )


def read_bronze(spark: SparkSession, name: str):
    return spark.read.format("delta").load(str(BRONZE_DIR / name))


if __name__ == "__main__":
    from music_man.generator.catalog import load_catalog
    from music_man.generator.events import generate_dataset
    from music_man.generator.users import load_users
    from music_man.spark_session import get_spark

    catalog = load_catalog()
    users = load_users()
    events, fraud_labels = generate_dataset()  # full defaults: 2M organic events, 40 fraud cases

    spark = get_spark("bronze")
    write_bronze(spark, catalog, users, events, fraud_labels)
    print("Bronze row counts:")
    for name in ["catalog", "users", "play_events", "fraud_labels"]:
        print(f"  {name}: {read_bronze(spark, name).count():,}")
    spark.stop()
