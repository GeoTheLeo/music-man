"""
Trains a supervised fraud classifier on the gold-layer artist/day features,
using the injected fraud_labels as ground truth. Logs params/metrics/model
to MLflow, then writes anomaly_score back into the artist_daily_metrics
gold table.
"""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from music_man.paths import GOLD_DIR, MLRUNS_DIR
from music_man.pipeline.bronze import read_bronze
from music_man.pipeline.gold import read_gold

FEATURE_COLUMNS = [
    "total_plays",
    "unique_listeners",
    "unique_devices",
    "unique_ips",
    "device_concentration_ratio",
    "play_velocity_max",
    "duration_variance",
    "geo_impossible_count",
]


def build_training_frame(spark: SparkSession) -> pd.DataFrame:
    metrics = read_gold(spark, "artist_daily_metrics")
    labels = (
        read_bronze(spark, "fraud_labels")
        .select("artist_id", "period")
        .withColumn("is_fraud", F.lit(1))
    )

    joined = metrics.join(
        labels,
        (metrics.artist_id == labels.artist_id) & (metrics.date.cast("string") == labels.period),
        how="left",
    ).select(metrics["*"], F.coalesce(labels["is_fraud"], F.lit(0)).alias("is_fraud"))

    return joined.toPandas()


def train(spark: SparkSession) -> tuple[GradientBoostingClassifier, dict, pd.DataFrame]:
    df = build_training_frame(spark)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    eval_metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("music_man_fraud_detection")
    with mlflow.start_run():
        mlflow.log_params({"model": "GradientBoostingClassifier", "n_features": len(FEATURE_COLUMNS)})
        mlflow.log_metrics(eval_metrics)
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name="music_man_fraud_classifier"
        )

    # Refit on all data for scoring, now that eval metrics are captured
    # from the held-out split above.
    model.fit(X, y)
    df["anomaly_score"] = model.predict_proba(X)[:, 1]

    return model, eval_metrics, df


def write_scored_gold(spark: SparkSession, scored_df: pd.DataFrame) -> None:
    final_df = scored_df.drop(columns=["is_fraud"])
    spark.createDataFrame(final_df).write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).save(str(GOLD_DIR / "artist_daily_metrics"))
    # A plain Parquet mirror for the agent's tools: they need fast reads
    # without paying Spark/JVM startup cost on every query - the same
    # split real deployments make between batch ETL (Spark) and a fast
    # serving layer (a Databricks SQL Warehouse, in the real-Databricks
    # version of this project).
    final_df.to_parquet(GOLD_DIR / "artist_daily_metrics.snapshot.parquet", index=False)


if __name__ == "__main__":
    from music_man.spark_session import get_spark

    spark = get_spark("train_model")
    model, eval_metrics, scored_df = train(spark)
    print("Eval metrics (held-out test split):", eval_metrics)
    write_scored_gold(spark, scored_df)
    print("anomaly_score written back to artist_daily_metrics gold table.")
    spark.stop()
