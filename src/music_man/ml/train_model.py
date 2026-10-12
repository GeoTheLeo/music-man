"""
Trains a supervised fraud classifier on the gold-layer artist/day features,
using the injected fraud_labels as ground truth. Logs params/metrics/model
to MLflow, then writes anomaly_score back into the artist_daily_metrics
gold table.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from music_man.paths import GOLD_DIR, MLRUNS_DIR, PROJECT_ROOT
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

# Ring features from the Neo4j account-device graph (graph/build_graph.py).
GRAPH_FEATURE_COLUMNS = [
    "ring_play_share",
    "max_ring_accounts",
    "max_ring_devices",
    "max_ring_artist_reach",
]

FEATURE_SETS = {
    "baseline": FEATURE_COLUMNS,
    "graph": FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS,
}

COMPARISON_JSON = PROJECT_ROOT / "docs" / "graph_feature_comparison.json"


def build_training_frame(spark: SparkSession) -> pd.DataFrame:
    metrics = read_gold(spark, "artist_daily_metrics")
    labels = (
        read_bronze(spark, "fraud_labels")
        .select("artist_id", "period")
        .dropDuplicates()
        .withColumn("is_fraud", F.lit(1))
    )

    joined = metrics.join(
        labels,
        (metrics.artist_id == labels.artist_id) & (metrics.date.cast("string") == labels.period),
        how="left",
    ).select(metrics["*"], F.coalesce(labels["is_fraud"], F.lit(0)).alias("is_fraud"))

    df = joined.toPandas()

    graph_path = GOLD_DIR / "artist_daily_graph_features.parquet"
    if graph_path.exists():
        graph = pd.read_parquet(graph_path)
        df["date"] = df["date"].astype(str)
        df = df.drop(columns=[c for c in GRAPH_FEATURE_COLUMNS if c in df.columns])
        df = df.merge(graph, on=["artist_id", "date"], how="left")

    return df


def _split(df: pd.DataFrame):
    """The one held-out split every feature set is scored on."""
    return train_test_split(df.index, test_size=0.25, stratify=df["is_fraud"], random_state=42)


def _evaluate(df: pd.DataFrame, columns: list[str], train_idx, test_idx) -> dict:
    X, y = df[columns].fillna(0), df["is_fraud"]

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X.loc[train_idx], y.loc[train_idx])
    y_pred = model.predict(X.loc[test_idx])
    y_test = y.loc[test_idx]

    cv = cross_validate(
        GradientBoostingClassifier(random_state=42), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=["precision", "recall", "f1"],
    )
    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "false_positives": int(((y_pred == 1) & (y_test == 0)).sum()),
        "false_negatives": int(((y_pred == 0) & (y_test == 1)).sum()),
        "test_fraud_cases": int(y_test.sum()),
        "cv5_precision": float(cv["test_precision"].mean()),
        "cv5_recall": float(cv["test_recall"].mean()),
        "cv5_f1": float(cv["test_f1"].mean()),
    }


def compare(spark: SparkSession) -> dict:
    """Score every feature set on the same split (plus 5-fold CV) and log each to MLflow."""
    df = build_training_frame(spark)
    train_idx, test_idx = _split(df)

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("music_man_fraud_detection")

    results = {}
    for name, columns in FEATURE_SETS.items():
        if any(c not in df.columns for c in columns):
            continue
        results[name] = _evaluate(df, columns, train_idx, test_idx)
        with mlflow.start_run(run_name=f"compare-{name}"):
            mlflow.log_params({"model": "GradientBoostingClassifier", "feature_set": name, "n_features": len(columns)})
            mlflow.log_metrics(results[name])

    COMPARISON_JSON.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_JSON.write_text(json.dumps(results, indent=2))
    return results


def train(
    spark: SparkSession, feature_set: str = "baseline"
) -> tuple[GradientBoostingClassifier, dict, pd.DataFrame]:
    columns = FEATURE_SETS[feature_set]
    df = build_training_frame(spark)
    df[columns] = df[columns].fillna(0)

    X = df[columns]
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
        mlflow.log_params(
            {"model": "GradientBoostingClassifier", "feature_set": feature_set, "n_features": len(columns)}
        )
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
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.date  # keep the gold table's date type
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=sorted(FEATURE_SETS), default="baseline")
    parser.add_argument("--compare", action="store_true", help="score every feature set on the same split; no write-back")
    args = parser.parse_args()

    spark = get_spark("train_model")
    if args.compare:
        print(json.dumps(compare(spark), indent=2))
        spark.stop()
        raise SystemExit
    model, eval_metrics, scored_df = train(spark, args.features)
    print("Eval metrics (held-out test split):", eval_metrics)
    write_scored_gold(spark, scored_df)
    print("anomaly_score written back to artist_daily_metrics gold table.")
    spark.stop()
