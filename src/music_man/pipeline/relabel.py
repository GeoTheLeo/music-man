"""
One-off migration: rewrite bronze fraud_labels with one row per day each
injected case touches, for datasets generated before that fix.

The generator is seeded, so re-running inject_fraud_events reproduces the
same 40 cases (targets, patterns, and event timestamps). This script checks
that every recomputed case's first day matches an existing label before it
overwrites anything, so it can't silently relabel a different dataset.

    python -m music_man.pipeline.relabel
"""

from __future__ import annotations

from music_man.generator.catalog import load_catalog
from music_man.generator.events import inject_fraud_events
from music_man.paths import BRONZE_DIR
from music_man.pipeline.bronze import read_bronze

N_FRAUD_CASES = 40
START_DATE, END_DATE = "2026-01-01", "2026-03-31"


def relabel(spark) -> tuple[int, int]:
    old = read_bronze(spark, "fraud_labels").toPandas()
    _, new = inject_fraud_events(load_catalog(), N_FRAUD_CASES, START_DATE, END_DATE, seed=42)

    first_days = new.sort_values("period").groupby("case_id").first().reset_index()
    recomputed = set(zip(first_days["artist_id"], first_days["period"]))
    existing = set(zip(old["artist_id"], old["period"]))
    if "case_id" in old.columns:
        raise SystemExit("fraud_labels already has one row per day; nothing to migrate")
    if recomputed != existing:
        raise SystemExit("recomputed cases don't match the existing labels; regenerate the dataset instead")

    spark.createDataFrame(new).write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).save(str(BRONZE_DIR / "fraud_labels"))
    return len(old), len(new)


if __name__ == "__main__":
    from music_man.spark_session import get_spark

    spark = get_spark("relabel")
    before, after = relabel(spark)
    print(f"fraud_labels: {before} first-day rows -> {after} rows (every day each case touches)")
    spark.stop()
