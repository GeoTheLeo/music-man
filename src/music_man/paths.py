"""
Canonical filesystem paths for Music Man.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SEED_DIR = DATA_DIR / "seed"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

CATALOG_PARQUET = SEED_DIR / "catalog.parquet"
USERS_PARQUET = SEED_DIR / "users.parquet"

QUEUE_DB_PATH = DATA_DIR / "queue.db"
AUDIT_LOG_PATH = DATA_DIR / "audit_log.csv"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
