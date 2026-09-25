"""
Synthetic user/subscription dimension.
"""

from __future__ import annotations

import random

import pandas as pd
from faker import Faker

from music_man.paths import SEED_DIR, USERS_PARQUET

_COUNTRIES = ["US", "GB", "DE", "FR", "BR", "JP", "AU", "CA", "MX", "IN"]
_PLAN_TIERS = ["free", "individual", "family", "student"]
_PLAN_WEIGHTS = [0.35, 0.4, 0.15, 0.1]


def generate_users(n_users: int = 20_000, seed: int = 42) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)

    rows = []
    for i in range(n_users):
        signup = fake.date_between(start_date="-3y", end_date="today")
        rows.append(
            {
                "user_id": f"user_{i:07d}",
                "country": rng.choice(_COUNTRIES),
                "signup_date": signup,
                "plan_tier": rng.choices(_PLAN_TIERS, weights=_PLAN_WEIGHTS, k=1)[0],
            }
        )

    df = pd.DataFrame(rows)
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(USERS_PARQUET, index=False)
    return df


def load_users() -> pd.DataFrame:
    if not USERS_PARQUET.exists():
        return generate_users()
    return pd.read_parquet(USERS_PARQUET)


if __name__ == "__main__":
    df = generate_users()
    print(f"Users: {len(df):,}")
    print(df.head())
