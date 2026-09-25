"""
Downloads and normalizes the seed music catalog from Kaggle.

Requires a one-time Kaggle account + API token at ~/.kaggle/kaggle.json
(kaggle.com -> Account Settings -> API -> Create New Token). Uses
`maharshipandya/-spotify-tracks-dataset` for real genre/popularity
distributions - see the project README for why a real dataset was chosen
over a fully synthetic one.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from music_man.paths import CATALOG_PARQUET, SEED_DIR

KAGGLE_DATASET = "maharshipandya/-spotify-tracks-dataset"

# The raw dataset's column names, normalized to our canonical catalog schema.
# Kaggle dataset column names occasionally drift between snapshots - if
# download_catalog() raises a KeyError, print df.columns and adjust this map.
_COLUMN_ALIASES = {
    "track_id": ["track_id"],
    "artist_name": ["artists", "artist_name", "artist"],
    "track_name": ["track_name", "name"],
    "album_name": ["album_name", "album"],
    "genre": ["track_genre", "genre"],
    "popularity_score": ["popularity"],
    "duration_ms": ["duration_ms"],
}


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"None of {candidates} found in dataset columns: {list(df.columns)}"
    )


def _stable_artist_id(artist_name: str) -> str:
    return "artist_" + hashlib.sha1(artist_name.strip().lower().encode()).hexdigest()[:12]


def download_catalog() -> pd.DataFrame:
    """
    Downloads the Kaggle dataset (cached by kagglehub after the first call),
    loads its CSV, and normalizes it into the canonical catalog schema:
    artist_id, artist_name, track_id, track_name, album_name, genre,
    popularity_score.
    """
    import kagglehub

    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_candidates = list(__import__("pathlib").Path(dataset_dir).glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {dataset_dir}")

    raw = pd.read_csv(csv_candidates[0])

    normalized = pd.DataFrame(
        {
            "track_id": raw[_resolve_column(raw, _COLUMN_ALIASES["track_id"])],
            "artist_name": raw[_resolve_column(raw, _COLUMN_ALIASES["artist_name"])],
            "track_name": raw[_resolve_column(raw, _COLUMN_ALIASES["track_name"])],
            "album_name": raw[_resolve_column(raw, _COLUMN_ALIASES["album_name"])],
            "genre": raw[_resolve_column(raw, _COLUMN_ALIASES["genre"])],
            "popularity_score": raw[_resolve_column(raw, _COLUMN_ALIASES["popularity_score"])],
            "duration_ms": raw[_resolve_column(raw, _COLUMN_ALIASES["duration_ms"])],
        }
    )

    # Kaggle's version of this dataset repeats the same track once per genre
    # it's tagged with - dedupe on track_id, keeping the first occurrence.
    normalized = normalized.drop_duplicates(subset="track_id").reset_index(drop=True)
    # A handful of rows in the source data have a missing artist/track name
    # (NaN) - not usable for a per-artist fraud analysis, so drop them.
    normalized = normalized.dropna(subset=["track_id", "artist_name", "track_name"]).reset_index(drop=True)
    normalized["artist_id"] = normalized["artist_name"].map(_stable_artist_id)
    # Guard against zero/garbage durations in the source data breaking
    # downstream duration-ratio features.
    normalized["duration_ms"] = normalized["duration_ms"].clip(lower=30_000, upper=600_000)
    normalized = normalized[
        ["artist_id", "artist_name", "track_id", "track_name", "album_name", "genre", "popularity_score", "duration_ms"]
    ]

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(CATALOG_PARQUET, index=False)
    return normalized


def load_catalog() -> pd.DataFrame:
    """Loads the normalized catalog, downloading it first if not cached."""
    if not CATALOG_PARQUET.exists():
        return download_catalog()
    return pd.read_parquet(CATALOG_PARQUET)


if __name__ == "__main__":
    df = download_catalog()
    print(f"Catalog: {len(df):,} tracks across {df['artist_id'].nunique():,} artists")
    print(df.head())
