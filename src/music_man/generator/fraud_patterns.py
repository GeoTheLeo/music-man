"""
Injectable royalty-fraud patterns for the synthetic play-event generator.

Each function generates a batch of *fraudulent* events for one (track_id,
artist_id) target, all sharing one consistent signature so events.py can
apply them generically. These are real, well-documented streaming-fraud
signatures (bot-farm plays) - see the project README for background.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Callable, Literal

FraudPattern = Literal[
    "burst_velocity",
    "device_concentration",
    "duration_regularity",
    "geo_impossible",
]

FRAUD_PATTERNS: list[FraudPattern] = [
    "burst_velocity",
    "device_concentration",
    "duration_regularity",
    "geo_impossible",
]


def _event(
    track_id: str,
    artist_id: str,
    user_id: str,
    device_id: str,
    ip_address: str,
    played_at: datetime,
    duration_ms_played: int,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "track_id": track_id,
        "artist_id": artist_id,
        "played_at": played_at,
        "device_id": device_id,
        "ip_address": ip_address,
        "duration_ms_played": max(duration_ms_played, 0),
        "event_type": "play",
    }


def generate_burst_velocity(
    track_id: str, artist_id: str, track_duration_ms: int, window_start: datetime, rng: random.Random
) -> list[dict]:
    """An implausible spike of plays for one track within a few minutes."""
    n_events = rng.randint(3000, 6000)
    burst_seconds = rng.randint(120, 300)
    events = []
    for _ in range(n_events):
        played_at = window_start + timedelta(seconds=rng.uniform(0, burst_seconds))
        user_id = f"bot-burst-{rng.randint(1, 50)}"
        device_id = f"bot-burst-dev-{rng.randint(1, 15)}"
        ip_address = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        duration = int(track_duration_ms * rng.uniform(0.95, 1.0))
        events.append(_event(track_id, artist_id, user_id, device_id, ip_address, played_at, duration))
    return events


def generate_device_concentration(
    track_id: str, artist_id: str, track_duration_ms: int, window_start: datetime, rng: random.Random
) -> list[dict]:
    """A handful of devices/IPs account for a disproportionate share of plays."""
    n_events = rng.randint(2000, 4000)
    n_devices = rng.randint(2, 4)
    window_hours = rng.randint(6, 24)
    devices = [f"farm-dev-{uuid.uuid4().hex[:8]}" for _ in range(n_devices)]
    ips = [f"172.16.{rng.randint(0, 255)}.{rng.randint(1, 254)}" for _ in range(n_devices)]
    events = []
    for _ in range(n_events):
        idx = rng.randint(0, n_devices - 1)
        played_at = window_start + timedelta(seconds=rng.uniform(0, window_hours * 3600))
        user_id = f"bot-farm-{idx}"
        duration = int(track_duration_ms * rng.uniform(0.9, 1.0))
        events.append(_event(track_id, artist_id, user_id, devices[idx], ips[idx], played_at, duration))
    return events


def generate_duration_regularity(
    track_id: str, artist_id: str, track_duration_ms: int, window_start: datetime, rng: random.Random
) -> list[dict]:
    """Bots replay a track for (almost) an identical duration every time."""
    n_events = rng.randint(2000, 4000)
    window_hours = rng.randint(6, 24)
    fixed_duration = int(track_duration_ms * 0.98)
    events = []
    for _ in range(n_events):
        played_at = window_start + timedelta(seconds=rng.uniform(0, window_hours * 3600))
        user_id = f"bot-loop-{rng.randint(1, 150)}"
        device_id = f"bot-loop-dev-{rng.randint(1, 40)}"
        ip_address = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        duration = fixed_duration + rng.randint(-50, 50)  # near-zero variance, on purpose
        events.append(_event(track_id, artist_id, user_id, device_id, ip_address, played_at, duration))
    return events


# Roughly opposite hemispheres - far enough apart that real travel between
# them inside a short window is physically impossible.
_GEO_PREFIX_PAIRS = [
    ("198.51.100.", "203.0.113."),
    ("192.0.2.", "233.252.0."),
]


def generate_geo_impossible(
    track_id: str, artist_id: str, track_duration_ms: int, window_start: datetime, rng: random.Random
) -> list[dict]:
    """The same user/device streams from two far-apart locations minutes apart."""
    n_pairs = rng.randint(200, 500)
    window_hours = rng.randint(6, 24)
    events = []
    for i in range(n_pairs):
        user_id = f"bot-geo-{uuid.uuid4().hex[:8]}"
        device_id = f"bot-geo-dev-{uuid.uuid4().hex[:8]}"
        prefix_a, prefix_b = rng.choice(_GEO_PREFIX_PAIRS)
        t1 = window_start + timedelta(seconds=rng.uniform(0, window_hours * 3600))
        t2 = t1 + timedelta(minutes=rng.uniform(2, 8))  # too fast to be real travel
        duration = int(track_duration_ms * rng.uniform(0.9, 1.0))
        events.append(_event(track_id, artist_id, user_id, device_id, f"{prefix_a}{rng.randint(1, 254)}", t1, duration))
        events.append(_event(track_id, artist_id, user_id, device_id, f"{prefix_b}{rng.randint(1, 254)}", t2, duration))
    return events


PatternFn = Callable[[str, str, int, datetime, random.Random], list[dict]]

PATTERN_GENERATORS: dict[FraudPattern, PatternFn] = {
    "burst_velocity": generate_burst_velocity,
    "device_concentration": generate_device_concentration,
    "duration_regularity": generate_duration_regularity,
    "geo_impossible": generate_geo_impossible,
}
