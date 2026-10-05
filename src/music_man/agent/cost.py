"""
Per-run token and dollar accounting, so every agent run (and every eval
trial) reports what it cost before Finance has to ask.

Prices are Anthropic first-party list prices in USD per million tokens.
Cache writes (5-minute TTL) bill at 1.25x input, cache reads at 0.1x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def price_key(model: str) -> str | None:
    """Map a reported model name (which may carry a date suffix) to its price-table key."""
    if model in PRICES_PER_MTOK:
        return model
    return next((key for key in PRICES_PER_MTOK if model.startswith(key)), None)


def call_cost(model: str, usage) -> float:
    """USD cost of one API response, from its usage block."""
    if model not in PRICES_PER_MTOK:
        raise KeyError(f"no price configured for model {model!r}")
    input_price, output_price = PRICES_PER_MTOK[model]
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        usage.input_tokens * input_price
        + cache_write * input_price * CACHE_WRITE_MULTIPLIER
        + cache_read * input_price * CACHE_READ_MULTIPLIER
        + usage.output_tokens * output_price
    ) / 1_000_000


@dataclass
class UsageTracker:
    calls: list[dict] = field(default_factory=list)

    def add(self, phase: str, model: str, usage) -> None:
        self.calls.append(
            {
                "phase": phase,
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cost_usd": call_cost(model, usage),
            }
        )

    @property
    def total_cost(self) -> float:
        return sum(c["cost_usd"] for c in self.calls)

    def summary(self) -> dict:
        by_phase: dict[str, float] = {}
        by_model: dict[str, dict] = {}
        for c in self.calls:
            by_phase[c["phase"]] = by_phase.get(c["phase"], 0.0) + c["cost_usd"]
            m = by_model.setdefault(
                c["model"],
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0, "cost_usd": 0.0},
            )
            m["calls"] += 1
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                m[key] += c[key]
            m["cost_usd"] += c["cost_usd"]
        for m in by_model.values():
            m["cost_usd"] = round(m["cost_usd"], 6)
        return {
            "by_model": by_model,
            "api_calls": len(self.calls),
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "cache_read_input_tokens": sum(c["cache_read_input_tokens"] for c in self.calls),
            "cost_usd": round(self.total_cost, 6),
            "cost_usd_by_phase": {k: round(v, 6) for k, v in by_phase.items()},
        }
