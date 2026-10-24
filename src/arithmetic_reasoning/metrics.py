"""Cache accounting and metric accumulation."""

from typing import Any


def cache_bytes(cache: Any) -> int:
    return sum(
        t.numel() * t.element_size()
        for layer in cache.layers
        for t in (layer.keys, layer.values)
        if t is not None
    )


class CounterMetrics(dict):
    def __missing__(self, key: str) -> int:
        return 0

    def add(self, key: str, value: Any) -> None:
        self[key] = self[key] + int(value)

    def rate(self, numerator: str, denominator: str) -> float | None:
        return self[numerator] / self[denominator] if self[denominator] else None
