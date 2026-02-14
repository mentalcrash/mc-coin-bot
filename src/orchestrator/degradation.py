"""Page-Hinkley Degradation Detector.

CUSUM variant mean-shift detector for real-time strategy performance monitoring.
Detects sustained negative drift in daily returns, triggering WARNING state.

Algorithm (exponentially weighted, detects downward shift):
    x_mean_t = alpha * x_mean_{t-1} + (1 - alpha) * x_t
    m_t += x_mean_t - x_t - delta
    M_t = min(M_t, m_t)
    Detection: m_t - M_t > lambda_

References:
    Page, E.S. (1954). Continuous Inspection Schemes.
    Hinkley, D.V. (1971). Inference about the change-point from CUSUM tests.

Rules Applied:
    - #10 Python Standards: __slots__, type hints
    - Pure Python (no numpy dependency)
"""

from __future__ import annotations

import math


class PageHinkleyDetector:
    """Page-Hinkley mean-shift detector.

    Monitors a stream of values (e.g. daily returns) and detects
    sustained negative drift indicating strategy degradation.

    Args:
        delta: Magnitude threshold — minimum shift to detect. Default 0.005.
        lambda_: Detection threshold — higher = fewer false alarms. Default 50.0.
        alpha: EWMA smoothing factor for running mean. Default 0.9999.
    """

    __slots__ = ("_alpha", "_delta", "_lambda", "_m_min", "_m_t", "_n", "_x_mean")

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 0.9999,
    ) -> None:
        self._delta = delta
        self._lambda = lambda_
        self._alpha = alpha
        self._n: int = 0
        self._x_mean: float = 0.0
        self._m_t: float = 0.0
        self._m_min: float = 0.0

    @property
    def lambda_threshold(self) -> float:
        """Detection threshold (lambda)."""
        return self._lambda

    def update(self, value: float) -> bool:
        """Ingest a new observation and check for degradation.

        Args:
            value: New observation (e.g. daily return).

        Returns:
            True if degradation detected (m_t - m_min > lambda_).
        """
        if math.isnan(value) or math.isinf(value):
            return False

        self._n += 1

        if self._n == 1:
            # First observation: initialize running mean only
            self._x_mean = value
            return False

        # Exponentially weighted running mean
        self._x_mean = self._alpha * self._x_mean + (1.0 - self._alpha) * value

        # Cumulative sum: accumulate excess of mean over value (detects decrease)
        self._m_t += self._x_mean - value - self._delta

        # Track cumulative minimum
        self._m_min = min(self._m_min, self._m_t)

        # Detection check
        return (self._m_t - self._m_min) > self._lambda

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._n = 0
        self._x_mean = 0.0
        self._m_t = 0.0
        self._m_min = 0.0

    @property
    def score(self) -> float:
        """Current detection score (m_t - m_min).

        Useful for monitoring — higher score means closer to detection.
        """
        return self._m_t - self._m_min

    @property
    def n_observations(self) -> int:
        """Number of observations ingested."""
        return self._n

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, float | int]:
        """Serialize mutable state for persistence (config excluded)."""
        return {
            "n": self._n,
            "x_mean": self._x_mean,
            "m_t": self._m_t,
            "m_min": self._m_min,
        }

    def restore_from_dict(self, data: dict[str, float | int]) -> None:
        """Restore mutable state from persisted dict.

        Missing fields default to initial values (forward-compatible).
        """
        self._n = int(data.get("n", 0))
        self._x_mean = float(data.get("x_mean", 0.0))
        self._m_t = float(data.get("m_t", 0.0))
        self._m_min = float(data.get("m_min", 0.0))
