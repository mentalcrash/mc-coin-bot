"""Entropy-Gate Trend 4H 전략.

Permutation Entropy 게이팅 + 3-scale Donchian breakout 앙상블로 추세 추종.
Low entropy = predictable = 진입 허용. High entropy = random = flat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.entropy_gate_trend_4h.config import EntropyGateTrend4hConfig
from src.strategy.entropy_gate_trend_4h.preprocessor import preprocess
from src.strategy.entropy_gate_trend_4h.signal import generate_signals


@register("entropy-gate-trend-4h")
class EntropyGateTrend4hStrategy(BaseStrategy):
    """Entropy-Gate Trend 4H 전략 구현.

    Permutation Entropy로 시장 복잡도를 판단하고,
    low-entropy(predictable) 구간에서만 3-scale Donchian breakout을 활성화한다.
    """

    def __init__(self, config: EntropyGateTrend4hConfig | None = None) -> None:
        self._config = config or EntropyGateTrend4hConfig()

    @property
    def name(self) -> str:
        return "entropy-gate-trend-4h"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> BaseModel:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        return {
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = EntropyGateTrend4hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "entropy_window": str(self._config.entropy_window),
            "entropy_m": str(self._config.entropy_m),
            "entropy_threshold": str(self._config.entropy_threshold),
            "dc_scales": (
                f"{self._config.dc_scale_short}/"
                f"{self._config.dc_scale_mid}/"
                f"{self._config.dc_scale_long}"
            ),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
