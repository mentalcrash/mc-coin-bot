"""Directional-Asymmetric Multi-Scale Momentum 전략.

UP/DOWN 비대칭 lookback으로 상승은 느리게 확인, 하락은 빠르게 반응.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.asymmetric_trend_8h.config import AsymmetricTrend8hConfig
from src.strategy.asymmetric_trend_8h.preprocessor import preprocess
from src.strategy.asymmetric_trend_8h.signal import generate_signals


@register("asymmetric-trend-8h")
class AsymmetricTrend8hStrategy(BaseStrategy):
    """Directional-Asymmetric Multi-Scale Momentum 전략 구현.

    상승 모멘텀은 긴 lookback(15/30/63)으로 천천히 확인하고,
    하락 모멘텀은 짧은 lookback(3/6/15)으로 빠르게 반응하는
    비대칭 구조로 크립토 시장의 급락 특성에 적응한다.
    """

    def __init__(self, config: AsymmetricTrend8hConfig | None = None) -> None:
        self._config = config or AsymmetricTrend8hConfig()

    @property
    def name(self) -> str:
        return "asymmetric-trend-8h"

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
        config = AsymmetricTrend8hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "up_lookbacks": (
                f"{self._config.up_lookback_short}/"
                f"{self._config.up_lookback_mid}/"
                f"{self._config.up_lookback_long}"
            ),
            "dn_lookbacks": (
                f"{self._config.dn_lookback_short}/"
                f"{self._config.dn_lookback_mid}/"
                f"{self._config.dn_lookback_long}"
            ),
            "consensus_threshold": str(self._config.consensus_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
