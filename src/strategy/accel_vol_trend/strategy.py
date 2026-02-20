"""Acceleration-Volatility Trend 전략.

가격 가속도(2차 미분) + GK vol 정규화로 모멘텀 품질 측정.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.accel_vol_trend.config import AccelVolTrendConfig
from src.strategy.accel_vol_trend.preprocessor import preprocess
from src.strategy.accel_vol_trend.signal import generate_signals


@register("accel-vol-trend")
class AccelVolTrendStrategy(BaseStrategy):
    """Acceleration-Volatility Trend 전략 구현.

    가격 가속도를 GK vol로 정규화하여 vol-regime 무관한 모멘텀 품질 캡처.
    """

    def __init__(self, config: AccelVolTrendConfig | None = None) -> None:
        self._config = config or AccelVolTrendConfig()

    @property
    def name(self) -> str:
        return "accel-vol-trend"

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
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = AccelVolTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "accel_fast": str(self._config.accel_fast),
            "accel_slow": str(self._config.accel_slow),
            "gk_window": str(self._config.gk_window),
            "accel_long_threshold": str(self._config.accel_long_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
