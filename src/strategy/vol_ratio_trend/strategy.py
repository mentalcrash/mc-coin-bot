"""Volatility Ratio Trend 전략.

단기/장기 RV 비율(vol term structure)로 시장 스트레스 감지 후 모멘텀 방향 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vol_ratio_trend.config import VolRatioTrendConfig
from src.strategy.vol_ratio_trend.preprocessor import preprocess
from src.strategy.vol_ratio_trend.signal import generate_signals


@register("vol-ratio-trend")
class VolRatioTrendStrategy(BaseStrategy):
    """Volatility Ratio Trend 전략 구현.

    Contango/Backwardation 상태에 따라 모멘텀 추종 여부 결정.
    """

    def __init__(self, config: VolRatioTrendConfig | None = None) -> None:
        self._config = config or VolRatioTrendConfig()

    @property
    def name(self) -> str:
        return "vol-ratio-trend"

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
        config = VolRatioTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "short_vol_window": str(self._config.short_vol_window),
            "long_vol_window": str(self._config.long_vol_window),
            "contango_threshold": str(self._config.contango_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
