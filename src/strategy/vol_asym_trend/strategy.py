"""Volatility Asymmetry Trend 전략.

상승/하락 수익률의 변동성 비율로 추세 방향 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vol_asym_trend.config import VolAsymTrendConfig
from src.strategy.vol_asym_trend.preprocessor import preprocess
from src.strategy.vol_asym_trend.signal import generate_signals


@register("vol-asym-trend")
class VolAsymTrendStrategy(BaseStrategy):
    """Volatility Asymmetry Trend 전략 구현.

    Up/down semivariance 비율로 추세 방향성 conviction 측정.
    """

    def __init__(self, config: VolAsymTrendConfig | None = None) -> None:
        self._config = config or VolAsymTrendConfig()

    @property
    def name(self) -> str:
        return "vol-asym-trend"

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
        config = VolAsymTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "asym_window": str(self._config.asym_window),
            "asym_long_threshold": str(self._config.asym_long_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
