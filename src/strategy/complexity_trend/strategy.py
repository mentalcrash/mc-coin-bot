"""Complexity-Filtered Trend 전략.

Fractal dimension + Hurst + efficiency ratio로 시장 복잡도 필터링.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.complexity_trend.config import ComplexityTrendConfig
from src.strategy.complexity_trend.preprocessor import preprocess
from src.strategy.complexity_trend.signal import generate_signals


@register("complexity-trend")
class ComplexityTrendStrategy(BaseStrategy):
    """Complexity-Filtered Trend 전략 구현.

    정보이론 지표(Hurst/Fractal/ER)로 복잡도를 정량화하여
    예측 가능한 구간에서만 추세추종 시그널 활성화.
    """

    def __init__(self, config: ComplexityTrendConfig | None = None) -> None:
        self._config = config or ComplexityTrendConfig()

    @property
    def name(self) -> str:
        return "complexity-trend"

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
        config = ComplexityTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "hurst_window": str(self._config.hurst_window),
            "fractal_period": str(self._config.fractal_period),
            "er_period": str(self._config.er_period),
            "hurst_threshold": str(self._config.hurst_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
