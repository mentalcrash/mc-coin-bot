"""Autocorrelation Momentum 전략.

lag-1 자기상관으로 모멘텀 레짐 감지 후 ROC 방향으로 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.autocorr_mom.config import AutocorrMomConfig
from src.strategy.autocorr_mom.preprocessor import preprocess
from src.strategy.autocorr_mom.signal import generate_signals


@register("autocorr-mom")
class AutocorrMomStrategy(BaseStrategy):
    """Autocorrelation Momentum 전략 구현.

    수익률 lag-1 자기상관이 양이면 모멘텀 레짐 -> 추세추종 유효.
    """

    def __init__(self, config: AutocorrMomConfig | None = None) -> None:
        self._config = config or AutocorrMomConfig()

    @property
    def name(self) -> str:
        return "autocorr-mom"

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
        config = AutocorrMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "autocorr_window": str(self._config.autocorr_window),
            "momentum_window": str(self._config.momentum_window),
            "autocorr_threshold": str(self._config.autocorr_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
