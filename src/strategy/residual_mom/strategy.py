"""Residual Momentum 전략.

시장 factor를 제거한 잔차의 모멘텀으로 자산 고유 alpha 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.residual_mom.config import ResidualMomConfig
from src.strategy.residual_mom.preprocessor import preprocess
from src.strategy.residual_mom.signal import generate_signals


@register("residual-mom")
class ResidualMomStrategy(BaseStrategy):
    """Residual Momentum 전략 구현.

    Rolling OLS로 시장 factor를 회귀 제거한 잔차의 모멘텀을 활용.
    CTREND와 정의상 orthogonal하여 앙상블 분산 가치 극대화.
    """

    def __init__(self, config: ResidualMomConfig | None = None) -> None:
        self._config = config or ResidualMomConfig()

    @property
    def name(self) -> str:
        return "residual-mom"

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
        config = ResidualMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "regression_window": str(self._config.regression_window),
            "residual_lookback": str(self._config.residual_lookback),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
