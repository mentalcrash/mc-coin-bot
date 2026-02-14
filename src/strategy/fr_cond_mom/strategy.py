"""FR Conditional Momentum 전략.

FR z-score 극단치=과밀 포지셔닝 시 모멘텀 conviction 조건부 조절.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fr_cond_mom.config import FrCondMomConfig
from src.strategy.fr_cond_mom.preprocessor import preprocess
from src.strategy.fr_cond_mom.signal import generate_signals


@register("fr-cond-mom")
class FrCondMomStrategy(BaseStrategy):
    """FR Conditional Momentum 전략 구현.

    모멘텀 시그널의 conviction을 FR z-score로 조절하여
    과밀 포지셔닝 리스크를 회피.
    """

    def __init__(self, config: FrCondMomConfig | None = None) -> None:
        self._config = config or FrCondMomConfig()

    @property
    def name(self) -> str:
        return "fr-cond-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

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
        config = FrCondMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mom_lookback": str(self._config.mom_lookback),
            "fr_ma_window": str(self._config.fr_ma_window),
            "fr_extreme_threshold": str(self._config.fr_extreme_threshold),
            "fr_dampening": str(self._config.fr_dampening),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
