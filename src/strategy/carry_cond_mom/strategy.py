"""Carry-Conditional Momentum 전략.

가격 모멘텀과 funding rate level의 agreement로 모멘텀 지속 가능성 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.carry_cond_mom.config import CarryCondMomConfig
from src.strategy.carry_cond_mom.preprocessor import preprocess
from src.strategy.carry_cond_mom.signal import generate_signals


@register("carry-cond-mom")
class CarryCondMomStrategy(BaseStrategy):
    """Carry-Conditional Momentum 전략 구현.

    mom(+) AND FR(+) = bullish consensus → confident long.
    불일치 시 position 축소. BIS WP #1087 실증 기반.
    """

    def __init__(self, config: CarryCondMomConfig | None = None) -> None:
        self._config = config or CarryCondMomConfig()

    @property
    def name(self) -> str:
        return "carry-cond-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "high", "low", "volume", "funding_rate"]

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
        config = CarryCondMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mom_lookback": str(self._config.mom_lookback),
            "fr_lookback": str(self._config.fr_lookback),
            "agreement_boost": str(self._config.agreement_boost),
            "disagreement_penalty": str(self._config.disagreement_penalty),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
