"""Weekend-Momentum 전략.

주말 returns에 가중치를 부여한 multi-scale momentum으로
기관 부재 시 retail behavioral momentum persistence 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.weekend_mom.config import WeekendMomConfig
from src.strategy.weekend_mom.preprocessor import preprocess
from src.strategy.weekend_mom.signal import generate_signals


@register("weekend-mom-12h")
class WeekendMomStrategy(BaseStrategy):
    """Weekend-Momentum 전략 구현.

    크립토 주말 momentum이 2-3x 강화되는 ACR 2025 연구를 기반으로,
    주말 returns에 가중치를 부여한 multi-scale momentum 시그널 생성.
    """

    def __init__(self, config: WeekendMomConfig | None = None) -> None:
        self._config = config or WeekendMomConfig()

    @property
    def name(self) -> str:
        return "weekend-mom-12h"

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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = WeekendMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fast_lookback": str(self._config.fast_lookback),
            "slow_lookback": str(self._config.slow_lookback),
            "weekend_boost": str(self._config.weekend_boost),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
