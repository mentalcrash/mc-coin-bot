"""Adaptive FR Carry 전략.

FR 극단 구간에서만 캐리 수취 + vol 필터로 캐스케이드 회피.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.adaptive_fr_carry.config import AdaptiveFrCarryConfig
from src.strategy.adaptive_fr_carry.preprocessor import preprocess
from src.strategy.adaptive_fr_carry.signal import generate_signals


@register("adaptive-fr-carry")
class AdaptiveFrCarryStrategy(BaseStrategy):
    """Adaptive FR Carry 전략 구현.

    FR z-score 극단 → carry 방향 진입, vol 필터로 캐스케이드 회피.
    """

    def __init__(self, config: AdaptiveFrCarryConfig | None = None) -> None:
        self._config = config or AdaptiveFrCarryConfig()

    @property
    def name(self) -> str:
        return "adaptive-fr-carry"

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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = AdaptiveFrCarryConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fr_entry_threshold": str(self._config.fr_entry_threshold),
            "fr_exit_threshold": str(self._config.fr_exit_threshold),
            "vol_ratio_exit": str(self._config.vol_ratio_exit),
            "er_max": str(self._config.er_max),
            "max_hold_bars": str(self._config.max_hold_bars),
            "short_mode": self._config.short_mode.name,
        }
