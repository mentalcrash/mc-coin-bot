"""Stablecoin Velocity Regime 전략.

Volume 기반 velocity proxy의 가속/감속으로 자금 유입/유출 선행 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.stablecoin_velocity.config import StablecoinVelocityConfig
from src.strategy.stablecoin_velocity.preprocessor import preprocess
from src.strategy.stablecoin_velocity.signal import generate_signals


@register("stablecoin-velocity")
class StablecoinVelocityStrategy(BaseStrategy):
    """Stablecoin Velocity Regime 전략 구현.

    스테이블코인 velocity 가속 → 시장 진입 자금 증가 → 가격 상승 선행.
    Volume/close ratio로 velocity를 프록시.
    """

    def __init__(self, config: StablecoinVelocityConfig | None = None) -> None:
        self._config = config or StablecoinVelocityConfig()

    @property
    def name(self) -> str:
        return "stablecoin-velocity"

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
        config = StablecoinVelocityConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "velocity_fast_window": str(self._config.velocity_fast_window),
            "velocity_slow_window": str(self._config.velocity_slow_window),
            "zscore_entry_threshold": str(self._config.zscore_entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
