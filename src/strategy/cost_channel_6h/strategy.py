"""Cost-Penalized Multi-Scale Channel 전략.

3종 채널(Donchian/Keltner/BB) x 3스케일(15/45/120) 앙상블 breakout + 비용 페널티로 거래 비용 최소화.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.cost_channel_6h.config import CostChannel6hConfig
from src.strategy.cost_channel_6h.preprocessor import preprocess
from src.strategy.cost_channel_6h.signal import generate_signals


@register("cost-channel-6h")
class CostChannel6hStrategy(BaseStrategy):
    """Cost-Penalized Multi-Scale Channel 전략 구현.

    3종 채널 유형(Donchian/Keltner/BB) x 3스케일(15/45/120)의 9개 sub-signal
    consensus에 비용 페널티를 적용하여 불필요한 포지션 전환을 억제한다.
    """

    def __init__(self, config: CostChannel6hConfig | None = None) -> None:
        self._config = config or CostChannel6hConfig()

    @property
    def name(self) -> str:
        return "cost-channel-6h"

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
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = CostChannel6hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "scales": f"{self._config.scale_short}/{self._config.scale_mid}/{self._config.scale_long}",
            "entry_threshold": str(self._config.entry_threshold),
            "cost_penalty_theta": str(self._config.cost_penalty_theta),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
