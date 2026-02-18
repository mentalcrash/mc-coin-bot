"""Liquidation Cascade Reversal 전략.

레버리지 캐스케이드 후 가격 오버슈트 평균회귀.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.liq_cascade_rev.config import LiqCascadeRevConfig
from src.strategy.liq_cascade_rev.preprocessor import preprocess
from src.strategy.liq_cascade_rev.signal import generate_signals


@register("liq-cascade-rev")
class LiqCascadeRevStrategy(BaseStrategy):
    """Liquidation Cascade Reversal 전략 구현.

    FR buildup → cascade event → reversal entry.
    """

    def __init__(self, config: LiqCascadeRevConfig | None = None) -> None:
        self._config = config or LiqCascadeRevConfig()

    @property
    def name(self) -> str:
        return "liq-cascade-rev"

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
        config = LiqCascadeRevConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fr_buildup_threshold": str(self._config.fr_buildup_threshold),
            "cascade_return_multiplier": str(self._config.cascade_return_multiplier),
            "vol_expansion_ratio": str(self._config.vol_expansion_ratio),
            "max_hold_bars": str(self._config.max_hold_bars),
            "short_mode": self._config.short_mode.name,
        }
