"""SuperTrend 전략.

ATR 기반 동적 지지/저항선으로 추세 전환을 감지하는 추세추종.
ShortMode로 Long-only / Long-Short 전환 가능.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.supertrend.config import SuperTrendConfig
from src.strategy.supertrend.preprocessor import preprocess
from src.strategy.supertrend.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("supertrend")
class SuperTrendStrategy(BaseStrategy):
    """SuperTrend 전략.

    SuperTrend 초록(상승추세) → Long 전체 진입
    SuperTrend 빨강(하락추세) → Short(FULL) 또는 청산(DISABLED)
    ShortMode로 Long-only / Long-Short 전환.
    """

    def __init__(self, config: SuperTrendConfig | None = None) -> None:
        self._config = config or SuperTrendConfig()

    @property
    def name(self) -> str:
        return "supertrend"

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
            "max_leverage_cap": 1.0,
            "rebalance_threshold": 0.05,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = SuperTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        from src.strategy.supertrend.config import ShortMode

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        info = {
            "atr_period": str(self._config.atr_period),
            "multiplier": str(self._config.multiplier),
            "mode": mode_map.get(self._config.short_mode, "Unknown"),
        }
        if self._config.use_adx_filter:
            info["adx_period"] = str(self._config.adx_period)
            info["adx_threshold"] = str(self._config.adx_threshold)
        if self._config.use_risk_sizing:
            info["risk_per_trade"] = str(self._config.risk_per_trade)
            info["atr_stop_multiplier"] = str(self._config.atr_stop_multiplier)
        return info
