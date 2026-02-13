"""Skew-Gated Momentum 전략.

수익률 분포 비대칭성(skewness)으로 모멘텀 방향 지속 가능성을 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.skew_mom.config import SkewMomConfig
from src.strategy.skew_mom.preprocessor import preprocess
from src.strategy.skew_mom.signal import generate_signals


@register("skew-mom")
class SkewMomStrategy(BaseStrategy):
    """Skew-Gated Momentum 전략 구현.

    양의 skew + 상승 모멘텀 = 상방 tail 가능성 확인,
    음의 skew 전환 = 크래시 전조 → 숏 진입.
    """

    def __init__(self, config: SkewMomConfig | None = None) -> None:
        self._config = config or SkewMomConfig()

    @property
    def name(self) -> str:
        return "skew-mom"

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
        config = SkewMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "skew_window": str(self._config.skew_window),
            "mom_lookback": str(self._config.mom_lookback),
            "skew_long_threshold": str(self._config.skew_long_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
