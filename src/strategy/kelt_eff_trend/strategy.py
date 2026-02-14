"""Keltner Efficiency Trend 전략.

ATR 기반 Keltner Channel 돌파 + Efficiency Ratio 품질 게이트.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.kelt_eff_trend.config import KeltEffTrendConfig
from src.strategy.kelt_eff_trend.preprocessor import preprocess
from src.strategy.kelt_eff_trend.signal import generate_signals


@register("kelt-eff-trend")
class KeltEffTrendStrategy(BaseStrategy):
    """Keltner Efficiency Trend 전략 구현.

    KC는 vol 적응형 밴드, ER은 추세 품질 게이트.
    이진 결정으로 conviction scalar 천장 회피.
    """

    def __init__(self, config: KeltEffTrendConfig | None = None) -> None:
        self._config = config or KeltEffTrendConfig()

    @property
    def name(self) -> str:
        return "kelt-eff-trend"

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
        config = KeltEffTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "kc_ema_period": str(self._config.kc_ema_period),
            "kc_atr_period": str(self._config.kc_atr_period),
            "kc_multiplier": str(self._config.kc_multiplier),
            "er_period": str(self._config.er_period),
            "er_threshold": str(self._config.er_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
