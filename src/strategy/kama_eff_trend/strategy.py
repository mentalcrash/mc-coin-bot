"""KAMA Efficiency Trend 전략.

ER이 높은 구간에서 KAMA 방향 추종, adaptive smoothing으로 whipsaw 방지.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.kama_eff_trend.config import KamaEffTrendConfig
from src.strategy.kama_eff_trend.preprocessor import preprocess
from src.strategy.kama_eff_trend.signal import generate_signals


@register("kama-eff-trend")
class KamaEffTrendStrategy(BaseStrategy):
    """KAMA Efficiency Trend 전략 구현.

    Efficiency Ratio가 높은 구간(정보 비대칭 활용)에서 KAMA slope 방향으로
    추세 추종. KAMA-price ATR 정규화 거리를 conviction으로 사용.
    """

    def __init__(self, config: KamaEffTrendConfig | None = None) -> None:
        self._config = config or KamaEffTrendConfig()

    @property
    def name(self) -> str:
        return "kama-eff-trend"

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
        config = KamaEffTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "er_period": str(self._config.er_period),
            "kama_period": str(self._config.kama_period),
            "er_threshold": str(self._config.er_threshold),
            "slope_window": str(self._config.slope_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
