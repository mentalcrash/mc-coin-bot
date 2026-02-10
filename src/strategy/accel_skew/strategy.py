"""Acceleration-Skewness Signal 전략.

가격 가속도와 rolling skewness를 결합하여 momentum quality를 필터링.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.accel_skew.config import AccelSkewConfig
from src.strategy.accel_skew.preprocessor import preprocess
from src.strategy.accel_skew.signal import generate_signals


@register("accel-skew")
class AccelSkewStrategy(BaseStrategy):
    """Acceleration-Skewness Signal 전략 구현.

    가격 가속도가 양(+)이고 rolling skewness도 양(+)이면 우상향 테일이 reward로 전환.
    Skewness가 음(-)이면 crash risk -> 거래 중단.
    """

    def __init__(self, config: AccelSkewConfig | None = None) -> None:
        self._config = config or AccelSkewConfig()

    @property
    def name(self) -> str:
        return "accel-skew"

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
        config = AccelSkewConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "acc_smooth_window": str(self._config.acc_smooth_window),
            "skew_window": str(self._config.skew_window),
            "skew_threshold": str(self._config.skew_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
