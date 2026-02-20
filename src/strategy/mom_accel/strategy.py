"""Momentum Acceleration 전략.

모멘텀의 2차 미분(가속도)으로 추세 성숙도 측정, velocity-acceleration alignment 시 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.mom_accel.config import MomAccelConfig
from src.strategy.mom_accel.preprocessor import preprocess
from src.strategy.mom_accel.signal import generate_signals


@register("mom-accel")
class MomAccelStrategy(BaseStrategy):
    """Momentum Acceleration 전략 구현.

    속도와 가속도가 같은 방향일 때만 추종.
    """

    def __init__(self, config: MomAccelConfig | None = None) -> None:
        self._config = config or MomAccelConfig()

    @property
    def name(self) -> str:
        return "mom-accel"

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
        config = MomAccelConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fast_roc": str(self._config.fast_roc),
            "slow_roc": str(self._config.slow_roc),
            "accel_window": str(self._config.accel_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
