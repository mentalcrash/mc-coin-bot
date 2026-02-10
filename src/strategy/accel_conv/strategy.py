"""Acceleration-Conviction Momentum 전략.

가격 가속도(2차 미분)와 캔들 body conviction의 결합으로 추세 지속을 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.accel_conv.config import AccelConvConfig
from src.strategy.accel_conv.preprocessor import preprocess
from src.strategy.accel_conv.signal import generate_signals


@register("accel-conv")
class AccelConvStrategy(BaseStrategy):
    """Acceleration-Conviction Momentum 전략 구현.

    가격 가속도(2차 미분)와 캔들 body conviction이 동시에 양(+)이면
    추세 지속 확률이 극대화된다.
    """

    def __init__(self, config: AccelConvConfig | None = None) -> None:
        self._config = config or AccelConvConfig()

    @property
    def name(self) -> str:
        return "accel-conv"

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
        config = AccelConvConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "smooth_window": str(self._config.smooth_window),
            "signal_threshold": str(self._config.signal_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
