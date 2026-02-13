"""Volume-Impulse Momentum 전략.

비정상 거래량 spike + 방향성 bar 기반 informed trading continuation 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vol_impulse_mom.config import VolImpulseMomConfig
from src.strategy.vol_impulse_mom.preprocessor import preprocess
from src.strategy.vol_impulse_mom.signal import generate_signals


@register("vol-impulse-mom")
class VolImpulseMomStrategy(BaseStrategy):
    """Volume-Impulse Momentum 전략 구현.

    Volume spike + directional bar = informed trading → continuation.
    """

    def __init__(self, config: VolImpulseMomConfig | None = None) -> None:
        self._config = config or VolImpulseMomConfig()

    @property
    def name(self) -> str:
        return "vol-impulse-mom"

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
        config = VolImpulseMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "vol_spike_window": str(self._config.vol_spike_window),
            "vol_spike_multiplier": str(self._config.vol_spike_multiplier),
            "body_ratio_threshold": str(self._config.body_ratio_threshold),
            "hold_bars": str(self._config.hold_bars),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
