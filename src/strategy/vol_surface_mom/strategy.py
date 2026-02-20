"""Volatility Surface Momentum 전략.

GK/YZ/Parkinson vol 비율로 시장 미시구조 정보를 인코딩하여 모멘텀 시그널 생성.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vol_surface_mom.config import VolSurfaceMomConfig
from src.strategy.vol_surface_mom.preprocessor import preprocess
from src.strategy.vol_surface_mom.signal import generate_signals


@register("vol-surface-mom")
class VolSurfaceMomStrategy(BaseStrategy):
    """Volatility Surface Momentum 전략 구현.

    GK/Parkinson/YZ vol 비율 변화로 미시구조 정보를 인코딩하여 모멘텀 품질 판단.
    """

    def __init__(self, config: VolSurfaceMomConfig | None = None) -> None:
        self._config = config or VolSurfaceMomConfig()

    @property
    def name(self) -> str:
        return "vol-surface-mom"

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
        config = VolSurfaceMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "gk_window": str(self._config.gk_window),
            "pk_window": str(self._config.pk_window),
            "yz_window": str(self._config.yz_window),
            "gk_pk_long_threshold": str(self._config.gk_pk_long_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
