"""Vol-of-Vol Momentum 전략.

GK realized vol의 VoV로 vol regime 안정성 측정하여 모멘텀 필터링.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vov_mom.config import VovMomConfig
from src.strategy.vov_mom.preprocessor import preprocess
from src.strategy.vov_mom.signal import generate_signals


@register("vov-mom")
class VovMomStrategy(BaseStrategy):
    """Vol-of-Vol Momentum 전략 구현.

    Low VoV=안정적 추세→모멘텀 신뢰, High VoV=불안정→관망.
    Du et al. 2025 VoV risk premium 기반.
    """

    def __init__(self, config: VovMomConfig | None = None) -> None:
        self._config = config or VovMomConfig()

    @property
    def name(self) -> str:
        return "vov-mom"

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
        config = VovMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "gk_window": str(self._config.gk_window),
            "vov_window": str(self._config.vov_window),
            "vov_threshold": str(self._config.vov_threshold),
            "mom_lookback": str(self._config.mom_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
