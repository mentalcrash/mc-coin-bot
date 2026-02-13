"""Vol-Efficiency Momentum 전략.

Parkinson(High-Low) vol과 Close-to-Close vol 비율로 방향성 conviction 측정.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.dvr_mom.config import DvrMomConfig
from src.strategy.dvr_mom.preprocessor import preprocess
from src.strategy.dvr_mom.signal import generate_signals


@register("dvr-mom")
class DvrMomStrategy(BaseStrategy):
    """Vol-Efficiency Momentum 전략 구현.

    DVR(CC vol / Parkinson vol) 낮으면 방향성 바 → 모멘텀 conviction 강화.
    """

    def __init__(self, config: DvrMomConfig | None = None) -> None:
        self._config = config or DvrMomConfig()

    @property
    def name(self) -> str:
        return "dvr-mom"

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
        config = DvrMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "dvr_window": str(self._config.dvr_window),
            "mom_lookback": str(self._config.mom_lookback),
            "dvr_threshold": str(self._config.dvr_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
