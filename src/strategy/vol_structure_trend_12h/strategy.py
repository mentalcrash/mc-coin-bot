"""Vol-Structure-Trend 12H 전략.

3종 변동성 추정기(GK/PK/YZ) 합의 x 3-scale 앙상블 추세 전략.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_structure_trend_12h.config import VolStructureTrendConfig
from src.strategy.vol_structure_trend_12h.preprocessor import preprocess
from src.strategy.vol_structure_trend_12h.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("vol-structure-trend-12h")
class VolStructureTrend12hStrategy(BaseStrategy):
    """3종 변동성 추정기 합의 기반 추세 전략."""

    def __init__(self, config: VolStructureTrendConfig | None = None) -> None:
        self._config = config or VolStructureTrendConfig()

    @property
    def name(self) -> str:
        return "vol-structure-trend-12h"

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
        """PortfolioManagerConfig kwargs."""
        return {
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> VolStructureTrend12hStrategy:
        """Parameter sweep용 팩토리."""
        config = VolStructureTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        """CLI startup panel 표시 정보."""
        return {
            "scale_short": str(self._config.scale_short),
            "scale_mid": str(self._config.scale_mid),
            "scale_long": str(self._config.scale_long),
            "roc_lookback": str(self._config.roc_lookback),
            "vol_agreement_threshold": str(self._config.vol_agreement_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
