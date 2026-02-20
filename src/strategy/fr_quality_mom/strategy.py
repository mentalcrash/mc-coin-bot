"""FR Quality Momentum 전략.

Funding Rate를 crowding risk meter로 활용하여 모멘텀 품질 필터링.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fr_quality_mom.config import FrQualityMomConfig
from src.strategy.fr_quality_mom.preprocessor import preprocess
from src.strategy.fr_quality_mom.signal import generate_signals


@register("fr-quality-mom")
class FrQualityMomStrategy(BaseStrategy):
    """FR Quality Momentum 전략 구현.

    FR 극단 = 과밀 포지셔닝 → 모멘텀 품질 저하 감지.
    """

    def __init__(self, config: FrQualityMomConfig | None = None) -> None:
        self._config = config or FrQualityMomConfig()

    @property
    def name(self) -> str:
        return "fr-quality-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

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
        config = FrQualityMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "momentum_window": str(self._config.momentum_window),
            "fr_lookback": str(self._config.fr_lookback),
            "fr_crowd_threshold": str(self._config.fr_crowd_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
