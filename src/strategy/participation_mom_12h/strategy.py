"""Participation Momentum 전략.

거래 참여도(tflow_intensity) Z-score가 높을 때 모멘텀 지속성을 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.participation_mom_12h.config import ParticipationMomConfig
from src.strategy.participation_mom_12h.preprocessor import preprocess
from src.strategy.participation_mom_12h.signal import generate_signals


@register("participation-mom-12h")
class ParticipationMomStrategy(BaseStrategy):
    """Participation Momentum 전략 구현.

    aggTrades 기반 거래 참여도(trades/hr) Z-score와 모멘텀 방향을 결합하여
    정보 비대칭 + 군집행동 비효율을 활용한다.
    """

    def __init__(self, config: ParticipationMomConfig | None = None) -> None:
        self._config = config or ParticipationMomConfig()

    @property
    def name(self) -> str:
        return "participation-mom-12h"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> BaseModel:
        return self._config

    @property
    def required_enrichments(self) -> list[str]:
        """tflow_intensity는 핵심 alpha 원천이므로 필수 enrichment로 선언."""
        return ["tflow_intensity"]

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
            "use_intrabar_trailing_stop": False,
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = ParticipationMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mom_lookback": str(self._config.mom_lookback),
            "intensity_zscore_window": str(self._config.intensity_zscore_window),
            "intensity_long_z": str(self._config.intensity_long_z),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
