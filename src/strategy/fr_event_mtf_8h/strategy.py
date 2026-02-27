"""Funding Rate Event Trigger + 12H Momentum Context 전략 (8H TF).

펀딩비 극단 z-score 이벤트와 EMA 추세 컨텍스트를 결합하여 crowded positioning reversal 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fr_event_mtf_8h.config import FrEventMtf8hConfig
from src.strategy.fr_event_mtf_8h.preprocessor import preprocess
from src.strategy.fr_event_mtf_8h.signal import generate_signals


@register("fr-event-mtf-8h")
class FrEventMtf8hStrategy(BaseStrategy):
    """Funding Rate Event Trigger + 12H Momentum Context 전략 구현.

    8H TF에서 펀딩비 극단 z-score(crowded positioning)가
    EMA 추세와 반대 방향일 때 이벤트 트리거로 진입한다.
    """

    def __init__(self, config: FrEventMtf8hConfig | None = None) -> None:
        self._config = config or FrEventMtf8hConfig()

    @property
    def name(self) -> str:
        return "fr-event-mtf-8h"

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
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = FrEventMtf8hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fr_ma_window": str(self._config.fr_ma_window),
            "fr_zscore_window": str(self._config.fr_zscore_window),
            "fr_extreme_threshold": str(self._config.fr_extreme_threshold),
            "ema_fast/slow": f"{self._config.ema_fast_period}/{self._config.ema_slow_period}",
            "min_hold_bars": str(self._config.min_hold_bars),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
