"""Cascade Momentum 전략.

연속 동방향 bar streak 기반 허딩/FOMO 캐스케이드 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.cascade_mom.config import CascadeMomConfig
from src.strategy.cascade_mom.preprocessor import preprocess
from src.strategy.cascade_mom.signal import generate_signals


@register("cascade-mom")
class CascadeMomStrategy(BaseStrategy):
    """Cascade Momentum 전략 구현.

    Streak count x body/ATR 정규화로 herding cascade 포착.
    """

    def __init__(self, config: CascadeMomConfig | None = None) -> None:
        self._config = config or CascadeMomConfig()

    @property
    def name(self) -> str:
        return "cascade-mom"

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
        config = CascadeMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "min_streak": str(self._config.min_streak),
            "score_threshold": str(self._config.score_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
