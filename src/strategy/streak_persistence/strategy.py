"""Return Streak Persistence 전략.

연속 양봉/음봉 streak에서 군집 행동(FOMO/패닉)으로 인한 추세 지속 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.streak_persistence.config import StreakPersistenceConfig
from src.strategy.streak_persistence.preprocessor import preprocess
from src.strategy.streak_persistence.signal import generate_signals


@register("streak-persistence")
class StreakPersistenceStrategy(BaseStrategy):
    """Return Streak Persistence 전략 구현.

    연속 양봉 3일+ → FOMO 군집 → 추가 상승.
    연속 음봉 3일+ → 패닉 확산 → 추가 하락.
    """

    def __init__(self, config: StreakPersistenceConfig | None = None) -> None:
        self._config = config or StreakPersistenceConfig()

    @property
    def name(self) -> str:
        return "streak-persistence"

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
        config = StreakPersistenceConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "streak_threshold": str(self._config.streak_threshold),
            "max_streak_cap": str(self._config.max_streak_cap),
            "momentum_lookback": str(self._config.momentum_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
