"""Drawdown-Recovery Phase 전략.

드로다운 후 회복 시 손실회피 과소반응으로 인한 follow-through 모멘텀 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.dd_recovery_phase.config import DDRecoveryPhaseConfig
from src.strategy.dd_recovery_phase.preprocessor import preprocess
from src.strategy.dd_recovery_phase.signal import generate_signals


@register("dd-recovery-phase")
class DDRecoveryPhaseStrategy(BaseStrategy):
    """Drawdown-Recovery Phase 전략 구현.

    Prospect theory 기반: 드로다운 후 50%+ 회복 시
    손실회피 투매자의 과소반응으로 follow-through 모멘텀 발생.
    """

    def __init__(self, config: DDRecoveryPhaseConfig | None = None) -> None:
        self._config = config or DDRecoveryPhaseConfig()

    @property
    def name(self) -> str:
        return "dd-recovery-phase"

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
        config = DDRecoveryPhaseConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "dd_threshold": str(self._config.dd_threshold),
            "recovery_ratio": str(self._config.recovery_ratio),
            "dd_lookback": str(self._config.dd_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
