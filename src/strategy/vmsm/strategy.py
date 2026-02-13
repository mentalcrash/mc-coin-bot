"""Volume-Gated Multi-Scale Momentum 전략.

다중 시간 수평 ROC 앙상블 + 볼륨 게이트 기반 모멘텀.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vmsm.config import VmsmConfig
from src.strategy.vmsm.preprocessor import preprocess
from src.strategy.vmsm.signal import generate_signals


@register("vmsm")
class VmsmStrategy(BaseStrategy):
    """Volume-Gated Multi-Scale Momentum 전략 구현.

    3-scale ROC 앙상블 + 볼륨 게이트로 conviction 강화.
    """

    def __init__(self, config: VmsmConfig | None = None) -> None:
        self._config = config or VmsmConfig()

    @property
    def name(self) -> str:
        return "vmsm"

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
        config = VmsmConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "roc_short": str(self._config.roc_short),
            "roc_mid": str(self._config.roc_mid),
            "roc_long": str(self._config.roc_long),
            "vol_gate_multiplier": str(self._config.vol_gate_multiplier),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
