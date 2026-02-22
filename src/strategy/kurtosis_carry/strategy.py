"""Kurtosis Carry 전략.

고첨도(fat tail) 리스크 프리미엄의 축적과 정상화 과정에서 carry 수취.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.kurtosis_carry.config import KurtosisCarryConfig
from src.strategy.kurtosis_carry.preprocessor import preprocess
from src.strategy.kurtosis_carry.signal import generate_signals


@register("kurtosis-carry")
class KurtosisCarryStrategy(BaseStrategy):
    """Kurtosis Carry 전략 구현.

    고첨도 → 리스크 프리미엄 확대 → 저첨도 전환 시 프리미엄 수취.
    Amaya et al. (JFE 2015) 학술 근거.
    """

    def __init__(self, config: KurtosisCarryConfig | None = None) -> None:
        self._config = config or KurtosisCarryConfig()

    @property
    def name(self) -> str:
        return "kurtosis-carry"

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
        config = KurtosisCarryConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "kurtosis_window": str(self._config.kurtosis_window),
            "kurtosis_long_window": str(self._config.kurtosis_long_window),
            "high_kurtosis_zscore": str(self._config.high_kurtosis_zscore),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
