"""Donchian Filtered 전략.

Donch-Multi 3-scale consensus에 funding rate crowd filter를 적용하여
과열 포지셔닝 시 진입을 억제한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.donch_filtered.config import DonchFilteredConfig
from src.strategy.donch_filtered.preprocessor import preprocess
from src.strategy.donch_filtered.signal import generate_signals


@register("donch-filtered")
class DonchFilteredStrategy(BaseStrategy):
    """Donchian Filtered 전략 구현.

    Donch-Multi 3-scale consensus + funding rate crowd filter.
    Derivatives 데이터 없으면 pure donch-multi로 동작 (graceful degradation).
    """

    def __init__(self, config: DonchFilteredConfig | None = None) -> None:
        self._config = config or DonchFilteredConfig()

    @property
    def name(self) -> str:
        return "donch-filtered"

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
        config = DonchFilteredConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "lookbacks": (
                f"{self._config.lookback_short}/{self._config.lookback_mid}"
                f"/{self._config.lookback_long}"
            ),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
            "fr_suppress_threshold": str(self._config.fr_suppress_threshold),
        }
