"""Donchian Multi-Scale 전략.

3-scale Donchian breakout 앙상블(20/40/80)로 regime-robust 추세 추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.donch_multi.config import DonchMultiConfig
from src.strategy.donch_multi.preprocessor import preprocess
from src.strategy.donch_multi.signal import generate_signals


@register("donch-multi")
class DonchMultiStrategy(BaseStrategy):
    """Donchian Multi-Scale 전략 구현.

    3개 Donchian Channel(20/40/80)의 breakout consensus로
    regime-robust한 추세 방향을 도출한다.
    """

    def __init__(self, config: DonchMultiConfig | None = None) -> None:
        self._config = config or DonchMultiConfig()

    @property
    def name(self) -> str:
        return "donch-multi"

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
        config = DonchMultiConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "lookbacks": f"{self._config.lookback_short}/{self._config.lookback_mid}/{self._config.lookback_long}",
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
