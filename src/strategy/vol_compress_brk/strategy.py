"""Vol-Compression Breakout 전략.

ATR compression/expansion 전이 감지로 breakout continuation 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vol_compress_brk.config import VolCompressBrkConfig
from src.strategy.vol_compress_brk.preprocessor import preprocess
from src.strategy.vol_compress_brk.signal import generate_signals


@register("vol-compress-brk")
class VolCompressBrkStrategy(BaseStrategy):
    """Vol-Compression Breakout 전략 구현.

    ATR fast/slow ratio로 compression → expansion 전이 포착.
    """

    def __init__(self, config: VolCompressBrkConfig | None = None) -> None:
        self._config = config or VolCompressBrkConfig()

    @property
    def name(self) -> str:
        return "vol-compress-brk"

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
        config = VolCompressBrkConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "atr_fast": str(self._config.atr_fast),
            "atr_slow": str(self._config.atr_slow),
            "compress_threshold": str(self._config.compress_threshold),
            "expand_threshold": str(self._config.expand_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
