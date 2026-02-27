"""Volatility Compression Breakout + Multi-TF 전략 (8H).

Yang-Zhang 변동성 압축 감지 → Donchian 돌파 + 모멘텀 합의로 진입,
변동성 팽창 시 퇴장. 8H TF에서 12H 모멘텀을 근사하는 Multi-TF 접근.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vol_compress_mtf_8h.config import VolCompressMtf8hConfig
from src.strategy.vol_compress_mtf_8h.preprocessor import preprocess
from src.strategy.vol_compress_mtf_8h.signal import generate_signals


@register("vol-compress-mtf-8h")
class VolCompressMtf8hStrategy(BaseStrategy):
    """Volatility Compression Breakout + Multi-TF 전략 구현 (8H).

    Yang-Zhang 변동성의 단기/장기 비율로 압축 구간을 감지하고,
    Donchian 돌파 + 모멘텀 방향 합의 시 진입한다.
    변동성 팽창 시 포지션을 청산한다.
    """

    def __init__(self, config: VolCompressMtf8hConfig | None = None) -> None:
        self._config = config or VolCompressMtf8hConfig()

    @property
    def name(self) -> str:
        return "vol-compress-mtf-8h"

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
        config = VolCompressMtf8hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "yz_windows": f"{self._config.yz_short_window}/{self._config.yz_long_window}",
            "compression_threshold": str(self._config.compression_threshold),
            "expansion_threshold": str(self._config.expansion_threshold),
            "dc_lookback": str(self._config.dc_lookback),
            "mom_lookback": str(self._config.mom_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
