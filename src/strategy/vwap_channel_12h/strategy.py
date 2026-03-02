"""VWAP-Channel Multi-Scale 전략.

VWAP 기반 3-scale channel breakout consensus로 volume-weighted 추세 추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vwap_channel_12h.config import VwapChannelConfig
from src.strategy.vwap_channel_12h.preprocessor import preprocess
from src.strategy.vwap_channel_12h.signal import generate_signals


@register("vwap-channel-12h")
class VwapChannelStrategy(BaseStrategy):
    """VWAP-Channel Multi-Scale 전략 구현.

    VWAP(Volume-Weighted Average Price) 기반 3-scale channel breakout 앙상블.
    거래량이 동의한 가격 수준 대비 breakout consensus로 로버스트한 추세 방향을 도출.
    """

    def __init__(self, config: VwapChannelConfig | None = None) -> None:
        self._config = config or VwapChannelConfig()

    @property
    def name(self) -> str:
        return "vwap-channel-12h"

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
        config = VwapChannelConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "scales": (
                f"{self._config.scale_short}/{self._config.scale_mid}/{self._config.scale_long}"
            ),
            "band_multiplier": str(self._config.band_multiplier),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
