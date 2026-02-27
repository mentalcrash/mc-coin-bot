"""Wavelet-Channel 8H 전략.

DWT denoised 3종 채널(Donchian/Keltner/BB) x 3스케일(22/66/132) 앙상블 breakout으로
노이즈 제거된 로버스트 추세 추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.wavelet_channel_8h.config import WaveletChannel8hConfig
from src.strategy.wavelet_channel_8h.preprocessor import preprocess
from src.strategy.wavelet_channel_8h.signal import generate_signals


@register("wavelet-channel-8h")
class WaveletChannel8hStrategy(BaseStrategy):
    """Wavelet-Channel 8H 전략 구현.

    DWT denoising으로 close price 노이즈를 제거한 후
    3종 채널(DC/KC/BB) x 3스케일(22/66/132)의 9개 sub-signal
    consensus로 로버스트한 추세 방향을 도출한다.
    """

    def __init__(self, config: WaveletChannel8hConfig | None = None) -> None:
        self._config = config or WaveletChannel8hConfig()

    @property
    def name(self) -> str:
        return "wavelet-channel-8h"

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
        config = WaveletChannel8hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "wavelet": f"{self._config.wavelet_family}/L{self._config.wavelet_level}",
            "scales": f"{self._config.scale_short}/{self._config.scale_mid}/{self._config.scale_long}",
            "entry_threshold": str(self._config.entry_threshold),
            "bb_std": str(self._config.bb_std),
            "kc_mult": str(self._config.kc_mult),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
