"""MAD-Channel Multi-Scale Trend 전략.

3종 채널(Donchian/Keltner/MAD) x 3스케일(20/60/150) 앙상블 breakout으로 로버스트 추세 추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.mad_channel_trend.config import MadChannelTrendConfig
from src.strategy.mad_channel_trend.preprocessor import preprocess
from src.strategy.mad_channel_trend.signal import generate_signals


@register("mad-channel-trend")
class MadChannelTrendStrategy(BaseStrategy):
    """MAD-Channel Multi-Scale Trend 전략 구현.

    3종 채널 유형(Donchian/Keltner/MAD) x 3스케일(20/60/150)의 9개 sub-signal
    consensus로 로버스트한 추세 방향을 도출한다.
    """

    def __init__(self, config: MadChannelTrendConfig | None = None) -> None:
        self._config = config or MadChannelTrendConfig()

    @property
    def name(self) -> str:
        return "mad-channel-trend"

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
        config = MadChannelTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "scales": f"{self._config.scale_short}/{self._config.scale_mid}/{self._config.scale_long}",
            "entry_threshold": str(self._config.entry_threshold),
            "mad_multiplier": str(self._config.mad_multiplier),
            "keltner_multiplier": str(self._config.keltner_multiplier),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
