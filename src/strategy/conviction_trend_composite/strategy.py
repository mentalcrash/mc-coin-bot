"""Conviction Trend Composite 전략.

가격 모멘텀 + OBV/RV composite conviction + 레짐 적응형 사이징.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.conviction_trend_composite.config import (
    ConvictionTrendCompositeConfig,
    ShortMode,
)
from src.strategy.conviction_trend_composite.preprocessor import preprocess
from src.strategy.conviction_trend_composite.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("conviction-trend-composite")
class ConvictionTrendCompositeStrategy(BaseStrategy):
    """Conviction Trend Composite 전략 구현.

    가격 모멘텀이 방향, OBV 거래량 구조 + RV ratio가 conviction modifier.
    독립 데이터 소스 합의 기반 진입, 레짐 확률 가중 사이징.
    """

    def __init__(self, config: ConvictionTrendCompositeConfig | None = None) -> None:
        self._config = config or ConvictionTrendCompositeConfig()

    @property
    def name(self) -> str:
        return "conviction-trend-composite"

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
        config = ConvictionTrendCompositeConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "mom": f"{cfg.mom_fast}/{cfg.mom_slow}",
            "obv": f"{cfg.obv_fast}/{cfg.obv_slow}",
            "rv": f"{cfg.rv_short_window}/{cfg.rv_long_window}",
            "conviction_threshold": f"{cfg.conviction_threshold:.1f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "short_mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
