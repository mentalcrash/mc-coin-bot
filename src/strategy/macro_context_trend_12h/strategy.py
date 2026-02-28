"""Macro-Context-Trend 12H 전략.

12H EMA 추세 시그널 + 1D 매크로 리스크 선호도 컨텍스트 사이징.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.macro_context_trend_12h.config import MacroContextTrendConfig
from src.strategy.macro_context_trend_12h.preprocessor import preprocess
from src.strategy.macro_context_trend_12h.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("macro-context-trend-12h")
class MacroContextTrend12hStrategy(BaseStrategy):
    """12H EMA 추세 + 매크로 컨텍스트 사이징 전략."""

    def __init__(self, config: MacroContextTrendConfig | None = None) -> None:
        self._config = config or MacroContextTrendConfig()

    @property
    def name(self) -> str:
        return "macro-context-trend-12h"

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
        """PortfolioManagerConfig kwargs."""
        return {
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> MacroContextTrend12hStrategy:
        """Parameter sweep용 팩토리."""
        config = MacroContextTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        """CLI startup panel 표시 정보."""
        return {
            "ema_fast": str(self._config.ema_fast),
            "ema_slow": str(self._config.ema_slow),
            "macro_risk_weight": str(self._config.macro_risk_weight),
            "macro_window": str(self._config.macro_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
