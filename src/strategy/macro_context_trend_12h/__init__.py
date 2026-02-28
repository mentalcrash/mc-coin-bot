"""Macro-Context-Trend 12H: EMA 추세 + 매크로 리스크 선호도 컨텍스트."""

from src.strategy.macro_context_trend_12h.config import MacroContextTrendConfig, ShortMode
from src.strategy.macro_context_trend_12h.preprocessor import preprocess
from src.strategy.macro_context_trend_12h.signal import generate_signals
from src.strategy.macro_context_trend_12h.strategy import MacroContextTrend12hStrategy

__all__ = [
    "MacroContextTrend12hStrategy",
    "MacroContextTrendConfig",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
