"""Regime-Adaptive Multi-Lookback Momentum 전략.

다중 lookback 모멘텀을 RegimeService 확률 기반으로 연속 가중 혼합.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.regime_adaptive_mom.config import RegimeAdaptiveMomConfig, ShortMode
from src.strategy.regime_adaptive_mom.preprocessor import preprocess
from src.strategy.regime_adaptive_mom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("regime-adaptive-momentum")
class RegimeAdaptiveMomStrategy(BaseStrategy):
    """Regime-Adaptive Multi-Lookback Momentum 전략 구현.

    다중 lookback(20d/60d/120d) 모멘텀을 레짐 확률 기반으로 연속 가중 혼합.
    Trending -> 빠른 반응, Volatile -> 안정적 장기 추세.
    """

    def __init__(self, config: RegimeAdaptiveMomConfig | None = None) -> None:
        self._config = config or RegimeAdaptiveMomConfig()

    @property
    def name(self) -> str:
        return "regime-adaptive-momentum"

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
        config = RegimeAdaptiveMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "lookbacks": f"{cfg.fast_lookback}/{cfg.mid_lookback}/{cfg.slow_lookback}",
            "signal_threshold": f"{cfg.signal_threshold:.3f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "short_mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
