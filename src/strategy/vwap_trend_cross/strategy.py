"""VWAP Trend Crossover strategy.

Short/long rolling VWAP crossover detects shifts in average participant entry price.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vwap_trend_cross.config import ShortMode, VwapTrendCrossConfig
from src.strategy.vwap_trend_cross.preprocessor import preprocess
from src.strategy.vwap_trend_cross.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vwap-trend-cross")
class VwapTrendCrossStrategy(BaseStrategy):
    """VWAP Trend Crossover strategy implementation.

    Uses rolling VWAP (volume-weighted average price) crossover to detect
    shifts in the average participant entry price. Crypto whale concentration
    makes VWAP shifts a meaningful trend signal.
    """

    def __init__(self, config: VwapTrendCrossConfig | None = None) -> None:
        self._config = config or VwapTrendCrossConfig()

    @property
    def name(self) -> str:
        return "vwap-trend-cross"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VwapTrendCrossConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        return self._config.warmup_periods()

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
    def from_params(cls, **params: Any) -> VwapTrendCrossStrategy:
        config = VwapTrendCrossConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "vwap_short_window": f"{cfg.vwap_short_window} bars",
            "vwap_long_window": f"{cfg.vwap_long_window} bars",
            "spread_clip": f"{cfg.spread_clip:.3f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
