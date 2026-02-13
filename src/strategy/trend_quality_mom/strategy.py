"""Trend Quality Momentum strategy.

Uses R^2 of linear regression as trend quality conviction scaler for momentum signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.trend_quality_mom.config import ShortMode, TrendQualityMomConfig
from src.strategy.trend_quality_mom.preprocessor import preprocess
from src.strategy.trend_quality_mom.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("trend-quality-mom")
class TrendQualityMomStrategy(BaseStrategy):
    """Trend Quality Momentum strategy implementation.

    R^2 of rolling linear regression measures how orderly a trend is.
    High R^2 + positive slope = strong long conviction.
    High R^2 + negative slope = strong short conviction (if allowed).
    Low R^2 = noisy market = reduced/zero position.
    """

    def __init__(self, config: TrendQualityMomConfig | None = None) -> None:
        self._config = config or TrendQualityMomConfig()

    @property
    def name(self) -> str:
        return "trend-quality-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> TrendQualityMomConfig:
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
    def from_params(cls, **params: Any) -> TrendQualityMomStrategy:
        config = TrendQualityMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "regression_lookback": f"{cfg.regression_lookback} bars",
            "r2_threshold": f"{cfg.r2_threshold:.2f}",
            "mom_lookback": f"{cfg.mom_lookback} bars",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
