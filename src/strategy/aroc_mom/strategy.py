"""Adaptive ROC Momentum strategy.

Dynamically adjust momentum lookback based on volatility regime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.aroc_mom.config import ArocMomConfig, ShortMode
from src.strategy.aroc_mom.preprocessor import preprocess
from src.strategy.aroc_mom.signal import generate_signals
from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("aroc-mom")
class ArocMomStrategy(BaseStrategy):
    """Adaptive ROC Momentum strategy implementation.

    High vol -> short ROC lookback (fast reaction).
    Low vol -> long ROC lookback (noise filter).
    """

    def __init__(self, config: ArocMomConfig | None = None) -> None:
        self._config = config or ArocMomConfig()

    @property
    def name(self) -> str:
        return "aroc-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> ArocMomConfig:
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
    def from_params(cls, **params: Any) -> ArocMomStrategy:
        config = ArocMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "fast_lookback": f"{cfg.fast_lookback} bars",
            "slow_lookback": f"{cfg.slow_lookback} bars",
            "vol_rank_window": f"{cfg.vol_rank_window} bars",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
