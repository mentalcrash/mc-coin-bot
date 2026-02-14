"""Efficiency Breakout strategy.

Kaufman Efficiency Ratio threshold breakout for noise-to-trend transition detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.eff_brk.config import EffBrkConfig, ShortMode
from src.strategy.eff_brk.preprocessor import preprocess
from src.strategy.eff_brk.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("eff-brk")
class EffBrkStrategy(BaseStrategy):
    """Efficiency Breakout strategy implementation.

    Uses Kaufman ER as primary breakout detector to identify trend initiation.
    """

    def __init__(self, config: EffBrkConfig | None = None) -> None:
        self._config = config or EffBrkConfig()

    @property
    def name(self) -> str:
        return "eff-brk"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> EffBrkConfig:
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
    def from_params(cls, **params: Any) -> EffBrkStrategy:
        config = EffBrkConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "er_period": f"{cfg.er_period} bars",
            "er_threshold": f"{cfg.er_threshold:.2f}",
            "er_exit_threshold": f"{cfg.er_exit_threshold:.2f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
