"""Fractal-Filtered Momentum 전략.

Fractal dimension으로 deterministic regime 감지, 해당 구간에서만 trend following.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.fractal_mom.config import FractalMomConfig, ShortMode
from src.strategy.fractal_mom.preprocessor import preprocess
from src.strategy.fractal_mom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("fractal-mom")
class FractalMomStrategy(BaseStrategy):
    """Fractal-Filtered Momentum 전략 구현.

    D < 1.5 = deterministic regime에서만 trend following 활성화.
    """

    def __init__(self, config: FractalMomConfig | None = None) -> None:
        self._config = config or FractalMomConfig()

    @property
    def name(self) -> str:
        return "fractal-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> FractalMomConfig:
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
    def from_params(cls, **params: Any) -> FractalMomStrategy:
        config = FractalMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "fractal_period": f"{cfg.fractal_period} bars",
            "fractal_threshold": f"{cfg.fractal_threshold:.2f}",
            "mom_fast/slow": f"{cfg.mom_fast}/{cfg.mom_slow}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
