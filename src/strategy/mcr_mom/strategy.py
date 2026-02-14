"""Momentum Crash Filter strategy.

Standard momentum + VoV-based crash filter as defensive override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.mcr_mom.config import McrMomConfig, ShortMode
from src.strategy.mcr_mom.preprocessor import preprocess
from src.strategy.mcr_mom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("mcr-mom")
class McrMomStrategy(BaseStrategy):
    """Momentum Crash Filter strategy implementation.

    Uses VoV as defensive crash filter, not as alpha source.
    """

    def __init__(self, config: McrMomConfig | None = None) -> None:
        self._config = config or McrMomConfig()

    @property
    def name(self) -> str:
        return "mcr-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> McrMomConfig:
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
    def from_params(cls, **params: Any) -> McrMomStrategy:
        config = McrMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "mom_lookback": f"{cfg.mom_lookback} bars",
            "vov_window": f"{cfg.vov_window} bars",
            "vov_crash_threshold": f"{cfg.vov_crash_threshold:.0%}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
