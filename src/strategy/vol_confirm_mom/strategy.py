"""Volume-Confirmed Momentum strategy.

Momentum + volume trend confirmation: enter only when rising volume confirms momentum direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_confirm_mom.config import ShortMode, VolConfirmMomConfig
from src.strategy.vol_confirm_mom.preprocessor import preprocess
from src.strategy.vol_confirm_mom.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-confirm-mom")
class VolConfirmMomStrategy(BaseStrategy):
    """Volume-Confirmed Momentum strategy implementation.

    Enters momentum trades only when volume trend confirms the direction.
    Rising volume (short SMA > long SMA) = increasing participation = fuel for momentum.
    Volume ratio provides continuous conviction scaling.
    """

    def __init__(self, config: VolConfirmMomConfig | None = None) -> None:
        self._config = config or VolConfirmMomConfig()

    @property
    def name(self) -> str:
        return "vol-confirm-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolConfirmMomConfig:
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
    def from_params(cls, **params: Any) -> VolConfirmMomStrategy:
        config = VolConfirmMomConfig(**params)
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
            "vol_short_window": f"{cfg.vol_short_window} bars",
            "vol_long_window": f"{cfg.vol_long_window} bars",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
