"""Asymmetric Volume Response 전략.

Volume-price impact ratio 비대칭으로 informed flow 감지.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.asym_vol_resp.config import AsymVolRespConfig, ShortMode
from src.strategy.asym_vol_resp.preprocessor import preprocess
from src.strategy.asym_vol_resp.signal import generate_signals
from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("asym-vol-resp")
class AsymVolRespStrategy(BaseStrategy):
    """Asymmetric Volume Response 전략 구현.

    Volume-price impact ratio로 informed flow 감지.
    """

    def __init__(self, config: AsymVolRespConfig | None = None) -> None:
        self._config = config or AsymVolRespConfig()

    @property
    def name(self) -> str:
        return "asym-vol-resp"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> AsymVolRespConfig:
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
    def from_params(cls, **params: Any) -> AsymVolRespStrategy:
        config = AsymVolRespConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "impact_window": f"{cfg.impact_window} bars",
            "asym_threshold": f"{cfg.asym_threshold:.1f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
