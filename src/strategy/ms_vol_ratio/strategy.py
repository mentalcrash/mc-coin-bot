"""Multi-Scale Volatility Ratio 전략.

단기/장기 변동성 비율로 정보 도착 → breakout timing 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ms_vol_ratio.config import MSVolRatioConfig, ShortMode
from src.strategy.ms_vol_ratio.preprocessor import preprocess
from src.strategy.ms_vol_ratio.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("ms-vol-ratio")
class MSVolRatioStrategy(BaseStrategy):
    """Multi-Scale Volatility Ratio 전략 구현.

    단기vol/장기vol ratio로 breakout timing 포착.
    """

    def __init__(self, config: MSVolRatioConfig | None = None) -> None:
        self._config = config or MSVolRatioConfig()

    @property
    def name(self) -> str:
        return "ms-vol-ratio"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> MSVolRatioConfig:
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
    def from_params(cls, **params: Any) -> MSVolRatioStrategy:
        config = MSVolRatioConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "short/long_vol": f"{cfg.short_vol_window}/{cfg.long_vol_window}",
            "ratio_upper": f"{cfg.ratio_upper:.2f}",
            "ratio_lower": f"{cfg.ratio_lower:.2f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
