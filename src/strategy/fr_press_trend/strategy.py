"""Funding Pressure Trend 전략.

가격 추세(SMA cross)가 primary alpha,
FR z-score는 포지셔닝 압력/군중 밀집도 리스크 필터로만 사용.
과도 레버리지 청산 캐스케이드 회피.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.fr_press_trend.config import FrPressTrendConfig, ShortMode
from src.strategy.fr_press_trend.preprocessor import preprocess
from src.strategy.fr_press_trend.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("fr-press-trend")
class FrPressTrendStrategy(BaseStrategy):
    """Funding Pressure Trend 전략 구현.

    SMA cross로 추세 방향을 결정하고,
    FR z-score로 과열/극단 포지셔닝 리스크를 필터링.
    """

    def __init__(self, config: FrPressTrendConfig | None = None) -> None:
        self._config = config or FrPressTrendConfig()

    @property
    def name(self) -> str:
        return "fr-press-trend"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

    @property
    def config(self) -> FrPressTrendConfig:
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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> FrPressTrendStrategy:
        config = FrPressTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "sma_fast": str(cfg.sma_fast),
            "sma_slow": str(cfg.sma_slow),
            "er_threshold": f"{cfg.er_threshold:.2f}",
            "fr_aligned_threshold": f"{cfg.fr_aligned_threshold:.1f}",
            "fr_extreme_threshold": f"{cfg.fr_extreme_threshold:.1f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
