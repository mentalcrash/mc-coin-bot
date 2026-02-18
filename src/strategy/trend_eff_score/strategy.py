"""Trend Efficiency Scorer 전략.

ER로 추세 품질을 측정, 품질 높을 때만 다중 수평선 모멘텀 합의로 방향 결정.
행동 편향(군집 행동) 기반 비효율 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.trend_eff_score.config import ShortMode, TrendEffScoreConfig
from src.strategy.trend_eff_score.preprocessor import preprocess
from src.strategy.trend_eff_score.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("trend-eff-score")
class TrendEffScoreStrategy(BaseStrategy):
    """Trend Efficiency Scorer 전략 구현.

    ER(Efficiency Ratio)로 추세 품질을 측정하고,
    품질이 높을 때만 다중 수평선 ROC 합의(scoring)로 방향을 결정.
    """

    def __init__(self, config: TrendEffScoreConfig | None = None) -> None:
        self._config = config or TrendEffScoreConfig()

    @property
    def name(self) -> str:
        return "trend-eff-score"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> TrendEffScoreConfig:
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
    def from_params(cls, **params: Any) -> TrendEffScoreStrategy:
        config = TrendEffScoreConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "er_window": str(cfg.er_window),
            "roc_periods": f"{cfg.roc_short}/{cfg.roc_medium}/{cfg.roc_long}",
            "er_threshold": f"{cfg.er_threshold:.2f}",
            "min_score": str(cfg.min_score),
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
