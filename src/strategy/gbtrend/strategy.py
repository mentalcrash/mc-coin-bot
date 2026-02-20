"""GBTrend 전략.

모멘텀 중심 12-feature GradientBoosting trend prediction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.gbtrend.config import GBTrendConfig, ShortMode
from src.strategy.gbtrend.preprocessor import preprocess
from src.strategy.gbtrend.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("gbtrend")
class GBTrendStrategy(BaseStrategy):
    """GBTrend (Gradient Boosting Momentum Trend) 전략.

    12개 모멘텀 중심 feature를 GradientBoostingRegressor로 결합.
    CTREND 대비 축소된 feature set으로 과적합 억제 + 상관도 차별화.
    """

    def __init__(self, config: GBTrendConfig | None = None) -> None:
        self._config = config or GBTrendConfig()

    @property
    def name(self) -> str:
        return "gbtrend"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> GBTrendConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def run_incremental(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySignals]:
        """Incremental 모드: 마지막 시그널만 계산."""
        self.validate_input(df)
        processed_df = self.preprocess(df)
        signals = generate_signals(processed_df, self._config, predict_last_only=True)
        return processed_df, signals

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
    def from_params(cls, **params: Any) -> GBTrendStrategy:
        config = GBTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "training_window": f"{cfg.training_window}d",
            "n_estimators": str(cfg.n_estimators),
            "max_depth": str(cfg.max_depth),
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
