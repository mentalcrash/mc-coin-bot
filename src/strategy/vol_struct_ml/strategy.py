"""Volatility Structure ML 전략.

Vol clustering + VoV premium 기반 13종 vol feature를 Elastic Net으로 결합.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_struct_ml.config import ShortMode, VolStructMLConfig
from src.strategy.vol_struct_ml.preprocessor import preprocess
from src.strategy.vol_struct_ml.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-struct-ml")
class VolStructMLStrategy(BaseStrategy):
    """Volatility Structure ML 전략 구현.

    GK/Parkinson/YZ vol, VoV, fractal_dimension, hurst, ER, ADX 등
    13종 vol 기반 feature를 Elastic Net으로 결합하여 방향성 예측.
    """

    def __init__(self, config: VolStructMLConfig | None = None) -> None:
        self._config = config or VolStructMLConfig()

    @property
    def name(self) -> str:
        return "vol-struct-ml"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolStructMLConfig:
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
    def from_params(cls, **params: Any) -> VolStructMLStrategy:
        config = VolStructMLConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "training_window": f"{cfg.training_window} bars",
            "alpha": f"{cfg.alpha:.2f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
