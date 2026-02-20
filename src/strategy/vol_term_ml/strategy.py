"""Vol-Term ML 전략.

다중 RV term structure + Ridge regression 기반 방향 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_term_ml.config import ShortMode, VolTermMLConfig
from src.strategy.vol_term_ml.preprocessor import preprocess
from src.strategy.vol_term_ml.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-term-ml")
class VolTermMLStrategy(BaseStrategy):
    """Vol-Term ML (Volatility Term Structure ML) 전략.

    5종 RV + 3종 vol ratio + Parkinson + GK = 10개 feature를
    Rolling Ridge로 결합하여 forward return 방향 예측.
    """

    def __init__(self, config: VolTermMLConfig | None = None) -> None:
        self._config = config or VolTermMLConfig()

    @property
    def name(self) -> str:
        return "vol-term-ml"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolTermMLConfig:
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
    def from_params(cls, **params: Any) -> VolTermMLStrategy:
        config = VolTermMLConfig(**params)
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
            "ridge_alpha": f"{cfg.ridge_alpha:.2f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
