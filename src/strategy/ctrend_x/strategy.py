"""CTREND-X 전략.

GradientBoosting 기반 28-feature trend prediction. CTREND의 비선형 확장.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ctrend_x.config import CTRENDXConfig, ShortMode
from src.strategy.ctrend_x.preprocessor import preprocess
from src.strategy.ctrend_x.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("ctrend-x")
class CTRENDXStrategy(BaseStrategy):
    """CTREND-X (GradientBoosting Trend Factor) 전략.

    CTREND와 동일한 28개 기술적 지표를 GradientBoostingRegressor로
    결합하여 비선형 패턴을 캡처합니다.
    """

    def __init__(self, config: CTRENDXConfig | None = None) -> None:
        self._config = config or CTRENDXConfig()

    @property
    def name(self) -> str:
        return "ctrend-x"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> CTRENDXConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def run_incremental(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySignals]:
        """Incremental 모드: 마지막 시그널만 효율적으로 계산."""
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
    def from_params(cls, **params: Any) -> CTRENDXStrategy:
        config = CTRENDXConfig(**params)
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
            "learning_rate": f"{cfg.learning_rate:.3f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
