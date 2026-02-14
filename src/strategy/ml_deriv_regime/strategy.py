"""ML Derivatives Regime 전략.

파생상품 데이터에 ML Elastic Net을 적용하여 CTREND 직교 alpha 추구.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ml_deriv_regime.config import MlDerivRegimeConfig, ShortMode
from src.strategy.ml_deriv_regime.preprocessor import preprocess
from src.strategy.ml_deriv_regime.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("ml-deriv-regime")
class MlDerivRegimeStrategy(BaseStrategy):
    """ML Derivatives Regime 전략 구현.

    파생상품 포지셔닝 데이터(FR)에 CTREND 검증된 ML Elastic Net 적용.
    기술지표와 독립적인 derivatives-only features로 CTREND 직교 alpha 추구.
    """

    def __init__(self, config: MlDerivRegimeConfig | None = None) -> None:
        self._config = config or MlDerivRegimeConfig()

    @property
    def name(self) -> str:
        return "ml-deriv-regime"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

    @property
    def config(self) -> BaseModel:
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

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
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
    def from_params(cls, **params: Any) -> MlDerivRegimeStrategy:
        config = MlDerivRegimeConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "training_window": f"{cfg.training_window}d",
            "prediction_horizon": f"{cfg.prediction_horizon}d",
            "alpha": f"{cfg.alpha:.2f}",
            "fr_lookback": f"{cfg.fr_lookback_short}/{cfg.fr_lookback_long}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
