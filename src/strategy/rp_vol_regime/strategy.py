"""Realized-Parkinson Vol Regime 전략.

RV/PV 비율로 시장 미시구조 상태 측정 후 모멘텀 방향 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.rp_vol_regime.config import RpVolRegimeConfig
from src.strategy.rp_vol_regime.preprocessor import preprocess
from src.strategy.rp_vol_regime.signal import generate_signals


@register("rp-vol-regime")
class RpVolRegimeStrategy(BaseStrategy):
    """Realized-Parkinson Vol Regime 전략 구현.

    PV/RV 비율로 시장 미시구조 상태(축적/추세)를 식별하여 시그널 생성.
    """

    def __init__(self, config: RpVolRegimeConfig | None = None) -> None:
        self._config = config or RpVolRegimeConfig()

    @property
    def name(self) -> str:
        return "rp-vol-regime"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> BaseModel:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

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
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = RpVolRegimeConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "rv_window": str(self._config.rv_window),
            "pv_window": str(self._config.pv_window),
            "ratio_zscore_window": str(self._config.ratio_zscore_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
