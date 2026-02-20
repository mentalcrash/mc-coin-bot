"""EMA Ribbon Momentum 전략 — 피보나치 EMA 리본 정렬도 + ROC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ema_ribbon_mom.config import EmaRibbonMomConfig
from src.strategy.ema_ribbon_mom.preprocessor import preprocess
from src.strategy.ema_ribbon_mom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("ema-ribbon-momentum")
class EmaRibbonMomStrategy(BaseStrategy):
    """피보나치 EMA 리본 정렬도 + ROC 모멘텀 전략.

    5개 피보나치 EMA(8,13,21,34,55)의 정렬도를 측정하여
    추세 성숙도를 판단. alignment > 0.7이고 ROC가 같은 방향일 때 진입.
    """

    def __init__(self, config: EmaRibbonMomConfig | None = None) -> None:
        self._config = config or EmaRibbonMomConfig()

    @property
    def name(self) -> str:
        return "ema-ribbon-momentum"

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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> EmaRibbonMomStrategy:
        config = EmaRibbonMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "ema_periods": str(self._config.ema_periods),
            "roc_period": str(self._config.roc_period),
            "alignment_threshold": str(self._config.alignment_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
