"""Capital Gains Overhang Momentum 전략.

Turnover 기반 평균 매입단가 추정 후 현재가 괴리로 disposition effect 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.cgo_mom.config import CgoMomConfig
from src.strategy.cgo_mom.preprocessor import preprocess
from src.strategy.cgo_mom.signal import generate_signals


@register("cgo-mom")
class CgoMomStrategy(BaseStrategy):
    """Capital Gains Overhang Momentum 전략 구현.

    높은 CGO(미실현 이익)는 disposition effect로 인한 매도 압력을 유발하며,
    역설적으로 모멘텀 지속을 예측. Grinblatt & Han 2005 기반.
    """

    def __init__(self, config: CgoMomConfig | None = None) -> None:
        self._config = config or CgoMomConfig()

    @property
    def name(self) -> str:
        return "cgo-mom"

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
        config = CgoMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "turnover_window": str(self._config.turnover_window),
            "cgo_zscore_window": str(self._config.cgo_zscore_window),
            "cgo_threshold": str(self._config.cgo_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
