"""CVD Divergence 8H 전략.

Price-CVD (Cumulative Volume Delta) 괴리로 추세 반전/지속 신호 포착.
CVD 데이터 없으면 graceful degradation → pure EMA trend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.cvd_diverge_8h.config import CvdDiverge8hConfig
from src.strategy.cvd_diverge_8h.preprocessor import preprocess
from src.strategy.cvd_diverge_8h.signal import generate_signals


@register("cvd-diverge-8h")
class CvdDiverge8hStrategy(BaseStrategy):
    """CVD Divergence 8H 전략 구현.

    Coinalyze CVD buy_volume와 가격의 방향 괴리를 감지하여
    추세 반전 또는 지속 확인 신호로 활용한다.
    CVD 컬럼(dext_cvd_buy_vol) 없으면 pure EMA trend로 graceful degradation.
    BTC/ETH 전용.
    """

    def __init__(self, config: CvdDiverge8hConfig | None = None) -> None:
        self._config = config or CvdDiverge8hConfig()

    @property
    def name(self) -> str:
        return "cvd-diverge-8h"

    @property
    def required_columns(self) -> list[str]:
        # CVD는 optional (graceful degradation)
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
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = CvdDiverge8hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "cvd_lookback": str(self._config.cvd_lookback),
            "cvd_ma_window": str(self._config.cvd_ma_window),
            "divergence_threshold": str(self._config.divergence_threshold),
            "trend_ema_window": str(self._config.trend_ema_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
