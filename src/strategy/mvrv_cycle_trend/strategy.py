"""MVRV Cycle Trend 전략.

MVRV Z-Score 사이클 레짐 필터 + 12H multi-lookback momentum.
BTC/ETH 전용 on-chain valuation 기반 추세 추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.mvrv_cycle_trend.config import MvrvCycleTrendConfig
from src.strategy.mvrv_cycle_trend.preprocessor import preprocess
from src.strategy.mvrv_cycle_trend.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("mvrv-cycle-trend")
class MvrvCycleTrendStrategy(BaseStrategy):
    """MVRV Cycle Trend 전략 구현.

    MVRV Z-Score가 사이클 위치를 판단하고,
    multi-lookback momentum이 진입/청산 타이밍을 결정한다.
    BTC/ETH 전용 (CoinMetrics MVRV scope).
    """

    def __init__(self, config: MvrvCycleTrendConfig | None = None) -> None:
        self._config = config or MvrvCycleTrendConfig()

    @property
    def name(self) -> str:
        return "mvrv-cycle-trend"

    @property
    def required_columns(self) -> list[str]:
        # OHLCV만 필수. On-chain (oc_mvrv)은 optional — Graceful Degradation
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
    def from_params(cls, **params: Any) -> MvrvCycleTrendStrategy:
        config = MvrvCycleTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mvrv_bull": str(self._config.mvrv_bull_threshold),
            "mvrv_bear": str(self._config.mvrv_bear_threshold),
            "mom_fast": str(self._config.mom_fast),
            "mom_slow": str(self._config.mom_slow),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
