"""OnFlow Trend 전략.

거래소 순입출금 + MVRV z-score 기반 추세추종. BTC/ETH 전용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.onflow_trend.config import OnflowTrendConfig
from src.strategy.onflow_trend.preprocessor import preprocess
from src.strategy.onflow_trend.signal import generate_signals


@register("onflow-trend")
class OnflowTrendStrategy(BaseStrategy):
    """OnFlow Trend 전략 구현.

    거래소 순입출금(Exchange Netflow)으로 informed trader 축적/분배 탐지.
    MVRV z-score로 macro cycle 확인. 8H OHLCV + 1D On-chain context.
    BTC/ETH 전용, HEDGE_ONLY.
    """

    def __init__(self, config: OnflowTrendConfig | None = None) -> None:
        self._config = config or OnflowTrendConfig()

    @property
    def name(self) -> str:
        return "onflow-trend"

    @property
    def required_columns(self) -> list[str]:
        # On-chain은 optional (Graceful Degradation)
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
        config = OnflowTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "flow_zscore_window": str(self._config.flow_zscore_window),
            "flow_long_z": str(self._config.flow_long_z),
            "flow_exit_z": str(self._config.flow_exit_z),
            "mvrv_undervalued": str(self._config.mvrv_undervalued),
            "mvrv_overheated": str(self._config.mvrv_overheated),
            "trend_ema_fast": str(self._config.trend_ema_fast),
            "trend_ema_slow": str(self._config.trend_ema_slow),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
