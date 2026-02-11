"""Quarter-Day TSMOM 전략.

이전 6H session return이 다음 session return을 양(+)으로 예측하는 intraday momentum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.qd_mom.config import QdMomConfig
from src.strategy.qd_mom.preprocessor import preprocess
from src.strategy.qd_mom.signal import generate_signals


@register("qd-mom")
class QdMomStrategy(BaseStrategy):
    """Quarter-Day TSMOM 전략 구현.

    이전 6H session return이 다음 session return을 예측.
    Late-informed trader의 정보 흡수 지연 메커니즘 활용.
    """

    def __init__(self, config: QdMomConfig | None = None) -> None:
        self._config = config or QdMomConfig()

    @property
    def name(self) -> str:
        return "qd-mom"

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
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = QdMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "vol_filter_lookback": str(self._config.vol_filter_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
