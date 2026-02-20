"""Exchange Flow Momentum 전략.

거래소 순유출 모멘텀 기반 축적/분배 포착. BTC/ETH 전용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.ex_flow_mom.config import ExFlowMomConfig
from src.strategy.ex_flow_mom.preprocessor import preprocess
from src.strategy.ex_flow_mom.signal import generate_signals


@register("ex-flow-mom")
class ExFlowMomStrategy(BaseStrategy):
    """Exchange Flow Momentum 전략 구현.

    거래소 순유출 가속 = 축적 시그널. BTC/ETH 전용.
    """

    def __init__(self, config: ExFlowMomConfig | None = None) -> None:
        self._config = config or ExFlowMomConfig()

    @property
    def name(self) -> str:
        return "ex-flow-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "oc_flow_net"]

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
        config = ExFlowMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "flow_window": str(self._config.flow_window),
            "flow_mom_window": str(self._config.flow_mom_window),
            "flow_threshold": str(self._config.flow_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
