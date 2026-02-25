"""Capital Flow Momentum 전략.

12H 듀얼스피드 ROC 모멘텀 + 1D Stablecoin supply ROC 확신도 가중.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.cap_flow_mom.config import CapFlowMomConfig
from src.strategy.cap_flow_mom.preprocessor import preprocess
from src.strategy.cap_flow_mom.signal import generate_signals


@register("cap-flow-mom")
class CapFlowMomStrategy(BaseStrategy):
    """Capital Flow Momentum 전략 구현.

    12H OHLCV 듀얼스피드 ROC + 1D Stablecoin supply ROC 확신도 가중.
    자본 유입/유출 방향과 가격 모멘텀 정렬 시 확신도 강화, 괴리 시 감쇄.
    """

    def __init__(self, config: CapFlowMomConfig | None = None) -> None:
        self._config = config or CapFlowMomConfig()

    @property
    def name(self) -> str:
        return "cap-flow-mom"

    @property
    def required_columns(self) -> list[str]:
        # OHLCV만 필수. On-chain은 optional (부재 시 중립 처리)
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
    def from_params(cls, **params: Any) -> CapFlowMomStrategy:
        config = CapFlowMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fast_roc_period": str(self._config.fast_roc_period),
            "slow_roc_period": str(self._config.slow_roc_period),
            "roc_threshold": str(self._config.roc_threshold),
            "stablecoin_boost": str(self._config.stablecoin_boost),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
