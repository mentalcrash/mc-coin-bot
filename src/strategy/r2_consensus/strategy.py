"""R2 Consensus Trend 전략.

3개 스케일의 선형회귀 R2 투표 consensus로 추세 진입 판단.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.r2_consensus.config import R2ConsensusConfig
from src.strategy.r2_consensus.preprocessor import preprocess
from src.strategy.r2_consensus.signal import generate_signals


@register("r2-consensus")
class R2ConsensusStrategy(BaseStrategy):
    """R2 Consensus Trend 전략 구현.

    3개 스케일(20/50/120)의 선형회귀 R^2 투표로 추세 일관성을 검증하여 진입.
    Multi-scale consensus는 단일 스케일 대비 파라미터 로버스트니스 향상.
    """

    def __init__(self, config: R2ConsensusConfig | None = None) -> None:
        self._config = config or R2ConsensusConfig()

    @property
    def name(self) -> str:
        return "r2-consensus"

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
        config = R2ConsensusConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "lookback_short": str(self._config.lookback_short),
            "lookback_mid": str(self._config.lookback_mid),
            "lookback_long": str(self._config.lookback_long),
            "r2_threshold": str(self._config.r2_threshold),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
