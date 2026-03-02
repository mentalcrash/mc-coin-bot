"""CCI Consensus Multi-Scale Trend 전략.

CCI(MAD 정규화) x 3스케일(20/60/150) consensus voting으로 로버스트 추세 추종.
CCI의 MAD 기반 정규화는 BB의 std와 수학적으로 직교적.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.cci_consensus.config import CciConsensusConfig
from src.strategy.cci_consensus.preprocessor import preprocess
from src.strategy.cci_consensus.signal import generate_signals


@register("cci-consensus")
class CciConsensusStrategy(BaseStrategy):
    """CCI Consensus Multi-Scale Trend 전략 구현.

    CCI x 3스케일(20/60/150)의 3개 sub-signal consensus voting으로
    MAD 정규화 기반 로버스트 추세 방향을 도출한다.
    """

    def __init__(self, config: CciConsensusConfig | None = None) -> None:
        self._config = config or CciConsensusConfig()

    @property
    def name(self) -> str:
        return "cci-consensus"

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
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = CciConsensusConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "scales": (
                f"{self._config.scale_short}/{self._config.scale_mid}/{self._config.scale_long}"
            ),
            "cci_upper": str(self._config.cci_upper),
            "cci_lower": str(self._config.cci_lower),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
