"""Multi-Horizon ROC Ensemble 전략.

다중 룩백 ROC 부호 투표로 robust한 추세 신호 생성.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.mh_roc.config import MhRocConfig
from src.strategy.mh_roc.preprocessor import preprocess
from src.strategy.mh_roc.signal import generate_signals


@register("mh-roc")
class MhRocStrategy(BaseStrategy):
    """Multi-Horizon ROC Ensemble 전략 구현.

    4개 시간축(6/18/42/90 bars) ROC 부호 투표로 추세 방향 결정.
    단일 모멘텀보다 robust하며 교훈 #1 앙상블 원리 적용.
    """

    def __init__(self, config: MhRocConfig | None = None) -> None:
        self._config = config or MhRocConfig()

    @property
    def name(self) -> str:
        return "mh-roc"

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
        config = MhRocConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "roc_horizons": (
                f"{self._config.roc_short}/{self._config.roc_medium_short}/"
                f"{self._config.roc_medium_long}/{self._config.roc_long}"
            ),
            "vote_threshold": str(self._config.vote_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
