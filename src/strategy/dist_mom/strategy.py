"""Return Distribution Momentum 전략.

수익률 분포 특성(양수비율, skewness)으로 모멘텀 품질 측정 후 방향 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.dist_mom.config import DistMomConfig
from src.strategy.dist_mom.preprocessor import preprocess
from src.strategy.dist_mom.signal import generate_signals


@register("dist-mom")
class DistMomStrategy(BaseStrategy):
    """Return Distribution Momentum 전략 구현.

    양수 수익률 비율 + skewness로 모멘텀 지속성 예측.
    """

    def __init__(self, config: DistMomConfig | None = None) -> None:
        self._config = config or DistMomConfig()

    @property
    def name(self) -> str:
        return "dist-mom"

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
        config = DistMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "dist_window": str(self._config.dist_window),
            "long_threshold": str(self._config.long_threshold),
            "short_threshold": str(self._config.short_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
