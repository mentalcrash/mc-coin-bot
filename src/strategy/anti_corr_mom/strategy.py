"""Anti-Correlation Momentum 전략.

에셋-BTC decorrelation 시 모멘텀 시그널 신뢰도 상승.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.anti_corr_mom.config import AntiCorrMomConfig
from src.strategy.anti_corr_mom.preprocessor import preprocess
from src.strategy.anti_corr_mom.signal import generate_signals


@register("anti-corr-mom")
class AntiCorrMomStrategy(BaseStrategy):
    """Anti-Correlation Momentum 전략 구현.

    BTC 대비 상관이 낮아지면 독자적 모멘텀 활용.
    """

    def __init__(self, config: AntiCorrMomConfig | None = None) -> None:
        self._config = config or AntiCorrMomConfig()

    @property
    def name(self) -> str:
        return "anti-corr-mom"

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
        config = AntiCorrMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "corr_window": str(self._config.corr_window),
            "corr_threshold": str(self._config.corr_threshold),
            "momentum_window": str(self._config.momentum_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
