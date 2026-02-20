"""F&G Asymmetric Momentum 전략.

Fear=역발상(contrarian), Greed=순응(momentum) 비대칭 접근.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fg_asym_mom.config import FgAsymMomConfig
from src.strategy.fg_asym_mom.preprocessor import preprocess
from src.strategy.fg_asym_mom.signal import generate_signals


@register("fg-asym-mom")
class FgAsymMomStrategy(BaseStrategy):
    """F&G Asymmetric Momentum 전략 구현.

    Fear 구간: 역발상 매수 (contrarian).
    Greed 구간: 모멘텀 유지 (momentum).
    비대칭 herding 구조를 활용한 하이브리드.
    """

    def __init__(self, config: FgAsymMomConfig | None = None) -> None:
        self._config = config or FgAsymMomConfig()

    @property
    def name(self) -> str:
        return "fg-asym-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "oc_fear_greed"]

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
        config = FgAsymMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fear_threshold": str(self._config.fear_threshold),
            "greed_threshold": str(self._config.greed_threshold),
            "greed_hold_threshold": str(self._config.greed_hold_threshold),
            "sma_short": str(self._config.sma_short),
            "sma_long": str(self._config.sma_long),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
