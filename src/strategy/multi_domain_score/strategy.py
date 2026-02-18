"""Multi-Domain Score 전략.

4차원(추세/볼륨/파생상품/변동성) soft scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.multi_domain_score.config import MultiDomainScoreConfig
from src.strategy.multi_domain_score.preprocessor import preprocess
from src.strategy.multi_domain_score.signal import generate_signals


@register("multi-domain-score")
class MultiDomainScoreStrategy(BaseStrategy):
    """Multi-Domain Score 전략 구현.

    4차원 독립 약한 알파 → soft scoring → noise 상쇄.
    """

    def __init__(self, config: MultiDomainScoreConfig | None = None) -> None:
        self._config = config or MultiDomainScoreConfig()

    @property
    def name(self) -> str:
        return "multi-domain-score"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

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
        config = MultiDomainScoreConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "w_trend": str(self._config.w_trend),
            "w_volume": str(self._config.w_volume),
            "w_derivatives": str(self._config.w_derivatives),
            "w_volatility": str(self._config.w_volatility),
            "entry_threshold": str(self._config.entry_threshold),
            "short_mode": self._config.short_mode.name,
        }
