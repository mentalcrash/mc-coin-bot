"""Multi-Source Directional Composite 전략.

3개 직교 데이터 소스(OHLCV momentum + stablecoin flow proxy + F&G sentiment)의
majority vote 기반 방향 결정. 약한 알파의 포트폴리오 결합 이론.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig
from src.strategy.multi_source_composite.preprocessor import preprocess
from src.strategy.multi_source_composite.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("multi-source-composite")
class MultiSourceCompositeStrategy(BaseStrategy):
    """Multi-Source Directional Composite 전략 구현.

    3개 직교 소스(momentum, velocity, F&G)의 majority vote로 방향을 결정하고
    합의 수준에 따라 conviction을 조절하여 약한 알파를 결합한다.
    """

    def __init__(self, config: MultiSourceCompositeConfig | None = None) -> None:
        self._config = config or MultiSourceCompositeConfig()

    @property
    def name(self) -> str:
        return "multi-source-composite"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "oc_fear_greed"]

    @property
    def config(self) -> MultiSourceCompositeConfig:
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
        config = MultiSourceCompositeConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mom_fast": str(self._config.mom_fast),
            "mom_slow": str(self._config.mom_slow),
            "velocity_fast_window": str(self._config.velocity_fast_window),
            "fg_delta_window": str(self._config.fg_delta_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
