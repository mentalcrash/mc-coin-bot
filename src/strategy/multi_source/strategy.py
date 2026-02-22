"""Multi-Source 전략 — 다중 데이터 소스 결합 전략.

On-chain, Macro, Options, Derivatives 등 다양한 데이터 소스의
서브시그널을 결합하여 복합 트레이딩 시그널을 생성합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.multi_source.config import MultiSourceConfig
from src.strategy.multi_source.preprocessor import preprocess
from src.strategy.multi_source.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("multi-source")
class MultiSourceStrategy(BaseStrategy):
    """다중 데이터 소스 결합 전략.

    enriched DataFrame의 비-OHLCV 컬럼(oc_*, macro_*, opt_*, deriv_*)을
    서브시그널로 변환하고 결합하여 복합 시그널을 생성합니다.
    """

    def __init__(self, config: MultiSourceConfig | None = None) -> None:
        if config is None:
            # 기본 설정: 최소 2개 서브시그널 필요
            from src.strategy.multi_source.config import SubSignalSpec

            config = MultiSourceConfig(
                signals=(
                    SubSignalSpec(column="oc_fear_greed"),
                    SubSignalSpec(column="oc_mvrv"),
                ),
            )
        self._config = config

    @property
    def name(self) -> str:
        return "multi-source"

    @property
    def required_columns(self) -> list[str]:
        """OHLCV + 서브시그널 컬럼."""
        base = ["close"]
        signal_cols = [s.column for s in self._config.signals]
        return base + signal_cols

    @property
    def config(self) -> MultiSourceConfig:
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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = MultiSourceConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "combine_method": self._config.combine_method.value,
            "n_signals": str(len(self._config.signals)),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
