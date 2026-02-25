"""T-Stat Momentum 전략.

수익률 t-statistic 기반 통계적 유의성 모멘텀. Multi-lookback t-stat blend + tanh 연속 포지션.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.t_stat_mom.config import TStatMomConfig
from src.strategy.t_stat_mom.preprocessor import preprocess
from src.strategy.t_stat_mom.signal import generate_signals


@register("t-stat-mom")
class TStatMomStrategy(BaseStrategy):
    """T-Stat Momentum 전략 구현.

    수익률의 t-statistic으로 통계적 유의성 있는 모멘텀을 포착.
    std/sqrt(N) 정규화로 vol 자동 적응이 내재.
    """

    def __init__(self, config: TStatMomConfig | None = None) -> None:
        self._config = config or TStatMomConfig()

    @property
    def name(self) -> str:
        return "t-stat-mom"

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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = TStatMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fast_lookback": str(self._config.fast_lookback),
            "mid_lookback": str(self._config.mid_lookback),
            "slow_lookback": str(self._config.slow_lookback),
            "entry_threshold": str(self._config.entry_threshold),
            "tanh_scale": str(self._config.tanh_scale),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
