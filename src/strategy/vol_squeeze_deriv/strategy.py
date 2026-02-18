"""Vol Squeeze + Derivatives 전략.

Vol percentile rank 압축 → breakout + FR 방향 확인.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vol_squeeze_deriv.config import VolSqueezeDerivConfig
from src.strategy.vol_squeeze_deriv.preprocessor import preprocess
from src.strategy.vol_squeeze_deriv.signal import generate_signals


@register("vol-squeeze-deriv")
class VolSqueezeDerivStrategy(BaseStrategy):
    """Vol Squeeze + Derivatives 전략 구현.

    Vol 압축 구간 → breakout 방향 + FR contrarian 확인.
    """

    def __init__(self, config: VolSqueezeDerivConfig | None = None) -> None:
        self._config = config or VolSqueezeDerivConfig()

    @property
    def name(self) -> str:
        return "vol-squeeze-deriv"

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
        config = VolSqueezeDerivConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "vol_rank_window": str(self._config.vol_rank_window),
            "squeeze_threshold": str(self._config.squeeze_threshold),
            "expansion_ratio": str(self._config.expansion_ratio),
            "vol_exit_rank": str(self._config.vol_exit_rank),
            "short_mode": self._config.short_mode.name,
        }
