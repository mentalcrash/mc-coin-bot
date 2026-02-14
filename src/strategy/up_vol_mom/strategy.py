"""Realized Semivariance Momentum 전략.

상방 반분산 비율 우위 시 정보 기반 매수 활동 반영 -> 모멘텀 시그널.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.up_vol_mom.config import UpVolMomConfig
from src.strategy.up_vol_mom.preprocessor import preprocess
from src.strategy.up_vol_mom.signal import generate_signals


@register("up-vol-mom")
class UpVolMomStrategy(BaseStrategy):
    """Realized Semivariance Momentum 전략 구현.

    Upside semivariance 우위 + 모멘텀 확인 -> 정보 기반 매수 활동 포착.
    """

    def __init__(self, config: UpVolMomConfig | None = None) -> None:
        self._config = config or UpVolMomConfig()

    @property
    def name(self) -> str:
        return "up-vol-mom"

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
        config = UpVolMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "semivar_window": str(self._config.semivar_window),
            "ratio_ma_window": str(self._config.ratio_ma_window),
            "ratio_threshold": str(self._config.ratio_threshold),
            "mom_lookback": str(self._config.mom_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
