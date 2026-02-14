"""Half-Day Momentum-Reversal 전략.

12H 전반부 return→후반부 방향 예측. 정상일 momentum, jump일 reversal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.hd_mom_rev.config import HdMomRevConfig
from src.strategy.hd_mom_rev.preprocessor import preprocess
from src.strategy.hd_mom_rev.signal import generate_signals


@register("hd-mom-rev")
class HdMomRevStrategy(BaseStrategy):
    """Half-Day Momentum-Reversal 전략 구현.

    12H intra-bar return의 크기에 따라 모멘텀/리버설 분기.
    정상일 -> 방향 유지, 급변일 -> 방향 반전.
    """

    def __init__(self, config: HdMomRevConfig | None = None) -> None:
        self._config = config or HdMomRevConfig()

    @property
    def name(self) -> str:
        return "hd-mom-rev"

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
        config = HdMomRevConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "jump_threshold": str(self._config.jump_threshold),
            "half_return_ma": str(self._config.half_return_ma),
            "confidence_cap": str(self._config.confidence_cap),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
