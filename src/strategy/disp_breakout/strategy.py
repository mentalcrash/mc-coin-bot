"""Disposition Breakout 전략.

처분 효과 기반 rolling high 돌파 시 추세추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.disp_breakout.config import DispBreakoutConfig
from src.strategy.disp_breakout.preprocessor import preprocess
from src.strategy.disp_breakout.signal import generate_signals


@register("disp-breakout")
class DispBreakoutStrategy(BaseStrategy):
    """Disposition Breakout 전략 구현.

    투자자 처분 효과 활용: rolling high 돌파 시 regret aversion 추격 매수.
    """

    def __init__(self, config: DispBreakoutConfig | None = None) -> None:
        self._config = config or DispBreakoutConfig()

    @property
    def name(self) -> str:
        return "disp-breakout"

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
        config = DispBreakoutConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "high_window": str(self._config.high_window),
            "proximity_threshold": str(self._config.proximity_threshold),
            "breakout_threshold": str(self._config.breakout_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
