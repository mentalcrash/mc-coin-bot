"""RV-Jump Continuation 전략.

Realized variance에서 bipower variation을 초과하는 jump 성분으로 continuation 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.rv_jump_cont.config import RvJumpContConfig
from src.strategy.rv_jump_cont.preprocessor import preprocess
from src.strategy.rv_jump_cont.signal import generate_signals


@register("rv-jump-cont")
class RvJumpContStrategy(BaseStrategy):
    """RV-Jump Continuation 전략 구현.

    Jump ratio(RV/BV) > threshold 시 momentum 방향으로 continuation.
    """

    def __init__(self, config: RvJumpContConfig | None = None) -> None:
        self._config = config or RvJumpContConfig()

    @property
    def name(self) -> str:
        return "rv-jump-cont"

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
        config = RvJumpContConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "rv_window": str(self._config.rv_window),
            "jump_threshold": str(self._config.jump_threshold),
            "mom_lookback": str(self._config.mom_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
