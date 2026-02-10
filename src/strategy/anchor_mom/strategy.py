"""Anchored Momentum 전략.

Rolling high 대비 근접도(nearness)와 모멘텀 방향을 결합하여 심리적 앵커링 효과를 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.anchor_mom.config import AnchorMomConfig
from src.strategy.anchor_mom.preprocessor import preprocess
from src.strategy.anchor_mom.signal import generate_signals


@register("anchor-mom")
class AnchorMomStrategy(BaseStrategy):
    """Anchored Momentum 전략 구현.

    투자자가 최근 고점을 심리적 앵커로 사용하여,
    고점 근처에서 under-reaction → 이후 추가 상승을 포착.
    """

    def __init__(self, config: AnchorMomConfig | None = None) -> None:
        self._config = config or AnchorMomConfig()

    @property
    def name(self) -> str:
        return "anchor-mom"

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
        config = AnchorMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "nearness_lookback": str(self._config.nearness_lookback),
            "mom_lookback": str(self._config.mom_lookback),
            "strong_nearness": str(self._config.strong_nearness),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
