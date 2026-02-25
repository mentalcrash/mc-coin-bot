"""Composite Momentum 전략.

OHLCV 3축 직교 분해(모멘텀 x 거래량 x GK변동성)로 복합 시그널 생성.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.comp_mom.config import CompMomConfig
from src.strategy.comp_mom.preprocessor import preprocess
from src.strategy.comp_mom.signal import generate_signals


@register("comp-mom")
class CompMomStrategy(BaseStrategy):
    """Composite Momentum 전략 구현.

    가격 모멘텀(방향) x 거래량(참여도) x GK변동성(환경) 3축 정렬 시
    강한 시그널, 불일치 시 자연 감쇄하는 Risk-Managed Momentum.
    """

    def __init__(self, config: CompMomConfig | None = None) -> None:
        self._config = config or CompMomConfig()

    @property
    def name(self) -> str:
        return "comp-mom"

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
        config = CompMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mom_period": str(self._config.mom_period),
            "composite_threshold": str(self._config.composite_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
