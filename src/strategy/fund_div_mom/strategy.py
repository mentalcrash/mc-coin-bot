"""Funding Divergence Momentum 전략.

가격 모멘텀과 funding rate 추세의 divergence로 추세 지속 가능성 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fund_div_mom.config import FundDivMomConfig
from src.strategy.fund_div_mom.preprocessor import preprocess
from src.strategy.fund_div_mom.signal import generate_signals


@register("fund-div-mom")
class FundDivMomStrategy(BaseStrategy):
    """Funding Divergence Momentum 전략 구현.

    FR 하락+가격 상승=유기적 수요(지속), FR 급등+가격 상승=투기과열(청산위험).
    크립토 영구선물 고유 edge 활용.
    """

    def __init__(self, config: FundDivMomConfig | None = None) -> None:
        self._config = config or FundDivMomConfig()

    @property
    def name(self) -> str:
        return "fund-div-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "high", "low", "volume", "funding_rate"]

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
        config = FundDivMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mom_lookback": str(self._config.mom_lookback),
            "fr_lookback": str(self._config.fr_lookback),
            "divergence_threshold": str(self._config.divergence_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
