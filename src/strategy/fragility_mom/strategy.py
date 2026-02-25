"""Fragility-Aware Momentum 전략.

VoV가 regime 전환 감지기. Low VoV + Low GK vol = 안정 + 잠잠 → 높은 확신 모멘텀.
High VoV 또는 High GK vol = 불안정/과열 → 축소. 행동편향(과신→취약성) 활용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fragility_mom.config import FragilityMomConfig
from src.strategy.fragility_mom.preprocessor import preprocess
from src.strategy.fragility_mom.signal import generate_signals


@register("fragility-mom")
class FragilityMomStrategy(BaseStrategy):
    """Fragility-Aware Momentum 전략 구현.

    VoV(변동성의 변동성) + GK vol percentile 결합으로
    시장 취약성(fragility)을 인식하는 모멘텀 전략.
    """

    def __init__(self, config: FragilityMomConfig | None = None) -> None:
        self._config = config or FragilityMomConfig()

    @property
    def name(self) -> str:
        return "fragility-mom"

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
        config = FragilityMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "gk_window": str(self._config.gk_window),
            "vov_window": str(self._config.vov_window),
            "vov_threshold": str(self._config.vov_threshold),
            "mom_lookback": str(self._config.mom_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
