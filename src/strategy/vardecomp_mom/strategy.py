"""Variance Decomposition Momentum 전략.

Realized variance를 good/bad semivariance로 분해하여 모멘텀 품질 측정.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vardecomp_mom.config import VardecompMomConfig
from src.strategy.vardecomp_mom.preprocessor import preprocess
from src.strategy.vardecomp_mom.signal import generate_signals


@register("vardecomp-mom")
class VardecompMomStrategy(BaseStrategy):
    """Variance Decomposition Momentum 전략 구현.

    Good_var 지배적 추세는 지속, bad_var 지배적 추세는 붕괴.
    Prospect theory + JFQA 2024 실증 기반.
    """

    def __init__(self, config: VardecompMomConfig | None = None) -> None:
        self._config = config or VardecompMomConfig()

    @property
    def name(self) -> str:
        return "vardecomp-mom"

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
        config = VardecompMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "semivar_window": str(self._config.semivar_window),
            "mom_lookback": str(self._config.mom_lookback),
            "var_ratio_threshold": str(self._config.var_ratio_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
