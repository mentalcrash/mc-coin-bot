"""Disposition CGO 전략.

Grinblatt-Han(2005) CGO + Frazzini(2006) overhang spread로
disposition effect underreaction drift를 포착하는 행동재무학 전략.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.disposition_cgo.config import DispositionCgoConfig
from src.strategy.disposition_cgo.preprocessor import preprocess
from src.strategy.disposition_cgo.signal import generate_signals


@register("disposition-cgo")
class DispositionCgoStrategy(BaseStrategy):
    """Disposition CGO 전략 구현.

    Turnover-weighted reference price 기반 CGO와 momentum overhang spread를
    결합하여 crypto retail 시장의 disposition effect underreaction drift 포착.
    """

    def __init__(self, config: DispositionCgoConfig | None = None) -> None:
        self._config = config or DispositionCgoConfig()

    @property
    def name(self) -> str:
        return "disposition-cgo"

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
        config = DispositionCgoConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "turnover_window": str(self._config.turnover_window),
            "cgo_entry_threshold": str(self._config.cgo_entry_threshold),
            "spread_confirm_threshold": str(self._config.spread_confirm_threshold),
            "momentum_window": str(self._config.momentum_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
