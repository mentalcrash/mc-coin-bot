"""Basis-Momentum 전략.

FR 변화율(가속도) z-score 기반 모멘텀 — FR level 전략 19개+ 전멸 후 대안.
FR level = lagging, FR acceleration = leading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.basis_momentum.config import BasisMomentumConfig
from src.strategy.basis_momentum.preprocessor import preprocess
from src.strategy.basis_momentum.signal import generate_signals


@register("basis-momentum")
class BasisMomentumStrategy(BaseStrategy):
    """Basis-Momentum 전략 구현.

    FR 수준이 아닌 FR 변화율(1st derivative)의 z-score로
    심리 전환의 초기 신호를 포착한다.
    derivatives 데이터(funding_rate) 부재 시 graceful degradation (flat 시그널).
    """

    def __init__(self, config: BasisMomentumConfig | None = None) -> None:
        self._config = config or BasisMomentumConfig()

    @property
    def name(self) -> str:
        return "basis-momentum"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> BaseModel:
        return self._config

    @property
    def required_enrichments(self) -> list[str]:
        """funding_rate 컬럼 필요 (derivatives enrichment에서 제공)."""
        return ["funding_rate"]

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
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = BasisMomentumConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fr_change_window": str(self._config.fr_change_window),
            "fr_std_window": str(self._config.fr_std_window),
            "entry_zscore": str(self._config.entry_zscore),
            "exit_zscore": str(self._config.exit_zscore),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
