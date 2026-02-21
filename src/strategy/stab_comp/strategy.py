"""Stablecoin Composition Shift Strategy Implementation.

USDT/USDC 구성비 변화 기반 리스크 선호도 시그널.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.stab_comp.config import StabCompConfig
from src.strategy.stab_comp.preprocessor import preprocess
from src.strategy.stab_comp.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("stab-comp")
class StabCompStrategy(BaseStrategy):
    """Stablecoin Composition Shift Strategy.

    USDT/(USDT+USDC) 비율의 7D/30D ROC 방향으로 리스크 선호도 판단.
    - 양쪽 ROC > 0 → USDT 비중 증가 → 리테일 risk-on → LONG
    - 양쪽 ROC < 0 → USDC 비중 증가 → 기관 cautious → SHORT
    - 혼합 → FLAT

    Example:
        >>> strategy = StabCompStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: StabCompConfig | None = None) -> None:
        self._config = config or StabCompConfig()

    @classmethod
    def from_params(cls, **params: Any) -> StabCompStrategy:
        """파라미터로 인스턴스 생성."""
        config = StabCompConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "Stab-Comp"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> StabCompConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """권장 PortfolioManagerConfig."""
        return {
            "max_leverage_cap": 1.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.5,
        }

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        return {
            "roc_windows": f"[{cfg.roc_short_window}D, {cfg.roc_long_window}D]",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": cfg.short_mode.name,
        }
