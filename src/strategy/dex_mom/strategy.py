"""DEX Activity Momentum Strategy Implementation.

DEX 거래량 7D/30D 변화율 기반 on-chain 활동 모멘텀.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.dex_mom.config import DexMomConfig
from src.strategy.dex_mom.preprocessor import preprocess
from src.strategy.dex_mom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("dex-mom")
class DexMomStrategy(BaseStrategy):
    """DEX Activity Momentum Strategy.

    DEX 거래량의 7D/30D ROC 방향으로 on-chain 활동 모멘텀 판단.
    - 양쪽 ROC > 0 → DEX 활동 확대 → risk-on → LONG
    - 양쪽 ROC < 0 → DEX 활동 축소 → risk-off → SHORT
    - 혼합 → FLAT

    Example:
        >>> strategy = DexMomStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: DexMomConfig | None = None) -> None:
        self._config = config or DexMomConfig()

    @classmethod
    def from_params(cls, **params: Any) -> DexMomStrategy:
        """파라미터로 인스턴스 생성."""
        config = DexMomConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "Dex-Mom"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> DexMomConfig:
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
