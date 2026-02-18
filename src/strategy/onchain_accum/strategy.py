"""On-chain Accumulation Strategy Implementation.

MVRV + Exchange Flow + Stablecoin 3지표 다수결.
BTC/ETH 전용, Long-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.onchain_accum.config import OnchainAccumConfig
from src.strategy.onchain_accum.preprocessor import preprocess
from src.strategy.onchain_accum.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("onchain-accum")
class OnchainAccumStrategy(BaseStrategy):
    """On-chain Accumulation Strategy.

    2/3 majority vote:
    - MVRV < 1.0 → undervalued (+1)
    - Net flow z < -1.0 → accumulation (+1)
    - Stablecoin ROC > 2% → dry powder (+1)

    Composite >= 2 → LONG. DISABLED short mode.
    BTC/ETH 전용 (CoinMetrics MVRV/Flow scope).

    Example:
        >>> strategy = OnchainAccumStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: OnchainAccumConfig | None = None) -> None:
        self._config = config or OnchainAccumConfig()

    @classmethod
    def from_params(cls, **params: Any) -> OnchainAccumStrategy:
        """파라미터로 인스턴스 생성."""
        config = OnchainAccumConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "Onchain-Accum"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> OnchainAccumConfig:
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
            "mvrv_range": f"[{cfg.mvrv_undervalued}, {cfg.mvrv_overvalued}]",
            "flow_threshold": f"z={cfg.flow_threshold}",
            "stablecoin_roc": f"{cfg.stablecoin_roc_threshold:.1%}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": cfg.short_mode.name,
        }
