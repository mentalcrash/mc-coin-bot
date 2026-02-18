"""OI-Price Divergence Strategy Implementation.

OI-가격 괴리 + Funding Rate z-score 기반.
BTC/ETH 전용 derivatives 전략.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.oi_diverge.config import OiDivergeConfig
from src.strategy.oi_diverge.preprocessor import preprocess
from src.strategy.oi_diverge.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("oi-diverge")
class OiDivergeStrategy(BaseStrategy):
    """OI-Price Divergence Strategy.

    - Short squeeze: OI↑ + 가격↓ + FR 극단 음수 → LONG
    - Long liquidation: OI↑ + 가격↑ + FR 극단 양수 → SHORT

    BTC/ETH 전용. ``funding_rate``, ``open_interest`` 컬럼은
    EDA StrategyEngine._enrich_derivatives()에서 주입됩니다.

    Example:
        >>> strategy = OiDivergeStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: OiDivergeConfig | None = None) -> None:
        self._config = config or OiDivergeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> OiDivergeStrategy:
        """파라미터로 인스턴스 생성."""
        config = OiDivergeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "OI-Diverge"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> OiDivergeConfig:
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
            "max_leverage_cap": 1.5,
            "system_stop_loss": 0.08,
            "rebalance_threshold": 0.05,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 2.5,
        }

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        return {
            "div_window": str(cfg.divergence_window),
            "fr_threshold": f"z={cfg.fr_zscore_threshold}",
            "div_threshold": f"{cfg.divergence_threshold}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": cfg.short_mode.name,
        }
