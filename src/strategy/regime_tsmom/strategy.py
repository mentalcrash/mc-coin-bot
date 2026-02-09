"""Regime-Adaptive TSMOM Strategy Implementation.

TSMOM + 레짐 적응적 포지션 사이징.
trending → 공격적, ranging → 보수적, volatile → 최소 노출.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.regime_tsmom.config import RegimeTSMOMConfig
from src.strategy.regime_tsmom.preprocessor import preprocess
from src.strategy.regime_tsmom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("regime-tsmom")
class RegimeTSMOMStrategy(BaseStrategy):
    """Regime-Adaptive TSMOM Strategy.

    기존 VW-TSMOM에 레짐 감지를 결합하여 시장 상황에 따라
    포지션 사이징을 적응적으로 조절합니다.

    - trending: 공격적 (높은 vol_target, 높은 leverage_scale)
    - ranging: 보수적 (낮은 vol_target, 낮은 leverage_scale)
    - volatile: 초보수 (최소 vol_target, 최소 leverage_scale)

    Example:
        >>> strategy = RegimeTSMOMStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: RegimeTSMOMConfig | None = None) -> None:
        self._config = config or RegimeTSMOMConfig()

    @classmethod
    def from_params(cls, **params: Any) -> RegimeTSMOMStrategy:
        """파라미터로 RegimeTSMOMStrategy 생성."""
        config = RegimeTSMOMConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "Regime-TSMOM"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> RegimeTSMOMConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Regime-TSMOM 권장 PortfolioManagerConfig."""
        return {
            "max_leverage_cap": 1.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        return {
            "lookback": f"{cfg.lookback}d",
            "base_vol_target": f"{cfg.vol_target:.0%}",
            "trending_vt": f"{cfg.trending_vol_target:.0%}",
            "ranging_vt": f"{cfg.ranging_vol_target:.0%}",
            "volatile_vt": f"{cfg.volatile_vol_target:.0%}",
            "mode": cfg.short_mode.name,
        }
