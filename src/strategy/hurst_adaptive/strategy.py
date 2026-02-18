"""Hurst-Adaptive Strategy Implementation.

Hurst Exponent + Efficiency Ratio 기반 레짐 감지.
OHLCV-only → 16 에셋 전체 적용 가능.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.hurst_adaptive.config import HurstAdaptiveConfig
from src.strategy.hurst_adaptive.preprocessor import preprocess
from src.strategy.hurst_adaptive.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("hurst-adaptive")
class HurstAdaptiveStrategy(BaseStrategy):
    """Hurst-Adaptive Strategy.

    - 추세 구간 (H > 0.55, ER > 0.40): 모멘텀 추종
    - 횡보 구간 (H < 0.45, ER < 0.40): 평균회귀
    - Dead zone (0.45~0.55): 시그널 없음

    Example:
        >>> strategy = HurstAdaptiveStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: HurstAdaptiveConfig | None = None) -> None:
        self._config = config or HurstAdaptiveConfig()

    @classmethod
    def from_params(cls, **params: Any) -> HurstAdaptiveStrategy:
        """파라미터로 인스턴스 생성."""
        config = HurstAdaptiveConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "Hurst-Adaptive"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> HurstAdaptiveConfig:
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
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        return {
            "hurst_window": str(cfg.hurst_window),
            "er_period": str(cfg.er_period),
            "vol_target": f"{cfg.vol_target:.0%}",
            "trend_threshold": f"H>{cfg.hurst_trend_threshold}, ER>{cfg.er_trend_threshold}",
            "mr_threshold": f"H<{cfg.hurst_mr_threshold}, ER<{cfg.er_trend_threshold}",
            "mode": cfg.short_mode.name,
        }
