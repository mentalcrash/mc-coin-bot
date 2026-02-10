"""Adaptive Kalman Trend Strategy Implementation.

칼만 필터 velocity로 adaptive trend 감지 후 포지션 결정.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.kalman_trend.config import KalmanTrendConfig, ShortMode
from src.strategy.kalman_trend.preprocessor import preprocess
from src.strategy.kalman_trend.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("kalman-trend")
class KalmanTrendStrategy(BaseStrategy):
    """Adaptive Kalman Trend Strategy.

    칼만 필터로 가격의 smoothed state와 velocity를 추정합니다.
    Adaptive Q parameter로 변동성 레짐에 자동 적응합니다.

    Key Features:
        - Bayesian noise separation (Kalman filter)
        - Adaptive process noise (Q) based on realized vol ratio
        - Velocity-based trend direction detection
        - ATR-based volatility targeting

    Attributes:
        _config: Kalman Trend 설정 (KalmanTrendConfig)
    """

    def __init__(self, config: KalmanTrendConfig | None = None) -> None:
        """KalmanTrendStrategy 초기화."""
        self._config = config or KalmanTrendConfig()

    @classmethod
    def from_params(cls, **params: Any) -> KalmanTrendStrategy:
        """파라미터로 KalmanTrendStrategy 생성."""
        config = KalmanTrendConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Kalman-Trend"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> KalmanTrendConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성."""
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Kalman Trend 전략에 권장되는 PortfolioManagerConfig 설정."""
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터."""
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "base_q": f"{cfg.base_q:.4f}",
            "observation_noise": f"{cfg.observation_noise:.2f}",
            "vel_threshold": f"{cfg.vel_threshold:.2f}",
            "vol_lookback": f"{cfg.vol_lookback}",
            "long_term_vol_lookback": f"{cfg.long_term_vol_lookback}",
            "mode": mode_str,
        }
