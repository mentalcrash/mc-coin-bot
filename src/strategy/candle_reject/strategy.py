"""Candlestick Rejection Momentum Strategy Implementation.

4H candle rejection wick + volume confirmation -> directional momentum signal.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.candle_reject.config import CandleRejectConfig, ShortMode
from src.strategy.candle_reject.preprocessor import preprocess
from src.strategy.candle_reject.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("candle-reject")
class CandleRejectStrategy(BaseStrategy):
    """Candlestick Rejection Momentum Strategy.

    4H candle의 긴 꼬리(rejection wick)로 가격 거부를 감지하고,
    거부 방향의 반대 = 시장의 진정한 방향으로 포지션을 구축합니다.

    Key Features:
        - Bar anatomy 기반 rejection ratio 계산
        - Volume Z-score 확인 (noise 필터링)
        - Consecutive rejection 부스트 (conviction 강화)
        - Body position reversal / timeout 기반 청산
        - Vol-target sizing

    Attributes:
        _config: Candlestick Rejection 설정 (CandleRejectConfig)
    """

    def __init__(self, config: CandleRejectConfig | None = None) -> None:
        """CandleRejectStrategy 초기화."""
        self._config = config or CandleRejectConfig()

    @classmethod
    def from_params(cls, **params: Any) -> CandleRejectStrategy:
        """파라미터로 CandleRejectStrategy 생성."""
        config = CandleRejectConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Candle-Reject"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> CandleRejectConfig:
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
        """Candle Reject 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "rejection_threshold": f"{cfg.rejection_threshold:.1%}",
            "volume_zscore_threshold": f"{cfg.volume_zscore_threshold:.1f}",
            "consecutive_boost": f"{cfg.consecutive_boost:.1f}x (>={cfg.consecutive_min})",
            "exit_timeout": f"{cfg.exit_timeout_bars} bars",
            "mode": mode_str,
        }
