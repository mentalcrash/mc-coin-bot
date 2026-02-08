"""Enhanced VW-TSMOM Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Enhanced VW-TSMOM 전략을 구현합니다.
볼륨 비율 정규화를 적용한 개선된 TSMOM 전략입니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.enhanced_tsmom.config import EnhancedTSMOMConfig, ShortMode
from src.strategy.enhanced_tsmom.preprocessor import preprocess
from src.strategy.enhanced_tsmom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("enhanced-tsmom")
class EnhancedTSMOMStrategy(BaseStrategy):
    """Enhanced Volume-Weighted Time Series Momentum Strategy.

    기존 TSMOM의 log1p(volume) 가중 대신 상대적 볼륨 비율(volume_ratio)을
    사용하여 모멘텀을 계산하는 개선된 전략입니다.

    Key Differences from TSMOM:
        - Volume Normalization: volume / rolling_mean(volume) 비율 사용
        - Clip Max: 거래량 비율 상한 제한으로 이상치 방어
        - 시장 구조 변화(볼륨 레벨 변화)에 더 강건한 시그널

    Attributes:
        _config: Enhanced TSMOM 설정 (EnhancedTSMOMConfig)

    Example:
        >>> from src.strategy.enhanced_tsmom import EnhancedTSMOMStrategy
        >>> strategy = EnhancedTSMOMStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: EnhancedTSMOMConfig | None = None) -> None:
        """EnhancedTSMOMStrategy 초기화.

        Args:
            config: Enhanced TSMOM 설정. None이면 기본 설정 사용.
        """
        self._config = config or EnhancedTSMOMConfig()

    @classmethod
    def from_params(cls, **params: Any) -> EnhancedTSMOMStrategy:
        """파라미터로 EnhancedTSMOMStrategy 생성.

        Args:
            **params: EnhancedTSMOMConfig 파라미터

        Returns:
            새 EnhancedTSMOMStrategy 인스턴스
        """
        config = EnhancedTSMOMConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Enhanced-VW-TSMOM"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> EnhancedTSMOMConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

        Returns:
            지표가 추가된 DataFrame
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성.

        Args:
            df: 전처리된 DataFrame (preprocess() 출력)

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Enhanced VW-TSMOM 전략에 권장되는 PortfolioManagerConfig 설정.

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터.

        Returns:
            파라미터명-값 딕셔너리
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short ({cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        result = {
            "lookback": f"{cfg.lookback}d",
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_window": f"{cfg.vol_window}d",
            "volume_lookback": f"{cfg.volume_lookback}d",
            "volume_clip_max": f"{cfg.volume_clip_max:.1f}x",
            "mode": mode_str,
        }

        if cfg.short_mode == ShortMode.HEDGE_ONLY:
            result["hedge_strength"] = f"{cfg.hedge_strength_ratio:.0%}"

        return result

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def conservative(cls) -> EnhancedTSMOMStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 EnhancedTSMOMStrategy
        """
        return cls(EnhancedTSMOMConfig.conservative())

    @classmethod
    def aggressive(cls) -> EnhancedTSMOMStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 EnhancedTSMOMStrategy
        """
        return cls(EnhancedTSMOMConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> EnhancedTSMOMStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 EnhancedTSMOMStrategy
        """
        config = EnhancedTSMOMConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
