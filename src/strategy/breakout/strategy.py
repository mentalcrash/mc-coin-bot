"""Adaptive Breakout Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Adaptive Breakout 전략을 구현합니다.
백테스팅, EDA, 라이브 트레이딩 모두에서 동일하게 사용됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.strategy.base import BaseStrategy
from src.strategy.breakout.config import AdaptiveBreakoutConfig
from src.strategy.breakout.preprocessor import preprocess
from src.strategy.breakout.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("adaptive-breakout")
class AdaptiveBreakoutStrategy(BaseStrategy):
    """Adaptive Breakout 전략.

    Donchian Channel 기반 돌파 전략으로, ATR을 활용하여
    변동성에 적응하는 임계값을 사용합니다.

    Key Features:
        - Donchian Channel: N일간 고/저점으로 형성된 채널 돌파 감지
        - ATR 기반 임계값: 변동성에 따라 동적으로 조절되는 진입 기준
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절
        - Trailing Stop: ATR 기반 손절매 (선택적)

    Attributes:
        _config: Adaptive Breakout 설정 (AdaptiveBreakoutConfig)

    Example:
        >>> from src.strategy.breakout import AdaptiveBreakoutStrategy
        >>>
        >>> # 기본 설정으로 생성
        >>> strategy = AdaptiveBreakoutStrategy()
        >>>
        >>> # 보수적 설정으로 생성
        >>> strategy = AdaptiveBreakoutStrategy.conservative()
        >>>
        >>> # 전략 실행
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: AdaptiveBreakoutConfig | None = None) -> None:
        """AdaptiveBreakoutStrategy 초기화.

        Args:
            config: 전략 설정. None이면 기본 설정 사용.
        """
        self._config = config or AdaptiveBreakoutConfig()

    @property
    def name(self) -> str:
        """전략 이름."""
        return "AdaptiveBreakout"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록.

        Donchian Channel과 ATR 계산에 OHLC 데이터가 필요합니다.
        """
        return ["open", "high", "low", "close"]

    @property
    def config(self) -> AdaptiveBreakoutConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 Adaptive Breakout 지표를 계산하여 추가합니다.

        Calculated Columns:
            - upper_band: Donchian Channel 상단
            - lower_band: Donchian Channel 하단
            - middle_band: Donchian Channel 중심선
            - atr: Average True Range
            - realized_vol: 실현 변동성
            - vol_scalar: 변동성 스케일러
            - threshold: 돌파 확인 임계값

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

        Returns:
            지표가 추가된 DataFrame
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성.

        전처리된 데이터에서 진입/청산 시그널을 생성합니다.

        Args:
            df: 전처리된 DataFrame (preprocess() 출력)

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Adaptive Breakout 전략에 권장되는 PortfolioManagerConfig 설정.

        Breakout 전략의 특성에 맞게 최적화된 설정:
            - 높은 변동성을 활용하므로 레버리지 2.5x 허용
            - Breakout은 손절이 중요하므로 8% system stop loss
            - 빈번한 진입/청산으로 낮은 rebalance threshold (3%)

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.5,
            "system_stop_loss": 0.08,
            "rebalance_threshold": 0.03,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터.

        Returns:
            파라미터명-값 딕셔너리 (사용자 친화적 포맷)
        """
        cfg = self._config
        info: dict[str, str] = {
            "channel_period": f"{cfg.channel_period}일",
            "k_value": f"{cfg.k_value}x ATR",
            "atr_period": f"{cfg.atr_period}일",
            "vol_target": f"{cfg.vol_target:.0%}",
        }
        if cfg.adaptive_threshold:
            info["adaptive_threshold"] = "활성화"
        # NOTE: Trailing Stop은 PortfolioManagerConfig에서 관리
        info["mode"] = "Long-Only" if cfg.long_only else "Long/Short"
        return info

    @classmethod
    def conservative(cls) -> AdaptiveBreakoutStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        긴 기간, 높은 k_value로 확실한 돌파만 진입합니다.

        Returns:
            보수적 파라미터의 AdaptiveBreakoutStrategy
        """
        return cls(AdaptiveBreakoutConfig.conservative())

    @classmethod
    def aggressive(cls) -> AdaptiveBreakoutStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        짧은 기간, 낮은 k_value로 더 많은 진입 기회를 추구합니다.

        Returns:
            공격적 파라미터의 AdaptiveBreakoutStrategy
        """
        return cls(AdaptiveBreakoutConfig.aggressive())

    @classmethod
    def for_timeframe(
        cls,
        timeframe: str,
        **kwargs: object,
    ) -> AdaptiveBreakoutStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "15m", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 AdaptiveBreakoutStrategy
        """
        config = AdaptiveBreakoutConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()
