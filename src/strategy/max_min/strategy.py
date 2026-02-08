"""MAX/MIN Combined Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 MAX/MIN 복합 전략을 구현합니다.
백테스팅, EDA, 라이브 트레이딩 모두에서 동일하게 사용됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.max_min.config import MaxMinConfig
from src.strategy.max_min.preprocessor import preprocess
from src.strategy.max_min.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("max-min")
class MaxMinStrategy(BaseStrategy):
    """MAX/MIN Combined Strategy.

    신고가 매수(trend-following)와 신저가 매수(mean-reversion)를
    가중 결합하여 매매 시그널을 생성하는 전략입니다.

    Key Features:
        - MAX 시그널: 신고가 돌파 시 trend-following 매수
        - MIN 시그널: 신저가 돌파 시 mean-reversion 매수
        - 가중 결합: max_weight/min_weight로 두 시그널 배합
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절

    Attributes:
        _config: MAX/MIN 설정 (MaxMinConfig)

    Example:
        >>> from src.strategy.max_min import MaxMinStrategy, MaxMinConfig
        >>>
        >>> # 기본 설정으로 생성
        >>> strategy = MaxMinStrategy()
        >>>
        >>> # 커스텀 설정으로 생성
        >>> config = MaxMinConfig(lookback=15, max_weight=0.7, min_weight=0.3)
        >>> strategy = MaxMinStrategy(config)
        >>>
        >>> # 전략 실행
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: MaxMinConfig | None = None) -> None:
        """MaxMinStrategy 초기화.

        Args:
            config: MAX/MIN 설정. None이면 기본 설정 사용.
        """
        self._config = config or MaxMinConfig()

    @classmethod
    def from_params(cls, **params: Any) -> MaxMinStrategy:
        """파라미터로 MaxMinStrategy 생성."""
        config = MaxMinConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "MAX-MIN"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> MaxMinConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 MAX/MIN 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - vol_scalar: 변동성 스케일러
            - rolling_max: 전봉 기준 rolling 최고가 (shift(1) 적용)
            - rolling_min: 전봉 기준 rolling 최저가 (shift(1) 적용)
            - atr: Average True Range

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
        """MAX/MIN 전략에 권장되는 PortfolioManagerConfig 설정.

        - 복합 전략으로 중간 수준의 레버리지 (2.0x)
        - 10% system stop loss로 큰 손실 방지
        - 5% rebalance threshold로 잦은 거래 방지

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터.

        Returns:
            파라미터명-값 딕셔너리 (사용자 친화적 포맷)
        """
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "lookback": f"{cfg.lookback}일",
            "max_weight": f"{cfg.max_weight:.0%}",
            "min_weight": f"{cfg.min_weight:.0%}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_window": f"{cfg.vol_window}일",
            "mode": mode_str,
        }

    @classmethod
    def conservative(cls) -> MaxMinStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 MaxMinStrategy
        """
        return cls(MaxMinConfig.conservative())

    @classmethod
    def aggressive(cls) -> MaxMinStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 MaxMinStrategy
        """
        return cls(MaxMinConfig.aggressive())

    @classmethod
    def for_timeframe(
        cls,
        timeframe: str,
        **kwargs: object,
    ) -> MaxMinStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "15m", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 MaxMinStrategy
        """
        config = MaxMinConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()
