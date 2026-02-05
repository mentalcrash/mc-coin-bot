"""VW-TSMOM Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 VW-TSMOM 전략을 구현합니다.
백테스팅, EDA, 라이브 트레이딩 모두에서 동일하게 사용됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode, TSMOMConfig
from src.strategy.tsmom.preprocessor import preprocess
from src.strategy.tsmom.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("tsmom")
class TSMOMStrategy(BaseStrategy):
    """Volume-Weighted Time Series Momentum Strategy.

    거래량 가중 시계열 모멘텀을 기반으로 매매 시그널을 생성하는 전략입니다.
    학술 연구(SSRN #4825389)에 기반한 검증된 접근법을 사용합니다.

    Key Features:
        - 거래량 가중 모멘텀: 거래량이 큰 가격 변화에 높은 가중치
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절
        - 레버리지 제한: 최대 레버리지로 리스크 관리

    Attributes:
        _config: TSMOM 설정 (TSMOMConfig)

    Example:
        >>> from src.strategy.tsmom import TSMOMStrategy, TSMOMConfig
        >>>
        >>> # 기본 설정으로 생성
        >>> strategy = TSMOMStrategy()
        >>>
        >>> # 커스텀 설정으로 생성
        >>> config = TSMOMConfig(lookback=48, vol_target=0.20)
        >>> strategy = TSMOMStrategy(config)
        >>>
        >>> # 전략 실행
        >>> processed_df, signals = strategy.run(ohlcv_df)
        >>> print(signals.entries.sum())  # 진입 시그널 수
    """

    def __init__(self, config: TSMOMConfig | None = None) -> None:
        """TSMOMStrategy 초기화.

        Args:
            config: TSMOM 설정. None이면 기본 설정 사용.
        """
        self._config = config or TSMOMConfig()

    @property
    def name(self) -> str:
        """전략 이름."""
        return "VW-TSMOM"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "volume"]

    @property
    def config(self) -> TSMOMConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 VW-TSMOM 지표를 계산하여 추가합니다.

        Note:
            레버리지 클램핑은 PortfolioManagerConfig에서 처리됩니다.
            전략은 순수한 raw_signal만 생성합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - vw_momentum: 거래량 가중 모멘텀
            - vol_scalar: 변동성 스케일러
            - raw_signal: 원시 시그널 (레버리지 무제한)

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
    def conservative(cls) -> TSMOMStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 TSMOMStrategy
        """
        return cls(TSMOMConfig.conservative())

    @classmethod
    def aggressive(cls) -> TSMOMStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 TSMOMStrategy
        """
        return cls(TSMOMConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> TSMOMStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "15m")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 TSMOMStrategy
        """
        config = TSMOMConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """VW-TSMOM 전략에 권장되는 PortfolioManagerConfig 설정.

        - Momentum은 추세 추종으로 느린 진입/청산
        - 레버리지 2.0x로 보수적 운용
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
        effective_mode = cfg.effective_short_mode()

        # 숏 모드 문자열 변환
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (≤{cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(effective_mode, "Unknown")

        result = {
            "lookback": f"{cfg.lookback}일",
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_window": f"{cfg.vol_window}일",
            "mode": mode_str,
        }

        # 헤지 모드일 때 추가 정보
        if effective_mode == ShortMode.HEDGE_ONLY:
            result["hedge_strength"] = f"{cfg.hedge_strength_ratio:.0%}"

        return result
