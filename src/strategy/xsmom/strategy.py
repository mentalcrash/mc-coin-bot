"""XSMOM Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 XSMOM 전략을 구현합니다.
백테스팅, EDA, 라이브 트레이딩 모두에서 동일하게 사용됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode
from src.strategy.xsmom.config import XSMOMConfig
from src.strategy.xsmom.preprocessor import preprocess
from src.strategy.xsmom.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("xsmom")
class XSMOMStrategy(BaseStrategy):
    """Cross-Sectional Momentum Strategy.

    코인별 rolling return과 vol-target sizing을 기반으로
    매매 시그널을 생성하는 횡단면 모멘텀 전략입니다.

    Key Features:
        - Rolling return: lookback 기간 수익률로 모멘텀 측정
        - Holding period: 시그널 리밸런싱 주기 제어
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절
        - Cross-sectional ranking: 멀티에셋 백테스트에서 수행

    Attributes:
        _config: XSMOM 설정 (XSMOMConfig)

    Example:
        >>> from src.strategy.xsmom import XSMOMStrategy, XSMOMConfig
        >>>
        >>> # 기본 설정으로 생성
        >>> strategy = XSMOMStrategy()
        >>>
        >>> # 커스텀 설정으로 생성
        >>> config = XSMOMConfig(lookback=21, holding_period=7)
        >>> strategy = XSMOMStrategy(config)
        >>>
        >>> # 전략 실행
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: XSMOMConfig | None = None) -> None:
        """XSMOMStrategy 초기화.

        Args:
            config: XSMOM 설정. None이면 기본 설정 사용.
        """
        self._config = config or XSMOMConfig()

    @classmethod
    def from_params(cls, **params: Any) -> XSMOMStrategy:
        """파라미터로 XSMOMStrategy 생성."""
        config = XSMOMConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "XSMOM"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> XSMOMConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 XSMOM 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - rolling_return: lookback 기간 수익률
            - vol_scalar: 변동성 스케일러
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
    def conservative(cls) -> XSMOMStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 XSMOMStrategy
        """
        return cls(XSMOMConfig.conservative())

    @classmethod
    def aggressive(cls) -> XSMOMStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 XSMOMStrategy
        """
        return cls(XSMOMConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> XSMOMStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 XSMOMStrategy
        """
        config = XSMOMConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """XSMOM 전략에 권장되는 PortfolioManagerConfig 설정.

        - 횡단면 모멘텀은 long-short 전략
        - 레버리지 2.0x로 보수적 운용
        - 10% system stop loss
        - 10% rebalance threshold (holding_period와 조합)

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터.

        Returns:
            파라미터명-값 딕셔너리
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "lookback": f"{cfg.lookback}d",
            "holding_period": f"{cfg.holding_period}d",
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_window": f"{cfg.vol_window}d",
            "mode": mode_str,
        }
