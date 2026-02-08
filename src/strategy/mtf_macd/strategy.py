"""MTF MACD Strategy Implementation.

MACD(12,26,9) 기반 추세 필터 + crossover 진입 전략입니다.
BaseStrategy를 상속하고 @register("mtf-macd")로 등록됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.mtf_macd.config import MtfMacdConfig, ShortMode
from src.strategy.mtf_macd.preprocessor import preprocess
from src.strategy.mtf_macd.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("mtf-macd")
class MtfMacdStrategy(BaseStrategy):
    """MTF MACD 전략.

    MACD crossover + trend filter를 이용한 추세 추종 전략입니다.
    MACD > 0 (bullish trend)에서 crossover 시 롱 진입,
    MACD < 0 (bearish trend)에서 crossover 시 숏 진입합니다.

    Key Features:
        - MACD(12,26,9) crossover 진입
        - MACD > 0 / < 0 trend filter
        - Candle color 기반 청산 (bearish for long, bullish for short)
        - 변동성 스케일링 (vol_target / realized_vol)

    Attributes:
        _config: MTF MACD 설정 (MtfMacdConfig)

    Example:
        >>> from src.strategy.mtf_macd import MtfMacdStrategy, MtfMacdConfig
        >>>
        >>> strategy = MtfMacdStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: MtfMacdConfig | None = None) -> None:
        """MtfMacdStrategy 초기화.

        Args:
            config: MTF MACD 설정. None이면 기본 설정 사용.
        """
        self._config = config or MtfMacdConfig()

    @classmethod
    def from_params(cls, **params: Any) -> MtfMacdStrategy:
        """파라미터로 MtfMacdStrategy 생성 (parameter sweep용).

        Args:
            **params: MtfMacdConfig 파라미터

        Returns:
            새 MtfMacdStrategy 인스턴스
        """
        config = MtfMacdConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "MTF-MACD"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close"]

    @property
    def config(self) -> MtfMacdConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        Calculated Columns:
            - macd_line: MACD Line
            - signal_line: Signal Line
            - macd_histogram: MACD Histogram
            - realized_vol: 실현 변동성 (연환산)
            - vol_scalar: 변동성 스케일러

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
        """MTF MACD 전략에 권장되는 PortfolioManagerConfig 설정.

        추세 추종 전략 특성:
            - 레버리지 2.0x
            - 10% system stop loss
            - 5% rebalance threshold

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
            파라미터명-값 딕셔너리
        """
        cfg = self._config
        mode_str = "Long-Only" if cfg.short_mode == ShortMode.DISABLED else "Long/Short"

        return {
            "macd": f"({cfg.fast_period},{cfg.slow_period},{cfg.signal_period})",
            "vol_window": f"{cfg.vol_window}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def conservative(cls) -> MtfMacdStrategy:
        """보수적 설정 (표준 MACD, 낮은 vol target).

        Returns:
            보수적 파라미터의 MtfMacdStrategy
        """
        return cls(MtfMacdConfig.conservative())

    @classmethod
    def aggressive(cls) -> MtfMacdStrategy:
        """공격적 설정 (빠른 MACD, 높은 vol target).

        Returns:
            공격적 파라미터의 MtfMacdStrategy
        """
        return cls(MtfMacdConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> MtfMacdStrategy:
        """타임프레임별 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 MtfMacdStrategy
        """
        config = MtfMacdConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
