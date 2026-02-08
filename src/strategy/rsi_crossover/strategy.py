"""RSI Crossover Strategy Implementation.

RSI 크로스오버 기반 평균회귀 전략입니다.
과매도(30) 상향 돌파 시 롱, 과매수(70) 하향 돌파 시 숏,
40/60 도달 시 청산하는 명확한 규칙 기반 전략입니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.rsi_crossover.config import RSICrossoverConfig
from src.strategy.rsi_crossover.preprocessor import preprocess
from src.strategy.rsi_crossover.signal import generate_signals
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("rsi-crossover")
class RSICrossoverStrategy(BaseStrategy):
    """RSI Crossover 전략.

    RSI 크로스오버를 이용한 평균회귀 전략입니다.
    과매도/과매수 레벨 크로스오버로 진입하고 중립 레벨에서 청산합니다.

    Key Features:
        - RSI 30/70 crossover 진입 (명확한 크로스오버 시그널)
        - RSI 40/60 도달 시 청산 (중립 레벨 복귀)
        - 상태 머신 기반 포지션 추적 (Donchian 패턴 재사용)
        - 변동성 스케일링 (vol_target / realized_vol)

    Attributes:
        _config: RSI Crossover 설정 (RSICrossoverConfig)

    Example:
        >>> from src.strategy.rsi_crossover import RSICrossoverStrategy
        >>>
        >>> strategy = RSICrossoverStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: RSICrossoverConfig | None = None) -> None:
        """RSICrossoverStrategy 초기화.

        Args:
            config: RSI Crossover 설정. None이면 기본 설정 사용.
        """
        self._config = config or RSICrossoverConfig()

    @classmethod
    def from_params(cls, **params: Any) -> RSICrossoverStrategy:
        """파라미터로 RSICrossoverStrategy 생성 (parameter sweep용).

        Args:
            **params: RSICrossoverConfig 파라미터

        Returns:
            새 RSICrossoverStrategy 인스턴스
        """
        config = RSICrossoverConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "RSI-Crossover"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> RSICrossoverConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        Calculated Columns:
            - rsi: RSI (0-100)
            - returns: 로그 수익률
            - realized_vol: 실현 변동성 (연환산)
            - vol_scalar: 변동성 스케일러
            - atr: Average True Range
            - drawdown: 최고점 대비 하락률

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
        """RSI Crossover 전략에 권장되는 PortfolioManagerConfig 설정.

        평균회귀 전략 특성:
            - 레버리지 1.5x로 보수적 운용
            - 8% system stop loss (MR은 빠른 손절)
            - 3% rebalance threshold (잦은 리밸런싱)
            - Trailing stop 활성화 (1.5x ATR)

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "execution_mode": "orders",
            "max_leverage_cap": 1.5,
            "system_stop_loss": 0.08,
            "rebalance_threshold": 0.03,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 1.5,
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

        return {
            "rsi_period": f"{cfg.rsi_period}",
            "entry": f"{cfg.entry_oversold:.0f}/{cfg.entry_overbought:.0f}",
            "exit": f"{cfg.exit_long:.0f}/{cfg.exit_short:.0f}",
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
    def conservative(cls) -> RSICrossoverStrategy:
        """보수적 설정 (넓은 entry 범위).

        Returns:
            보수적 파라미터의 RSICrossoverStrategy
        """
        return cls(RSICrossoverConfig.conservative())

    @classmethod
    def aggressive(cls) -> RSICrossoverStrategy:
        """공격적 설정 (좁은 entry 범위).

        Returns:
            공격적 파라미터의 RSICrossoverStrategy
        """
        return cls(RSICrossoverConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> RSICrossoverStrategy:
        """타임프레임별 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 RSICrossoverStrategy
        """
        config = RSICrossoverConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
