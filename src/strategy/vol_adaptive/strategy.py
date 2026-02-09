"""Vol-Adaptive Trend Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Vol-Adaptive Trend 전략을 구현합니다.
EMA crossover + RSI confirm + ADX filter + ATR vol-target sizing으로
추세를 추종합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_adaptive.config import ShortMode, VolAdaptiveConfig
from src.strategy.vol_adaptive.preprocessor import preprocess
from src.strategy.vol_adaptive.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-adaptive")
class VolAdaptiveStrategy(BaseStrategy):
    """Vol-Adaptive Trend Strategy.

    EMA crossover로 추세 방향을 판별하고, RSI와 ADX로 확인/필터링한 후,
    ATR 기반 변동성 타겟팅으로 포지션 사이징하는 전략입니다.

    Key Features:
        - EMA crossover 추세 판별 (fast/slow)
        - RSI 추세 확인 (방향 일치 검증)
        - ADX 추세 강도 필터 (약한 추세 제외)
        - 변동성 타겟팅 포지션 사이징

    Attributes:
        _config: Vol-Adaptive 설정 (VolAdaptiveConfig)

    Example:
        >>> from src.strategy.vol_adaptive import VolAdaptiveStrategy, VolAdaptiveConfig
        >>> config = VolAdaptiveConfig(ema_fast=10, ema_slow=50)
        >>> strategy = VolAdaptiveStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: VolAdaptiveConfig | None = None) -> None:
        """VolAdaptiveStrategy 초기화.

        Args:
            config: Vol-Adaptive 설정. None이면 기본 설정 사용.
        """
        self._config = config or VolAdaptiveConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VolAdaptiveStrategy:
        """파라미터로 VolAdaptiveStrategy 생성.

        Args:
            **params: VolAdaptiveConfig 생성 파라미터

        Returns:
            VolAdaptiveStrategy 인스턴스
        """
        config = VolAdaptiveConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Vol-Adaptive"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolAdaptiveConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 Vol-Adaptive Trend 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - vol_scalar: 변동성 스케일러
            - ema_fast: 빠른 EMA
            - ema_slow: 느린 EMA
            - rsi: RSI
            - adx: ADX
            - atr: Average True Range
            - drawdown: 롤링 최고점 대비 드로다운

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

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Vol-Adaptive 전략에 권장되는 PortfolioManagerConfig 설정.

        - execution_mode: orders 모드
        - 레버리지 2.0x로 보수적 운용
        - 10% system stop loss로 큰 손실 방지
        - 10% rebalance threshold로 거래비용 절감
        - Trailing Stop 활성화 (3.0x ATR)

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "execution_mode": "orders",
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
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "ema": f"fast={cfg.ema_fast}, slow={cfg.ema_slow}",
            "rsi": f"period={cfg.rsi_period}, upper={cfg.rsi_upper}, lower={cfg.rsi_lower}",
            "adx": f"period={cfg.adx_period}, threshold={cfg.adx_threshold}",
            "vol_target": f"{cfg.vol_target:.0%} (window={cfg.vol_window})",
            "atr_period": f"{cfg.atr_period}",
            "mode": mode_str,
        }
