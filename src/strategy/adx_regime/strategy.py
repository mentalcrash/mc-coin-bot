"""ADX Regime Filter Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 ADX Regime Filter 전략을 구현합니다.
ADX 기반으로 momentum/mean-reversion을 자동 블렌딩하여
시장 레짐에 적응하는 전략입니다.

Rules Applied:
    - #03 Strategy Architecture: BaseStrategy 상속, @register 등록
    - #11 Pydantic Modeling: Config 기반 파라미터 관리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.adx_regime.config import ADXRegimeConfig
from src.strategy.adx_regime.preprocessor import preprocess
from src.strategy.adx_regime.signal import generate_signals
from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("adx-regime")
class ADXRegimeStrategy(BaseStrategy):
    """ADX Regime Filter Strategy.

    ADX(Average Directional Index) 기반으로 시장 레짐을 판별하고,
    추세장에서는 momentum, 횡보장에서는 mean-reversion 시그널을
    자동으로 블렌딩합니다.

    Key Features:
        - ADX 기반 자동 레짐 판별 (추세/횡보/전환)
        - Momentum leg: VW-TSMOM 방식 거래량 가중 모멘텀
        - MR leg: Z-Score 기반 평균회귀
        - 선형 블렌딩: ADX 수준에 따라 두 시그널의 가중 합산
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절

    Example:
        >>> strategy = ADXRegimeStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: ADXRegimeConfig | None = None) -> None:
        """ADXRegimeStrategy 초기화.

        Args:
            config: ADX Regime 설정. None이면 기본 설정 사용.
        """
        self._config = config or ADXRegimeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> ADXRegimeStrategy:
        """파라미터로 전략 생성 (parameter sweep용).

        Args:
            **params: ADXRegimeConfig 파라미터

        Returns:
            새 ADXRegimeStrategy 인스턴스
        """
        config = ADXRegimeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "ADX-Regime"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> ADXRegimeConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 ADX Regime Filter 지표를 계산하여 추가합니다.

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

        Returns:
            지표가 추가된 DataFrame
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성.

        전처리된 데이터에서 레짐 적응형 진입/청산 시그널을 생성합니다.

        Args:
            df: 전처리된 DataFrame (preprocess() 출력)

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """권장 PortfolioManagerConfig 설정.

        ADX Regime은 moderate 레버리지와 trailing stop 활용을 권장합니다.
        레짐 전환 시 빈번한 거래를 방지하기 위해 rebalance_threshold를 설정합니다.

        Returns:
            PortfolioManagerConfig 파라미터 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터.

        Returns:
            파라미터명-값 딕셔너리 (사용자 친화적 포맷)
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short ({cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "adx_band": f"[{cfg.adx_low:.0f}, {cfg.adx_high:.0f}]",
            "mom_lookback": f"{cfg.mom_lookback}d",
            "mr_lookback": f"{cfg.mr_lookback}d (z={cfg.mr_entry_z:.1f}/{cfg.mr_exit_z:.1f})",
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
    def conservative(cls) -> ADXRegimeStrategy:
        """보수적 설정."""
        return cls(ADXRegimeConfig.conservative())

    @classmethod
    def aggressive(cls) -> ADXRegimeStrategy:
        """공격적 설정."""
        return cls(ADXRegimeConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> ADXRegimeStrategy:
        """타임프레임별 설정.

        Args:
            timeframe: 타임프레임 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 ADXRegimeStrategy
        """
        config = ADXRegimeConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
