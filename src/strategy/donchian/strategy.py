"""Donchian Channel Breakout Strategy Implementation.

터틀 트레이딩 기반 Donchian Channel 돌파 전략.
BaseStrategy를 상속하고 @register("donchian")으로 등록됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.strategy.base import BaseStrategy
from src.strategy.donchian.config import DonchianConfig, ShortMode
from src.strategy.donchian.preprocessor import preprocess
from src.strategy.donchian.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("donchian")
class DonchianStrategy(BaseStrategy):
    """Donchian Channel Breakout 전략.

    터틀 트레이딩 규칙을 따르는 채널 돌파 전략입니다.
    Entry Channel과 Exit Channel을 분리하여 운용합니다.

    Key Features:
        - Entry Channel (N일): 채널 돌파 시 진입
        - Exit Channel (M일): 반대 채널 터치 시 청산 (N > M)
        - ATR 기반 포지션 사이징

    Attributes:
        _config: 전략 설정 (DonchianConfig)

    Example:
        >>> from src.strategy.donchian import DonchianStrategy, DonchianConfig
        >>>
        >>> # 기본 설정 (20/10 터틀 시스템1)
        >>> strategy = DonchianStrategy()
        >>>
        >>> # 보수적 설정 (55/20 터틀 시스템2)
        >>> strategy = DonchianStrategy.conservative()
        >>>
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: DonchianConfig | None = None) -> None:
        """DonchianStrategy 초기화.

        Args:
            config: Donchian 설정. None이면 기본 설정 사용.
        """
        self._config = config or DonchianConfig()

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Donchian"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼."""
        return ["high", "low", "close"]

    @property
    def config(self) -> DonchianConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리.

        Calculated Columns:
            - entry_upper/entry_lower: Entry Channel
            - exit_upper/exit_lower: Exit Channel
            - atr: Average True Range
            - realized_vol: 실현 변동성
            - vol_scalar: 변동성 스케일러

        Args:
            df: OHLCV DataFrame

        Returns:
            지표가 추가된 DataFrame
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성.

        Args:
            df: 전처리된 DataFrame

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """권장 PortfolioManagerConfig 설정.

        Donchian 전략 특성:
            - 추세 추종으로 긴 보유 기간
            - 레버리지 2.0x로 보수적 운용
            - 10% stop loss

        Returns:
            PortfolioManagerConfig 키워드 인자
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널 정보."""
        cfg = self._config
        mode_str = "Long-Only" if cfg.short_mode == ShortMode.DISABLED else "Long/Short"

        return {
            "entry_period": f"{cfg.entry_period}일",
            "exit_period": f"{cfg.exit_period}일",
            "atr_period": f"{cfg.atr_period}일",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }

    @classmethod
    def conservative(cls) -> DonchianStrategy:
        """보수적 설정 (55/20 터틀 시스템2).

        Returns:
            보수적 파라미터의 DonchianStrategy
        """
        return cls(DonchianConfig.conservative())

    @classmethod
    def aggressive(cls) -> DonchianStrategy:
        """공격적 설정 (20/10 터틀 시스템1).

        Returns:
            공격적 파라미터의 DonchianStrategy
        """
        return cls(DonchianConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> DonchianStrategy:
        """타임프레임별 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 DonchianStrategy
        """
        config = DonchianConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

    def warmup_periods(self) -> int:
        """워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()
