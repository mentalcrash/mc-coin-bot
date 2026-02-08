"""Donchian Ensemble Strategy Implementation.

BaseStrategy를 상속받아 9개 lookback Donchian Channel 앙상블 전략을 구현합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig, ShortMode
from src.strategy.donchian_ensemble.preprocessor import preprocess
from src.strategy.donchian_ensemble.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("donchian-ensemble")
class DonchianEnsembleStrategy(BaseStrategy):
    """Donchian Ensemble Strategy.

    9개 lookback의 Donchian Channel breakout 시그널을 평균하여
    추세 방향을 결정하고, 변동성 스케일링으로 포지션을 조절합니다.

    Key Features:
        - 다중 타임스케일 앙상블: 5~360일 범위의 9개 lookback
        - Breakout 신호 평균: 각 채널별 +1/0/-1 신호의 산술 평균
        - Vol-target scaling: 실현 변동성 대비 목표 변동성 비율

    Attributes:
        _config: 전략 설정 (DonchianEnsembleConfig)

    Example:
        >>> from src.strategy.donchian_ensemble import DonchianEnsembleStrategy
        >>> strategy = DonchianEnsembleStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: DonchianEnsembleConfig | None = None) -> None:
        """DonchianEnsembleStrategy 초기화.

        Args:
            config: 전략 설정. None이면 기본 설정 사용.
        """
        self._config = config or DonchianEnsembleConfig()

    @classmethod
    def from_params(cls, **params: Any) -> DonchianEnsembleStrategy:
        """파라미터로 DonchianEnsembleStrategy 생성."""
        config = DonchianEnsembleConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Donchian-Ensemble"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록 (high, low, close)."""
        return ["high", "low", "close"]

    @property
    def config(self) -> DonchianEnsembleConfig:
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

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Donchian Ensemble 전략에 권장되는 PortfolioManagerConfig 설정.

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    @classmethod
    def conservative(cls) -> DonchianEnsembleStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 DonchianEnsembleStrategy
        """
        return cls(DonchianEnsembleConfig.conservative())

    @classmethod
    def aggressive(cls) -> DonchianEnsembleStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 DonchianEnsembleStrategy
        """
        return cls(DonchianEnsembleConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> DonchianEnsembleStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 DonchianEnsembleStrategy
        """
        config = DonchianEnsembleConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터.

        Returns:
            파라미터명-값 딕셔너리
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        lookbacks_str = ", ".join(str(lb) for lb in cfg.lookbacks)

        return {
            "lookbacks": f"[{lookbacks_str}]",
            "num_channels": str(len(cfg.lookbacks)),
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }
