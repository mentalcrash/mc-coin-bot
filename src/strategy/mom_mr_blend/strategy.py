"""Momentum + Mean Reversion Blend Strategy.

Momentum Z-Score와 Mean Reversion Z-Score를 블렌딩하여
직교적 알파 소스를 포착하는 전략입니다.

Rules Applied:
    - #03 Strategy Architecture: BaseStrategy 상속, @register 등록
    - #11 Pydantic Modeling: Config 기반 파라미터 관리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.mom_mr_blend.config import MomMrBlendConfig, ShortMode
from src.strategy.mom_mr_blend.preprocessor import preprocess
from src.strategy.mom_mr_blend.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("mom-mr-blend")
class MomMrBlendStrategy(BaseStrategy):
    """Momentum + Mean Reversion Blend 전략.

    Momentum Z-Score(28d)와 Mean Reversion Z-Score(14d)를 50/50 블렌딩하여
    추세장에서는 모멘텀, 횡보장에서는 평균회귀 알파를 포착합니다.

    TSMOM과의 차이:
        - TSMOM: 순수 추세 추종 (단일 모멘텀)
        - Mom-MR Blend: 모멘텀 + 평균회귀 직교 블렌딩
        - 두 전략은 서로 다른 시장 레짐에서 보완적 성과

    Example:
        >>> strategy = MomMrBlendStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: MomMrBlendConfig | None = None) -> None:
        """MomMrBlendStrategy 초기화.

        Args:
            config: Mom-MR Blend 설정. None이면 기본 설정 사용.
        """
        self._config = config or MomMrBlendConfig()

    @classmethod
    def from_params(cls, **params: Any) -> MomMrBlendStrategy:
        """파라미터로 전략 생성 (parameter sweep용).

        Args:
            **params: MomMrBlendConfig 파라미터

        Returns:
            새 MomMrBlendStrategy 인스턴스
        """
        config = MomMrBlendConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Mom-MR-Blend"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록 (close만 필요)."""
        return ["close"]

    @property
    def config(self) -> MomMrBlendConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """전처리 (지표 계산)."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """시그널 생성."""
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """권장 PortfolioManagerConfig 설정.

        블렌드 전략은 중간 수준의 레버리지와 리밸런싱을 권장합니다.

        Returns:
            PortfolioManagerConfig 파라미터 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    @classmethod
    def conservative(cls) -> MomMrBlendStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 MomMrBlendStrategy
        """
        return cls(MomMrBlendConfig.conservative())

    @classmethod
    def aggressive(cls) -> MomMrBlendStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 MomMrBlendStrategy
        """
        return cls(MomMrBlendConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> MomMrBlendStrategy:
        """타임프레임별 설정.

        Args:
            timeframe: 타임프레임 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 MomMrBlendStrategy
        """
        config = MomMrBlendConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods()

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 파라미터."""
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }

        return {
            "mom_lookback": f"{cfg.mom_lookback}d",
            "mr_lookback": f"{cfg.mr_lookback}d",
            "blend_weights": f"Mom={cfg.mom_weight:.0%} / MR={cfg.mr_weight:.0%}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
