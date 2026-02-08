"""Z-Score Mean Reversion Strategy.

동적 lookback z-score 기반 평균회귀 전략입니다.
변동성 레짐에 따라 적응적으로 lookback을 전환하여
극단적 z-score 구간에서 평균회귀를 포착합니다.

Rules Applied:
    - #03 Strategy Architecture: BaseStrategy 상속, @register 등록
    - #11 Pydantic Modeling: Config 기반 파라미터 관리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.zscore_mr.config import ShortMode, ZScoreMRConfig
from src.strategy.zscore_mr.preprocessor import preprocess
from src.strategy.zscore_mr.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("zscore-mr")
class ZScoreMRStrategy(BaseStrategy):
    """Z-Score 평균회귀 전략.

    변동성 레짐에 따라 short/long lookback을 전환하여
    적응적 z-score 기반 평균회귀 시그널을 생성합니다.

    TSMOM/Breakout과의 차이:
        - TSMOM/Breakout: 추세 추종 (모멘텀 활용)
        - Z-Score MR: 평균회귀 (극단적 이탈 후 복귀 포착)
        - 두 전략은 상호 보완적

    Example:
        >>> strategy = ZScoreMRStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: ZScoreMRConfig | None = None) -> None:
        self._config = config or ZScoreMRConfig()

    @classmethod
    def from_params(cls, **params: Any) -> ZScoreMRStrategy:
        """파라미터로 전략 생성 (parameter sweep용).

        Args:
            **params: ZScoreMRConfig 파라미터

        Returns:
            새 ZScoreMRStrategy 인스턴스
        """
        config = ZScoreMRConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "Z-Score MR"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> ZScoreMRConfig:
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

        평균회귀는 보수적인 레버리지와 빈번한 리밸런싱을 권장합니다.

        Returns:
            PortfolioManagerConfig 파라미터 딕셔너리
        """
        return {
            "execution_mode": "orders",
            "max_leverage_cap": 1.5,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.03,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 2.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 파라미터."""
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short ({cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "lookback": f"S={cfg.short_lookback}d / L={cfg.long_lookback}d",
            "entry/exit_z": f"{cfg.entry_z:.1f} / {cfg.exit_z:.1f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "high_vol_pct": f"{cfg.high_vol_percentile:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods()

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> ZScoreMRStrategy:
        """타임프레임별 설정."""
        config = ZScoreMRConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
