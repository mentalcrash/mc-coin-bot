"""KAMA Trend Following Strategy.

Kaufman Adaptive Moving Average 기반 추세 추종 전략입니다.
KAMA는 Efficiency Ratio를 사용하여 추세장과 횡보장을 자동으로 구분합니다.

Rules Applied:
    - #03 Strategy Architecture: BaseStrategy 상속, @register 등록
    - #11 Pydantic Modeling: Config 기반 파라미터 관리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.kama.config import KAMAConfig, ShortMode
from src.strategy.kama.preprocessor import preprocess
from src.strategy.kama.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("kama")
class KAMAStrategy(BaseStrategy):
    """KAMA 추세 추종 전략.

    Kaufman Adaptive Moving Average를 사용하여 추세 방향을 판단하고,
    ATR 필터로 잡음을 제거합니다.

    TSMOM과의 차이:
        - TSMOM: 고정 룩백 기반 모멘텀 추종
        - KAMA: Efficiency Ratio로 적응형 이동평균 사용
        - KAMA는 횡보장에서 자동으로 둔감해짐

    Example:
        >>> strategy = KAMAStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: KAMAConfig | None = None) -> None:
        self._config = config or KAMAConfig()

    @classmethod
    def from_params(cls, **params: Any) -> KAMAStrategy:
        """파라미터로 전략 생성 (parameter sweep용).

        Args:
            **params: KAMAConfig 파라미터

        Returns:
            새 KAMAStrategy 인스턴스
        """
        config = KAMAConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "KAMA"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> KAMAConfig:
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

        KAMA 추세 추종은 적응형 이동평균 기반이므로
        적절한 레버리지와 trailing stop을 권장합니다.

        Returns:
            PortfolioManagerConfig 파라미터 딕셔너리
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
        """CLI 시작 패널에 표시할 파라미터."""
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short ({cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "er_lookback": f"{cfg.er_lookback}d",
            "fast/slow": f"{cfg.fast_period}/{cfg.slow_period}",
            "atr_multiplier": f"{cfg.atr_multiplier:.1f}x ATR",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods()
