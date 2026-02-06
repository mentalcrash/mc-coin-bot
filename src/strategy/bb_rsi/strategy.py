"""BB+RSI Mean Reversion Strategy.

볼린저밴드 + RSI 기반 평균회귀 전략입니다.
횡보장(ADX < 25)에서 과매수/과매도 구간의 평균회귀를 포착합니다.
TSMOM과 반대 환경에서 작동하여 상호 보완적입니다.

Rules Applied:
    - #03 Strategy Architecture: BaseStrategy 상속, @register 등록
    - #11 Pydantic Modeling: Config 기반 파라미터 관리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.bb_rsi.config import BBRSIConfig, ShortMode
from src.strategy.bb_rsi.preprocessor import preprocess
from src.strategy.bb_rsi.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("bb-rsi")
class BBRSIStrategy(BaseStrategy):
    """BB+RSI 평균회귀 전략.

    볼린저밴드와 RSI를 조합하여 과매수/과매도 구간에서
    평균회귀 시그널을 생성합니다.

    TSMOM과의 차이:
        - TSMOM: 추세 추종 (ADX < 25에서 약함)
        - BB+RSI: 평균회귀 (ADX >= 25에서 약함)
        - 두 전략은 상호 보완적

    Example:
        >>> strategy = BBRSIStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: BBRSIConfig | None = None) -> None:
        self._config = config or BBRSIConfig()

    @classmethod
    def from_params(cls, **params: Any) -> BBRSIStrategy:
        """파라미터로 전략 생성 (parameter sweep용).

        Args:
            **params: BBRSIConfig 파라미터

        Returns:
            새 BBRSIStrategy 인스턴스
        """
        config = BBRSIConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        return "BB-RSI"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> BBRSIConfig:
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

        평균회귀는 보수적인 레버리지와 trailing stop 활용을 권장합니다.

        Returns:
            PortfolioManagerConfig 파라미터 딕셔너리
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
        """CLI 시작 패널에 표시할 파라미터."""
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short ({cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        result = {
            "bb_period": f"{cfg.bb_period}d (+-{cfg.bb_std}std)",
            "rsi_period": f"{cfg.rsi_period}d ({cfg.rsi_oversold:.0f}/{cfg.rsi_overbought:.0f})",
            "vol_target": f"{cfg.vol_target:.0%}",
            "weights": f"BB {cfg.bb_weight:.0%} / RSI {cfg.rsi_weight:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
        if cfg.use_adx_filter:
            result["adx_filter"] = (
                f"ADX>={cfg.adx_threshold:.0f} → {cfg.trending_position_scale:.0%}"
            )
        return result

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods()

    @classmethod
    def conservative(cls) -> BBRSIStrategy:
        """보수적 설정."""
        return cls(BBRSIConfig.conservative())

    @classmethod
    def aggressive(cls) -> BBRSIStrategy:
        """공격적 설정."""
        return cls(BBRSIConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> BBRSIStrategy:
        """타임프레임별 설정."""
        config = BBRSIConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
