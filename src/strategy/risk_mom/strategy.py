"""Risk-Managed Momentum Strategy Implementation.

BaseStrategy를 상속받아 Risk-Managed Momentum 전략을 구현합니다.
TSMOM + Barroso-Santa-Clara (2015) variance scaling.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.risk_mom.config import RiskMomConfig
from src.strategy.risk_mom.preprocessor import preprocess
from src.strategy.risk_mom.signal import generate_signals
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("risk-mom")
class RiskMomStrategy(BaseStrategy):
    """Risk-Managed Momentum Strategy.

    거래량 가중 모멘텀에 Barroso-Santa-Clara variance scaling을 적용하여
    리스크를 조절하는 전략입니다. 실현 분산이 높을 때 포지션을 축소하고,
    낮을 때 포지션을 확대합니다.

    Key Differences from TSMOM:
        - vol_scalar 대신 bsc_scaling 사용
        - bsc_scaling = vol_target^2 / realized_var
        - 6개월(126일) 분산 윈도우로 더 안정적인 스케일링

    Attributes:
        _config: Risk-Mom 설정 (RiskMomConfig)

    Example:
        >>> from src.strategy.risk_mom import RiskMomStrategy, RiskMomConfig
        >>> config = RiskMomConfig(lookback=30, var_window=126)
        >>> strategy = RiskMomStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: RiskMomConfig | None = None) -> None:
        """RiskMomStrategy 초기화.

        Args:
            config: Risk-Mom 설정. None이면 기본 설정 사용.
        """
        self._config = config or RiskMomConfig()

    @classmethod
    def from_params(cls, **params: Any) -> RiskMomStrategy:
        """파라미터로 RiskMomStrategy 생성."""
        config = RiskMomConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Risk-Mom"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록 (OHLCV 전체 필요 — ATR 계산용)."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> RiskMomConfig:
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
        """Risk-Mom 전략에 권장되는 PortfolioManagerConfig 설정.

        모멘텀 전략 특성상 TSMOM과 동일한 PM 설정을 권장합니다.

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    @classmethod
    def conservative(cls) -> RiskMomStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 RiskMomStrategy
        """
        return cls(RiskMomConfig.conservative())

    @classmethod
    def aggressive(cls) -> RiskMomStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 RiskMomStrategy
        """
        return cls(RiskMomConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> RiskMomStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "15m")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 RiskMomStrategy
        """
        config = RiskMomConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)

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

        result: dict[str, str] = {
            "lookback": f"{cfg.lookback}d",
            "var_window": f"{cfg.var_window}d (BSC)",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }

        if cfg.short_mode == ShortMode.HEDGE_ONLY:
            result["hedge_strength"] = f"{cfg.hedge_strength_ratio:.0%}"

        return result
