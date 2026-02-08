"""Vol-Regime Adaptive Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Vol-Regime Adaptive 전략을 구현합니다.
변동성 regime별 파라미터 자동 전환으로 시장 상황에 적응합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_regime.config import ShortMode, VolRegimeConfig
from src.strategy.vol_regime.preprocessor import preprocess
from src.strategy.vol_regime.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-regime")
class VolRegimeStrategy(BaseStrategy):
    """Vol-Regime Adaptive Strategy.

    변동성 regime(high/normal/low)별 TSMOM 파라미터를 자동 전환하여
    시장 상황에 적응하는 전략입니다.

    Key Features:
        - 변동성 regime 자동 판별 (percentile rank 기반)
        - Regime별 최적화된 모멘텀 lookback 및 vol target
        - 고변동성: 보수적 (긴 lookback, 낮은 vol target)
        - 저변동성: 공격적 (짧은 lookback, 높은 vol target)

    Attributes:
        _config: Vol-Regime 설정 (VolRegimeConfig)

    Example:
        >>> from src.strategy.vol_regime import VolRegimeStrategy, VolRegimeConfig
        >>> config = VolRegimeConfig(high_vol_threshold=0.8, low_vol_threshold=0.2)
        >>> strategy = VolRegimeStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: VolRegimeConfig | None = None) -> None:
        """VolRegimeStrategy 초기화.

        Args:
            config: Vol-Regime 설정. None이면 기본 설정 사용.
        """
        self._config = config or VolRegimeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VolRegimeStrategy:
        """파라미터로 VolRegimeStrategy 생성.

        Args:
            **params: VolRegimeConfig 생성 파라미터

        Returns:
            VolRegimeStrategy 인스턴스
        """
        config = VolRegimeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Vol-Regime"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolRegimeConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 Vol-Regime Adaptive 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - vol_regime: 변동성 percentile rank
            - regime_strength: regime별 모멘텀 * vol scalar
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
        """Vol-Regime 전략에 권장되는 PortfolioManagerConfig 설정.

        - 레버리지 2.0x로 보수적 운용
        - 10% system stop loss로 큰 손실 방지
        - 5% rebalance threshold로 regime 전환 시 빠른 반응

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
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
            "vol_lookback": f"{cfg.vol_lookback}d",
            "vol_rank_lookback": f"{cfg.vol_rank_lookback}d",
            "high_vol": f"lookback={cfg.high_vol_lookback}, target={cfg.high_vol_target:.0%}",
            "normal": f"lookback={cfg.normal_lookback}, target={cfg.normal_vol_target:.0%}",
            "low_vol": f"lookback={cfg.low_vol_lookback}, target={cfg.low_vol_target:.0%}",
            "thresholds": f"high>{cfg.high_vol_threshold:.0%}, low<{cfg.low_vol_threshold:.0%}",
            "mode": mode_str,
        }
