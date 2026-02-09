"""Hurst/ER Regime Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Hurst/ER Regime 전략을 구현합니다.
Efficiency Ratio와 Hurst exponent로 시장 regime을 판별하고
regime별 최적 전략(momentum/mean-reversion)을 적용합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.hurst_regime.config import HurstRegimeConfig, ShortMode
from src.strategy.hurst_regime.preprocessor import preprocess
from src.strategy.hurst_regime.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("hurst-regime")
class HurstRegimeStrategy(BaseStrategy):
    """Hurst/ER Regime Strategy.

    Efficiency Ratio + R/S Hurst exponent로 시장 regime
    (trending/mean-reverting/neutral)을 판별하고,
    regime별 최적 전략을 적용합니다.

    Key Features:
        - ER: 가격 효율성 측정 (방향성 / 변동 합)
        - Hurst: 시계열 기억 특성 측정 (persistent vs anti-persistent)
        - Trending regime: momentum following
        - MR regime: z-score fading
        - Neutral: reduced momentum

    Attributes:
        _config: Hurst/ER Regime 설정 (HurstRegimeConfig)

    Example:
        >>> from src.strategy.hurst_regime import HurstRegimeStrategy, HurstRegimeConfig
        >>> config = HurstRegimeConfig(er_lookback=20, hurst_window=100)
        >>> strategy = HurstRegimeStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: HurstRegimeConfig | None = None) -> None:
        """HurstRegimeStrategy 초기화.

        Args:
            config: Hurst/ER Regime 설정. None이면 기본 설정 사용.
        """
        self._config = config or HurstRegimeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> HurstRegimeStrategy:
        """파라미터로 HurstRegimeStrategy 생성.

        Args:
            **params: HurstRegimeConfig 생성 파라미터

        Returns:
            HurstRegimeStrategy 인스턴스
        """
        config = HurstRegimeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Hurst-Regime"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> HurstRegimeConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 Hurst/ER Regime 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - er: Efficiency Ratio
            - hurst: Rolling Hurst exponent
            - momentum: 누적 수익률
            - z_score: Z-Score
            - realized_vol: 실현 변동성
            - vol_scalar: 변동성 스케일러
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

        전처리된 데이터에서 regime을 판별하고 진입/청산 시그널을 생성합니다.

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
        """Hurst/ER Regime 전략에 권장되는 PortfolioManagerConfig 설정.

        - 레버리지 2.0x로 보수적 운용
        - 10% system stop loss
        - 10% rebalance threshold (regime 전환 반영)
        - Trailing stop 활성화 (3.0x ATR)

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
        """
        return {
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
            "er_lookback": f"{cfg.er_lookback}d",
            "hurst_window": f"{cfg.hurst_window}d",
            "er_thresholds": f"trend>{cfg.er_trend_threshold:.2f}, mr<{cfg.er_mr_threshold:.2f}",
            "hurst_thresholds": f"trend>{cfg.hurst_trend_threshold:.2f}, mr<{cfg.hurst_mr_threshold:.2f}",
            "mode": mode_str,
        }
