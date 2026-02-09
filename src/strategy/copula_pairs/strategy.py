"""Copula Pairs Trading Strategy Implementation.

Engle-Granger cointegration -> spread -> z-score -> mean-reversion signals.
Full copula fitting is deferred to a later phase.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.copula_pairs.config import CopulaPairsConfig
from src.strategy.copula_pairs.preprocessor import preprocess
from src.strategy.copula_pairs.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("copula-pairs")
class CopulaPairsStrategy(BaseStrategy):
    """Copula Pairs Trading Strategy.

    Engle-Granger cointegration 기반 페어 트레이딩 전략입니다.
    Spread z-score의 평균 회귀 성질을 이용하여 시그널을 생성합니다.

    Key Features:
        - Rolling OLS hedge ratio로 동적 베타 추정
        - Spread z-score 기반 진입/청산/스탑
        - Stateful 시그널 (ffill state machine)
        - Vol-target 포지션 사이징

    Attributes:
        _config: CopulaPairs 설정 (CopulaPairsConfig)

    Example:
        >>> from src.strategy.copula_pairs import CopulaPairsStrategy, CopulaPairsConfig
        >>> config = CopulaPairsConfig(formation_window=63, zscore_entry=2.0)
        >>> strategy = CopulaPairsStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: CopulaPairsConfig | None = None) -> None:
        """CopulaPairsStrategy 초기화.

        Args:
            config: CopulaPairs 설정. None이면 기본 설정 사용.
        """
        self._config = config or CopulaPairsConfig()

    @classmethod
    def from_params(cls, **params: Any) -> CopulaPairsStrategy:
        """파라미터로 CopulaPairsStrategy 생성.

        Args:
            **params: CopulaPairsConfig 생성 파라미터

        Returns:
            CopulaPairsStrategy 인스턴스
        """
        config = CopulaPairsConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Copula Pairs"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "high", "low", "volume", "pair_close"]

    @property
    def config(self) -> CopulaPairsConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV + pair_close 데이터에 페어 트레이딩 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - hedge_ratio: Rolling OLS beta
            - spread: close - beta * pair_close
            - spread_zscore: Rolling z-score of spread
            - vol_scalar: 변동성 스케일러
            - atr: Average True Range

        Args:
            df: OHLCV + pair_close DataFrame (DatetimeIndex 필수)

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
        """Copula Pairs 전략에 권장되는 PortfolioManagerConfig 설정.

        - 레버리지 2.0x로 보수적 운용
        - 5% rebalance threshold
        - 10% system stop loss
        - Trailing stop 활성화 (3.0x ATR)

        Returns:
            PortfolioManagerConfig 생성에 필요한 키워드 인자 딕셔너리
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
            파라미터명-값 딕셔너리
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "formation_window": f"{cfg.formation_window}d",
            "zscore_entry": f"{cfg.zscore_entry:.1f}",
            "zscore_exit": f"{cfg.zscore_exit:.1f}",
            "zscore_stop": f"{cfg.zscore_stop:.1f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }
