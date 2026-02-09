"""Multi-Factor Ensemble Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Multi-Factor Ensemble 전략을 구현합니다.
3개의 직교 팩터(모멘텀, 거래량 충격, 역변동성)를 균등 가중 결합합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.multi_factor.config import MultiFactorConfig
from src.strategy.multi_factor.preprocessor import preprocess
from src.strategy.multi_factor.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("multi-factor")
class MultiFactorStrategy(BaseStrategy):
    """Multi-Factor Ensemble Strategy.

    3개의 직교 팩터(모멘텀, 거래량 충격, 역변동성)를 z-score 정규화 후
    균등 가중 결합하여 복합 시그널을 생성하는 전략입니다.

    Key Features:
        - 모멘텀 팩터: 가격 추세 추종
        - 거래량 충격 팩터: 비정상 거래량 감지
        - 역변동성 팩터: Low Volatility Premium 포착
        - Z-score 정규화: 팩터 간 동일 스케일
        - 변동성 타겟팅: 포지션 사이징

    Attributes:
        _config: Multi-Factor 설정 (MultiFactorConfig)

    Example:
        >>> from src.strategy.multi_factor import MultiFactorStrategy, MultiFactorConfig
        >>> config = MultiFactorConfig(momentum_lookback=21, vol_target=0.35)
        >>> strategy = MultiFactorStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: MultiFactorConfig | None = None) -> None:
        """MultiFactorStrategy 초기화.

        Args:
            config: Multi-Factor 설정. None이면 기본 설정 사용.
        """
        self._config = config or MultiFactorConfig()

    @classmethod
    def from_params(cls, **params: Any) -> MultiFactorStrategy:
        """파라미터로 MultiFactorStrategy 생성.

        Args:
            **params: MultiFactorConfig 생성 파라미터

        Returns:
            MultiFactorStrategy 인스턴스
        """
        config = MultiFactorConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Multi-Factor Ensemble"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> MultiFactorConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 팩터 계산.

        OHLCV 데이터에 Multi-Factor 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률 (로그)
            - realized_vol: 실현 변동성 (연환산)
            - vol_scalar: 변동성 스케일러
            - momentum_factor: 모멘텀 팩터 z-score
            - volume_shock_factor: 거래량 충격 팩터 z-score
            - volatility_factor: 역변동성 팩터 z-score
            - combined_score: 3개 팩터 균등 가중 평균
            - atr: Average True Range

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

        Returns:
            팩터가 추가된 DataFrame
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
        """Multi-Factor 전략에 권장되는 PortfolioManagerConfig 설정.

        - 레버리지 2.0x로 보수적 운용
        - 10% system stop loss로 큰 손실 방지
        - 10% rebalance threshold로 거래비용 절감
        - Trailing Stop 활성화 (3.0x ATR)

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
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "momentum_lookback": f"{cfg.momentum_lookback}",
            "volume_shock_window": f"{cfg.volume_shock_window}",
            "vol_window": f"{cfg.vol_window}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "zscore_window": f"{cfg.zscore_window}",
            "mode": mode_str,
        }
