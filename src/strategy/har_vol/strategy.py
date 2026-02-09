"""HAR Volatility Overlay Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 HAR-RV 변동성 예측 기반 전략을 구현합니다.
HAR 모델로 변동성을 예측하고 vol surprise(realized - forecast)로 매매합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.har_vol.config import HARVolConfig
from src.strategy.har_vol.preprocessor import preprocess
from src.strategy.har_vol.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("har-vol")
class HARVolStrategy(BaseStrategy):
    """HAR Volatility Overlay Strategy.

    HAR-RV(Heterogeneous Autoregressive Realized Volatility) 모델을 사용하여
    변동성을 예측하고, 실현 변동성과 예측의 차이(vol surprise)를 기반으로
    모멘텀/평균회귀 시그널을 생성하는 전략입니다.

    Key Features:
        - Parkinson volatility: High-Low range 기반 효율적 변동성 추정
        - HAR model: 이질적 시장 참여자(daily/weekly/monthly)를 반영
        - Vol surprise: realized > forecast → momentum, forecast > realized → mean-reversion
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절

    Attributes:
        _config: HAR Volatility 설정 (HARVolConfig)

    Example:
        >>> from src.strategy.har_vol import HARVolStrategy, HARVolConfig
        >>> config = HARVolConfig(training_window=252, vol_target=0.35)
        >>> strategy = HARVolStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: HARVolConfig | None = None) -> None:
        """HARVolStrategy 초기화.

        Args:
            config: HAR Volatility 설정. None이면 기본 설정 사용.
        """
        self._config = config or HARVolConfig()

    @classmethod
    def from_params(cls, **params: Any) -> HARVolStrategy:
        """파라미터로 HARVolStrategy 생성.

        Args:
            **params: HARVolConfig 생성 파라미터

        Returns:
            HARVolStrategy 인스턴스
        """
        config = HARVolConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "HAR Volatility"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> HARVolConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 HAR-RV 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - parkinson_vol: Parkinson volatility
            - rv_daily, rv_weekly, rv_monthly: HAR features
            - har_forecast: HAR-RV OLS forecast
            - vol_surprise: realized - forecast
            - realized_vol: 실현 변동성 (연환산)
            - vol_scalar: 변동성 스케일러
            - atr: Average True Range

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
        """HAR Volatility 전략에 권장되는 PortfolioManagerConfig 설정.

        - 레버리지 2.0x로 보수적 운용
        - 10% system stop loss로 큰 손실 방지
        - 10% rebalance threshold로 regime 전환 시 반응
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
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "daily_window": f"{cfg.daily_window}d",
            "weekly_window": f"{cfg.weekly_window}d",
            "monthly_window": f"{cfg.monthly_window}d",
            "training_window": f"{cfg.training_window}d",
            "vol_surprise_threshold": f"{cfg.vol_surprise_threshold:.4f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }
