"""VW-TSMOM Pure Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 VW-TSMOM Pure 전략을 구현합니다.
백테스팅, EDA, 라이브 트레이딩 모두에서 동일하게 사용됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode
from src.strategy.vw_tsmom.config import VWTSMOMConfig
from src.strategy.vw_tsmom.preprocessor import preprocess
from src.strategy.vw_tsmom.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vw-tsmom")
class VWTSMOMStrategy(BaseStrategy):
    """Volume-Weighted Time Series Momentum Pure Strategy.

    거래량 가중 수익률(VW Returns)만을 사용하는 순수 VW-TSMOM 전략입니다.
    기존 tsmom 전략의 간소화 변형으로, VW returns에만 집중합니다.

    Key Features:
        - 거래량 가중 수익률: log1p(volume)으로 가중된 returns
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절
        - ShortMode 지원: DISABLED, HEDGE_ONLY, FULL

    Attributes:
        _config: VW-TSMOM 설정 (VWTSMOMConfig)

    Example:
        >>> from src.strategy.vw_tsmom import VWTSMOMStrategy, VWTSMOMConfig
        >>> config = VWTSMOMConfig(lookback=21, vol_target=0.35)
        >>> strategy = VWTSMOMStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: VWTSMOMConfig | None = None) -> None:
        """VWTSMOMStrategy 초기화.

        Args:
            config: VW-TSMOM 설정. None이면 기본 설정 사용.
        """
        self._config = config or VWTSMOMConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VWTSMOMStrategy:
        """파라미터로 VWTSMOMStrategy 생성."""
        config = VWTSMOMConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "VW-TSMOM Pure"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "volume"]

    @property
    def config(self) -> VWTSMOMConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 VW-TSMOM Pure 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - vw_returns: 거래량 가중 수익률
            - vol_scalar: 변동성 스케일러
            - drawdown: 드로다운
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
        """VW-TSMOM Pure 전략에 권장되는 PortfolioManagerConfig 설정.

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
            파라미터명-값 딕셔너리 (사용자 친화적 포맷)
        """
        cfg = self._config

        # 숏 모드 문자열 변환
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short ({cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        result = {
            "lookback": f"{cfg.lookback}d",
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_window": f"{cfg.vol_window}d",
            "mode": mode_str,
        }

        # 헤지 모드일 때 추가 정보
        if cfg.short_mode == ShortMode.HEDGE_ONLY:
            result["hedge_strength"] = f"{cfg.hedge_strength_ratio:.0%}"

        return result

    @classmethod
    def conservative(cls) -> VWTSMOMStrategy:
        """보수적 설정의 전략 인스턴스 생성.

        Returns:
            보수적 파라미터의 VWTSMOMStrategy
        """
        return cls(VWTSMOMConfig.conservative())

    @classmethod
    def aggressive(cls) -> VWTSMOMStrategy:
        """공격적 설정의 전략 인스턴스 생성.

        Returns:
            공격적 파라미터의 VWTSMOMStrategy
        """
        return cls(VWTSMOMConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> VWTSMOMStrategy:
        """특정 타임프레임에 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "15m")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 VWTSMOMStrategy
        """
        config = VWTSMOMConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
