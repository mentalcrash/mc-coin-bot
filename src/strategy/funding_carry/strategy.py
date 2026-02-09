"""Funding Rate Carry Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Funding Rate Carry 전략을 구현합니다.
백테스팅, EDA, 라이브 트레이딩 모두에서 동일하게 사용됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.funding_carry.config import FundingCarryConfig
from src.strategy.funding_carry.preprocessor import preprocess
from src.strategy.funding_carry.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("funding-carry")
class FundingCarryStrategy(BaseStrategy):
    """Funding Rate Carry Strategy.

    Funding rate의 부호를 기반으로 캐리 시그널을 생성하는 전략입니다.
    Positive FR -> short (funding 수취), Negative FR -> long.

    Key Features:
        - Funding rate carry: 양의 FR -> 숏 (캐리 수취)
        - Z-score 정규화: 상대적 FR 크기 파악
        - 변동성 스케일링: 목표 변동성 대비 포지션 크기 조절
        - Entry threshold: 최소 FR 크기 필터

    Attributes:
        _config: Funding Carry 설정 (FundingCarryConfig)

    Example:
        >>> from src.strategy.funding_carry import FundingCarryStrategy, FundingCarryConfig
        >>> config = FundingCarryConfig(lookback=3, vol_target=0.35)
        >>> strategy = FundingCarryStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_with_funding_df)
    """

    def __init__(self, config: FundingCarryConfig | None = None) -> None:
        """FundingCarryStrategy 초기화.

        Args:
            config: Funding Carry 설정. None이면 기본 설정 사용.
        """
        self._config = config or FundingCarryConfig()

    @classmethod
    def from_params(cls, **params: Any) -> FundingCarryStrategy:
        """파라미터로 FundingCarryStrategy 생성.

        Args:
            **params: FundingCarryConfig 생성 파라미터

        Returns:
            FundingCarryStrategy 인스턴스
        """
        config = FundingCarryConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Funding Rate Carry"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "high", "low", "volume", "funding_rate"]

    @property
    def config(self) -> FundingCarryConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV + funding_rate 데이터에 Funding Carry 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - realized_vol: 실현 변동성
            - avg_funding_rate: 평균 펀딩비
            - funding_zscore: 펀딩비 Z-score
            - vol_scalar: 변동성 스케일러
            - atr: Average True Range

        Args:
            df: OHLCV + funding_rate DataFrame (DatetimeIndex 필수)

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
        """Funding Carry 전략에 권장되는 PortfolioManagerConfig 설정.

        - 레버리지 2.0x로 보수적 운용
        - 10% system stop loss로 큰 손실 방지
        - 5% rebalance threshold로 잦은 거래 방지
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
            파라미터명-값 딕셔너리 (사용자 친화적 포맷)
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "lookback": f"{cfg.lookback} periods",
            "zscore_window": f"{cfg.zscore_window}d",
            "vol_target": f"{cfg.vol_target:.0%}",
            "entry_threshold": f"{cfg.entry_threshold:.4f}",
            "mode": mode_str,
        }
