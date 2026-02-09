"""Vol Structure Regime Strategy Implementation.

이 모듈은 BaseStrategy를 상속받아 Vol Structure Regime 전략을 구현합니다.
Short/long vol ratio와 normalized momentum으로 3 regime을 분류합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_structure.config import ShortMode, VolStructureConfig
from src.strategy.vol_structure.preprocessor import preprocess
from src.strategy.vol_structure.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-structure")
class VolStructureStrategy(BaseStrategy):
    """Vol Structure Regime Strategy.

    Short/long volatility ratio와 normalized momentum을 사용하여
    expansion / neutral / contraction 3가지 regime을 분류하고
    각 regime별 포지셔닝을 수행하는 전략입니다.

    Key Features:
        - Vol ratio(단기/장기)로 변동성 구조 파악
        - Normalized momentum으로 추세 강도 측정
        - Expansion: 강한 추세 추종 (±1)
        - Contraction: 관망 (0)
        - Neutral: 보수적 참여 (±0.5)

    Attributes:
        _config: Vol Structure 설정 (VolStructureConfig)

    Example:
        >>> from src.strategy.vol_structure import VolStructureStrategy, VolStructureConfig
        >>> config = VolStructureConfig(vol_short_window=10, vol_long_window=60)
        >>> strategy = VolStructureStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: VolStructureConfig | None = None) -> None:
        """VolStructureStrategy 초기화.

        Args:
            config: Vol Structure 설정. None이면 기본 설정 사용.
        """
        self._config = config or VolStructureConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VolStructureStrategy:
        """파라미터로 VolStructureStrategy 생성.

        Args:
            **params: VolStructureConfig 생성 파라미터

        Returns:
            VolStructureStrategy 인스턴스
        """
        config = VolStructureConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Vol-Structure"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolStructureConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        OHLCV 데이터에 Vol Structure Regime 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: 수익률
            - vol_short: 단기 변동성
            - vol_long: 장기 변동성
            - vol_ratio: 단기/장기 변동성 비율
            - norm_momentum: 정규화된 모멘텀
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
        """Vol Structure 전략에 권장되는 PortfolioManagerConfig 설정.

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
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "vol_short_window": f"{cfg.vol_short_window}d",
            "vol_long_window": f"{cfg.vol_long_window}d",
            "expansion_threshold": f"vol_ratio>{cfg.expansion_vol_ratio}, mom>{cfg.expansion_mom_threshold}",
            "contraction_threshold": f"vol_ratio<{cfg.contraction_vol_ratio}, mom<{cfg.contraction_mom_threshold}",
            "mode": mode_str,
        }
