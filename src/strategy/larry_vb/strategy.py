"""Larry Williams Volatility Breakout Strategy Implementation.

전일 변동폭 기반 돌파 전략입니다.
BaseStrategy를 상속하고 @register("larry-vb")으로 등록됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.larry_vb.config import LarryVBConfig, ShortMode
from src.strategy.larry_vb.preprocessor import preprocess
from src.strategy.larry_vb.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("larry-vb")
class LarryVBStrategy(BaseStrategy):
    """Larry Williams Volatility Breakout 전략.

    전일 변동폭(High - Low)에 k_factor를 곱한 만큼을 당일 시가에 더한 레벨을
    돌파하면 진입합니다. 1-bar hold 패턴으로 다음 바에서 청산합니다.

    Key Features:
        - 전일 변동폭 기반 돌파 레벨 산출
        - k_factor로 돌파 민감도 조절
        - 1-bar hold: 단기 보유 후 청산
        - Vol Scaling: 목표 변동성 대비 포지션 크기 조절
        - Long/Short 모드 (기본: FULL)

    Attributes:
        _config: Larry VB 설정 (LarryVBConfig)

    Example:
        >>> from src.strategy.larry_vb import LarryVBStrategy
        >>>
        >>> strategy = LarryVBStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: LarryVBConfig | None = None) -> None:
        """LarryVBStrategy 초기화.

        Args:
            config: Larry VB 설정. None이면 기본 설정 사용.
        """
        self._config = config or LarryVBConfig()

    @classmethod
    def from_params(cls, **params: Any) -> LarryVBStrategy:
        """파라미터로 LarryVBStrategy 생성 (parameter sweep용).

        Args:
            **params: LarryVBConfig 파라미터

        Returns:
            새 LarryVBStrategy 인스턴스
        """
        config = LarryVBConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Larry-VB"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼.

        변동폭 계산에 OHLC 데이터가 필요합니다.
        """
        return ["open", "high", "low", "close"]

    @property
    def config(self) -> LarryVBConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 (지표 계산).

        Calculated Columns:
            - prev_range, breakout_upper, breakout_lower
            - realized_vol, vol_scalar

        Args:
            df: OHLCV DataFrame

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

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Larry VB 전략에 권장되는 PortfolioManagerConfig 설정.

        Larry VB 전략 특성:
            - 단기 돌파 전략이므로 레버리지 2.0x
            - 10% system stop loss
            - 5% rebalance threshold

        Returns:
            PortfolioManagerConfig 키워드 인자 딕셔너리
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 파라미터."""
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "k_factor": f"{cfg.k_factor:.2f}",
            "vol_window": f"{cfg.vol_window}d",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }

    def warmup_periods(self) -> int:
        """워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()
