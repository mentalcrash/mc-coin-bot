"""GK Volatility Breakout Strategy Implementation.

Garman-Klass 변동성 압축 후 Donchian 채널 돌파 전략.
BaseStrategy를 상속하고 @register("gk-breakout")으로 등록됩니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.gk_breakout.config import GKBreakoutConfig, ShortMode
from src.strategy.gk_breakout.preprocessor import preprocess
from src.strategy.gk_breakout.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("gk-breakout")
class GKBreakoutStrategy(BaseStrategy):
    """GK Volatility Breakout 전략.

    Garman-Klass variance로 변동성 압축 구간을 감지하고,
    Donchian Channel 돌파 시 진입하는 전략입니다.

    Key Features:
        - GK Variance: OHLC 4가지 가격 활용으로 효율적 변동성 추정
        - Vol Compression: 단기/장기 변동성 비율로 압축 구간 감지
        - Donchian Breakout: 압축 후 채널 돌파 시 추세 진입
        - Vol Scaling: 목표 변동성 대비 포지션 크기 조절

    Attributes:
        _config: GK Breakout 설정 (GKBreakoutConfig)

    Example:
        >>> from src.strategy.gk_breakout import GKBreakoutStrategy
        >>>
        >>> strategy = GKBreakoutStrategy()
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: GKBreakoutConfig | None = None) -> None:
        """GKBreakoutStrategy 초기화.

        Args:
            config: GK Breakout 설정. None이면 기본 설정 사용.
        """
        self._config = config or GKBreakoutConfig()

    @classmethod
    def from_params(cls, **params: Any) -> GKBreakoutStrategy:
        """파라미터로 GKBreakoutStrategy 생성 (parameter sweep용).

        Args:
            **params: GKBreakoutConfig 파라미터

        Returns:
            새 GKBreakoutStrategy 인스턴스
        """
        config = GKBreakoutConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "GK-Breakout"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼.

        GK variance 계산에 OHLC, 거래량 데이터가 필요합니다.
        """
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> GKBreakoutConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 (지표 계산).

        Calculated Columns:
            - returns, realized_vol, vol_scalar
            - gk_var, vol_ratio
            - dc_upper, dc_lower
            - atr, drawdown

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
        """GK Breakout 전략에 권장되는 PortfolioManagerConfig 설정.

        GK Breakout 전략 특성:
            - 변동성 압축 후 돌파로 레버리지 2.0x
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
            ShortMode.HEDGE_ONLY: f"Hedge-Short ({cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "gk_lookback": f"{cfg.gk_lookback}d",
            "compression_threshold": f"{cfg.compression_threshold:.2f}",
            "breakout_lookback": f"{cfg.breakout_lookback}d",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }

    def warmup_periods(self) -> int:
        """워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()
