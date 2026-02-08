"""TTM Squeeze Strategy Implementation.

BaseStrategy를 상속받아 TTM Squeeze 전략을 구현합니다.
Bollinger Bands + Keltner Channels squeeze 해제 시 momentum 방향 진입.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.ttm_squeeze.config import ShortMode, TtmSqueezeConfig
from src.strategy.ttm_squeeze.preprocessor import preprocess
from src.strategy.ttm_squeeze.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("ttm-squeeze")
class TtmSqueezeStrategy(BaseStrategy):
    """TTM Squeeze Strategy.

    Bollinger Bands가 Keltner Channels 안으로 수축(squeeze) 후
    해제될 때 momentum 방향으로 진입하는 전략입니다.

    Key Features:
        - BB/KC squeeze 감지 (저변동성 -> 변동성 확장)
        - Momentum = close - donchian midline (방향 결정)
        - Exit SMA cross (청산 조건)
        - 변동성 스케일링 (vol_target / realized_vol)

    Attributes:
        _config: TTM Squeeze 설정 (TtmSqueezeConfig)

    Example:
        >>> from src.strategy.ttm_squeeze import TtmSqueezeStrategy, TtmSqueezeConfig
        >>> config = TtmSqueezeConfig(bb_period=20, kc_mult=1.5)
        >>> strategy = TtmSqueezeStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: TtmSqueezeConfig | None = None) -> None:
        """TtmSqueezeStrategy 초기화.

        Args:
            config: TTM Squeeze 설정. None이면 기본 설정 사용.
        """
        self._config = config or TtmSqueezeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> TtmSqueezeStrategy:
        """파라미터로 TtmSqueezeStrategy 생성 (parameter sweep용).

        Args:
            **params: TtmSqueezeConfig 파라미터

        Returns:
            새 TtmSqueezeStrategy 인스턴스
        """
        config = TtmSqueezeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "TTM-Squeeze"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close"]

    @property
    def config(self) -> TtmSqueezeConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산.

        Calculated Columns:
            - bb_upper, bb_lower: Bollinger Bands
            - kc_upper, kc_lower: Keltner Channels
            - squeeze_on: Squeeze 상태 (bool)
            - momentum: close - donchian midline
            - exit_sma: 청산용 SMA
            - realized_vol: 실현 변동성
            - vol_scalar: 변동성 스케일러

        Args:
            df: OHLCV DataFrame (DatetimeIndex 필수)

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
        """TTM Squeeze 전략에 권장되는 PortfolioManagerConfig 설정.

        Squeeze breakout 전략 특성:
            - 레버리지 2.0x
            - 10% system stop loss
            - 5% rebalance threshold

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
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "bb": f"{cfg.bb_period}p / {cfg.bb_std:.1f}std",
            "kc": f"{cfg.kc_period}p / {cfg.kc_mult:.1f}x ATR",
            "momentum": f"{cfg.mom_period}p",
            "exit_sma": f"{cfg.exit_sma_period}p",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self._config.warmup_periods()

    @classmethod
    def conservative(cls) -> TtmSqueezeStrategy:
        """보수적 설정 (넓은 BB/KC).

        Returns:
            보수적 파라미터의 TtmSqueezeStrategy
        """
        return cls(TtmSqueezeConfig.conservative())

    @classmethod
    def aggressive(cls) -> TtmSqueezeStrategy:
        """공격적 설정 (좁은 BB/KC).

        Returns:
            공격적 파라미터의 TtmSqueezeStrategy
        """
        return cls(TtmSqueezeConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> TtmSqueezeStrategy:
        """타임프레임별 최적화된 전략 생성.

        Args:
            timeframe: 타임프레임 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 TtmSqueezeStrategy
        """
        config = TtmSqueezeConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
