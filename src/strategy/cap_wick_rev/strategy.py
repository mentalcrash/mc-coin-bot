"""Capitulation Wick Reversal 전략.

레버리지 청산 캐스케이드 후 가격 과잉반응 -> 48-72h 회복 패턴 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.cap_wick_rev.config import CapWickRevConfig, ShortMode
from src.strategy.cap_wick_rev.preprocessor import preprocess
from src.strategy.cap_wick_rev.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("cap-wick-rev")
class CapWickRevStrategy(BaseStrategy):
    """Capitulation Wick Reversal 전략 구현.

    레버리지 청산 캐스케이드로 인한 가격 과잉반응을 3중 필터
    (ATR spike + Volume surge + Wick ratio)로 감지하고
    confirmation bars 후 contrarian 진입합니다.

    Key Features:
        - 3중 필터로 진짜 capitulation/euphoria만 포착
        - Confirmation bars로 noise 필터링
        - Timeout-based exit (18-bar = 3일 @4H)
        - Vol-target sizing
        - Regime awareness (극단 변동성 환경에서 시그널 감쇠)

    Attributes:
        _config: 전략 설정 (CapWickRevConfig)
    """

    def __init__(self, config: CapWickRevConfig | None = None) -> None:
        """CapWickRevStrategy 초기화."""
        self._config = config or CapWickRevConfig()

    @classmethod
    def from_params(cls, **params: Any) -> CapWickRevStrategy:
        """파라미터로 CapWickRevStrategy 생성."""
        config = CapWickRevConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "cap-wick-rev"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> CapWickRevConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성."""
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Capitulation Wick Reversal 전략에 권장되는 PortfolioManagerConfig 설정."""
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터."""
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "atr_spike": f"{cfg.atr_spike_threshold:.1f}x",
            "vol_surge": f"{cfg.vol_surge_threshold:.1f}x",
            "wick_ratio": f"{cfg.wick_ratio_threshold:.1f}",
            "confirmation": f"{cfg.confirmation_bars} bars",
            "timeout": f"{cfg.exit_timeout_bars} bars",
            "mode": mode_str,
        }
