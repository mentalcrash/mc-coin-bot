"""Dual Momentum Strategy Implementation.

12H 최적화 횡단면 모멘텀 전략.
Cross-sectional ranking은 IntraPodAllocator(DUAL_MOMENTUM)에서 수행.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.dual_mom.config import DualMomConfig
from src.strategy.dual_mom.preprocessor import preprocess
from src.strategy.dual_mom.signal import generate_signals
from src.strategy.registry import register
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("dual-mom")
class DualMomStrategy(BaseStrategy):
    """Dual Momentum Strategy.

    Per-symbol momentum signal + vol-target sizing.
    Cross-sectional ranking은 IntraPodAllocator(DUAL_MOMENTUM)에서 수행.
    """

    def __init__(self, config: DualMomConfig | None = None) -> None:
        self._config = config or DualMomConfig()

    @classmethod
    def from_params(cls, **params: Any) -> DualMomStrategy:
        """파라미터로 DualMomStrategy 생성."""
        config = DualMomConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "DualMom"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["close", "high", "low", "volume"]

    @property
    def config(self) -> DualMomConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성."""
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods()

    def get_startup_info(self) -> dict[str, str]:
        """CLI 시작 패널에 표시할 핵심 파라미터."""
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "lookback": f"{cfg.lookback} bars",
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_window": f"{cfg.vol_window} bars",
            "annualization": f"{cfg.annualization_factor:.0f}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """권장 PortfolioManagerConfig 설정."""
        return {
            "max_leverage_cap": 1.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
        }
