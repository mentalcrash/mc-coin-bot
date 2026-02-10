"""Liquidity-Adjusted Momentum Strategy Implementation.

Amihud + RelVol → TSMOM conviction scaling (1H).

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.liq_momentum.config import LiqMomentumConfig, ShortMode
from src.strategy.liq_momentum.preprocessor import preprocess
from src.strategy.liq_momentum.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("liq-momentum")
class LiqMomentumStrategy(BaseStrategy):
    """Liquidity-Adjusted Momentum Strategy.

    Amihud illiquidity와 relative volume으로 유동성 상태를 분류하고,
    low liquidity에서 conviction을 확대하는 1H 전략입니다.

    Key Features:
        - Amihud illiquidity로 시장 미시구조 파악
        - Relative volume으로 유동성 상태 분류
        - 저유동성 환경에서 모멘텀 conviction 확대
        - 주말 효과 반영

    Attributes:
        _config: Liq Momentum 설정 (LiqMomentumConfig)
    """

    def __init__(self, config: LiqMomentumConfig | None = None) -> None:
        """LiqMomentumStrategy 초기화."""
        self._config = config or LiqMomentumConfig()

    @classmethod
    def from_params(cls, **params: Any) -> LiqMomentumStrategy:
        """파라미터로 LiqMomentumStrategy 생성."""
        config = LiqMomentumConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Liq-Momentum"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> LiqMomentumConfig:
        """전략 설정."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 및 지표 계산."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """매매 시그널 생성."""
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (1H bars)."""
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Liq Momentum 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "mom_lookback": f"{cfg.mom_lookback}h",
            "low_liq_mult": f"{cfg.low_liq_multiplier:.1f}x",
            "high_liq_mult": f"{cfg.high_liq_multiplier:.1f}x",
            "weekend_mult": f"{cfg.weekend_multiplier:.1f}x",
            "mode": mode_str,
        }
