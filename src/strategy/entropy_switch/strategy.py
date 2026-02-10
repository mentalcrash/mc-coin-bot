"""Entropy Regime Switch Strategy Implementation.

Shannon Entropy로 시장 예측가능성을 측정하여 trend-following filter 적용.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.entropy_switch.config import EntropySwitchConfig, ShortMode
from src.strategy.entropy_switch.preprocessor import preprocess
from src.strategy.entropy_switch.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("entropy-switch")
class EntropySwitchStrategy(BaseStrategy):
    """Entropy Regime Switch Strategy.

    Shannon Entropy로 시장 예측가능성을 측정합니다.
    Low entropy (규칙적 패턴) → trend following.
    High entropy (랜덤) → stop trading.

    Key Features:
        - Information theory 기반 regime 분류
        - Shannon Entropy rolling window
        - ADX 보조 필터
        - Low entropy: momentum 추종
        - High entropy: 관망

    Attributes:
        _config: Entropy Switch 설정 (EntropySwitchConfig)
    """

    def __init__(self, config: EntropySwitchConfig | None = None) -> None:
        """EntropySwitchStrategy 초기화."""
        self._config = config or EntropySwitchConfig()

    @classmethod
    def from_params(cls, **params: Any) -> EntropySwitchStrategy:
        """파라미터로 EntropySwitchStrategy 생성."""
        config = EntropySwitchConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Entropy-Switch"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> EntropySwitchConfig:
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
        """Entropy Switch 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "entropy_window": str(cfg.entropy_window),
            "entropy_bins": str(cfg.entropy_bins),
            "thresholds": f"[{cfg.entropy_low_threshold}, {cfg.entropy_high_threshold}]",
            "adx_filter": f"{cfg.use_adx_filter} (>{cfg.adx_threshold})",
            "mode": mode_str,
        }
