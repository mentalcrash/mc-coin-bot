"""Autocorrelation Regime-Adaptive Strategy Implementation.

Returns의 serial correlation으로 regime 감지 후 자동 전환.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.ac_regime.config import ACRegimeConfig, ShortMode
from src.strategy.ac_regime.preprocessor import preprocess
from src.strategy.ac_regime.signal import generate_signals
from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("ac-regime")
class ACRegimeStrategy(BaseStrategy):
    """Autocorrelation Regime-Adaptive Strategy.

    Returns의 serial correlation 부호로 regime을 분류합니다.
    양수 AC → trending (momentum), 음수 AC → mean-reverting (contrarian).

    Key Features:
        - Rolling autocorrelation으로 regime 실시간 감지
        - Bartlett bound로 통계적 유의성 검정
        - Trending regime: momentum 추종
        - Mean-reverting regime: contrarian 전략
        - Neutral: random walk → 관망

    Attributes:
        _config: AC Regime 설정 (ACRegimeConfig)
    """

    def __init__(self, config: ACRegimeConfig | None = None) -> None:
        """ACRegimeStrategy 초기화."""
        self._config = config or ACRegimeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> ACRegimeStrategy:
        """파라미터로 ACRegimeStrategy 생성."""
        config = ACRegimeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "AC-Regime"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> ACRegimeConfig:
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
        """AC Regime 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "ac_window": f"{cfg.ac_window}d",
            "ac_lag": str(cfg.ac_lag),
            "significance_z": f"{cfg.significance_z:.2f}",
            "mom_lookback": f"{cfg.mom_lookback}d",
            "mode": mode_str,
        }
