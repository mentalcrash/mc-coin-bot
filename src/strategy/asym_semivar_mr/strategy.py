"""Asymmetric Semivariance MR 전략.

방향별 semivariance 비율로 공포/탐욕 과잉반응 감지 -> contrarian 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.asym_semivar_mr.config import AsymSemivarMRConfig, ShortMode
from src.strategy.asym_semivar_mr.preprocessor import preprocess
from src.strategy.asym_semivar_mr.signal import generate_signals
from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("asym-semivar-mr")
class AsymSemivarMRStrategy(BaseStrategy):
    """Asymmetric Semivariance Mean Reversion 전략 구현.

    Downside/upside semivariance 비율의 Z-score로 극단적
    공포(capitulation)/탐욕(euphoria) 상태를 감지하고 contrarian 진입합니다.

    Key Features:
        - Rolling semivariance 비율로 시장 공포/탐욕 비대칭 측정
        - Z-score 기반 extremes 포착 (mean reversion entry/exit)
        - Timeout-based exit (30-bar)
        - Vol-target sizing
        - Regime awareness (극단 변동성 환경에서 시그널 감쇠)

    Attributes:
        _config: 전략 설정 (AsymSemivarMRConfig)
    """

    def __init__(self, config: AsymSemivarMRConfig | None = None) -> None:
        """AsymSemivarMRStrategy 초기화."""
        self._config = config or AsymSemivarMRConfig()

    @classmethod
    def from_params(cls, **params: Any) -> AsymSemivarMRStrategy:
        """파라미터로 AsymSemivarMRStrategy 생성."""
        config = AsymSemivarMRConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "asym-semivar-mr"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> AsymSemivarMRConfig:
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
        """Asymmetric Semivariance MR 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "semivar_window": str(cfg.semivar_window),
            "zscore_window": str(cfg.zscore_window),
            "entry_zscore": f"{cfg.entry_zscore:.1f}",
            "exit_zscore": f"{cfg.exit_zscore:.1f}",
            "timeout": f"{cfg.exit_timeout_bars} bars",
            "mode": mode_str,
        }
