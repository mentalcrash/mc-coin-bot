"""OU Mean Reversion Strategy Implementation.

Ornstein-Uhlenbeck process 파라미터 추정 기반 mean reversion 전략.
Half-life가 짧을 때만 거래하여 trend regime을 자동 회피합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ou_meanrev.config import OUMeanRevConfig, ShortMode
from src.strategy.ou_meanrev.preprocessor import preprocess
from src.strategy.ou_meanrev.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("ou-meanrev")
class OUMeanRevStrategy(BaseStrategy):
    """OU Mean Reversion Strategy.

    Ornstein-Uhlenbeck process의 half-life를 rolling OLS로 추정합니다.
    Half-life가 짧을 때(fast mean reversion) Z-score 기반 진입/청산을 수행합니다.
    Half-life가 길면(trend regime) 자동으로 거래를 중단합니다.

    Key Features:
        - Rolling OLS로 OU 파라미터(theta, mu, half-life) 추정
        - Half-life 필터로 trend regime 자동 회피
        - Z-score 기반 mean reversion entry/exit
        - Timeout-based exit (30-bar)
        - Vol-target sizing

    Attributes:
        _config: OU Mean Reversion 설정 (OUMeanRevConfig)
    """

    def __init__(self, config: OUMeanRevConfig | None = None) -> None:
        """OUMeanRevStrategy 초기화."""
        self._config = config or OUMeanRevConfig()

    @classmethod
    def from_params(cls, **params: Any) -> OUMeanRevStrategy:
        """파라미터로 OUMeanRevStrategy 생성."""
        config = OUMeanRevConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "OU-MeanRev"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> OUMeanRevConfig:
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
        """OU Mean Reversion 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "ou_window": str(cfg.ou_window),
            "entry_zscore": f"{cfg.entry_zscore:.1f}",
            "exit_zscore": f"{cfg.exit_zscore:.1f}",
            "max_half_life": str(cfg.max_half_life),
            "timeout": f"{cfg.exit_timeout_bars} bars",
            "mode": mode_str,
        }
