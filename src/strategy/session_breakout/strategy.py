"""Session Breakout Strategy Implementation.

Asian range breakout (00-08 UTC) 방향 추종 1H 전략.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.session_breakout.config import SessionBreakoutConfig, ShortMode
from src.strategy.session_breakout.preprocessor import preprocess
from src.strategy.session_breakout.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("session-breakout")
class SessionBreakoutStrategy(BaseStrategy):
    """Session Breakout Strategy.

    Asian session (00-08 UTC)의 high/low range를 계산하고,
    trade window에서 breakout 방향을 추종하는 1H 전략입니다.

    Key Features:
        - Asian session range breakout 감지
        - Range percentile로 squeeze (좁은 range) 필터링
        - Trade window / exit hour로 시간 제어
        - Vol-target 기반 포지션 사이징

    Attributes:
        _config: Session Breakout 설정 (SessionBreakoutConfig)
    """

    def __init__(self, config: SessionBreakoutConfig | None = None) -> None:
        """SessionBreakoutStrategy 초기화."""
        self._config = config or SessionBreakoutConfig()

    @classmethod
    def from_params(cls, **params: Any) -> SessionBreakoutStrategy:
        """파라미터로 SessionBreakoutStrategy 생성."""
        config = SessionBreakoutConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Session-Breakout"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> SessionBreakoutConfig:
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
        """Session Breakout 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "asian_session": f"{cfg.asian_start_hour:02d}-{cfg.asian_end_hour:02d} UTC",
            "trade_window": f"{cfg.asian_end_hour:02d}-{cfg.trade_end_hour:02d} UTC",
            "exit_hour": f"{cfg.exit_hour:02d}:00 UTC",
            "range_pctl_threshold": f"{cfg.range_pctl_threshold:.0f}th",
            "mode": mode_str,
        }
