"""Volume Climax Reversal Strategy Implementation.

극단적 거래량 스파이크에서 capitulation/euphoria 반전을 포착.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_climax.config import ShortMode, VolClimaxConfig
from src.strategy.vol_climax.preprocessor import preprocess
from src.strategy.vol_climax.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-climax")
class VolClimaxStrategy(BaseStrategy):
    """Volume Climax Reversal Strategy.

    극단적 거래량 스파이크(climax)는 집단적 항복(capitulation) 또는
    과열(euphoria)을 의미하며, 에너지 소진 후 단기 반전이 발생합니다.

    Key Features:
        - Volume Z-score 기반 climax 감지
        - OBV-Price divergence로 확신도 강화
        - Close position으로 capitulation/euphoria 분류
        - Vol-target 사이징
        - Timeout 기반 자동 청산

    Attributes:
        _config: Volume Climax 설정 (VolClimaxConfig)
    """

    def __init__(self, config: VolClimaxConfig | None = None) -> None:
        """VolClimaxStrategy 초기화."""
        self._config = config or VolClimaxConfig()

    @classmethod
    def from_params(cls, **params: Any) -> VolClimaxStrategy:
        """파라미터로 VolClimaxStrategy 생성."""
        config = VolClimaxConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Vol-Climax"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolClimaxConfig:
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
        """Volume Climax 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "vol_zscore_window": f"{cfg.vol_zscore_window}bar",
            "climax_threshold": f"{cfg.climax_threshold:.1f}",
            "obv_lookback": f"{cfg.obv_lookback}bar",
            "exit_timeout": f"{cfg.exit_timeout_bars}bar",
            "mode": mode_str,
        }
