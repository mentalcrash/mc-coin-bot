"""Hour Seasonality Strategy Implementation.

Per-hour rolling t-stat + volume confirm (1H).

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.hour_season.config import HourSeasonConfig, ShortMode
from src.strategy.hour_season.preprocessor import preprocess
from src.strategy.hour_season.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("hour-season")
class HourSeasonStrategy(BaseStrategy):
    """Hour Seasonality Strategy.

    각 시간대(0~23h)의 과거 N일간 수익률에 대해 rolling t-stat을 계산하고,
    통계적으로 유의미한 시간대에서만 진입하는 1H 전략입니다.

    Key Features:
        - Per-hour rolling t-statistic으로 계절성 유의성 검증
        - Relative volume confirm으로 노이즈 필터링
        - Vol-target 기반 포지션 사이징

    Attributes:
        _config: Hour Season 설정 (HourSeasonConfig)
    """

    def __init__(self, config: HourSeasonConfig | None = None) -> None:
        """HourSeasonStrategy 초기화."""
        self._config = config or HourSeasonConfig()

    @classmethod
    def from_params(cls, **params: Any) -> HourSeasonStrategy:
        """파라미터로 HourSeasonStrategy 생성."""
        config = HourSeasonConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """전략 이름."""
        return "Hour-Season"

    @property
    def required_columns(self) -> list[str]:
        """필수 컬럼 목록."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> HourSeasonConfig:
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
        """Hour Season 전략에 권장되는 PortfolioManagerConfig 설정."""
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
            "season_window": f"{cfg.season_window_days}d",
            "t_stat_threshold": f"{cfg.t_stat_threshold:.1f}",
            "vol_confirm": f"{cfg.vol_confirm_threshold:.1f}x",
            "mode": mode_str,
        }
