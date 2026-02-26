"""Donchian Cascade MTF 전략.

12H-equivalent Donchian breakout을 4H 해상도로 감지하여 진입 타이밍 최적화.
4H EMA confirmation으로 false breakout 필터링.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.donch_cascade.config import DonchCascadeConfig
from src.strategy.donch_cascade.preprocessor import preprocess
from src.strategy.donch_cascade.signal import generate_signals


@register("donch-cascade")
class DonchCascadeStrategy(BaseStrategy):
    """Donchian Cascade MTF 전략 구현.

    12H-equivalent Donchian Channel(3x lookback)의 breakout consensus로
    방향을 결정하고, 4H EMA confirmation으로 진입 타이밍을 최적화한다.

    Benefits:
        - 브레이크아웃 최대 8시간 조기 감지 (12H → 4H 해상도)
        - Momentum confirmation으로 false breakout 필터링
        - max_wait_bars 후 강제 진입 → 기존 대비 최악의 경우에도 동등
    """

    def __init__(self, config: DonchCascadeConfig | None = None) -> None:
        self._config = config or DonchCascadeConfig()

    @property
    def name(self) -> str:
        return "donch-cascade"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> BaseModel:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        return {
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = DonchCascadeConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        lbs = self._config.actual_lookbacks()
        return {
            "lookbacks (base)": (
                f"{self._config.lookback_short}/"
                f"{self._config.lookback_mid}/"
                f"{self._config.lookback_long}"
            ),
            "lookbacks (actual)": f"{lbs[0]}/{lbs[1]}/{lbs[2]}",
            "htf_multiplier": str(self._config.htf_multiplier),
            "entry_threshold": str(self._config.entry_threshold),
            "confirm_ema": str(self._config.confirm_ema_period),
            "max_wait": str(self._config.max_wait_bars),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
