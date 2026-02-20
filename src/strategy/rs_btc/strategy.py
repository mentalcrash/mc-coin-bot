"""Relative Strength vs BTC 전략.

BTC 대비 상대 강도 지표 기반 cross-sectional momentum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.rs_btc.config import RsBtcConfig
from src.strategy.rs_btc.preprocessor import preprocess
from src.strategy.rs_btc.signal import generate_signals


@register("rs-btc")
class RsBtcStrategy(BaseStrategy):
    """Relative Strength vs BTC 전략 구현.

    BTC 대비 상대 성과 지속 경향 활용.
    """

    def __init__(self, config: RsBtcConfig | None = None) -> None:
        self._config = config or RsBtcConfig()

    @property
    def name(self) -> str:
        return "rs-btc"

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
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = RsBtcConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "rs_window": str(self._config.rs_window),
            "rs_smooth_window": str(self._config.rs_smooth_window),
            "rs_threshold": str(self._config.rs_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
