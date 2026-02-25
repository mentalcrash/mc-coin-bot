"""Z-Momentum (MACD-V) 전략.

ATR-정규화 MACD (MACD-V) + 명시적 flat zone으로 Vol-anchoring bias 교정.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.z_mom.config import ZMomConfig
from src.strategy.z_mom.preprocessor import preprocess
from src.strategy.z_mom.signal import generate_signals


@register("z-mom")
class ZMomStrategy(BaseStrategy):
    """Z-Momentum (MACD-V) 전략 구현.

    MACD를 ATR로 정규화(MACD-V)하여 변동성 앵커링 편향을 교정하고,
    명시적 flat zone으로 noisy crossover를 필터링한다.
    """

    def __init__(self, config: ZMomConfig | None = None) -> None:
        self._config = config or ZMomConfig()

    @property
    def name(self) -> str:
        return "z-mom"

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
        config = ZMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "macd_fast": str(self._config.macd_fast),
            "macd_slow": str(self._config.macd_slow),
            "macd_signal": str(self._config.macd_signal),
            "atr_period": str(self._config.atr_period),
            "flat_zone": str(self._config.flat_zone),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
