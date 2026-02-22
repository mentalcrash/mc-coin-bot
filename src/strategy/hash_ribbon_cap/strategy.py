"""Hash-Ribbon Capitulation 전략.

BTC 채굴자 capitulation 후 매도 압력 소멸 → 회복 모멘텀 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.hash_ribbon_cap.config import HashRibbonCapConfig
from src.strategy.hash_ribbon_cap.preprocessor import preprocess
from src.strategy.hash_ribbon_cap.signal import generate_signals


@register("hash-ribbon-cap")
class HashRibbonCapStrategy(BaseStrategy):
    """Hash-Ribbon Capitulation 전략 구현.

    SMA fast/slow cross로 capitulation 구간을 프록시하고,
    회복 확인 후 long 진입.
    """

    def __init__(self, config: HashRibbonCapConfig | None = None) -> None:
        self._config = config or HashRibbonCapConfig()

    @property
    def name(self) -> str:
        return "hash-ribbon-cap"

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
        config = HashRibbonCapConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "hash_fast_window": str(self._config.hash_fast_window),
            "hash_slow_window": str(self._config.hash_slow_window),
            "recovery_confirm_bars": str(self._config.recovery_confirm_bars),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
