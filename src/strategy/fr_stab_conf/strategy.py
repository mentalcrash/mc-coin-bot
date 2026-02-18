"""Funding Rate + Stablecoin Confluence 전략.

FR 극단값과 Stablecoin flow 확인을 결합하여 과열/과냉 반전 포인트를 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fr_stab_conf.config import FrStabConfConfig
from src.strategy.fr_stab_conf.preprocessor import preprocess
from src.strategy.fr_stab_conf.signal import generate_signals


@register("fr-stab-conf")
class FrStabConfStrategy(BaseStrategy):
    """FR + Stablecoin Confluence 전략 구현.

    FR 양수 극단 + Stablecoin 유출 → SHORT (과열 반전)
    FR 음수 극단 + Stablecoin 유입 → LONG (과냉 반전)
    """

    def __init__(self, config: FrStabConfConfig | None = None) -> None:
        self._config = config or FrStabConfConfig()

    @property
    def name(self) -> str:
        return "fr-stab-conf"

    @property
    def required_columns(self) -> list[str]:
        return [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "funding_rate",
            "oc_stablecoin_total_circulating_usd",
        ]

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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = FrStabConfConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fr_ma_window": str(self._config.fr_ma_window),
            "fr_zscore_window": str(self._config.fr_zscore_window),
            "fr_short_threshold": str(self._config.fr_short_threshold),
            "fr_long_threshold": str(self._config.fr_long_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
