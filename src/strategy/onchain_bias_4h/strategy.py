"""On-chain Bias 4H 전략.

On-chain Phase(MVRV/Flow/Stablecoin) 1D gate + 4H momentum timing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.onchain_bias_4h.config import OnchainBias4hConfig
from src.strategy.onchain_bias_4h.preprocessor import preprocess
from src.strategy.onchain_bias_4h.signal import generate_signals


@register("onchain-bias-4h")
class OnchainBias4hStrategy(BaseStrategy):
    """On-chain Bias 4H 전략 구현.

    On-chain phase 판정 → directional bias + momentum timing.
    """

    def __init__(self, config: OnchainBias4hConfig | None = None) -> None:
        self._config = config or OnchainBias4hConfig()

    @property
    def name(self) -> str:
        return "onchain-bias-4h"

    @property
    def required_columns(self) -> list[str]:
        return [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "oc_mvrv",
            "oc_flow_in_ex_usd",
            "oc_flow_out_ex_usd",
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
        config = OnchainBias4hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "mvrv_accumulation": str(self._config.mvrv_accumulation),
            "mvrv_distribution": str(self._config.mvrv_distribution),
            "er_min": str(self._config.er_min),
            "roc_threshold": str(self._config.roc_threshold),
            "short_mode": self._config.short_mode.name,
        }
