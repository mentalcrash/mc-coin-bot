"""BTC-Lead Follower Signal 전략.

BTC t-1 수익률로 altcoin t 수익률 방향 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.btc_lead.config import BtcLeadConfig
from src.strategy.btc_lead.preprocessor import preprocess
from src.strategy.btc_lead.signal import generate_signals


@register("btc-lead")
class BtcLeadStrategy(BaseStrategy):
    """BTC-Lead Follower Signal 전략 구현.

    BTC 정보 전파 지연 활용: BTC 수익률이 altcoin 방향 선행.
    """

    def __init__(self, config: BtcLeadConfig | None = None) -> None:
        self._config = config or BtcLeadConfig()

    @property
    def name(self) -> str:
        return "btc-lead"

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
        config = BtcLeadConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "btc_mom_window": str(self._config.btc_mom_window),
            "btc_threshold": str(self._config.btc_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
