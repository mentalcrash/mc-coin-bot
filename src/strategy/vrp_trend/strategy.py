"""VRP-Trend 전략.

DVOL(IV) vs RV 스프레드(VRP)와 추세 확인으로 시장 방향 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vrp_trend.config import VrpTrendConfig
from src.strategy.vrp_trend.preprocessor import preprocess
from src.strategy.vrp_trend.signal import generate_signals


@register("vrp-trend")
class VrpTrendStrategy(BaseStrategy):
    """VRP-Trend 전략 구현.

    크립토 옵션 시장 DVOL(IV) vs RV 스프레드로 시장 방향 예측.
    고VRP=과공포 프리미엄 수취(롱), 저VRP=실제위험(숏).
    """

    def __init__(self, config: VrpTrendConfig | None = None) -> None:
        self._config = config or VrpTrendConfig()

    @property
    def name(self) -> str:
        return "vrp-trend"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "dvol"]

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
        config = VrpTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "rv_window": str(self._config.rv_window),
            "vrp_ma_window": str(self._config.vrp_ma_window),
            "vrp_zscore_window": str(self._config.vrp_zscore_window),
            "vrp_entry_z": str(self._config.vrp_entry_z),
            "vrp_exit_z": str(self._config.vrp_exit_z),
            "trend_sma_window": str(self._config.trend_sma_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
