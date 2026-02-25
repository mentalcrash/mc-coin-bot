"""VRP-Regime Trend 전략.

BTC/ETH 옵션시장 VRP(IV-RV spread)를 레짐 indicator로 활용한 추세추종 전략.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vrp_regime_trend.config import VrpRegimeTrendConfig
from src.strategy.vrp_regime_trend.preprocessor import preprocess
from src.strategy.vrp_regime_trend.signal import generate_signals


@register("vrp-regime-trend")
class VrpRegimeTrendStrategy(BaseStrategy):
    """VRP-Regime Trend 전략 구현.

    Deribit DVOL(IV) vs Garman-Klass RV 스프레드(VRP) + EMA 추세 확인.
    고VRP = 과공포 프리미엄 수취(롱), 저VRP = 실제위험(숏). 8H TF.
    """

    def __init__(self, config: VrpRegimeTrendConfig | None = None) -> None:
        self._config = config or VrpRegimeTrendConfig()

    @property
    def name(self) -> str:
        return "vrp-regime-trend"

    @property
    def required_columns(self) -> list[str]:
        # opt_dvol은 optional (Graceful Degradation)
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
        config = VrpRegimeTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "gk_rv_window": str(self._config.gk_rv_window),
            "vrp_ma_window": str(self._config.vrp_ma_window),
            "vrp_zscore_window": str(self._config.vrp_zscore_window),
            "vrp_high_z": str(self._config.vrp_high_z),
            "vrp_low_z": str(self._config.vrp_low_z),
            "trend_ema_fast": str(self._config.trend_ema_fast),
            "trend_ema_slow": str(self._config.trend_ema_slow),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
