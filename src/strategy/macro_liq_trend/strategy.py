"""Macro-Liquidity Adaptive Trend 전략.

글로벌 유동성(DXY, VIX, Stablecoin, SPY)과 가격 모멘텀 정렬로 크립토 방향 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.macro_liq_trend.config import MacroLiqTrendConfig
from src.strategy.macro_liq_trend.preprocessor import preprocess
from src.strategy.macro_liq_trend.signal import generate_signals


@register("macro-liq-trend")
class MacroLiqTrendStrategy(BaseStrategy):
    """Macro-Liquidity Adaptive Trend 전략 구현.

    글로벌 유동성 composite score(DXY 하락 + VIX 하락 + Stablecoin 증가 + SPY 상승)와
    가격 모멘텀(SMA cross)의 정렬로 크립토 방향성을 예측한다.
    """

    def __init__(self, config: MacroLiqTrendConfig | None = None) -> None:
        self._config = config or MacroLiqTrendConfig()

    @property
    def name(self) -> str:
        return "macro-liq-trend"

    @property
    def required_columns(self) -> list[str]:
        return [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "macro_dxy",
            "macro_vix",
            "macro_spy",
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
        config = MacroLiqTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "dxy_roc_period": str(self._config.dxy_roc_period),
            "vix_roc_period": str(self._config.vix_roc_period),
            "spy_roc_period": str(self._config.spy_roc_period),
            "stab_change_period": str(self._config.stab_change_period),
            "zscore_window": str(self._config.zscore_window),
            "price_mom_period": str(self._config.price_mom_period),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
