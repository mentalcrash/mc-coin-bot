"""Regime-Gated Multi-Factor Mean Reversion 전략.

Ranging 레짐에서만 활성화되는 멀티팩터 평균회귀.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.regime_mf_mr.config import RegimeMfMrConfig, ShortMode
from src.strategy.regime_mf_mr.preprocessor import preprocess
from src.strategy.regime_mf_mr.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("regime-mf-mr")
class RegimeMfMrStrategy(BaseStrategy):
    """Regime-Gated Multi-Factor MR 전략 구현.

    크립토 MR alpha가 ranging 레짐에서만 존재한다는 가설 기반.
    p_ranging 확률 게이팅으로 trending 구간 손실을 근본 차단.
    """

    def __init__(self, config: RegimeMfMrConfig | None = None) -> None:
        self._config = config or RegimeMfMrConfig()

    @property
    def name(self) -> str:
        return "regime-mf-mr"

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

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간."""
        return self._config.warmup_periods()

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
    def from_params(cls, **params: Any) -> RegimeMfMrStrategy:
        config = RegimeMfMrConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "bb_period": str(cfg.bb_period),
            "zscore_window": str(cfg.zscore_window),
            "rsi_period": str(cfg.rsi_period),
            "regime_gate": f"{cfg.regime_gate_threshold:.0%}",
            "min_factors": str(cfg.min_factor_agreement),
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
