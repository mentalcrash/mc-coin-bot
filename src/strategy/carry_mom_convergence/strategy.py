"""Carry-Momentum Convergence 전략.

가격 모멘텀 PRIMARY + FR z-score conviction modifier.
가격-FR 수렴 시 높은 conviction, 발산 시 낮은 conviction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.carry_mom_convergence.config import CarryMomConvergenceConfig, ShortMode
from src.strategy.carry_mom_convergence.preprocessor import preprocess
from src.strategy.carry_mom_convergence.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("carry-mom-convergence")
class CarryMomConvergenceStrategy(BaseStrategy):
    """Carry-Momentum Convergence 전략 구현.

    가격 모멘텀이 alpha source, FR z-score는 추세 건강도 conviction modifier.
    가격-FR 수렴 시 강한 추세, 발산 시 추세 피로로 판단.
    """

    def __init__(self, config: CarryMomConvergenceConfig | None = None) -> None:
        self._config = config or CarryMomConvergenceConfig()

    @property
    def name(self) -> str:
        return "carry-mom-convergence"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

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
        config = CarryMomConvergenceConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "mom_lookback": str(cfg.mom_lookback),
            "ema": f"{cfg.mom_fast}/{cfg.mom_slow}",
            "fr_lookback": str(cfg.fr_lookback),
            "convergence": f"boost={cfg.convergence_boost}/penalty={cfg.divergence_penalty}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "short_mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
