"""FR-Pred (Funding Rate Prediction) 전략.

FR z-score 평균회귀 + FR momentum 이중 시그널 기반 방향 예측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.fr_pred.config import FRPredConfig, ShortMode
from src.strategy.fr_pred.preprocessor import preprocess
from src.strategy.fr_pred.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("fr-pred")
class FRPredStrategy(BaseStrategy):
    """FR-Pred (Funding Rate Prediction) 전략.

    FR z-score mean-reversion + FR MA crossover momentum 이중 시그널.
    극단 FR에서 contrarian 진입, FR 추세에서 carry 추종.
    """

    def __init__(self, config: FRPredConfig | None = None) -> None:
        self._config = config or FRPredConfig()

    @property
    def name(self) -> str:
        return "fr-pred"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

    @property
    def config(self) -> FRPredConfig:
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
    def from_params(cls, **params: Any) -> FRPredStrategy:
        config = FRPredConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "fr_mr_threshold": f"{cfg.fr_mr_threshold:.1f}",
            "fr_mom": f"{cfg.fr_mom_fast}/{cfg.fr_mom_slow}",
            "weights": f"MR={cfg.mr_weight:.1f}/Mom={cfg.mom_weight:.1f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
