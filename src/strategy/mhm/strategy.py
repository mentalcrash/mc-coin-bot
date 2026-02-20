"""MHM (Multi-Horizon Momentum) 전략.

다중 horizon 모멘텀 역변동성 가중 합산 + agreement 기반 conviction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.mhm.config import MHMConfig, ShortMode
from src.strategy.mhm.preprocessor import preprocess
from src.strategy.mhm.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("mhm")
class MHMStrategy(BaseStrategy):
    """MHM (Multi-Horizon Momentum) 전략.

    5개 horizon(5/10/21/63/126)의 모멘텀을 역변동성 가중 합산.
    horizon 간 부호 일치(agreement)로 conviction 조절.
    """

    def __init__(self, config: MHMConfig | None = None) -> None:
        self._config = config or MHMConfig()

    @property
    def name(self) -> str:
        return "mhm"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> MHMConfig:
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
    def from_params(cls, **params: Any) -> MHMStrategy:
        config = MHMConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "horizons": f"{cfg.lookback_1}/{cfg.lookback_2}/{cfg.lookback_3}/{cfg.lookback_4}/{cfg.lookback_5}",
            "agreement": str(cfg.agreement_threshold),
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
