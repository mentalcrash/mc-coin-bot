"""Carry-Sentiment Gate 전략.

Funding Rate carry premium을 F&G sentiment gate로 타이밍 개선.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.carry_sent.config import CarrySentConfig, ShortMode
from src.strategy.carry_sent.preprocessor import preprocess
from src.strategy.carry_sent.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("carry-sent")
class CarrySentStrategy(BaseStrategy):
    """Carry-Sentiment Gate 전략 구현.

    FR carry premium을 F&G sentiment gate로 필터링하여 타이밍 개선.
    F&G 극단에서는 contrarian override로 행동편향 포착.
    """

    def __init__(self, config: CarrySentConfig | None = None) -> None:
        self._config = config or CarrySentConfig()

    @property
    def name(self) -> str:
        return "carry-sent"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate", "oc_fear_greed"]

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
        config = CarrySentConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "fr_lookback": str(cfg.fr_lookback),
            "fg_fear/greed": f"{cfg.fg_fear_threshold}/{cfg.fg_greed_threshold}",
            "fg_gate": f"[{cfg.fg_gate_low}, {cfg.fg_gate_high}]",
            "vol_target": f"{cfg.vol_target:.0%}",
            "short_mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
