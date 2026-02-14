"""Funding Rate Carry (Vol-Conditioned) 전략.

극단 FR에서 contrarian carry 수취, 저변동성 환경에서만 활성화.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.fr_carry_vol.config import FRCarryVolConfig, ShortMode
from src.strategy.fr_carry_vol.preprocessor import preprocess
from src.strategy.fr_carry_vol.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("fr-carry-vol")
class FRCarryVolStrategy(BaseStrategy):
    """Funding Rate Carry (Vol-Conditioned) 전략 구현.

    극단 FR에서 contrarian carry로 보험료 수취.
    저변동성 환경에서만 carry 포지션 유지.
    """

    def __init__(self, config: FRCarryVolConfig | None = None) -> None:
        self._config = config or FRCarryVolConfig()

    @property
    def name(self) -> str:
        return "fr-carry-vol"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "funding_rate"]

    @property
    def config(self) -> FRCarryVolConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        return {
            "stop_loss_pct": 0.08,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.5,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
        }

    @classmethod
    def from_params(cls, **params: Any) -> FRCarryVolStrategy:
        config = FRCarryVolConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "fr_lookback": f"{cfg.fr_lookback} bars",
            "fr_extreme_zscore": f"{cfg.fr_extreme_zscore:.1f}",
            "vol_condition_pctile": f"{cfg.vol_condition_pctile:.0%}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
