"""Entropy-Carry-Momentum 전략.

Shannon entropy로 시장 예측 가능성 측정하여 모멘텀과 FR carry 가중치 적응적 조절.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.entropy_carry_mom.config import EntropyCarryMomConfig, ShortMode
from src.strategy.entropy_carry_mom.preprocessor import preprocess
from src.strategy.entropy_carry_mom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("entropy-carry-mom")
class EntropyCarryMomStrategy(BaseStrategy):
    """Entropy-Carry-Momentum 전략 구현.

    Shannon entropy로 시장 예측가능성을 측정하여,
    낮은 entropy(규칙적) -> momentum 우위, 높은 entropy(무질서) -> FR carry 우위로
    적응적 multi-factor 가중치를 조절한다.
    """

    def __init__(self, config: EntropyCarryMomConfig | None = None) -> None:
        self._config = config or EntropyCarryMomConfig()

    @property
    def name(self) -> str:
        return "entropy-carry-mom"

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
        config = EntropyCarryMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "entropy_window": str(cfg.entropy_window),
            "mom_lookback": str(cfg.mom_lookback),
            "fr_lookback": str(cfg.fr_lookback),
            "vol_target": f"{cfg.vol_target:.0%}",
            "short_mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
