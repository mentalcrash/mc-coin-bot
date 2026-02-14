"""Jump Drift Momentum 전략.

4H bipower variation 기반 jump 감지 후 post-jump drift 추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.jump_drift_mom.config import JumpDriftMomConfig
from src.strategy.jump_drift_mom.preprocessor import preprocess
from src.strategy.jump_drift_mom.signal import generate_signals


@register("jump-drift-mom")
class JumpDriftMomStrategy(BaseStrategy):
    """Jump Drift Momentum 전략 구현.

    크립토 jump은 정보 이벤트(고래, 규제, liquidation cascade)를 반영.
    Post-jump drift로 정보 이벤트 후 가격 조정 방향 추종.
    """

    def __init__(self, config: JumpDriftMomConfig | None = None) -> None:
        self._config = config or JumpDriftMomConfig()

    @property
    def name(self) -> str:
        return "jump-drift-mom"

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
        config = JumpDriftMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "rv_window": str(self._config.rv_window),
            "bpv_window": str(self._config.bpv_window),
            "jump_zscore_threshold": str(self._config.jump_zscore_threshold),
            "drift_lookback": str(self._config.drift_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
