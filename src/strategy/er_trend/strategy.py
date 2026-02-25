"""ER Trend 전략.

Multi-lookback Signed ER 가중 합성으로 추세 품질과 방향을 동시에 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.er_trend.config import ErTrendConfig
from src.strategy.er_trend.preprocessor import preprocess
from src.strategy.er_trend.signal import generate_signals


@register("er-trend")
class ErTrendStrategy(BaseStrategy):
    """ER Trend 전략 구현.

    Multi-lookback Efficiency Ratio에 방향 정보(sign)를 곱한 Signed ER을
    가중 합성하여 추세의 품질과 방향을 동시에 판단한다.
    군집행동/레버리지 청산 증폭 시 정보 캐스케이드를 활용.
    """

    def __init__(self, config: ErTrendConfig | None = None) -> None:
        self._config = config or ErTrendConfig()

    @property
    def name(self) -> str:
        return "er-trend"

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
        config = ErTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "er_fast": str(self._config.er_fast),
            "er_mid": str(self._config.er_mid),
            "er_slow": str(self._config.er_slow),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
