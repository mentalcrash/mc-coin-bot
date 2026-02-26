"""VWR Asymmetric Multi-Scale 전략.

3-scale VWR 앙상블 + 비대칭 임계값으로 crypto drift 비대칭성 활용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.vwr_asym_multi.config import VwrAsymMultiConfig
from src.strategy.vwr_asym_multi.preprocessor import preprocess
from src.strategy.vwr_asym_multi.signal import generate_signals


@register("vwr-asym-multi")
class VwrAsymMultiStrategy(BaseStrategy):
    """VWR Asymmetric Multi-Scale 전략 구현.

    12H Volume-Weighted Returns 다중스케일(10/21/42) z-score 앙상블에
    비대칭 long/short 임계값을 적용하여 crypto 구조적 drift 비대칭을 반영한다.
    """

    def __init__(self, config: VwrAsymMultiConfig | None = None) -> None:
        self._config = config or VwrAsymMultiConfig()

    @property
    def name(self) -> str:
        return "vwr-asym-multi"

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
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = VwrAsymMultiConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "lookbacks": (
                f"{self._config.lookback_short}/"
                f"{self._config.lookback_mid}/"
                f"{self._config.lookback_long}"
            ),
            "thresholds": (f"L={self._config.long_threshold}/S={self._config.short_threshold}"),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
