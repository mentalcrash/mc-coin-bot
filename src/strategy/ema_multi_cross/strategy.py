"""EMA Multi-Cross 전략 — 3쌍 EMA 크로스 합의 투표."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ema_multi_cross.config import EmaMultiCrossConfig
from src.strategy.ema_multi_cross.preprocessor import preprocess
from src.strategy.ema_multi_cross.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("ema-multi-cross")
class EmaMultiCrossStrategy(BaseStrategy):
    """3쌍 EMA 크로스 합의 투표 전략.

    단기(8/21), 중기(20/50), 장기(50/100) EMA pair의 방향을 투표.
    2/3 이상 합의 시 진입, 불일치 시 관망.
    미니 앙상블 접근으로 단일 크로스 대비 whipsaw 감소.
    """

    def __init__(self, config: EmaMultiCrossConfig | None = None) -> None:
        self._config = config or EmaMultiCrossConfig()

    @property
    def name(self) -> str:
        return "ema-multi-cross"

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
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> EmaMultiCrossStrategy:
        config = EmaMultiCrossConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "pair1": f"{self._config.pair1_fast}/{self._config.pair1_slow}",
            "pair2": f"{self._config.pair2_fast}/{self._config.pair2_slow}",
            "pair3": f"{self._config.pair3_fast}/{self._config.pair3_slow}",
            "min_votes": str(self._config.min_votes),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
