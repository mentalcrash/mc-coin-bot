"""Liquidity-Confirmed Trend 전략.

On-chain 유동성 복합지수가 가격 모멘텀 방향을 확인할 때만 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.liq_conf_trend.config import LiqConfTrendConfig, ShortMode
from src.strategy.liq_conf_trend.preprocessor import preprocess
from src.strategy.liq_conf_trend.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("liq-conf-trend")
class LiqConfTrendStrategy(BaseStrategy):
    """Liquidity-Confirmed Trend 전략 구현.

    On-chain 유동성(stablecoin + TVL) 성장이 가격 모멘텀 방향을 확인할 때 진입.
    F&G 극단에서는 contrarian override로 행동편향 포착.
    On-chain 데이터 미존재 시 graceful degradation (neutral score).
    """

    def __init__(self, config: LiqConfTrendConfig | None = None) -> None:
        self._config = config or LiqConfTrendConfig()

    @property
    def name(self) -> str:
        return "liq-conf-trend"

    @property
    def required_columns(self) -> list[str]:
        # On-chain columns are optional (graceful degradation)
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
        config = LiqConfTrendConfig(**params)
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
            "liq_threshold": str(cfg.liq_score_threshold),
            "fg_fear/greed": f"{cfg.fg_fear_threshold}/{cfg.fg_greed_threshold}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "short_mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
