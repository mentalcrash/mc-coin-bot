"""Vol Squeeze Breakout 전략.

변동성 극저 스퀴즈 후 방향성 탈출 포착.
변동성 군집(GARCH) 기반 구조적 비효율, 이벤트 드리븐 극저빈도 거래.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vol_squeeze_brk.config import ShortMode, VolSqueezeBrkConfig
from src.strategy.vol_squeeze_brk.preprocessor import preprocess
from src.strategy.vol_squeeze_brk.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vol-squeeze-brk")
class VolSqueezeBrkStrategy(BaseStrategy):
    """Vol Squeeze Breakout 전략 구현.

    변동성 극저 스퀴즈 후 방향성 탈출을 포착.
    BB width percentile + ATR ratio로 스퀴즈 감지,
    BB 돌파 + 거래량 서지로 breakout 확인.
    """

    def __init__(self, config: VolSqueezeBrkConfig | None = None) -> None:
        self._config = config or VolSqueezeBrkConfig()

    @property
    def name(self) -> str:
        return "vol-squeeze-brk"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VolSqueezeBrkConfig:
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
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> VolSqueezeBrkStrategy:
        config = VolSqueezeBrkConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "bb_period": str(cfg.bb_period),
            "bb_pct_threshold": f"{cfg.bb_pct_threshold:.0%}",
            "atr_ratio_threshold": f"{cfg.atr_ratio_threshold:.2f}",
            "vol_surge_multiplier": f"{cfg.vol_surge_multiplier:.1f}x",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
