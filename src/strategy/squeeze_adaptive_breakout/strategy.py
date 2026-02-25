"""Squeeze-Adaptive Breakout 전략.

BB inside KC squeeze 해제 시점에 KAMA 적응적 방향 + BB position conviction으로
레버리지 캐스케이드 과대 이동을 포착한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.squeeze_adaptive_breakout.config import ShortMode, SqueezeAdaptiveBreakoutConfig
from src.strategy.squeeze_adaptive_breakout.preprocessor import preprocess
from src.strategy.squeeze_adaptive_breakout.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("squeeze-adaptive-breakout")
class SqueezeAdaptiveBreakoutStrategy(BaseStrategy):
    """Squeeze-Adaptive Breakout 전략 구현.

    변동성 압축(BB inside KC) 해제 시점에 KAMA 적응적 방향 결정 +
    BB position conviction 스케일링으로 breakout 이동을 포착한다.
    """

    def __init__(self, config: SqueezeAdaptiveBreakoutConfig | None = None) -> None:
        self._config = config or SqueezeAdaptiveBreakoutConfig()

    @property
    def name(self) -> str:
        return "squeeze-adaptive-breakout"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> SqueezeAdaptiveBreakoutConfig:
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
    def from_params(cls, **params: Any) -> SqueezeAdaptiveBreakoutStrategy:
        config = SqueezeAdaptiveBreakoutConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: "Hedge-Short",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "bb": f"{cfg.bb_period}p / {cfg.bb_std:.1f}std",
            "kc": f"{cfg.kc_period}p / {cfg.kc_mult:.1f}x ATR",
            "kama": f"ER={cfg.kama_er_lookback}, fast={cfg.kama_fast}, slow={cfg.kama_slow}",
            "bb_pos_thresholds": f"L>{cfg.bb_pos_long_threshold:.1f} / S<{cfg.bb_pos_short_threshold:.1f}",
            "squeeze_lookback": str(cfg.squeeze_lookback),
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
