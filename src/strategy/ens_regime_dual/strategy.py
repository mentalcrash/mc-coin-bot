"""Regime-Adaptive Dual-Alpha Ensemble 전략.

CTREND + regime-mf-mr 결합 메타 앙상블.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.ens_regime_dual.config import EnsRegimeDualConfig
from src.strategy.ens_regime_dual.preprocessor import preprocess
from src.strategy.ens_regime_dual.signal import generate_signals
from src.strategy.ensemble.config import ShortMode
from src.strategy.registry import get_strategy, register

if TYPE_CHECKING:
    from src.strategy.types import StrategySignals

logger = logging.getLogger(__name__)


@register("ens-regime-dual")
class EnsRegimeDualStrategy(BaseStrategy):
    """Regime-Adaptive Dual-Alpha Ensemble 전략 구현.

    CTREND(기술 ML 모멘텀) + regime-mf-mr(레짐 게이트 평균회귀)을
    inverse_volatility로 결합하여 전 시장 환경에 대응.
    """

    def __init__(self, config: EnsRegimeDualConfig | None = None) -> None:
        self._config = config or EnsRegimeDualConfig()
        self._sub_strategies: list[BaseStrategy] = []
        self._strategy_names: list[str] = []
        self._weights = pd.Series(dtype=float)
        self._init_sub_strategies()

    def _init_sub_strategies(self) -> None:
        """Registry에서 서브 전략 인스턴스를 생성."""
        sub_specs = [
            ("ctrend", self._config.ctrend_weight),
            ("regime-mf-mr", self._config.regime_mf_mr_weight),
        ]

        names: list[str] = []
        strategies: list[BaseStrategy] = []
        weight_dict: dict[str, float] = {}

        for name, weight in sub_specs:
            strategy_cls = get_strategy(name)
            strategy = strategy_cls()
            names.append(name)
            strategies.append(strategy)
            weight_dict[name] = weight

        self._strategy_names = names
        self._sub_strategies = strategies
        self._weights = pd.Series(weight_dict)

        logger.info(
            "EnsRegimeDual initialized | strategies=%s, aggregation=%s",
            names,
            self._config.aggregation.value,
        )

    @property
    def name(self) -> str:
        return "ens-regime-dual"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def config(self) -> EnsRegimeDualConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(
            df,
            self._sub_strategies,
            self._strategy_names,
            self._weights,
            self._config,
        )

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간.

        max(sub-strategy warmup) + aggregation lookback.
        """
        sub_warmups = [
            strategy.warmup_periods()  # type: ignore[attr-defined]
            for strategy in self._sub_strategies
            if hasattr(strategy, "warmup_periods")
        ]
        max_sub_warmup = max(sub_warmups) if sub_warmups else 0
        return max_sub_warmup + self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    @classmethod
    def from_params(cls, **params: Any) -> EnsRegimeDualStrategy:
        config = EnsRegimeDualConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "strategies": ", ".join(self._strategy_names),
            "num_strategies": str(len(self._sub_strategies)),
            "aggregation": cfg.aggregation.value,
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
