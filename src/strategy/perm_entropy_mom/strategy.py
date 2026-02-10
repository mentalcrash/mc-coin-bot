"""Permutation Entropy Momentum Strategy Implementation.

Low PE = orderly price pattern = high conviction momentum following.
High PE = noise = reduced/zero position.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.perm_entropy_mom.config import PermEntropyMomConfig, ShortMode
from src.strategy.perm_entropy_mom.preprocessor import preprocess
from src.strategy.perm_entropy_mom.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("perm-entropy-mom")
class PermEntropyMomStrategy(BaseStrategy):
    """Permutation Entropy Momentum Strategy.

    Uses Permutation Entropy to measure the orderliness of price patterns.
    Low PE indicates trending/orderly markets where momentum is more reliable.
    High PE indicates noisy/random markets where positions are reduced.

    Key Features:
        - PE measures pattern complexity via ordinal rankings
        - Short-term PE (5d) for tactical conviction scaling
        - Long-term PE (10d) for regime context
        - Momentum direction + vol-target sizing + PE conviction scaling
        - Noise gate: PE > 0.95 -> zero position (pure noise zone)

    Attributes:
        _config: Strategy configuration (PermEntropyMomConfig)
    """

    def __init__(self, config: PermEntropyMomConfig | None = None) -> None:
        """Initialize PermEntropyMomStrategy."""
        self._config = config or PermEntropyMomConfig()

    @classmethod
    def from_params(cls, **params: Any) -> PermEntropyMomStrategy:
        """Create PermEntropyMomStrategy from parameters."""
        config = PermEntropyMomConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "Perm-Entropy-Mom"

    @property
    def required_columns(self) -> list[str]:
        """Required columns."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> PermEntropyMomConfig:
        """Strategy configuration."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data and calculate indicators."""
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """Generate trading signals."""
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        """Required warmup period (in bars)."""
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Recommended PortfolioManagerConfig for PE Momentum strategy."""
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
            "use_trailing_stop": True,
            "trailing_stop_atr_multiplier": 3.0,
        }

    def get_startup_info(self) -> dict[str, str]:
        """Key parameters for CLI startup panel."""
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "pe_order": str(cfg.pe_order),
            "pe_short_window": f"{cfg.pe_short_window} bars",
            "pe_long_window": f"{cfg.pe_long_window} bars",
            "mom_lookback": f"{cfg.mom_lookback} bars",
            "noise_threshold": f"{cfg.noise_threshold:.2f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }
