"""Stochastic Momentum Hybrid Strategy Implementation.

BaseStrategy subclass combining Stochastic %K/%D crossover,
SMA trend filter, and ATR-based dynamic position sizing.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.stoch_mom.config import ShortMode, StochMomConfig
from src.strategy.stoch_mom.preprocessor import preprocess
from src.strategy.stoch_mom.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("stoch-mom")
class StochMomStrategy(BaseStrategy):
    """Stochastic Momentum Hybrid Strategy.

    Combines Stochastic oscillator (%K/%D crossover) with SMA trend filter
    and ATR-based volatility ratio for dynamic position sizing. The trend
    filter ensures entries are aligned with the prevailing trend while the
    stochastic oscillator provides precise timing.

    Key Features:
        - Stochastic %K/%D crossover for entry timing
        - SMA(30) trend filter for direction confirmation
        - ATR/close vol_ratio for dynamic position sizing
        - Volatility scalar for risk-adjusted strength

    Attributes:
        _config: Strategy configuration (StochMomConfig)

    Example:
        >>> from src.strategy.stoch_mom import StochMomStrategy, StochMomConfig
        >>> config = StochMomConfig(k_period=14, sma_period=30)
        >>> strategy = StochMomStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: StochMomConfig | None = None) -> None:
        """Initialize StochMomStrategy.

        Args:
            config: Strategy configuration. Uses defaults if None.
        """
        self._config = config or StochMomConfig()

    @classmethod
    def from_params(cls, **params: Any) -> StochMomStrategy:
        """Create strategy from parameter dict (parameter sweep compatible).

        Args:
            **params: StochMomConfig parameters

        Returns:
            New StochMomStrategy instance
        """
        config = StochMomConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "Stoch-Mom"

    @property
    def required_columns(self) -> list[str]:
        """Required input columns (OHLC for stochastic + ATR)."""
        return ["open", "high", "low", "close"]

    @property
    def config(self) -> StochMomConfig:
        """Strategy configuration."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data preprocessing and indicator calculation.

        Calculated Columns:
            - pct_k: Stochastic %K (0-100)
            - pct_d: Stochastic %D (0-100)
            - sma: Simple Moving Average
            - atr: Average True Range
            - returns: Log returns
            - realized_vol: Realized volatility (annualized)
            - vol_scalar: Volatility scalar
            - vol_ratio: ATR/close ratio

        Args:
            df: OHLCV DataFrame (DatetimeIndex required)

        Returns:
            DataFrame with indicators added
        """
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        """Generate trading signals.

        Args:
            df: Preprocessed DataFrame (preprocess() output)

        Returns:
            StrategySignals NamedTuple
        """
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Recommended PortfolioManagerConfig for Stoch-Mom strategy.

        Stochastic crossover strategies work well with:
            - Moderate leverage (2.0x)
            - 10% system stop loss
            - 5% rebalance threshold

        Returns:
            Keyword args dict for PortfolioManagerConfig
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.05,
        }

    def get_startup_info(self) -> dict[str, str]:
        """Key parameters for CLI startup panel.

        Returns:
            Parameter name-value dict
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "k_period": str(cfg.k_period),
            "d_period": str(cfg.d_period),
            "sma_period": str(cfg.sma_period),
            "vol_target": f"{cfg.vol_target:.0%}",
            "vol_ratio": f"[{cfg.min_vol_ratio:.2f}, {cfg.max_vol_ratio:.2f}]",
            "mode": mode_str,
        }

    def warmup_periods(self) -> int:
        """Required warmup period (number of candles).

        Returns:
            Minimum number of candles needed for strategy calculation
        """
        return self._config.warmup_periods()

    @classmethod
    def conservative(cls) -> StochMomStrategy:
        """Conservative preset instance.

        Returns:
            StochMomStrategy with conservative parameters
        """
        return cls(StochMomConfig.conservative())

    @classmethod
    def aggressive(cls) -> StochMomStrategy:
        """Aggressive preset instance.

        Returns:
            StochMomStrategy with aggressive parameters
        """
        return cls(StochMomConfig.aggressive())

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> StochMomStrategy:
        """Create strategy optimized for a specific timeframe.

        Args:
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            **kwargs: Additional setting overrides

        Returns:
            StochMomStrategy optimized for the timeframe
        """
        config = StochMomConfig.for_timeframe(timeframe, **kwargs)
        return cls(config)
