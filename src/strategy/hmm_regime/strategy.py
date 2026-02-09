"""HMM Regime Strategy Implementation.

GaussianHMM 3-state (Bull/Bear/Sideways) regime classification 기반 전략.
Expanding window training으로 look-ahead bias를 방지합니다.

Rules Applied:
    - #01 Project Structure: Strategy Layer responsibility
    - #12 Data Engineering: Vectorization (HMM training loop은 ML 예외)
    - #26 VectorBT Standards: Compatible output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.hmm_regime.config import HMMRegimeConfig, ShortMode
from src.strategy.hmm_regime.preprocessor import preprocess
from src.strategy.hmm_regime.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("hmm-regime")
class HMMRegimeStrategy(BaseStrategy):
    """HMM Regime Strategy.

    GaussianHMM을 사용하여 시장을 Bull/Bear/Sideways 3가지 regime으로 분류하고,
    regime에 따라 포지션 방향과 강도를 결정하는 전략입니다.

    Key Features:
        - GaussianHMM 3-state regime classification
        - Expanding window training (look-ahead bias 방지)
        - Regime probability 기반 포지션 강도 조절
        - Volatility targeting으로 리스크 관리

    Attributes:
        _config: HMM Regime 설정 (HMMRegimeConfig)

    Example:
        >>> from src.strategy.hmm_regime import HMMRegimeStrategy, HMMRegimeConfig
        >>> config = HMMRegimeConfig(n_states=3, min_train_window=252)
        >>> strategy = HMMRegimeStrategy(config)
        >>> processed_df, signals = strategy.run(ohlcv_df)
    """

    def __init__(self, config: HMMRegimeConfig | None = None) -> None:
        """HMMRegimeStrategy initialization.

        Args:
            config: HMM Regime configuration. Uses defaults if None.
        """
        self._config = config or HMMRegimeConfig()

    @classmethod
    def from_params(cls, **params: Any) -> HMMRegimeStrategy:
        """Create HMMRegimeStrategy from parameters.

        Args:
            **params: HMMRegimeConfig creation parameters

        Returns:
            HMMRegimeStrategy instance
        """
        config = HMMRegimeConfig(**params)
        return cls(config)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "HMM-Regime"

    @property
    def required_columns(self) -> list[str]:
        """Required columns list."""
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> HMMRegimeConfig:
        """Strategy configuration."""
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data preprocessing and indicator calculation.

        OHLCV 데이터에 HMM Regime 지표를 계산하여 추가합니다.

        Calculated Columns:
            - returns: Log/simple returns
            - rolling_vol: Rolling volatility
            - regime: HMM regime label (-1, 0, 1)
            - regime_prob: Regime posterior probability
            - realized_vol: Annualized realized volatility
            - vol_scalar: Volatility scalar
            - atr: Average True Range
            - drawdown: Rolling max drawdown

        Args:
            df: OHLCV DataFrame (DatetimeIndex required)

        Returns:
            DataFrame with calculated indicators
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

    def warmup_periods(self) -> int:
        """Required warmup period (number of candles).

        Returns:
            Minimum candles needed for strategy calculation
        """
        return self._config.warmup_periods()

    @classmethod
    def recommended_config(cls) -> dict[str, object]:
        """Recommended PortfolioManagerConfig settings for HMM Regime strategy.

        - Leverage 2.0x for conservative operation
        - 10% system stop loss for large loss prevention
        - 10% rebalance threshold for regime transition response

        Returns:
            Keyword arguments dict for PortfolioManagerConfig creation
        """
        return {
            "max_leverage_cap": 2.0,
            "system_stop_loss": 0.10,
            "rebalance_threshold": 0.10,
        }

    def get_startup_info(self) -> dict[str, str]:
        """Key parameters for CLI startup panel.

        Returns:
            Parameter name-value dict
        """
        cfg = self._config

        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        mode_str = mode_map.get(cfg.short_mode, "Unknown")

        return {
            "n_states": str(cfg.n_states),
            "min_train_window": f"{cfg.min_train_window}d",
            "retrain_interval": f"{cfg.retrain_interval}d",
            "n_iter": str(cfg.n_iter),
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_str,
        }
