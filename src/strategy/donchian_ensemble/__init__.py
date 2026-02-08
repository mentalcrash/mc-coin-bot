"""Donchian Ensemble Strategy.

9개 lookback의 Donchian 신호를 평균내는 앙상블 전략.

Example:
    >>> from src.strategy.donchian_ensemble import DonchianEnsembleStrategy, DonchianEnsembleConfig
    >>> strategy = DonchianEnsembleStrategy()
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig, ShortMode
from src.strategy.donchian_ensemble.strategy import DonchianEnsembleStrategy

__all__ = [
    "DonchianEnsembleConfig",
    "DonchianEnsembleStrategy",
    "ShortMode",
]
