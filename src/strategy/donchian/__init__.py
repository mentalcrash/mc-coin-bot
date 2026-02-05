"""Donchian Channel Breakout Strategy.

터틀 트레이딩 기반 채널 돌파 전략입니다.
Entry/Exit Channel을 분리하여 추세 추종합니다.

Example:
    >>> from src.strategy.donchian import DonchianStrategy, DonchianConfig
    >>>
    >>> strategy = DonchianStrategy()  # 20/10 터틀 시스템1
    >>> processed_df, signals = strategy.run(ohlcv_df)
"""

from src.strategy.donchian.config import DonchianConfig, ShortMode
from src.strategy.donchian.strategy import DonchianStrategy

__all__ = [
    "DonchianConfig",
    "DonchianStrategy",
    "ShortMode",
]
