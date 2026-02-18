"""On-chain Accumulation Strategy.

MVRV + Exchange Flow + Stablecoin 3지표 다수결 방식.
BTC/ETH 전용 (CoinMetrics MVRV/Flow scope). Long-only.

Components:
    - OnchainAccumConfig: Pydantic frozen config
    - preprocess: MVRV + flow zscore + stablecoin ROC 계산
    - generate_signals: 2/3 majority vote
    - OnchainAccumStrategy: @register("onchain-accum")
"""

from src.strategy.onchain_accum.config import OnchainAccumConfig
from src.strategy.onchain_accum.preprocessor import preprocess
from src.strategy.onchain_accum.signal import generate_signals
from src.strategy.onchain_accum.strategy import OnchainAccumStrategy

__all__ = [
    "OnchainAccumConfig",
    "OnchainAccumStrategy",
    "generate_signals",
    "preprocess",
]
