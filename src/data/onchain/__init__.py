"""On-chain data ingestion module (DeFiLlama, CoinMetrics, etc.).

Exports:
    - AsyncOnchainClient: Rate-limited HTTP client for on-chain APIs
    - OnchainFetcher: DeFiLlama stablecoin/TVL/DEX + Coin Metrics fetcher
    - OnchainBronzeStorage: Bronze layer storage for on-chain data
    - OnchainSilverProcessor: Silver layer processor for on-chain data
    - Models: Stablecoin, TVL, DEX, CoinMetrics records
"""

from src.data.onchain.client import AsyncOnchainClient
from src.data.onchain.fetcher import (
    CM_ASSETS,
    CM_METRICS,
    COINMETRICS_BASE_URL,
    DEFILLAMA_API_URL,
    OnchainFetcher,
)
from src.data.onchain.models import (
    CoinMetricsRecord,
    DexVolumeRecord,
    OnchainBatch,
    StablecoinChainRecord,
    StablecoinIndividualRecord,
    StablecoinSupplyRecord,
    TvlRecord,
)
from src.data.onchain.storage import OnchainBronzeStorage, OnchainSilverProcessor

__all__ = [
    "CM_ASSETS",
    "CM_METRICS",
    "COINMETRICS_BASE_URL",
    "DEFILLAMA_API_URL",
    "AsyncOnchainClient",
    "CoinMetricsRecord",
    "DexVolumeRecord",
    "OnchainBatch",
    "OnchainBronzeStorage",
    "OnchainFetcher",
    "OnchainSilverProcessor",
    "StablecoinChainRecord",
    "StablecoinIndividualRecord",
    "StablecoinSupplyRecord",
    "TvlRecord",
]
