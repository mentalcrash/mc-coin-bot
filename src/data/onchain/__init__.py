"""On-chain data ingestion module (DeFiLlama, CoinMetrics, mempool.space, etc.).

Exports:
    - AsyncOnchainClient: Rate-limited HTTP client for on-chain APIs
    - OnchainFetcher: DeFiLlama/CoinMetrics/Blockchain.com/Etherscan/mempool.space fetcher
    - OnchainBronzeStorage: Bronze layer storage for on-chain data
    - OnchainSilverProcessor: Silver layer processor for on-chain data
    - Models: Stablecoin, TVL, DEX, CoinMetrics, MempoolMining records
"""

from src.data.onchain.client import AsyncOnchainClient
from src.data.onchain.fetcher import (
    BC_CHARTS,
    BLOCKCHAIN_API_URL,
    CM_ASSETS,
    CM_METRICS,
    CM_RENAME_MAP,
    COINMETRICS_BASE_URL,
    DEFILLAMA_API_URL,
    ETHERSCAN_API_URL,
    FEAR_GREED_URL,
    MEMPOOL_API_URL,
    WEI_PER_ETH,
    OnchainFetcher,
)
from src.data.onchain.models import (
    BlockchainChartRecord,
    CoinMetricsRecord,
    DexVolumeRecord,
    EthSupplyRecord,
    FearGreedRecord,
    MempoolMiningRecord,
    OnchainBatch,
    StablecoinChainRecord,
    StablecoinIndividualRecord,
    StablecoinSupplyRecord,
    TvlRecord,
)
from src.data.onchain.service import ONCHAIN_BATCH_DEFINITIONS, OnchainDataService
from src.data.onchain.storage import OnchainBronzeStorage, OnchainSilverProcessor

__all__ = [
    "BC_CHARTS",
    "BLOCKCHAIN_API_URL",
    "CM_ASSETS",
    "CM_METRICS",
    "CM_RENAME_MAP",
    "COINMETRICS_BASE_URL",
    "DEFILLAMA_API_URL",
    "ETHERSCAN_API_URL",
    "FEAR_GREED_URL",
    "MEMPOOL_API_URL",
    "ONCHAIN_BATCH_DEFINITIONS",
    "WEI_PER_ETH",
    "AsyncOnchainClient",
    "BlockchainChartRecord",
    "CoinMetricsRecord",
    "DexVolumeRecord",
    "EthSupplyRecord",
    "FearGreedRecord",
    "MempoolMiningRecord",
    "OnchainBatch",
    "OnchainBronzeStorage",
    "OnchainDataService",
    "OnchainFetcher",
    "OnchainSilverProcessor",
    "StablecoinChainRecord",
    "StablecoinIndividualRecord",
    "StablecoinSupplyRecord",
    "TvlRecord",
]
