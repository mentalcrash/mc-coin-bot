"""Extended Derivatives data ingestion module (Coinalyze + Hyperliquid).

Exports:
    - AsyncCoinalyzeClient: Rate-limited HTTP client for Coinalyze API
    - AsyncHyperliquidClient: Rate-limited HTTP client for Hyperliquid API (POST)
    - CoinalyzeFetcher: Coinalyze data fetcher (8 datasets)
    - HyperliquidFetcher: Hyperliquid data fetcher (2 datasets)
    - DerivExtBronzeStorage: Bronze layer storage
    - DerivExtSilverProcessor: Silver layer processor (liquidation rollup + compound dedup)
    - DerivExtDataService: Data service (batch/load/enrich/precompute)
    - Models: AggOIRecord, AggFundingRecord, LiquidationRecord, CVDRecord,
              HLAssetContextRecord, HLPredictedFundingRecord, DerivExtBatch
"""

from src.data.deriv_ext.client import AsyncCoinalyzeClient, AsyncHyperliquidClient
from src.data.deriv_ext.fetcher import (
    COINALYZE_DATASETS,
    HYPERLIQUID_DATASETS,
    CoinalyzeFetcher,
    HyperliquidFetcher,
    route_fetch,
)
from src.data.deriv_ext.models import (
    AggFundingRecord,
    AggOIRecord,
    CVDRecord,
    DerivExtBatch,
    HLAssetContextRecord,
    HLPredictedFundingRecord,
    LiquidationRecord,
)
from src.data.deriv_ext.service import DERIV_EXT_BATCH_DEFINITIONS, DerivExtDataService
from src.data.deriv_ext.storage import DerivExtBronzeStorage, DerivExtSilverProcessor

__all__ = [
    "COINALYZE_DATASETS",
    "DERIV_EXT_BATCH_DEFINITIONS",
    "HYPERLIQUID_DATASETS",
    "AggFundingRecord",
    "AggOIRecord",
    "AsyncCoinalyzeClient",
    "AsyncHyperliquidClient",
    "CVDRecord",
    "CoinalyzeFetcher",
    "DerivExtBatch",
    "DerivExtBronzeStorage",
    "DerivExtDataService",
    "DerivExtSilverProcessor",
    "HLAssetContextRecord",
    "HLPredictedFundingRecord",
    "HyperliquidFetcher",
    "LiquidationRecord",
    "route_fetch",
]
