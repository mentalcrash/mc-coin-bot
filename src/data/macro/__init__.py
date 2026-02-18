"""Macro economic data ingestion module (FRED + yfinance + CoinGecko).

Exports:
    - AsyncMacroClient: Rate-limited HTTP client for FRED API
    - AsyncCoinGeckoClient: Rate-limited HTTP client for CoinGecko API
    - YFinanceClient: Async wrapper for yfinance downloads
    - MacroFetcher: FRED/yfinance/CoinGecko data fetcher
    - MacroBronzeStorage: Bronze layer storage for macro data
    - MacroSilverProcessor: Silver layer processor for macro data
    - MacroDataService: Macro data service (batch/load/enrich/precompute)
    - Models: FREDObservationRecord, YFinanceRecord, CoinGeckoGlobalRecord, CoinGeckoDefiRecord, MacroBatch
"""

from src.data.macro.client import AsyncCoinGeckoClient, AsyncMacroClient, YFinanceClient
from src.data.macro.fetcher import (
    COINGECKO_DATASETS,
    FRED_SERIES,
    YFINANCE_TICKERS,
    MacroFetcher,
    route_fetch,
)
from src.data.macro.models import (
    CoinGeckoDefiRecord,
    CoinGeckoGlobalRecord,
    FREDObservationRecord,
    MacroBatch,
    YFinanceRecord,
)
from src.data.macro.service import MACRO_BATCH_DEFINITIONS, MacroDataService
from src.data.macro.storage import MacroBronzeStorage, MacroSilverProcessor

__all__ = [
    "COINGECKO_DATASETS",
    "FRED_SERIES",
    "MACRO_BATCH_DEFINITIONS",
    "YFINANCE_TICKERS",
    "AsyncCoinGeckoClient",
    "AsyncMacroClient",
    "CoinGeckoDefiRecord",
    "CoinGeckoGlobalRecord",
    "FREDObservationRecord",
    "MacroBatch",
    "MacroBronzeStorage",
    "MacroDataService",
    "MacroFetcher",
    "MacroSilverProcessor",
    "YFinanceClient",
    "YFinanceRecord",
    "route_fetch",
]
