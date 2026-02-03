"""Data ingestion pipeline with Medallion Architecture (Bronze/Silver).

Exports:
    - BronzeStorage: Raw data storage
    - DataFetcher: Async data fetching from exchange
    - SilverProcessor: Gap-filled data processing
    - MarketDataService: Data access service (Repository Pattern)
    - MarketDataRequest: Data request DTO
    - MarketDataSet: Data response DTO with metadata
"""

from src.data.bronze import BronzeStorage
from src.data.fetcher import DataFetcher
from src.data.market_data import MarketDataRequest, MarketDataSet
from src.data.service import MarketDataService
from src.data.silver import SilverProcessor

__all__ = [
    "BronzeStorage",
    "DataFetcher",
    "MarketDataRequest",
    "MarketDataService",
    "MarketDataSet",
    "SilverProcessor",
]
