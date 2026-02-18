"""Deribit Options data ingestion module.

Exports:
    - AsyncOptionsClient: Rate-limited HTTP client for Deribit Public API
    - OptionsFetcher: Deribit data fetcher (5 datasets)
    - OptionsBronzeStorage: Bronze layer storage for options data
    - OptionsSilverProcessor: Silver layer processor for options data
    - OptionsDataService: Options data service (batch/load/enrich/precompute)
    - Models: DVolRecord, PutCallRatioRecord, HistoricalVolRecord,
              TermStructureRecord, MaxPainRecord, OptionsBatch
"""

from src.data.options.client import AsyncOptionsClient
from src.data.options.fetcher import (
    DERIBIT_DATASETS,
    OptionsFetcher,
    route_fetch,
)
from src.data.options.models import (
    DVolRecord,
    HistoricalVolRecord,
    MaxPainRecord,
    OptionsBatch,
    PutCallRatioRecord,
    TermStructureRecord,
)
from src.data.options.service import OPTIONS_BATCH_DEFINITIONS, OptionsDataService
from src.data.options.storage import OptionsBronzeStorage, OptionsSilverProcessor

__all__ = [
    "DERIBIT_DATASETS",
    "OPTIONS_BATCH_DEFINITIONS",
    "AsyncOptionsClient",
    "DVolRecord",
    "HistoricalVolRecord",
    "MaxPainRecord",
    "OptionsBatch",
    "OptionsBronzeStorage",
    "OptionsDataService",
    "OptionsFetcher",
    "OptionsSilverProcessor",
    "PutCallRatioRecord",
    "TermStructureRecord",
    "route_fetch",
]
