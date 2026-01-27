"""Data collection and storage module for cryptocurrency market data."""

from src.data.binance_client import BinanceClient
from src.data.collector import collect_symbols, collect_top_symbols, run_collection
from src.data.schemas import CandleRecord, RawBinanceKline, TickerInfo
from src.data.storage import (
    fill_missing_candles,
    list_available_data,
    load_candles_as_records,
    load_klines_from_parquet,
    save_klines_to_parquet,
    validate_candle_data,
)

__all__ = [
    # Schemas
    "RawBinanceKline",
    "CandleRecord",
    "TickerInfo",
    # Client
    "BinanceClient",
    # Collector
    "collect_symbols",
    "collect_top_symbols",
    "run_collection",
    # Storage
    "save_klines_to_parquet",
    "load_klines_from_parquet",
    "load_candles_as_records",
    "fill_missing_candles",
    "validate_candle_data",
    "list_available_data",
]
