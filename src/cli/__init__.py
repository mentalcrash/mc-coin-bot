"""CLI interface using Typer.

Available subcommands:
    - ingest: Data ingestion pipeline (Bronze -> Silver)
    - backtest: Strategy backtesting and optimization
"""

from src.cli.backtest import app as backtest_app
from src.cli.ingest import app as ingest_app

__all__ = ["backtest_app", "ingest_app"]
