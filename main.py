"""MC Coin Bot - Entry Point.

This module provides the main entry point for the trading bot.
Currently supports the data ingestion CLI.

Usage:
    python main.py ingest bronze BTC/USDT --year 2025
    python main.py ingest silver BTC/USDT --year 2025
    python main.py ingest pipeline BTC/USDT --year 2025
    python main.py ingest info
"""

import typer

from src.cli.backtest import app as backtest_app
from src.cli.eda import app as eda_app
from src.cli.ingest import app as ingest_app
from src.cli.pipeline import app as pipeline_app

# Main Typer Application
app = typer.Typer(
    name="mc-coin-bot",
    help="MC Coin Bot - 2026 Crypto Quant Trading System",
    no_args_is_help=True,
)

# Register sub-applications
app.add_typer(ingest_app, name="ingest", help="Data ingestion pipeline (Bronze/Silver)")
app.add_typer(backtest_app, name="backtest", help="Strategy backtesting (VectorBT)")
app.add_typer(eda_app, name="eda", help="EDA (Event-Driven Architecture) backtesting")
app.add_typer(pipeline_app, name="pipeline", help="Strategy pipeline management (YAML)")


if __name__ == "__main__":
    app()
