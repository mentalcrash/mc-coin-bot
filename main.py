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

from src.cli.ingest import app as ingest_app

# Main Typer Application
app = typer.Typer(
    name="mc-coin-bot",
    help="MC Coin Bot - 2026 Crypto Quant Trading System",
    no_args_is_help=True,
)

# Register sub-applications
app.add_typer(ingest_app, name="ingest", help="Data ingestion pipeline (Bronze/Silver)")


if __name__ == "__main__":
    app()
