"""Typer CLI for EDA (Event-Driven Architecture) backtesting.

Commands:
    - run: EDA 백테스트 실행
    - parity: VBT vs EDA 결과 비교

Rules Applied:
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
    - #15 Logging Standards: Loguru structured logging
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config.settings import get_settings
from src.core.exceptions import DataNotFoundError
from src.core.logger import setup_logger
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.eda.runner import EDARunner
from src.portfolio.config import PortfolioManagerConfig
from src.strategy import get_strategy

console = Console()
app = typer.Typer(help="EDA (Event-Driven Architecture) backtesting")


@app.command()
def run(
    strategy_name: Annotated[str, typer.Argument(help="Strategy name (e.g., tsmom)")],
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    start: Annotated[str, typer.Option(help="Start date (YYYY-MM-DD)")] = "2024-01-01",
    end: Annotated[str, typer.Option(help="End date (YYYY-MM-DD)")] = "2025-12-31",
    capital: Annotated[float, typer.Option(help="Initial capital (USD)")] = 10000.0,
    leverage: Annotated[float, typer.Option(help="Max leverage cap")] = 2.0,
    rebalance: Annotated[float, typer.Option(help="Rebalance threshold")] = 0.05,
    timeframe: Annotated[str, typer.Option(help="Timeframe")] = "1d",
) -> None:
    """Run EDA backtest for a single symbol."""
    setup_logger()
    settings = get_settings()

    try:
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
    except (KeyError, ValueError, TypeError) as e:
        console.print(f"[red]Strategy not found: {e}[/red]")
        raise typer.Exit(code=1) from e

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

    # Load data
    try:
        service = MarketDataService(settings)
        request = MarketDataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
        )
        data = service.get(request)
    except DataNotFoundError as e:
        console.print(f"[red]Data not found: {e}[/red]")
        raise typer.Exit(code=1) from e

    config = PortfolioManagerConfig(
        max_leverage_cap=leverage,
        rebalance_threshold=rebalance,
    )

    runner = EDARunner(
        strategy=strategy,
        data=data,
        config=config,
        initial_capital=capital,
    )

    logger.info("Running EDA backtest: {} {} {}-{}", strategy_name, symbol, start, end)
    metrics = asyncio.run(runner.run())

    # Display results
    table = Table(title=f"EDA Backtest: {strategy_name} / {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Return", f"{metrics.total_return:.2f}%")
    table.add_row("CAGR", f"{metrics.cagr:.2f}%")
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.4f}")
    table.add_row("Max Drawdown", f"{metrics.max_drawdown:.2f}%")
    table.add_row("Win Rate", f"{metrics.win_rate:.1f}%")
    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row("Winning Trades", str(metrics.winning_trades))
    table.add_row("Losing Trades", str(metrics.losing_trades))

    if metrics.volatility is not None:
        table.add_row("Volatility", f"{metrics.volatility:.2f}%")
    if metrics.profit_factor is not None:
        table.add_row("Profit Factor", f"{metrics.profit_factor:.4f}")

    console.print(table)
