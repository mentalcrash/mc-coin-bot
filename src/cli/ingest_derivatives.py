"""CLI commands for derivatives data ingestion.

Funding Rate, Open Interest, LS Ratio, Taker Ratio 데이터를
Bronze/Silver 파이프라인으로 수집합니다.

Commands:
    - bronze: Fetch raw derivatives data
    - silver: Process Bronze → Silver (forward-fill)
    - pipeline: Full Bronze → Silver pipeline
    - info: Display derivatives data info

Rules Applied:
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.derivatives_fetcher import DerivativesFetcher
from src.data.derivatives_storage import DerivativesBronzeStorage, DerivativesSilverProcessor
from src.exchange.binance_futures_client import BinanceFuturesClient

console = Console()

app = typer.Typer(
    name="derivatives",
    help="Derivatives data ingestion (Funding Rate, OI, LS Ratio, Taker Ratio)",
    no_args_is_help=True,
)


async def _fetch_bronze_deriv(symbol: str, years: list[int]) -> None:
    """Bronze 파생상품 데이터 수집."""
    settings = get_settings()
    settings.ensure_directories()

    async with BinanceFuturesClient(settings) as client:
        fetcher = DerivativesFetcher(client=client, settings=settings)
        bronze = DerivativesBronzeStorage(settings)

        for year in years:
            console.print(f"\n[bold blue]Fetching derivatives {symbol} {year}...[/bold blue]")
            batch = await fetcher.fetch_year(symbol, year)

            if batch.is_empty:
                console.print(f"[yellow]No derivatives data for {symbol} {year}[/yellow]")
                continue

            path = bronze.save(batch, year)
            body = (
                f"[green]✓[/green] Saved {batch.total_records:,} records to:\n{path}\n"
                f"  FR={len(batch.funding_rates)}, OI={len(batch.open_interest)}, "
                f"LS={len(batch.long_short_ratios)}, Taker={len(batch.taker_ratios)}"
            )
            console.print(
                Panel(
                    body,
                    title=f"Bronze Derivatives: {symbol} {year}",
                    border_style="green",
                )
            )


@app.command()
def bronze(
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to fetch"),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Fetch raw derivatives data and save to Bronze layer.

    Example:
        uv run mcbot ingest derivatives bronze BTC/USDT --year 2024
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    console.print(
        Panel.fit(
            f"[bold]Derivatives Bronze Ingestion[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}",
            border_style="blue",
        )
    )

    try:
        asyncio.run(_fetch_bronze_deriv(symbol, year))
        console.print("\n[bold green]✓ Derivatives bronze ingestion completed![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def silver(
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to process"),
    ],
    skip_validation: Annotated[
        bool,
        typer.Option("--skip-validation", help="Skip data validation"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Process Bronze derivatives data with forward-fill to Silver layer.

    Example:
        uv run mcbot ingest derivatives silver BTC/USDT --year 2024
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    processor = DerivativesSilverProcessor(settings)

    for y in year:
        console.print(f"\n[bold yellow]Processing derivatives {symbol} {y}...[/bold yellow]")
        try:
            report = processor.analyze_gaps(symbol, y)
            path = processor.process(symbol, y, validate=not skip_validation)

            table = Table(title=f"Silver Derivatives: {symbol} {y}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Total Rows", f"{report.total_rows:,}")
            for col, count in report.nan_counts.items():
                if count > 0:
                    table.add_row(f"NaN: {col}", str(count))
            table.add_row("Output Path", str(path))
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1) from e

    console.print("\n[bold green]✓ Derivatives silver processing completed![/bold green]")


@app.command()
def pipeline(
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to process"),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Run full pipeline: Bronze (fetch) → Silver (forward-fill).

    Example:
        uv run mcbot ingest derivatives pipeline BTC/USDT --year 2024 --year 2025
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    console.print(
        Panel.fit(
            f"[bold]Derivatives Pipeline[/bold]\nSymbol: {symbol}\nYears: {', '.join(map(str, year))}",
            border_style="magenta",
        )
    )

    # Step 1: Bronze
    console.print("\n[bold blue]Step 1/2: Bronze (Fetching)[/bold blue]")
    try:
        asyncio.run(_fetch_bronze_deriv(symbol, year))
    except Exception as e:
        console.print(f"[bold red]Pipeline failed at Bronze step:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Step 2: Silver
    console.print("\n[bold yellow]Step 2/2: Silver (Forward-fill)[/bold yellow]")
    processor = DerivativesSilverProcessor(settings)
    for y in year:
        try:
            path = processor.process(symbol, y)
            console.print(f"[green]✓[/green] Silver derivatives saved: {path}")
        except Exception as e:
            console.print(f"[bold red]Pipeline failed at Silver step:[/bold red] {e}")
            raise typer.Exit(code=1) from e

    console.print(
        Panel(
            "[bold green]✓ Derivatives pipeline completed![/bold green]",
            border_style="green",
        )
    )


@app.command()
def info(
    symbol: Annotated[str, typer.Argument(help="Trading symbol (e.g., BTC/USDT)")],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to inspect"),
    ],
) -> None:
    """Display derivatives data info for given symbol/year.

    Example:
        uv run mcbot ingest derivatives info BTC/USDT --year 2024
    """
    settings = get_settings()
    bronze = DerivativesBronzeStorage(settings)
    silver_proc = DerivativesSilverProcessor(settings)

    for y in year:
        table = Table(title=f"Derivatives Info: {symbol} {y}")
        table.add_column("Layer", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        # Bronze
        bronze_info = bronze.get_info(symbol, y)
        if bronze_info:
            size_mb = bronze_info["size_bytes"] / 1024 / 1024
            table.add_row("Bronze", "✓ Exists", f"{size_mb:.2f} MB")
        else:
            table.add_row("Bronze", "✗ Missing", "-")

        # Silver
        if silver_proc.exists(symbol, y):
            df = silver_proc.load(symbol, y)
            table.add_row("Silver", "✓ Exists", f"{len(df):,} rows")
        else:
            table.add_row("Silver", "✗ Missing", "-")

        console.print(table)
