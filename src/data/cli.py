"""CLI interface for Binance data collection."""

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.data.binance_client import BinanceClient
from src.data.collector import run_collection
from src.data.storage import list_available_data, validate_candle_data

app = typer.Typer(
    name="data",
    help="Binance candle data collection utilities",
)
console = Console()


@app.command()
def collect(
    years: Annotated[
        list[int],
        typer.Option(
            "--years",
            "-y",
            help="Years to collect data for (e.g., -y 2023 -y 2024 -y 2025)",
        ),
    ] = [2023, 2024, 2025],  # noqa: B006
    symbols: Annotated[
        list[str] | None,
        typer.Option(
            "--symbols",
            "-s",
            help="Specific symbols to collect (e.g., -s BTCUSDT -s ETHUSDT)",
        ),
    ] = None,
    top: Annotated[
        int,
        typer.Option(
            "--top",
            "-t",
            help="Number of top symbols by volume to collect (ignored if --symbols is set)",
        ),
    ] = 100,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help="Data directory path (default: data/binance/candles)",
        ),
    ] = None,
    delay: Annotated[
        int,
        typer.Option(
            "--delay",
            help="Delay between API requests in milliseconds",
        ),
    ] = 25,
    no_fill_gaps: Annotated[
        bool,
        typer.Option(
            "--no-fill-gaps",
            help="Do not fill missing candles with previous values",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force re-collection even if file exists",
        ),
    ] = False,
) -> None:
    """Collect candle data from Binance.

    Examples:
        # Collect top 100 symbols for 2023, 2024, 2025
        python -m src.data.cli collect

        # Collect specific symbols
        python -m src.data.cli collect -s BTCUSDT -s ETHUSDT -y 2024

        # Collect top 50 symbols
        python -m src.data.cli collect -t 50 -y 2024 -y 2025
    """
    console.print("[bold blue]Starting data collection...[/bold blue]")

    if symbols:
        console.print(f"Symbols: {symbols}")
    else:
        console.print(f"Top {top} symbols by volume")

    console.print(f"Years: {years}")
    console.print(f"Fill gaps: {not no_fill_gaps}")
    console.print(f"Skip existing: {not force}")

    try:
        results = run_collection(
            symbols=symbols,
            top_n=top if not symbols else None,
            years=years,
            data_dir=data_dir,
            delay_ms=delay,
            fill_gaps=not no_fill_gaps,
            skip_existing=not force,
        )

        # 결과 요약
        total_files = sum(len(files) for files in results.values())
        console.print("\n[bold green]Collection completed![/bold green]")
        console.print(f"Total files: {total_files}")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command()
def status(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help="Data directory path",
        ),
    ] = None,
) -> None:
    """Show status of collected data.

    Example:
        python -m src.data.cli status
    """
    data_list = list_available_data(data_dir)

    if not data_list:
        console.print("[yellow]No data files found.[/yellow]")
        return

    # 심볼별로 그룹화
    symbols: dict[str, list[dict[str, Any]]] = {}
    for item in data_list:
        symbol = item["symbol"]
        if symbol not in symbols:
            symbols[symbol] = []
        symbols[symbol].append(item)

    # 테이블 출력
    table = Table(title="Collected Data Status")
    table.add_column("Symbol", style="cyan")
    table.add_column("Years", style="green")
    table.add_column("Files", justify="right")
    table.add_column("Total Size (MB)", justify="right")

    for symbol, items in sorted(symbols.items()):
        years_str = ", ".join(str(item["year"]) for item in sorted(items, key=lambda x: x["year"]))
        total_size = sum(item["size_mb"] for item in items)
        table.add_row(
            symbol,
            years_str,
            str(len(items)),
            f"{total_size:.2f}",
        )

    console.print(table)
    console.print(f"\nTotal symbols: {len(symbols)}")
    console.print(f"Total files: {len(data_list)}")
    total_size_all = sum(item["size_mb"] for item in data_list)
    console.print(f"Total size: {total_size_all:.2f} MB")


@app.command()
def validate(
    symbol: Annotated[
        str,
        typer.Argument(help="Symbol to validate (e.g., BTCUSDT)"),
    ],
    year: Annotated[
        int,
        typer.Argument(help="Year to validate (e.g., 2024)"),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help="Data directory path",
        ),
    ] = None,
) -> None:
    """Validate candle data integrity.

    Example:
        python -m src.data.cli validate BTCUSDT 2024
    """
    result = validate_candle_data(symbol, year, data_dir)

    table = Table(title=f"Validation Result: {symbol} {year}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("File exists", "Yes" if result["file_exists"] else "No")
    table.add_row("Total candles", str(result["total_candles"]))
    table.add_row("Expected candles", str(result["expected_candles"]))
    table.add_row("Missing candles", str(result["missing_candles"]))
    table.add_row("Duplicate candles", str(result["duplicate_candles"]))
    table.add_row("Data range", result["data_range"] or "N/A")
    table.add_row("File size (MB)", str(result["file_size_mb"]))

    console.print(table)

    # 검증 결과 판정
    if not result["file_exists"]:
        console.print("[red]FAIL: File does not exist[/red]")
    elif result["missing_candles"] > 0:
        console.print(f"[yellow]WARNING: {result['missing_candles']} missing candles[/yellow]")
    elif result["duplicate_candles"] > 0:
        console.print(f"[yellow]WARNING: {result['duplicate_candles']} duplicate candles[/yellow]")
    else:
        console.print("[green]PASS: Data integrity check passed[/green]")


@app.command()
def list_symbols(
    top: Annotated[
        int,
        typer.Option(
            "--top",
            "-t",
            help="Number of top symbols to list",
        ),
    ] = 100,
) -> None:
    """List top symbols by trading volume.

    Example:
        python -m src.data.cli list-symbols -t 50
    """

    async def get_symbols() -> list[str]:
        async with BinanceClient(rate_limit=True) as client:
            return await client.get_top_usdt_tickers(top)

    console.print(f"[bold blue]Fetching top {top} USDT trading pairs...[/bold blue]")

    try:
        symbols = asyncio.run(get_symbols())

        table = Table(title=f"Top {top} USDT Trading Pairs")
        table.add_column("#", style="dim", justify="right")
        table.add_column("Symbol", style="cyan")

        for i, symbol in enumerate(symbols, 1):
            table.add_row(str(i), symbol)

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


def main() -> None:
    """CLI entry point."""
    # 로깅 설정
    logger.remove()
    logger.add(
        lambda msg: console.print(msg, end=""),
        format="<level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    app()


if __name__ == "__main__":
    main()
