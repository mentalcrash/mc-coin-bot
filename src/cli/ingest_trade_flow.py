"""CLI commands for Trade Flow (aggTrades) data ingestion.

Binance Futures aggTrades → 12H bar-level trade flow 피처 Silver 생성.
Bronze 생략 (raw aggTrades TB 단위 → 인메모리 처리).

Commands:
    - pipeline: 단일 심볼 + 연도 파이프라인
    - info: Silver 데이터 인벤토리
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

console = Console()

app = typer.Typer(
    name="trade-flow",
    help="Trade flow ingestion (Binance aggTrades → 12H bar features)",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# pipeline command
# ---------------------------------------------------------------------------


async def _run_pipeline(symbols: list[str], years: list[int]) -> None:
    """심볼 x 연도 파이프라인 실행."""
    from src.data.trade_flow.ingester import AggTradesIngester

    settings = get_settings()
    settings.ensure_directories()
    ingester = AggTradesIngester(settings)

    for symbol in symbols:
        for year in years:
            console.print(f"\n[blue]Processing[/blue] {symbol} {year}...")
            try:
                path = await ingester.ingest(symbol, year)
                console.print(f"[green]Silver saved:[/green] {path}")
            except ValueError as e:
                console.print(f"[yellow]Skipped:[/yellow] {symbol} {year} — {e}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {symbol} {year} — {e}")
                raise


@app.command()
def pipeline(
    symbols: Annotated[
        list[str],
        typer.Argument(help="Trading symbol(s) (e.g., BTC/USDT ETH/USDT)"),
    ],
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to process"),
    ] = [2024, 2025],  # noqa: B006
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Download aggTrades from data.binance.vision → 12H trade flow Silver.

    Bronze 생략: raw aggTrades는 인메모리 처리 후 Silver에 12H 집계만 저장.

    Example:
        uv run mcbot ingest trade-flow pipeline BTC/USDT --year 2024
        uv run mcbot ingest trade-flow pipeline BTC/USDT ETH/USDT -y 2024 -y 2025
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    console.print(
        Panel.fit(
            f"[bold]Trade Flow Pipeline[/bold]\nSymbols: {', '.join(symbols)}\nYears: {', '.join(map(str, year))}\nSource: data.binance.vision (aggTrades)",
            border_style="magenta",
        )
    )

    try:
        asyncio.run(_run_pipeline(symbols, year))
        console.print("\n[bold green]Trade flow pipeline completed![/bold green]")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------

# 주요 거래 심볼 목록
_DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "BNB/USDT",
]

_DEFAULT_YEARS = range(2020, 2027)


@app.command()
def info() -> None:
    """Display trade flow Silver data inventory.

    Example:
        uv run mcbot ingest trade-flow info
    """
    settings = get_settings()

    table = Table(title="Trade Flow Silver Inventory")
    table.add_column("Symbol", style="cyan")
    table.add_column("Year", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Path")

    found = 0
    for symbol in _DEFAULT_SYMBOLS:
        for year in _DEFAULT_YEARS:
            path = settings.get_trade_flow_silver_path(symbol, year)
            if path.exists():
                table.add_row(symbol, str(year), "[green]Exists[/green]", str(path))
                found += 1

    if found == 0:
        console.print("[yellow]No trade flow Silver data found.[/yellow]")
        console.print(
            "Run [bold]uv run mcbot ingest trade-flow pipeline BTC/USDT -y 2024[/bold] to get started."
        )
    else:
        console.print(table)
        console.print(f"\nTotal: {found} Silver files")
