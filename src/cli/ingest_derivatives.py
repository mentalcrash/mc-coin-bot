"""CLI commands for derivatives data ingestion.

Funding Rate, Open Interest, LS Ratio, Taker Ratio 데이터를
Bronze/Silver 파이프라인으로 수집합니다.

Commands:
    - bronze: Fetch raw derivatives data
    - silver: Process Bronze → Silver (forward-fill)
    - pipeline: Full Bronze → Silver pipeline
    - batch: Batch download multiple symbols/years
    - info: Display derivatives data info

Rules Applied:
    - #18 Typer CLI: Annotated syntax, Rich UI, async handling
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.derivatives_fetcher import DerivativesFetcher
from src.data.derivatives_storage import DerivativesBronzeStorage, DerivativesSilverProcessor
from src.exchange.binance_futures_client import BinanceFuturesClient

console = Console()
_MAX_DISPLAY_ROWS = 20

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


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------


@dataclass
class _BatchResult:
    """Batch download 결과 추적."""

    total: int = 0
    success: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


async def _batch_download(
    symbols: list[str],
    years: list[int],
    skip_existing: bool,
    progress: Progress,
) -> _BatchResult:
    """멀티 심볼/연도 derivatives 배치 다운로드.

    BinanceFuturesClient를 전체 배치에서 1번만 생성하여 재사용합니다.
    """
    settings = get_settings()
    settings.ensure_directories()

    result = _BatchResult(total=len(symbols) * len(years))

    task_id = progress.add_task(
        "[cyan]Derivatives Batch Progress[/cyan]",
        total=result.total,
    )

    async with BinanceFuturesClient(settings) as client:
        fetcher = DerivativesFetcher(client=client, settings=settings)
        bronze_storage = DerivativesBronzeStorage(settings)
        silver_proc = DerivativesSilverProcessor(settings)

        for idx, symbol in enumerate(symbols):
            for year in years:
                task_key = f"{symbol}_{year}"

                # skip-existing: Bronze 파일 존재 시 스킵
                if skip_existing and bronze_storage.exists(symbol, year):
                    result.skipped.append(task_key)
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[yellow]Skipped[/yellow] {symbol} {year}",
                    )
                    continue

                try:
                    # Bronze: 데이터 수집
                    progress.update(
                        task_id,
                        description=f"[blue]Fetching[/blue] {symbol} {year}",
                    )
                    batch = await fetcher.fetch_year(symbol, year)

                    if batch.is_empty:
                        result.skipped.append(task_key)
                        progress.update(
                            task_id,
                            advance=1,
                            description=f"[yellow]Empty[/yellow] {symbol} {year}",
                        )
                        continue

                    bronze_storage.save(batch, year)

                    # Silver: forward-fill
                    progress.update(
                        task_id,
                        description=f"[yellow]Processing[/yellow] {symbol} {year}",
                    )
                    silver_proc.process(symbol, year, validate=True)

                    result.success.append(task_key)
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[green]Done[/green] {symbol} {year}",
                    )

                    await asyncio.sleep(0.5)

                except Exception as e:
                    result.failed.append((task_key, str(e)))
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[red]Failed[/red] {symbol} {year}: {e}",
                    )
                    await asyncio.sleep(2.0)

            # 심볼 간 추가 대기
            if idx < len(symbols) - 1:
                await asyncio.sleep(1.0)

    return result


def _display_batch_result(result: _BatchResult) -> None:
    """배치 다운로드 결과 리포트 출력."""
    summary = Table(title="Derivatives Batch Summary", show_header=True)
    summary.add_column("Status", style="bold")
    summary.add_column("Count", justify="right")
    summary.add_column("Percentage", justify="right")

    total = result.total
    for label, count, style in [
        ("Success", len(result.success), "green"),
        ("Failed", len(result.failed), "red"),
        ("Skipped", len(result.skipped), "yellow"),
    ]:
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
        summary.add_row(f"[{style}]{label}[/{style}]", str(count), pct)

    summary.add_row("[bold]Total[/bold]", str(total), "100%", style="bold")
    console.print(summary)

    if result.failed:
        fail_table = Table(title="Failed Tasks", show_header=True)
        fail_table.add_column("Task", style="cyan")
        fail_table.add_column("Error", style="red")

        for task_key, error in result.failed[:_MAX_DISPLAY_ROWS]:
            fail_table.add_row(task_key, error[:80])

        if len(result.failed) > _MAX_DISPLAY_ROWS:
            fail_table.add_row("...", f"and {len(result.failed) - _MAX_DISPLAY_ROWS} more")

        console.print(fail_table)


@app.command()
def batch(
    symbols: Annotated[
        str,
        typer.Option(
            "--symbols",
            "-s",
            help="Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)",
        ),
    ] = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,DOGE/USDT,LINK/USDT,ADA/USDT,AVAX/USDT",
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s) to fetch (can specify multiple)"),
    ] = [2020, 2021, 2022, 2023, 2024, 2025],  # noqa: B006
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help="Skip if Bronze file already exists",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Only show target tasks without downloading"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Batch download derivatives data for multiple symbols and years.

    Runs full Bronze -> Silver pipeline for each symbol/year combination.
    Default symbols: 8 Tier-1/2 assets. Default years: 2020-2025.

    Example:
        # Download all 8 assets for 2020-2025 (skip existing)
        uv run mcbot ingest derivatives batch

        # Specific symbols and years
        uv run mcbot ingest derivatives batch -s BTC/USDT,ETH/USDT -y 2024 -y 2025

        # Preview without downloading
        uv run mcbot ingest derivatives batch --dry-run

        # Re-download (don't skip existing)
        uv run mcbot ingest derivatives batch --no-skip-existing
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    total_tasks = len(symbol_list) * len(year)

    # Header
    sym_str = ", ".join(symbol_list)
    yr_str = ", ".join(map(str, year))
    header_body = f"[bold]Derivatives Batch Download[/bold]\nSymbols: {sym_str} ({len(symbol_list)})\nYears: {yr_str}\nTotal tasks: {total_tasks}\nSkip existing: {skip_existing}"
    console.print(
        Panel.fit(
            header_body,
            border_style="magenta",
        )
    )

    # Symbol/Year matrix table
    matrix = Table(title="Target Matrix")
    matrix.add_column("Symbol", style="cyan")
    for y in year:
        matrix.add_column(str(y), justify="center")

    for sym in symbol_list:
        row: list[str] = [sym, *["[green]●[/green]" for _ in year]]
        matrix.add_row(*row)

    console.print(matrix)

    # Dry-run
    if dry_run:
        est_minutes = total_tasks * 8
        dry_body = f"[yellow]Dry-run mode[/yellow]\n\nTotal tasks: {total_tasks} ({len(symbol_list)} symbols x {len(year)} years)\nEstimated time: ~{est_minutes // 60}h {est_minutes % 60}m\n\nRun without --dry-run to start downloading."
        console.print(
            Panel(
                dry_body,
                border_style="yellow",
            )
        )
        return

    # Execute batch
    console.print(f"\n[bold cyan]Downloading {total_tasks} symbol-year combinations...[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        result = asyncio.run(
            _batch_download(
                symbols=symbol_list,
                years=year,
                skip_existing=skip_existing,
                progress=progress,
            )
        )

    # Results
    console.print("\n[bold cyan]Results[/bold cyan]")
    _display_batch_result(result)

    if len(result.failed) == 0:
        console.print(
            Panel(
                "[bold green]✓ Derivatives batch download completed![/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]⚠ Batch completed with {len(result.failed)} failures[/bold yellow]",
                border_style="yellow",
            )
        )
