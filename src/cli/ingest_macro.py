"""CLI commands for macro economic data ingestion.

FRED API (7 시리즈) + yfinance (6 ETF) 데이터를
Bronze/Silver 파이프라인으로 수집합니다.

Commands:
    - pipeline: Single dataset Bronze -> Silver pipeline
    - batch: Category-based batch download (fred/yfinance/all)
    - info: Data inventory display

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
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.macro.client import AsyncCoinGeckoClient, AsyncMacroClient
from src.data.macro.fetcher import MacroFetcher, route_fetch
from src.data.macro.service import MacroDataService, get_date_col
from src.data.macro.storage import MacroBronzeStorage, MacroSilverProcessor

console = Console()
_MAX_DISPLAY_ROWS = 20

# CoinGecko snapshot datasets (append mode)
_SNAPSHOT_DATASETS: set[str] = {"global_metrics", "defi_global"}

app = typer.Typer(
    name="macro",
    help="Macro economic data ingestion (FRED 7 series + yfinance 6 ETFs)",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# pipeline command
# ---------------------------------------------------------------------------


async def _fetch_pipeline(source: str, name: str) -> None:
    """단일 데이터셋 Bronze -> Silver 파이프라인."""
    settings = get_settings()
    settings.ensure_directories()

    bronze = MacroBronzeStorage(settings)
    silver = MacroSilverProcessor(settings)

    if source == "fred":
        api_key = settings.fred_api_key.get_secret_value()
        if not api_key:
            console.print("[bold red]Error:[/bold red] FRED_API_KEY is not set in .env")
            raise typer.Exit(code=1)

        async with AsyncMacroClient("fred") as client:
            fetcher = MacroFetcher(client, api_key)
            console.print(f"[blue]Fetching[/blue] {source}/{name}...")
            df = await route_fetch(fetcher, source, name)
    elif source == "yfinance":
        async with AsyncMacroClient("yfinance") as client:
            fetcher = MacroFetcher(client, api_key="")
            console.print(f"[blue]Fetching[/blue] {source}/{name}...")
            df = await route_fetch(fetcher, source, name)
    elif source == "coingecko":
        cg_key = settings.coingecko_api_key.get_secret_value()
        if not cg_key:
            console.print("[bold red]Error:[/bold red] COINGECKO_API_KEY is not set in .env")
            raise typer.Exit(code=1)

        async with (
            AsyncCoinGeckoClient(api_key=cg_key) as cg_client,
            AsyncMacroClient("coingecko") as dummy_client,
        ):
            fetcher = MacroFetcher(dummy_client, api_key="", coingecko_client=cg_client)
            console.print(f"[blue]Fetching[/blue] {source}/{name}...")
            df = await route_fetch(fetcher, source, name)
    else:
        console.print(f"[bold red]Error:[/bold red] Unknown source: {source}")
        raise typer.Exit(code=1)

    if df.empty:
        console.print(f"[yellow]No data returned for {source}/{name}[/yellow]")
        return

    # Bronze save: snapshot datasets use append, history uses save
    if name in _SNAPSHOT_DATASETS:
        path = bronze.append(df, source, name)
        console.print(f"[green]Bronze appended:[/green] {path} ({len(df):,} rows)")
    else:
        path = bronze.save(df, source, name)
        console.print(f"[green]Bronze saved:[/green] {path} ({len(df):,} rows)")

    # Silver process
    date_col = get_date_col(source)
    silver_path = silver.process(source, name, date_col=date_col, sort_col=date_col)
    console.print(f"[green]Silver saved:[/green] {silver_path}")


@app.command()
def pipeline(
    source: Annotated[str, typer.Argument(help="Data source (fred, yfinance)")],
    name: Annotated[str, typer.Argument(help="Data name (dxy, spy, ...)")],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """Run full pipeline: Fetch -> Bronze -> Silver for a single dataset.

    Example:
        uv run mcbot ingest macro pipeline fred dxy
        uv run mcbot ingest macro pipeline yfinance spy
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    console.print(
        Panel.fit(
            f"[bold]Macro Pipeline[/bold]\nSource: {source}\nName: {name}",
            border_style="magenta",
        )
    )

    try:
        asyncio.run(_fetch_pipeline(source, name))
        console.print("\n[bold green]Pipeline completed![/bold green]")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


@dataclass
class _BatchResult:
    """Batch download 결과 추적."""

    total: int = 0
    success: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


async def _batch_download(
    definitions: list[tuple[str, str]],
    progress: Progress,
) -> _BatchResult:
    """Source별로 그룹핑하여 batch download."""
    settings = get_settings()
    settings.ensure_directories()

    result = _BatchResult(total=len(definitions))
    bronze = MacroBronzeStorage(settings)
    silver = MacroSilverProcessor(settings)

    task_id = progress.add_task(
        "[cyan]Macro Batch Progress[/cyan]",
        total=result.total,
    )

    # source별 그룹핑
    groups: dict[str, list[str]] = {}
    for source, name in definitions:
        groups.setdefault(source, []).append(name)

    api_key = settings.fred_api_key.get_secret_value()
    cg_key = settings.coingecko_api_key.get_secret_value()

    for source, names in groups.items():
        if source == "coingecko":
            if not cg_key:
                for name in names:
                    result.failed.append((f"{source}/{name}", "COINGECKO_API_KEY not set"))
                    progress.update(task_id, advance=1)
                continue
            async with (
                AsyncCoinGeckoClient(api_key=cg_key) as cg_client,
                AsyncMacroClient(source) as client,
            ):
                fetcher = MacroFetcher(client, api_key="", coingecko_client=cg_client)
                await _batch_fetch_names(
                    fetcher, source, names, bronze, silver, result, progress, task_id
                )
        else:
            async with AsyncMacroClient(source) as client:
                fetcher = MacroFetcher(client, api_key)
                await _batch_fetch_names(
                    fetcher, source, names, bronze, silver, result, progress, task_id
                )

    return result


async def _batch_fetch_names(
    fetcher: MacroFetcher,
    source: str,
    names: list[str],
    bronze: MacroBronzeStorage,
    silver: MacroSilverProcessor,
    result: _BatchResult,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """source 내 개별 name 순회하여 fetch → bronze → silver."""
    for name in names:
        task_key = f"{source}/{name}"
        try:
            progress.update(
                task_id,
                description=f"[blue]Fetching[/blue] {task_key}",
            )
            df = await route_fetch(fetcher, source, name)

            if df.empty:
                result.skipped.append(task_key)
                progress.update(
                    task_id,
                    advance=1,
                    description=f"[yellow]Empty[/yellow] {task_key}",
                )
                continue

            # Bronze: snapshot datasets use append
            if name in _SNAPSHOT_DATASETS:
                bronze.append(df, source, name)
            else:
                bronze.save(df, source, name)

            # Silver
            progress.update(
                task_id,
                description=f"[yellow]Processing[/yellow] {task_key}",
            )
            date_col = get_date_col(source)
            silver.process(source, name, date_col=date_col, sort_col=date_col)

            result.success.append(task_key)
            progress.update(
                task_id,
                advance=1,
                description=f"[green]Done[/green] {task_key}",
            )

        except Exception as e:
            result.failed.append((task_key, str(e)))
            progress.update(
                task_id,
                advance=1,
                description=f"[red]Failed[/red] {task_key}: {e}",
            )


def _display_batch_result(result: _BatchResult) -> None:
    """배치 결과 summary table 출력."""
    summary = Table(title="Macro Batch Summary", show_header=True)
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
    batch_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Batch type: fred, yfinance, all",
        ),
    ] = "all",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Only show target datasets without fetching"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Batch download macro data by category.

    Example:
        uv run mcbot ingest macro batch --type all
        uv run mcbot ingest macro batch --type fred
        uv run mcbot ingest macro batch --dry-run
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    service = MacroDataService(settings)

    try:
        definitions = service.get_batch_definitions(batch_type)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Header
    console.print(
        Panel.fit(
            f"[bold]Macro Batch Download[/bold]\nType: {batch_type}\nTotal datasets: {len(definitions)}",
            border_style="magenta",
        )
    )

    # Target list
    target_table = Table(title="Target Datasets")
    target_table.add_column("#", style="dim", justify="right")
    target_table.add_column("Source", style="cyan")
    target_table.add_column("Name", style="green")

    for i, (source, name) in enumerate(definitions[:_MAX_DISPLAY_ROWS], 1):
        target_table.add_row(str(i), source, name)

    if len(definitions) > _MAX_DISPLAY_ROWS:
        target_table.add_row("...", "", f"and {len(definitions) - _MAX_DISPLAY_ROWS} more")

    console.print(target_table)

    # Dry-run
    if dry_run:
        console.print(
            Panel(
                f"[yellow]Dry-run mode[/yellow]\n\nTotal datasets: {len(definitions)}\n\nRun without --dry-run to start downloading.",
                border_style="yellow",
            )
        )
        return

    # Execute
    console.print(f"\n[bold cyan]Downloading {len(definitions)} datasets...[/bold cyan]")

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
        result = asyncio.run(_batch_download(definitions, progress))

    # Results
    console.print("\n[bold cyan]Results[/bold cyan]")
    _display_batch_result(result)

    if len(result.failed) == 0:
        console.print(
            Panel(
                "[bold green]Macro batch download completed![/bold green]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]Batch completed with {len(result.failed)} failures[/bold yellow]",
                border_style="yellow",
            )
        )


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


@app.command()
def info(
    batch_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Batch type: fred, yfinance, all",
        ),
    ] = "all",
) -> None:
    """Display macro data inventory.

    Example:
        uv run mcbot ingest macro info
        uv run mcbot ingest macro info --type fred
    """
    settings = get_settings()
    service = MacroDataService(settings)

    try:
        definitions = service.get_batch_definitions(batch_type)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    bronze = MacroBronzeStorage(settings)
    silver = MacroSilverProcessor(settings)

    table = Table(title=f"Macro Data Inventory ({batch_type})")
    table.add_column("Source", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Bronze", justify="center")
    table.add_column("Silver", justify="center")

    for source, name in definitions:
        bronze_status = (
            "[green]Exists[/green]" if bronze.exists(source, name) else "[red]Missing[/red]"
        )
        silver_status = (
            "[green]Exists[/green]" if silver.exists(source, name) else "[red]Missing[/red]"
        )
        table.add_row(source, name, bronze_status, silver_status)

    console.print(table)
