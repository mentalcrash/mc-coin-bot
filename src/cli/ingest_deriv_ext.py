"""CLI commands for Coinalyze Extended Derivatives data ingestion.

Aggregated OI, Funding Rate, Liquidation, CVD 데이터를
Bronze/Silver 파이프라인으로 수집합니다.

Commands:
    - pipeline: Single dataset Bronze -> Silver pipeline
    - batch: Batch download (coinalyze/all)
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
from src.data.deriv_ext.client import AsyncCoinalyzeClient, AsyncHyperliquidClient
from src.data.deriv_ext.fetcher import CoinalyzeFetcher, HyperliquidFetcher, route_fetch
from src.data.deriv_ext.service import DerivExtDataService, get_date_col
from src.data.deriv_ext.storage import DerivExtBronzeStorage, DerivExtSilverProcessor

console = Console()
_MAX_DISPLAY_ROWS = 20

# Hyperliquid snapshot datasets (append mode)
_SNAPSHOT_DATASETS: set[str] = {"hl_asset_contexts", "hl_predicted_fundings"}

app = typer.Typer(
    name="deriv-ext",
    help="Extended Derivatives ingestion (Coinalyze: OI, Funding, Liquidation, CVD)",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# pipeline command
# ---------------------------------------------------------------------------


async def _fetch_pipeline(source: str, name: str) -> None:
    """단일 데이터셋 Bronze -> Silver 파이프라인."""
    settings = get_settings()
    settings.ensure_directories()

    bronze = DerivExtBronzeStorage(settings)
    silver = DerivExtSilverProcessor(settings)

    if source == "coinalyze":
        api_key = settings.coinalyze_api_key.get_secret_value()
        if not api_key:
            console.print("[bold red]COINALYZE_API_KEY is not set.[/bold red]")
            raise typer.Exit(code=1)

        async with AsyncCoinalyzeClient(source, api_key=api_key) as client:
            fetcher = CoinalyzeFetcher(client)
            console.print(f"[blue]Fetching[/blue] {source}/{name}...")
            df = await route_fetch(fetcher, source, name)
    elif source == "hyperliquid":
        async with AsyncHyperliquidClient() as client:
            fetcher_hl = HyperliquidFetcher(client)
            console.print(f"[blue]Fetching[/blue] {source}/{name}...")
            df = await route_fetch(fetcher_hl, source, name)
    else:
        console.print(f"[bold red]Error:[/bold red] Unknown source: {source}")
        raise typer.Exit(code=1)

    if df.empty:
        console.print(f"[yellow]No data returned for {source}/{name}[/yellow]")
        return

    # Bronze save: 스냅샷 데이터는 append, 히스토리는 save
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
    source: Annotated[str, typer.Argument(help="Data source (coinalyze)")],
    name: Annotated[str, typer.Argument(help="Data name (btc_agg_oi, btc_agg_funding, ...)")],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """Run full pipeline: Fetch -> Bronze -> Silver for a single dataset.

    Example:
        uv run mcbot ingest deriv-ext pipeline coinalyze btc_agg_oi
        uv run mcbot ingest deriv-ext pipeline coinalyze btc_liquidations
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    console.print(
        Panel.fit(
            f"[bold]Deriv-Ext Pipeline[/bold]\nSource: {source}\nName: {name}",
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

    api_key = settings.coinalyze_api_key.get_secret_value()

    result = _BatchResult(total=len(definitions))
    bronze = DerivExtBronzeStorage(settings)
    silver = DerivExtSilverProcessor(settings)

    task_id = progress.add_task(
        "[cyan]Deriv-Ext Batch Progress[/cyan]",
        total=result.total,
    )

    # source별 그룹핑
    groups: dict[str, list[str]] = {}
    for source, name in definitions:
        groups.setdefault(source, []).append(name)

    for source, names in groups.items():
        if source == "coinalyze":
            async with AsyncCoinalyzeClient(source, api_key=api_key) as client:
                fetcher: CoinalyzeFetcher | HyperliquidFetcher = CoinalyzeFetcher(client)
                await _batch_fetch_names(
                    fetcher, source, names, bronze, silver, result, progress, task_id
                )
        elif source == "hyperliquid":
            async with AsyncHyperliquidClient() as hl_client:
                fetcher_hl: CoinalyzeFetcher | HyperliquidFetcher = HyperliquidFetcher(hl_client)
                await _batch_fetch_names(
                    fetcher_hl, source, names, bronze, silver, result, progress, task_id
                )

    return result


async def _batch_fetch_names(
    fetcher: CoinalyzeFetcher | HyperliquidFetcher,
    source: str,
    names: list[str],
    bronze: DerivExtBronzeStorage,
    silver: DerivExtSilverProcessor,
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

            # Bronze: 스냅샷 데이터는 append
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
    summary = Table(title="Deriv-Ext Batch Summary", show_header=True)
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
            help="Batch type: coinalyze, all",
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
    """Batch download extended derivatives data by category.

    Example:
        uv run mcbot ingest deriv-ext batch --type all
        uv run mcbot ingest deriv-ext batch --type coinalyze
        uv run mcbot ingest deriv-ext batch --dry-run
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    service = DerivExtDataService(settings)

    try:
        definitions = service.get_batch_definitions(batch_type)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Header
    console.print(
        Panel.fit(
            f"[bold]Deriv-Ext Batch Download[/bold]\nType: {batch_type}\nTotal datasets: {len(definitions)}",
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

    # Check API key
    api_key = settings.coinalyze_api_key.get_secret_value()
    if not api_key:
        console.print("[bold red]COINALYZE_API_KEY is not set.[/bold red]")
        raise typer.Exit(code=1)

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
                "[bold green]Deriv-ext batch download completed![/bold green]",
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
            help="Batch type: coinalyze, all",
        ),
    ] = "all",
) -> None:
    """Display extended derivatives data inventory.

    Example:
        uv run mcbot ingest deriv-ext info
        uv run mcbot ingest deriv-ext info --type coinalyze
    """
    settings = get_settings()
    service = DerivExtDataService(settings)

    try:
        definitions = service.get_batch_definitions(batch_type)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    bronze = DerivExtBronzeStorage(settings)
    silver = DerivExtSilverProcessor(settings)

    table = Table(title=f"Deriv-Ext Data Inventory ({batch_type})")
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
