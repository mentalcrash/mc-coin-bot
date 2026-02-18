"""CLI commands for Deribit Options data ingestion.

DVOL (BTC/ETH), Put/Call Ratio, Historical Vol, Term Structure, Max Pain
데이터를 Bronze/Silver 파이프라인으로 수집합니다.

Commands:
    - pipeline: Single dataset Bronze -> Silver pipeline
    - batch: Batch download (deribit/all)
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
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.options.client import AsyncOptionsClient
from src.data.options.fetcher import OptionsFetcher, route_fetch
from src.data.options.service import OptionsDataService, get_date_col
from src.data.options.storage import OptionsBronzeStorage, OptionsSilverProcessor

console = Console()
_MAX_DISPLAY_ROWS = 20

# 스냅샷 데이터셋 (히스토리 없음 → append 방식)
_SNAPSHOT_DATASETS = {"btc_pc_ratio", "btc_term_structure", "btc_max_pain"}

app = typer.Typer(
    name="options",
    help="Options data ingestion (Deribit DVOL, P/C Ratio, Term Structure, ...)",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# pipeline command
# ---------------------------------------------------------------------------


async def _fetch_pipeline(source: str, name: str) -> None:
    """단일 데이터셋 Bronze -> Silver 파이프라인."""
    settings = get_settings()
    settings.ensure_directories()

    bronze = OptionsBronzeStorage(settings)
    silver = OptionsSilverProcessor(settings)

    async with AsyncOptionsClient(source) as client:
        fetcher = OptionsFetcher(client)
        console.print(f"[blue]Fetching[/blue] {source}/{name}...")
        df = await route_fetch(fetcher, source, name)

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
    source: Annotated[str, typer.Argument(help="Data source (deribit)")],
    name: Annotated[str, typer.Argument(help="Data name (btc_dvol, btc_pc_ratio, ...)")],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """Run full pipeline: Fetch -> Bronze -> Silver for a single dataset.

    Example:
        uv run mcbot ingest options pipeline deribit btc_dvol
        uv run mcbot ingest options pipeline deribit btc_pc_ratio
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    console.print(
        Panel.fit(
            f"[bold]Options Pipeline[/bold]\nSource: {source}\nName: {name}",
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
    bronze = OptionsBronzeStorage(settings)
    silver = OptionsSilverProcessor(settings)

    task_id = progress.add_task(
        "[cyan]Options Batch Progress[/cyan]",
        total=result.total,
    )

    # source별 그룹핑
    groups: dict[str, list[str]] = {}
    for source, name in definitions:
        groups.setdefault(source, []).append(name)

    for source, names in groups.items():
        async with AsyncOptionsClient(source) as client:
            fetcher = OptionsFetcher(client)

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

    return result


def _display_batch_result(result: _BatchResult) -> None:
    """배치 결과 summary table 출력."""
    summary = Table(title="Options Batch Summary", show_header=True)
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
            help="Batch type: deribit, all",
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
    """Batch download options data by category.

    Example:
        uv run mcbot ingest options batch --type all
        uv run mcbot ingest options batch --type deribit
        uv run mcbot ingest options batch --dry-run
    """
    settings = get_settings()
    setup_logger(log_dir=settings.log_dir, console_level="DEBUG" if verbose else "INFO")

    service = OptionsDataService(settings)

    try:
        definitions = service.get_batch_definitions(batch_type)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    # Header
    console.print(
        Panel.fit(
            f"[bold]Options Batch Download[/bold]\nType: {batch_type}\nTotal datasets: {len(definitions)}",
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
                "[bold green]Options batch download completed![/bold green]",
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
            help="Batch type: deribit, all",
        ),
    ] = "all",
) -> None:
    """Display options data inventory.

    Example:
        uv run mcbot ingest options info
        uv run mcbot ingest options info --type deribit
    """
    settings = get_settings()
    service = OptionsDataService(settings)

    try:
        definitions = service.get_batch_definitions(batch_type)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    bronze = OptionsBronzeStorage(settings)
    silver = OptionsSilverProcessor(settings)

    table = Table(title=f"Options Data Inventory ({batch_type})")
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
