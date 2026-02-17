"""Typer CLI for Data Catalog.

Commands:
    - list: 데이터셋 목록 (유형/그룹 필터링)
    - show: 데이터셋 상세 정보
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.catalog.models import DataType
from src.catalog.store import DataCatalogStore

app = typer.Typer(no_args_is_help=True)
console = Console()

_TYPE_COLORS: dict[DataType, str] = {
    DataType.OHLCV: "cyan",
    DataType.DERIVATIVES: "blue",
    DataType.ONCHAIN: "green",
}


@app.command(name="list")
def list_datasets(
    data_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Filter by type (ohlcv, derivatives, onchain)"),
    ] = None,
    group: Annotated[
        str | None, typer.Option("--group", "-g", help="Filter by batch group")
    ] = None,
) -> None:
    """데이터셋 목록 (필터링 가능)."""
    store = DataCatalogStore()
    try:
        if data_type:
            try:
                dt = DataType(data_type)
            except ValueError:
                console.print(f"[red]Invalid type: {data_type}[/red]")
                console.print(f"Valid: {', '.join(t.value for t in DataType)}")
                raise typer.Exit(code=1) from None
            datasets = store.get_by_type(dt)
        elif group:
            datasets = store.get_by_group(group)
        else:
            datasets = store.load_all()
    except FileNotFoundError:
        console.print("[red]catalogs/datasets.yaml not found.[/red]")
        raise typer.Exit(code=1) from None

    if not datasets:
        console.print("[yellow]No datasets match the given filters.[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Data Catalog ({len(datasets)} datasets)",
    )
    table.add_column("ID", style="bold", min_width=12)
    table.add_column("Name", min_width=16)
    table.add_column("Type", width=12)
    table.add_column("Group", width=12)
    table.add_column("Source", width=14)
    table.add_column("Res.", width=4)
    table.add_column("Columns", width=5, justify="right")

    for ds in datasets:
        color = _TYPE_COLORS.get(ds.data_type, "white")
        table.add_row(
            ds.id,
            ds.name,
            f"[{color}]{ds.data_type}[/{color}]",
            ds.batch_group or "-",
            ds.source_id,
            ds.resolution,
            str(len(ds.columns)),
        )

    console.print(table)


@app.command()
def show(
    dataset_id: Annotated[str, typer.Argument(help="Dataset ID (e.g., btc_metrics)")],
) -> None:
    """데이터셋 상세 정보."""
    store = DataCatalogStore()
    try:
        ds = store.load(dataset_id)
    except FileNotFoundError:
        console.print("[red]catalogs/datasets.yaml not found.[/red]")
        raise typer.Exit(code=1) from None
    except KeyError:
        console.print(f"[red]Dataset not found: {dataset_id}[/red]")
        raise typer.Exit(code=1) from None

    # Source info
    try:
        source = store.get_source(ds.source_id)
        source_info = (
            f"{source.name} (lag: {source.lag_days}d, limit: {source.rate_limit_per_min}/min)"
        )
    except KeyError:
        source_info = ds.source_id

    color = _TYPE_COLORS.get(ds.data_type, "white")
    lines = [
        f"[bold]{ds.id}[/bold] — {ds.name}",
        "",
        ds.description,
        "",
        f"[bold]Type:[/bold] [{color}]{ds.data_type}[/{color}]",
        f"[bold]Group:[/bold] {ds.batch_group or '-'}",
        f"[bold]Source:[/bold] {source_info}",
        f"[bold]Resolution:[/bold] {ds.resolution}",
        f"[bold]Available Since:[/bold] {ds.available_since or '-'}",
        f"[bold]Storage:[/bold] {ds.storage_path or '-'}",
        f"[bold]Columns:[/bold] {', '.join(ds.columns) if ds.columns else '-'}",
    ]

    if ds.enrichment:
        lines.append("")
        lines.append(f"[bold]Enrichment Scope:[/bold] {ds.enrichment.scope}")
        if ds.enrichment.target_assets:
            lines.append(f"[bold]Target Assets:[/bold] {', '.join(ds.enrichment.target_assets)}")
        if ds.enrichment.columns:
            lines.append(f"[bold]Enrichment Columns:[/bold] {', '.join(ds.enrichment.columns)}")
        if ds.enrichment.rename_map:
            renames = [f"{k} → {v}" for k, v in ds.enrichment.rename_map.items()]
            lines.append(f"[bold]Rename:[/bold] {', '.join(renames)}")

    if ds.strategy_hints:
        lines.append("")
        lines.append("[bold]Strategy Hints:[/bold]")
        lines.extend(f"  [dim]•[/dim] {hint}" for hint in ds.strategy_hints)

    console.print(Panel("\n".join(lines), title=f"Dataset: {ds.id}"))
