"""Typer CLI for Data Catalog.

Commands:
    - list: 데이터셋 목록 (유형/그룹 필터링)
    - show: 데이터셋 상세 정보
    - failure-patterns: 실패 패턴 목록
    - failure-pattern-show: 실패 패턴 상세
    - indicators: 지표 목록
    - indicator-show: 지표 상세
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


# ─── Failure Patterns ────────────────────────────────────────────────


@app.command(name="failure-patterns")
def failure_patterns(
    gate: Annotated[
        str | None, typer.Option("--gate", "-g", help="Filter by affected gate (e.g., G1)")
    ] = None,
    freq: Annotated[
        str | None, typer.Option("--freq", "-f", help="Filter by frequency (high/medium/low)")
    ] = None,
) -> None:
    """실패 패턴 목록."""
    from src.catalog.failure_store import FailurePatternStore

    store = FailurePatternStore()
    try:
        if gate:
            patterns = store.filter_by_gate(gate)
        elif freq:
            patterns = store.filter_by_frequency(freq)
        else:
            patterns = store.load_all()
    except FileNotFoundError:
        console.print("[red]catalogs/failure_patterns.yaml not found.[/red]")
        raise typer.Exit(code=1) from None

    if not patterns:
        console.print("[yellow]No patterns match the given filters.[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Failure Patterns ({len(patterns)})",
    )
    table.add_column("ID", style="bold", min_width=12)
    table.add_column("Name", min_width=16)
    table.add_column("Freq", width=8)
    table.add_column("Gates", width=10)
    table.add_column("Examples", width=5, justify="right")
    table.add_column("Prevention", min_width=20)

    freq_colors = {"high": "red", "medium": "yellow", "low": "green"}
    for p in patterns:
        color = freq_colors.get(p.frequency, "white")
        table.add_row(
            p.id,
            p.name,
            f"[{color}]{p.frequency}[/{color}]",
            ", ".join(p.affected_gates),
            str(len(p.examples)),
            p.prevention[0] if p.prevention else "-",
        )

    console.print(table)


@app.command(name="failure-pattern-show")
def failure_pattern_show(
    pattern_id: Annotated[str, typer.Argument(help="Pattern ID (e.g., cost_erosion)")],
) -> None:
    """실패 패턴 상세 정보."""
    from src.catalog.failure_store import FailurePatternStore

    store = FailurePatternStore()
    try:
        p = store.load(pattern_id)
    except FileNotFoundError:
        console.print("[red]catalogs/failure_patterns.yaml not found.[/red]")
        raise typer.Exit(code=1) from None
    except KeyError:
        console.print(f"[red]Pattern not found: {pattern_id}[/red]")
        raise typer.Exit(code=1) from None

    freq_colors = {"high": "red", "medium": "yellow", "low": "green"}
    color = freq_colors.get(p.frequency, "white")

    lines = [
        f"[bold]{p.id}[/bold] — {p.name}",
        "",
        p.description,
        "",
        f"[bold]Frequency:[/bold] [{color}]{p.frequency}[/{color}]",
        f"[bold]Affected Gates:[/bold] {', '.join(p.affected_gates)}",
    ]

    if p.detection_rules:
        lines.append("")
        lines.append("[bold]Detection Rules:[/bold]")
        lines.extend(
            f"  [dim]•[/dim] {rule.metric} {rule.operator} {rule.threshold}"
            for rule in p.detection_rules
        )

    if p.prevention:
        lines.append("")
        lines.append("[bold]Prevention:[/bold]")
        lines.extend(f"  [dim]•[/dim] {tip}" for tip in p.prevention)

    if p.examples:
        lines.append("")
        lines.append(f"[bold]Examples:[/bold] {', '.join(p.examples)}")

    if p.related_lessons:
        lines.append(
            f"[bold]Related Lessons:[/bold] {', '.join(str(lid) for lid in p.related_lessons)}"
        )

    console.print(Panel("\n".join(lines), title=f"Failure Pattern: {p.id}"))


# ─── Indicators ──────────────────────────────────────────────────────


@app.command(name="indicators")
def list_indicators(
    category: Annotated[
        str | None, typer.Option("--category", "-c", help="Filter by category")
    ] = None,
    unused: Annotated[
        bool, typer.Option("--unused", help="Show only unused indicators")
    ] = False,
    potential: Annotated[
        str | None, typer.Option("--potential", "-p", help="Filter by alpha potential")
    ] = None,
) -> None:
    """지표 목록."""
    from src.catalog.indicator_store import IndicatorCatalogStore

    store = IndicatorCatalogStore()
    try:
        if category:
            indicators = store.get_by_category(category)
        elif unused:
            indicators = store.get_unused()
        elif potential:
            indicators = store.get_by_potential(potential)
        else:
            indicators = store.load_all()
    except FileNotFoundError:
        console.print("[red]catalogs/indicators.yaml not found.[/red]")
        raise typer.Exit(code=1) from None

    if not indicators:
        console.print("[yellow]No indicators match the given filters.[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Indicator Catalog ({len(indicators)})",
    )
    table.add_column("ID", style="bold", min_width=12)
    table.add_column("Name", min_width=16)
    table.add_column("Module", width=12)
    table.add_column("Category", width=12)
    table.add_column("Used By", width=15)
    table.add_column("Alpha", width=8)

    pot_colors = {"high": "green", "medium": "yellow", "low": "dim"}
    for ind in indicators:
        color = pot_colors.get(ind.alpha_potential, "white")
        table.add_row(
            ind.id,
            ind.name,
            ind.module,
            ind.category.value,
            ", ".join(ind.used_by) if ind.used_by else "[dim]-[/dim]",
            f"[{color}]{ind.alpha_potential}[/{color}]",
        )

    console.print(table)


@app.command(name="indicator-show")
def indicator_show(
    indicator_id: Annotated[str, typer.Argument(help="Indicator ID (e.g., hurst_exponent)")],
) -> None:
    """지표 상세 정보."""
    from src.catalog.indicator_store import IndicatorCatalogStore

    store = IndicatorCatalogStore()
    try:
        ind = store.load(indicator_id)
    except FileNotFoundError:
        console.print("[red]catalogs/indicators.yaml not found.[/red]")
        raise typer.Exit(code=1) from None
    except KeyError:
        console.print(f"[red]Indicator not found: {indicator_id}[/red]")
        raise typer.Exit(code=1) from None

    pot_colors = {"high": "green", "medium": "yellow", "low": "dim"}
    color = pot_colors.get(ind.alpha_potential, "white")

    lines = [
        f"[bold]{ind.id}[/bold] — {ind.name}",
        "",
        ind.description,
        "",
        f"[bold]Module:[/bold] {ind.module}",
        f"[bold]Category:[/bold] {ind.category.value}",
        f"[bold]Alpha Potential:[/bold] [{color}]{ind.alpha_potential}[/{color}]",
    ]

    if ind.default_params:
        params = ", ".join(f"{k}={v}" for k, v in ind.default_params.items())
        lines.append(f"[bold]Default Params:[/bold] {params}")

    if ind.used_by:
        lines.append(f"[bold]Used By:[/bold] {', '.join(ind.used_by)}")
    else:
        lines.append("[bold]Used By:[/bold] [dim]None (unused)[/dim]")

    if ind.notes:
        lines.append("")
        lines.append(f"[bold]Notes:[/bold] {ind.notes}")

    console.print(Panel("\n".join(lines), title=f"Indicator: {ind.id}"))
