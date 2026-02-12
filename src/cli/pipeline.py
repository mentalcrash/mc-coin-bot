"""Typer CLI for Strategy Pipeline Management.

Commands:
    - status: 전략 현황 요약 (상태별 카운트)
    - list: 전략 목록 (필터링)
    - show: 전략 상세 정보
    - create: 새 전략 YAML 생성 (CANDIDATE 상태)
    - record: Gate 결과 기록
    - update-status: 전략 상태 변경
    - report: Dashboard 자동 생성
    - migrate: Markdown → YAML 마이그레이션
    - table: 모든 전략 현황 표
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline.models import (
    Decision,
    GateId,
    GateResult,
    GateVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.store import StrategyStore

app = typer.Typer(no_args_is_help=True)
console = Console()

_STATUS_COLORS: dict[StrategyStatus, str] = {
    StrategyStatus.CANDIDATE: "cyan",
    StrategyStatus.IMPLEMENTED: "blue",
    StrategyStatus.TESTING: "yellow",
    StrategyStatus.ACTIVE: "green",
    StrategyStatus.RETIRED: "red",
}

_GATE_DISPLAY = ["G0A", "G0B", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]


def _gate_badge_colored(record: StrategyRecord, gid: GateId) -> str:
    """Gate 결과를 Rich 색상 문자로 변환."""
    result = record.gates.get(gid)
    if result is None:
        return "[dim]-[/dim]"
    if result.status == GateVerdict.PASS:
        return "[green]P[/green]"
    return "[red]F[/red]"


# ─── Commands ────────────────────────────────────────────────────────


@app.command()
def status() -> None:
    """전략 현황 요약 (상태별 카운트)."""
    store = StrategyStore()
    records = store.load_all()

    if not records:
        console.print("[yellow]No strategies found. Run 'pipeline migrate' first.[/yellow]")
        return

    counts: dict[StrategyStatus, int] = {}
    for r in records:
        counts[r.meta.status] = counts.get(r.meta.status, 0) + 1

    total = len(records)
    console.print(f"\n[bold]Strategy Pipeline Status[/bold] ({total} total)\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Bar")

    for s in StrategyStatus:
        count = counts.get(s, 0)
        color = _STATUS_COLORS[s]
        bar_len = int(count / max(total, 1) * 30)
        bar = f"[{color}]{'█' * bar_len}[/{color}]"
        table.add_row(f"[{color}]{s}[/{color}]", str(count), bar)

    console.print(table)


@app.command(name="list")
def list_strategies(
    status_filter: Annotated[
        str | None, typer.Option("--status", "-s", help="Filter by status")
    ] = None,
    gate: Annotated[str | None, typer.Option("--gate", "-g", help="Filter by current gate")] = None,
    verdict: Annotated[
        str | None, typer.Option("--verdict", "-v", help="PASS or FAIL at gate")
    ] = None,
    timeframe: Annotated[str | None, typer.Option("--tf", help="Filter by timeframe")] = None,
) -> None:
    """전략 목록 (필터링 가능)."""
    store = StrategyStore()
    records = store.load_all()

    if status_filter:
        try:
            sf = StrategyStatus(status_filter)
        except ValueError:
            console.print(f"[red]Invalid status: {status_filter}[/red]")
            raise typer.Exit(code=1) from None
        records = [r for r in records if r.meta.status == sf]

    if gate and verdict:
        gid = GateId(gate)
        gv = GateVerdict(verdict)
        records = [r for r in records if gid in r.gates and r.gates[gid].status == gv]
    elif gate:
        gid = GateId(gate)
        records = [r for r in records if r.current_gate == gid]

    if timeframe:
        records = [r for r in records if r.meta.timeframe == timeframe]

    if not records:
        console.print("[yellow]No strategies match the given filters.[/yellow]")
        return

    _print_strategy_table(records)


@app.command()
def show(name: Annotated[str, typer.Argument(help="Strategy name (kebab-case)")]) -> None:
    """전략 상세 정보."""
    store = StrategyStore()
    try:
        record = store.load(name)
    except FileNotFoundError:
        console.print(f"[red]Strategy not found: {name}[/red]")
        raise typer.Exit(code=1) from None

    _print_strategy_detail(record)


@app.command()
def create(
    name: Annotated[str, typer.Argument(help="Strategy name (kebab-case)")],
    display_name: Annotated[str, typer.Option("--display-name", help="표시 이름")],
    category: Annotated[str, typer.Option("--category", help="전략 카테고리")],
    timeframe: Annotated[str, typer.Option("--timeframe", "--tf", help="타임프레임")],
    short_mode: Annotated[str, typer.Option("--short-mode", help="DISABLED|HEDGE_ONLY|FULL")],
    rationale: Annotated[str, typer.Option("--rationale", "-r", help="경제적 논거")] = "",
    g0a_score: Annotated[int, typer.Option("--g0a-score", help="Gate 0A 점수")] = 0,
) -> None:
    """새 전략 YAML 생성 (CANDIDATE 상태)."""
    store = StrategyStore()
    if store.exists(name):
        console.print(f"[red]Already exists: {name}[/red]")
        raise typer.Exit(code=1)

    today = date.today()
    record_obj = StrategyRecord(
        meta=StrategyMeta(
            name=name,
            display_name=display_name,
            category=category,
            timeframe=timeframe,
            short_mode=short_mode,
            status=StrategyStatus.CANDIDATE,
            created_at=today,
            economic_rationale=rationale,
        ),
        gates={
            GateId.G0A: GateResult(
                status=GateVerdict.PASS,
                date=today,
                details={"score": g0a_score, "max_score": 30},
            ),
        },
        decisions=[
            Decision(
                date=today,
                gate=GateId.G0A,
                verdict=GateVerdict.PASS,
                rationale=f"{g0a_score}/30점",
            ),
        ],
    )
    store.save(record_obj)
    console.print(f"[green]Created: strategies/{name}.yaml (CANDIDATE)[/green]")


@app.command()
def record(
    name: Annotated[str, typer.Argument(help="Strategy name")],
    gate: Annotated[str, typer.Option("--gate", "-g", help="Gate ID (G0A, G1, ...)")],
    verdict: Annotated[str, typer.Option("--verdict", "-v", help="PASS or FAIL")],
    rationale: Annotated[str, typer.Option("--rationale", "-r", help="판정 사유")] = "",
    detail: Annotated[
        list[str] | None, typer.Option("--detail", "-d", help="key=value pairs")
    ] = None,
    no_retire: Annotated[bool, typer.Option("--no-retire", help="FAIL 시 자동 RETIRED 방지")] = False,
) -> None:
    """Gate 결과 기록."""
    store = StrategyStore()
    if not store.exists(name):
        console.print(f"[red]Strategy not found: {name}[/red]")
        raise typer.Exit(code=1) from None

    details: dict[str, object] = {}
    for d in detail or []:
        key, _, val = d.partition("=")
        try:
            details[key] = float(val)
        except ValueError:
            details[key] = val

    gid = GateId(gate)
    gv = GateVerdict(verdict)
    store.record_gate(name, gid, gv, details=details, rationale=rationale)
    console.print(f"[green]Recorded {gate} {verdict} for {name}[/green]")

    if gv == GateVerdict.FAIL and not no_retire:
        store.update_status(name, StrategyStatus.RETIRED)
        console.print("[yellow]Status → RETIRED[/yellow]")
    elif gv == GateVerdict.FAIL and no_retire:
        console.print("[yellow]FAIL recorded (status unchanged — --no-retire)[/yellow]")


@app.command(name="update-status")
def update_status_cmd(
    name: Annotated[str, typer.Argument(help="Strategy name")],
    status: Annotated[str, typer.Option("--status", "-s", help="New status")],
) -> None:
    """전략 상태 변경."""
    store = StrategyStore()
    if not store.exists(name):
        console.print(f"[red]Strategy not found: {name}[/red]")
        raise typer.Exit(code=1)
    new_status = StrategyStatus(status)
    store.update_status(name, new_status)
    console.print(f"[green]{name} → {new_status}[/green]")


@app.command()
def report() -> None:
    """Dashboard 자동 생성 (YAML → markdown)."""
    from src.pipeline.report import DashboardGenerator

    store = StrategyStore()
    generator = DashboardGenerator(store)
    content = generator.generate()

    from pathlib import Path

    output = Path("docs/strategy/dashboard.md")
    output.write_text(content, encoding="utf-8")
    console.print(f"[green]Dashboard generated: {output}[/green]")


@app.command()
def migrate() -> None:
    """Markdown → YAML 마이그레이션."""
    from src.pipeline.migrate import run_migration

    run_migration()
    console.print("[green]Migration complete.[/green]")


@app.command(name="table")
def full_table() -> None:
    """모든 전략의 현황을 Gate 진행도 표로 출력."""
    store = StrategyStore()
    records = store.load_all()

    if not records:
        console.print("[yellow]No strategies found. Run 'pipeline migrate' first.[/yellow]")
        return

    # Sort: ACTIVE first, then by name
    records.sort(key=lambda r: (r.meta.status != StrategyStatus.ACTIVE, r.meta.name))

    table = Table(
        title=f"Strategy Pipeline Overview ({len(records)} strategies)",
        show_header=True,
        header_style="bold",
        show_lines=False,
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Strategy", style="bold", min_width=16)
    table.add_column("TF", width=4)
    table.add_column("Status", width=12)
    table.add_column("Sharpe", justify="right", width=7)
    table.add_column("CAGR", justify="right", width=8)
    table.add_column("MDD", justify="right", width=7)
    table.add_column("Best Asset", width=11)
    for gname in _GATE_DISPLAY:
        table.add_column(gname, justify="center", width=3)

    for i, r in enumerate(records, 1):
        color = _STATUS_COLORS.get(r.meta.status, "white")
        sharpe_str = f"{r.best_sharpe:.2f}" if r.best_sharpe is not None else "-"
        best_asset_str = r.best_asset or "-"

        cagr_str = "-"
        mdd_str = "-"
        if r.asset_performance:
            best = max(r.asset_performance, key=lambda a: a.sharpe)
            cagr_str = f"+{best.cagr:.1f}%" if best.cagr > 0 else f"{best.cagr:.1f}%"
            mdd_str = f"-{best.mdd:.1f}%"

        gate_cells = [_gate_badge_colored(r, GateId(g)) for g in _GATE_DISPLAY]

        table.add_row(
            str(i),
            r.meta.display_name,
            r.meta.timeframe,
            f"[{color}]{r.meta.status}[/{color}]",
            sharpe_str,
            cagr_str,
            mdd_str,
            best_asset_str,
            *gate_cells,
        )

    console.print()
    console.print(table)

    # Summary
    active = sum(1 for r in records if r.meta.status == StrategyStatus.ACTIVE)
    retired = sum(1 for r in records if r.meta.status == StrategyStatus.RETIRED)
    summary = f"\n  [green]ACTIVE: {active}[/green] | [red]RETIRED: {retired}[/red] | Total: {len(records)}"
    console.print(summary)


# ─── Display helpers ─────────────────────────────────────────────────


def _print_strategy_table(records: list[StrategyRecord]) -> None:
    """전략 목록을 Rich Table로 출력."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="bold")
    table.add_column("TF")
    table.add_column("Status")
    table.add_column("Gate")
    table.add_column("Sharpe", justify="right")
    table.add_column("Best Asset")

    for r in records:
        color = _STATUS_COLORS.get(r.meta.status, "white")
        sharpe = f"{r.best_sharpe:.2f}" if r.best_sharpe is not None else "-"
        gate = str(r.current_gate) if r.current_gate else "-"
        if r.fail_gate:
            gate = f"{r.fail_gate} [red]FAIL[/red]"
        table.add_row(
            r.meta.display_name,
            r.meta.timeframe,
            f"[{color}]{r.meta.status}[/{color}]",
            gate,
            sharpe,
            r.best_asset or "-",
        )

    console.print(table)


def _print_strategy_detail(record: StrategyRecord) -> None:
    """전략 상세를 Rich Panel로 출력."""
    r = record
    color = _STATUS_COLORS.get(r.meta.status, "white")

    # Meta info
    meta_lines = [
        f"[bold]Name:[/bold] {r.meta.display_name} ({r.meta.name})",
        f"[bold]Category:[/bold] {r.meta.category}",
        f"[bold]TF:[/bold] {r.meta.timeframe}  |  [bold]Short:[/bold] {r.meta.short_mode}",
        f"[bold]Status:[/bold] [{color}]{r.meta.status}[/{color}]",
        f"[bold]Rationale:[/bold] {r.meta.economic_rationale}",
    ]
    console.print(Panel("\n".join(meta_lines), title=f"[bold]{r.meta.display_name}[/bold]"))

    # Gate progress
    gate_line = "  ".join(f"{g}: {_gate_badge_colored(r, GateId(g))}" for g in _GATE_DISPLAY)
    console.print(f"\n[bold]Gate Progress:[/bold] {gate_line}\n")

    # Asset table
    if r.asset_performance:
        at = Table(show_header=True, header_style="bold", title="Asset Performance")
        at.add_column("Symbol")
        at.add_column("Sharpe", justify="right")
        at.add_column("CAGR", justify="right")
        at.add_column("MDD", justify="right")
        at.add_column("Trades", justify="right")
        at.add_column("PF", justify="right")

        for a in sorted(r.asset_performance, key=lambda x: x.sharpe, reverse=True):
            at.add_row(
                a.symbol,
                f"{a.sharpe:.2f}",
                f"+{a.cagr:.1f}%" if a.cagr > 0 else f"{a.cagr:.1f}%",
                f"-{a.mdd:.1f}%",
                str(a.trades),
                f"{a.profit_factor:.2f}" if a.profit_factor else "-",
            )
        console.print(at)

    # Parameters
    if r.parameters:
        console.print(f"\n[bold]Parameters:[/bold] {r.parameters}")

    # Decisions
    if r.decisions:
        dt = Table(show_header=True, header_style="bold", title="Decision History")
        dt.add_column("Date")
        dt.add_column("Gate")
        dt.add_column("Verdict")
        dt.add_column("Rationale")
        for d in r.decisions:
            v_color = "green" if d.verdict == GateVerdict.PASS else "red"
            dt.add_row(str(d.date), str(d.gate), f"[{v_color}]{d.verdict}[/{v_color}]", d.rationale)
        console.print(dt)
