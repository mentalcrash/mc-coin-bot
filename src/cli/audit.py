"""Typer CLI for Architecture Audit Report System.

Commands:
    - list: 스냅샷 목록
    - show: 스냅샷 상세
    - latest: 최신 스냅샷
    - findings: 발견사항 목록 (필터링)
    - finding-show: 발견사항 상세
    - actions: 액션 목록 (필터링)
    - action-show: 액션 상세
    - trend: 스냅샷간 지표 추이
    - resolve-finding: 발견사항 해결 처리
    - update-action: 액션 상태 변경
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline.audit_models import (
    ActionItem,
    ActionPriority,
    ActionStatus,
    AuditSeverity,
    AuditSnapshot,
    Finding,
    FindingStatus,
    HealthLevel,
)
from src.pipeline.audit_store import AuditStore

app = typer.Typer(no_args_is_help=True)
console = Console()

_SEVERITY_COLORS: dict[AuditSeverity, str] = {
    AuditSeverity.CRITICAL: "red bold",
    AuditSeverity.HIGH: "red",
    AuditSeverity.MEDIUM: "yellow",
    AuditSeverity.LOW: "dim",
}

_STATUS_COLORS: dict[FindingStatus, str] = {
    FindingStatus.OPEN: "red",
    FindingStatus.IN_PROGRESS: "yellow",
    FindingStatus.RESOLVED: "green",
    FindingStatus.WONT_FIX: "dim",
    FindingStatus.DEFERRED: "cyan",
}

_HEALTH_COLORS: dict[HealthLevel, str] = {
    HealthLevel.GREEN: "green",
    HealthLevel.YELLOW: "yellow",
    HealthLevel.RED: "red",
}

_PRIORITY_COLORS: dict[ActionPriority, str] = {
    ActionPriority.P0: "red bold",
    ActionPriority.P1: "red",
    ActionPriority.P2: "yellow",
    ActionPriority.P3: "dim",
}

_ACTION_STATUS_COLORS: dict[ActionStatus, str] = {
    ActionStatus.PENDING: "yellow",
    ActionStatus.IN_PROGRESS: "cyan",
    ActionStatus.COMPLETED: "green",
    ActionStatus.CANCELLED: "dim",
}


# ─── Snapshot Commands ────────────────────────────────────────────────


@app.command(name="list")
def list_snapshots() -> None:
    """모든 감사 스냅샷 목록."""
    store = AuditStore()
    snapshots = store.load_all_snapshots()

    if not snapshots:
        console.print("[yellow]No audit snapshots found.[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Audit Snapshots ({len(snapshots)})",
    )
    table.add_column("Date", style="bold", width=12)
    table.add_column("Overall", justify="center", width=8)
    table.add_column("Tests", justify="right", width=6)
    table.add_column("Coverage", justify="right", width=9)
    table.add_column("Findings", justify="right", width=9)
    table.add_column("Actions", justify="right", width=8)
    table.add_column("Summary", min_width=20)

    for s in snapshots:
        summary_short = s.summary.split("\n")[0][:50] if s.summary else "-"
        table.add_row(
            str(s.date),
            s.grades.overall or "-",
            str(s.metrics.test_count),
            f"{s.metrics.coverage_pct:.0%}",
            str(len(s.new_findings)),
            str(len(s.new_actions)),
            summary_short,
        )

    console.print(table)


@app.command()
def show(
    date_str: Annotated[str, typer.Argument(help="스냅샷 날짜 (YYYY-MM-DD)")],
) -> None:
    """특정 감사 스냅샷 상세."""
    store = AuditStore()
    try:
        snapshot = store.load_snapshot(date_str)
    except FileNotFoundError:
        console.print(f"[red]Snapshot not found: {date_str}[/red]")
        raise typer.Exit(code=1) from None

    _print_snapshot_detail(store, snapshot)


@app.command()
def latest() -> None:
    """최신 감사 스냅샷 상세."""
    store = AuditStore()
    snapshot = store.latest_snapshot()
    if snapshot is None:
        console.print("[yellow]No audit snapshots found.[/yellow]")
        return
    _print_snapshot_detail(store, snapshot)


# ─── Finding Commands ─────────────────────────────────────────────────


@app.command()
def findings(
    status: Annotated[str | None, typer.Option("--status", "-s", help="Filter by status")] = None,
    severity: Annotated[str | None, typer.Option("--severity", help="Filter by severity")] = None,
    category: Annotated[
        str | None, typer.Option("--category", "-c", help="Filter by category")
    ] = None,
) -> None:
    """발견사항 목록 (필터링 가능)."""
    store = AuditStore()

    if status:
        try:
            fs = FindingStatus(status)
        except ValueError:
            console.print(f"[red]Invalid status: {status}[/red]")
            raise typer.Exit(code=1) from None
        records = store.findings_by_status(fs)
    elif severity:
        try:
            sev = AuditSeverity(severity)
        except ValueError:
            console.print(f"[red]Invalid severity: {severity}[/red]")
            raise typer.Exit(code=1) from None
        records = store.findings_by_severity(sev)
    elif category:
        from src.pipeline.audit_models import AuditCategory

        try:
            cat = AuditCategory(category)
        except ValueError:
            console.print(f"[red]Invalid category: {category}[/red]")
            raise typer.Exit(code=1) from None
        records = store.findings_by_category(cat)
    else:
        records = store.load_all_findings()

    if not records:
        console.print("[yellow]No findings match the given filters.[/yellow]")
        return

    _print_findings_table(records)


@app.command(name="finding-show")
def finding_show(
    finding_id: Annotated[int, typer.Argument(help="Finding ID")],
) -> None:
    """발견사항 상세."""
    store = AuditStore()
    try:
        f = store.load_finding(finding_id)
    except FileNotFoundError:
        console.print(f"[red]Finding not found: {finding_id}[/red]")
        raise typer.Exit(code=1) from None

    sev_color = _SEVERITY_COLORS.get(f.severity, "white")
    status_color = _STATUS_COLORS.get(f.status, "white")

    lines = [
        f"[bold]#{f.id}[/bold] {f.title}",
        "",
        f"[bold]Severity:[/bold] [{sev_color}]{f.severity}[/{sev_color}]",
        f"[bold]Category:[/bold] {f.category}",
        f"[bold]Status:[/bold] [{status_color}]{f.status}[/{status_color}]",
        f"[bold]Location:[/bold] {f.location or '-'}",
        "",
        f"[bold]Description:[/bold]\n{f.description}" if f.description else "",
        f"[bold]Impact:[/bold]\n{f.impact}" if f.impact else "",
        f"[bold]Proposed Fix:[/bold]\n{f.proposed_fix}" if f.proposed_fix else "",
        "",
        f"[bold]Effort:[/bold] {f.effort or '-'}",
        f"[bold]Related Actions:[/bold] {f.related_actions or '-'}",
        f"[bold]Tags:[/bold] {', '.join(f.tags) if f.tags else '-'}",
        f"[bold]Discovered:[/bold] {f.discovered_at}",
        f"[bold]Resolved:[/bold] {f.resolved_at or '-'}",
    ]
    console.print(Panel("\n".join(lines), title=f"Finding #{f.id}"))


# ─── Action Commands ──────────────────────────────────────────────────


@app.command()
def actions(
    status: Annotated[str | None, typer.Option("--status", "-s", help="Filter by status")] = None,
    priority: Annotated[
        str | None, typer.Option("--priority", "-p", help="Filter by priority")
    ] = None,
) -> None:
    """액션 아이템 목록 (필터링 가능)."""
    store = AuditStore()

    if status:
        try:
            action_status = ActionStatus(status)
        except ValueError:
            console.print(f"[red]Invalid status: {status}[/red]")
            raise typer.Exit(code=1) from None
        records = [a for a in store.load_all_actions() if a.status == action_status]
    elif priority:
        try:
            pri = ActionPriority(priority)
        except ValueError:
            console.print(f"[red]Invalid priority: {priority}[/red]")
            raise typer.Exit(code=1) from None
        records = store.actions_by_priority(pri)
    else:
        records = store.load_all_actions()

    if not records:
        console.print("[yellow]No actions match the given filters.[/yellow]")
        return

    _print_actions_table(records)


@app.command(name="action-show")
def action_show(
    action_id: Annotated[int, typer.Argument(help="Action ID")],
) -> None:
    """액션 상세."""
    store = AuditStore()
    try:
        a = store.load_action(action_id)
    except FileNotFoundError:
        console.print(f"[red]Action not found: {action_id}[/red]")
        raise typer.Exit(code=1) from None

    pri_color = _PRIORITY_COLORS.get(a.priority, "white")
    status_color = _ACTION_STATUS_COLORS.get(a.status, "white")

    lines = [
        f"[bold]#{a.id}[/bold] {a.title}",
        "",
        f"[bold]Priority:[/bold] [{pri_color}]{a.priority}[/{pri_color}]",
        f"[bold]Status:[/bold] [{status_color}]{a.status}[/{status_color}]",
        f"[bold]Phase:[/bold] {a.phase or '-'}",
        f"[bold]Assigned:[/bold] {a.assigned_to or '-'}",
        "",
        f"[bold]Description:[/bold]\n{a.description}" if a.description else "",
        "",
        f"[bold]Effort:[/bold] {a.estimated_effort or '-'}",
        f"[bold]Related Findings:[/bold] {a.related_findings or '-'}",
        f"[bold]Tags:[/bold] {', '.join(a.tags) if a.tags else '-'}",
        f"[bold]Created:[/bold] {a.created_at}",
        f"[bold]Started:[/bold] {a.started_at or '-'}",
        f"[bold]Completed:[/bold] {a.completed_at or '-'}",
        "",
        f"[bold]Verification:[/bold]\n{a.verification}" if a.verification else "",
    ]
    console.print(Panel("\n".join(lines), title=f"Action #{a.id}"))


# ─── Trend Command ───────────────────────────────────────────────────


@app.command()
def trend() -> None:
    """스냅샷간 지표 추이."""
    store = AuditStore()
    snapshots = store.load_all_snapshots()

    if not snapshots:
        console.print("[yellow]No audit snapshots found.[/yellow]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        title="Audit Trend",
    )
    table.add_column("Date", style="bold", width=12)
    table.add_column("Overall", justify="center", width=8)
    table.add_column("Tests", justify="right", width=7)
    table.add_column("Pass%", justify="right", width=7)
    table.add_column("Lint", justify="right", width=5)
    table.add_column("Type", justify="right", width=5)
    table.add_column("Cov%", justify="right", width=7)
    table.add_column("Strats", justify="right", width=6)
    table.add_column("Active", justify="right", width=6)
    table.add_column("New Find", justify="right", width=9)
    table.add_column("New Act", justify="right", width=8)

    for s in snapshots:
        table.add_row(
            str(s.date),
            s.grades.overall or "-",
            str(s.metrics.test_count),
            f"{s.metrics.test_pass_rate:.0%}",
            str(s.metrics.lint_errors),
            str(s.metrics.type_errors),
            f"{s.metrics.coverage_pct:.0%}",
            str(s.strategy_summary.total),
            str(s.strategy_summary.active),
            str(len(s.new_findings)),
            str(len(s.new_actions)),
        )

    console.print(table)

    # Open findings / pending actions summary
    all_findings = store.load_all_findings()
    open_count = sum(1 for f in all_findings if f.status == FindingStatus.OPEN)
    critical_open = sum(
        1
        for f in all_findings
        if f.status == FindingStatus.OPEN and f.severity == AuditSeverity.CRITICAL
    )
    all_actions = store.load_all_actions()
    pending_count = sum(1 for a in all_actions if a.status == ActionStatus.PENDING)

    trend_summary = (
        f"\n  Open findings: [red]{open_count}[/red] (critical: [red bold]{critical_open}[/red bold])"
        f" | Pending actions: [yellow]{pending_count}[/yellow]"
    )
    console.print(trend_summary)


# ─── Status Change Commands ──────────────────────────────────────────


@app.command(name="resolve-finding")
def resolve_finding(
    finding_id: Annotated[int, typer.Argument(help="Finding ID to resolve")],
) -> None:
    """발견사항 해결 처리."""
    store = AuditStore()
    try:
        store.resolve_finding(finding_id)
    except FileNotFoundError:
        console.print(f"[red]Finding not found: {finding_id}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Finding #{finding_id} resolved.[/green]")


@app.command(name="update-action")
def update_action(
    action_id: Annotated[int, typer.Argument(help="Action ID")],
    status: Annotated[str, typer.Option("--status", "-s", help="New status")],
) -> None:
    """액션 상태 변경."""
    try:
        new_status = ActionStatus(status)
    except ValueError:
        console.print(f"[red]Invalid status: {status}[/red]")
        raise typer.Exit(code=1) from None

    store = AuditStore()
    try:
        store.update_action_status(action_id, new_status)
    except FileNotFoundError:
        console.print(f"[red]Action not found: {action_id}[/red]")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Action #{action_id} → {new_status}[/green]")


# ─── Display helpers ─────────────────────────────────────────────────


def _print_snapshot_detail(store: AuditStore, snapshot: AuditSnapshot) -> None:
    """스냅샷 상세를 Rich Panel로 출력."""
    s = snapshot
    lines = [
        f"[bold]Date:[/bold] {s.date}",
        f"[bold]Git SHA:[/bold] {s.git_sha or '-'}",
        f"[bold]Auditor:[/bold] {s.auditor}",
        f"[bold]Scope:[/bold] {', '.join(c.value for c in s.scope) if s.scope else '-'}",
    ]
    console.print(Panel("\n".join(lines), title=f"Audit Snapshot {s.date}"))

    # Grades
    g = s.grades
    grade_line = (
        f"  Arch: {g.architecture or '-'} | Risk: {g.risk_safety or '-'}"
        f" | Quality: {g.code_quality or '-'} | Data: {g.data_pipeline or '-'}"
        f" | Test/Ops: {g.testing_ops or '-'} | [bold]Overall: {g.overall or '-'}[/bold]"
    )
    console.print(f"\n[bold]Grades:[/bold]\n{grade_line}")

    # Metrics
    m = s.metrics
    metrics_line = (
        f"\n[bold]Metrics:[/bold]"
        f"\n  Tests: {m.test_count} | Pass: {m.test_pass_rate:.0%}"
        f" | Lint: {m.lint_errors} | Type: {m.type_errors} | Coverage: {m.coverage_pct:.0%}"
    )
    console.print(metrics_line)

    # Module health
    if s.module_health:
        ht = Table(show_header=True, header_style="bold", title="Module Health")
        ht.add_column("Module", min_width=16)
        ht.add_column("Health", justify="center", width=8)
        ht.add_column("Coverage", justify="right", width=9)
        ht.add_column("Notes", min_width=20)

        for mh in s.module_health:
            color = _HEALTH_COLORS.get(mh.health, "white")
            cov = f"{mh.coverage_pct:.0%}" if mh.coverage_pct is not None else "-"
            ht.add_row(mh.module, f"[{color}]{mh.health}[/{color}]", cov, mh.notes)

        console.print()
        console.print(ht)

    # Strategy summary
    ss = s.strategy_summary
    strat_line = (
        f"\n[bold]Strategies:[/bold]"
        f" Total: {ss.total} | Active: {ss.active} | Testing: {ss.testing}"
        f" | Candidate: {ss.candidate} | Retired: {ss.retired}"
    )
    console.print(strat_line)

    # Summary
    if s.summary:
        console.print(f"\n[bold]Summary:[/bold]\n{s.summary}")

    # Related findings/actions
    if s.new_findings:
        console.print(f"\n[bold]New Findings:[/bold] {s.new_findings}")
        new_findings = []
        for fid in s.new_findings:
            try:
                new_findings.append(store.load_finding(fid))
            except FileNotFoundError:
                continue
        if new_findings:
            _print_findings_table(new_findings)

    if s.new_actions:
        console.print(f"\n[bold]New Actions:[/bold] {s.new_actions}")


def _print_findings_table(records: list[Finding]) -> None:
    """발견사항 목록을 Rich Table로 출력."""
    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Findings ({len(records)})",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Severity", width=10)
    table.add_column("Status", width=12)
    table.add_column("Category", width=14)
    table.add_column("Title", min_width=20)
    table.add_column("Location", width=20)

    for f in records:
        sev_color = _SEVERITY_COLORS.get(f.severity, "white")
        status_color = _STATUS_COLORS.get(f.status, "white")
        table.add_row(
            str(f.id),
            f"[{sev_color}]{f.severity}[/{sev_color}]",
            f"[{status_color}]{f.status}[/{status_color}]",
            f.category.value,
            f.title,
            f.location or "-",
        )

    console.print(table)


def _print_actions_table(records: list[ActionItem]) -> None:
    """액션 목록을 Rich Table로 출력."""
    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Actions ({len(records)})",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Priority", width=5)
    table.add_column("Status", width=12)
    table.add_column("Phase", width=6)
    table.add_column("Title", min_width=20)
    table.add_column("Effort", width=8)
    table.add_column("Findings", width=10)

    for a in records:
        pri_color = _PRIORITY_COLORS.get(a.priority, "white")
        status_color = _ACTION_STATUS_COLORS.get(a.status, "white")
        findings_str = (
            ", ".join(str(fid) for fid in a.related_findings) if a.related_findings else "-"
        )
        table.add_row(
            str(a.id),
            f"[{pri_color}]{a.priority}[/{pri_color}]",
            f"[{status_color}]{a.status}[/{status_color}]",
            a.phase or "-",
            a.title,
            a.estimated_effort or "-",
            findings_str,
        )

    console.print(table)
