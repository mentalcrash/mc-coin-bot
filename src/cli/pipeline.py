"""Typer CLI for Strategy Pipeline Management.

Commands:
    - status: 전략 현황 요약 (상태별 카운트)
    - list: 전략 목록 (필터링)
    - show: 전략 상세 정보
    - create: 새 전략 YAML 생성 (CANDIDATE 상태)
    - record: Phase 결과 기록
    - update-status: 전략 상태 변경
    - report: Dashboard 자동 생성
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
    PhaseId,
    PhaseResult,
    PhaseVerdict,
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

_PHASE_DISPLAY = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def _phase_badge_colored(record: StrategyRecord, pid: PhaseId) -> str:
    """Phase 결과를 Rich 색상 문자로 변환."""
    result = record.phases.get(pid)
    if result is None:
        return "[dim]-[/dim]"
    if result.status == PhaseVerdict.PASS:
        return "[green]P[/green]"
    return "[red]F[/red]"


# ─── Commands ────────────────────────────────────────────────────────


@app.command()
def status() -> None:
    """전략 현황 요약 (상태별 카운트)."""
    store = StrategyStore()
    records = store.load_all()

    if not records:
        console.print("[yellow]No strategies found.[/yellow]")
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

    # TESTING breakdown by next_phase
    testing = [r for r in records if r.meta.status == StrategyStatus.TESTING]
    if testing:
        breakdown: dict[str, int] = {}
        for r in testing:
            np = r.next_phase or "DONE"
            breakdown[np] = breakdown.get(np, 0) + 1
        parts = [f"  {phase}: {cnt}" for phase, cnt in sorted(breakdown.items())]
        console.print("\n[yellow]TESTING Breakdown:[/yellow]")
        for part in parts:
            console.print(part)


@app.command(name="list")
def list_strategies(
    status_filter: Annotated[
        str | None, typer.Option("--status", "-s", help="Filter by status")
    ] = None,
    phase: Annotated[str | None, typer.Option("--phase", "-g", help="Filter by current phase")] = None,
    verdict: Annotated[
        str | None, typer.Option("--verdict", "-v", help="PASS or FAIL at phase")
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

    if phase and verdict:
        pid = PhaseId(phase)
        pv = PhaseVerdict(verdict)
        records = [r for r in records if pid in r.phases and r.phases[pid].status == pv]
    elif phase:
        pid = PhaseId(phase)
        records = [r for r in records if r.current_phase == pid]

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


_P1_ITEM_MIN_SCORE = 1
_P1_ITEM_MAX_SCORE = 5


def _load_p1_criteria() -> tuple[int, int, list[str]]:
    """phase-criteria.yaml에서 P1 기준 로드.

    Returns:
        (pass_threshold, max_total, item_names) 튜플
    """
    from src.pipeline.phase_criteria_store import PhaseCriteriaStore

    try:
        store = PhaseCriteriaStore()
        p1 = store.load("P1")
        scoring = p1.scoring
        if scoring is None:
            return 21, 30, []
        return scoring.pass_threshold, scoring.max_total, [i.name for i in scoring.items]
    except (FileNotFoundError, KeyError):
        return 21, 30, []


def _validate_p1_items(
    items_json: str,
    item_names: list[str],
) -> dict[str, int]:
    """--p1-items JSON 파싱 + 항목명/범위 검증.

    Args:
        items_json: JSON 문자열
        item_names: phase-criteria.yaml에서 로드한 항목명 목록

    Returns:
        검증된 {항목명: 점수} 딕셔너리

    Raises:
        typer.Exit: 검증 실패 시 exit(1)
    """
    import json

    try:
        items: dict[str, object] = json.loads(items_json)
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON for --p1-items[/red]")
        raise typer.Exit(code=1) from None

    # 항목명 검증
    if item_names:
        expected = set(item_names)
        provided = set(items.keys())
        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            msg_parts = []
            if missing:
                msg_parts.append(f"누락: {', '.join(sorted(missing))}")
            if extra:
                msg_parts.append(f"불일치: {', '.join(sorted(extra))}")
            console.print(f"[red]P1 항목 불일치 — {'; '.join(msg_parts)}[/red]")
            raise typer.Exit(code=1)

    # 점수 범위 검증 (1-5)
    for item_name, item_score in items.items():
        if (
            not isinstance(item_score, int)
            or item_score < _P1_ITEM_MIN_SCORE
            or item_score > _P1_ITEM_MAX_SCORE
        ):
            console.print(
                f"[red]Invalid score for '{item_name}': {item_score} (valid: {_P1_ITEM_MIN_SCORE}~{_P1_ITEM_MAX_SCORE})[/red]"
            )
            raise typer.Exit(code=1)

    # After validation, all values are int
    return {k: int(v) for k, v in items.items()}  # type: ignore[arg-type]


def _build_p1_v2_details(
    items: dict[str, int],
    pass_threshold: int,
    max_total: int,
) -> tuple[PhaseVerdict, dict[str, object], str]:
    """v2 항목별 점수 → verdict, details, rationale_text 생성."""
    total_score = sum(items.values())
    verdict = PhaseVerdict.PASS if total_score >= pass_threshold else PhaseVerdict.FAIL
    details: dict[str, object] = {
        "version": 2,
        "score": total_score,
        "max_score": max_total,
        "items": items,
    }

    # 항목별 점수 테이블 출력
    score_table = Table(show_header=True, header_style="bold")
    score_table.add_column("항목", min_width=16)
    score_table.add_column("점수", justify="right", width=6)
    for iname, iscore in items.items():
        score_table.add_row(iname, f"{iscore}/{_P1_ITEM_MAX_SCORE}")
    verdict_str = "[green]PASS[/green]" if verdict == PhaseVerdict.PASS else "[red]FAIL[/red]"
    score_table.add_row(
        "[bold]합계[/bold]",
        f"[bold]{total_score}/{max_total}[/bold] ({verdict_str} ≥ {pass_threshold})",
    )
    console.print(score_table)

    return verdict, details, f"{total_score}/{max_total}점 (v2)"


@app.command()
def create(
    name: Annotated[str, typer.Argument(help="Strategy name (kebab-case)")],
    display_name: Annotated[str, typer.Option("--display-name", help="표시 이름")],
    category: Annotated[str, typer.Option("--category", help="전략 카테고리")],
    timeframe: Annotated[str, typer.Option("--timeframe", "--tf", help="타임프레임")],
    short_mode: Annotated[str, typer.Option("--short-mode", help="DISABLED|HEDGE_ONLY|FULL")],
    rationale: Annotated[str, typer.Option("--rationale", "-r", help="경제적 논거")] = "",
    p1_score: Annotated[int, typer.Option("--p1-score", help="Phase 1 점수 (v1 레거시)")] = 0,
    p1_items: Annotated[
        str | None,
        typer.Option(
            "--p1-items",
            help='항목별 점수 JSON (e.g., \'{"경제적 논거 고유성":4,"IC 사전 검증":5,...}\')',
        ),
    ] = None,
    rationale_category: Annotated[
        str | None, typer.Option("--rationale-category", help="학술 근거 카테고리")
    ] = None,
) -> None:
    """새 전략 YAML 생성 (CANDIDATE 상태)."""
    store = StrategyStore()
    if store.exists(name):
        console.print(f"[red]Already exists: {name}[/red]")
        raise typer.Exit(code=1)

    pass_threshold, max_total, item_names = _load_p1_criteria()

    if p1_items is not None:
        items = _validate_p1_items(p1_items, item_names)
        verdict, details, rationale_text = _build_p1_v2_details(items, pass_threshold, max_total)
        total_score = sum(items.values())
    else:
        # v1 레거시: --p1-score
        if p1_score < 0 or p1_score > max_total:
            console.print(f"[red]Invalid P1 score: {p1_score} (valid range: 0~{max_total})[/red]")
            raise typer.Exit(code=1)

        total_score = p1_score
        verdict = PhaseVerdict.PASS if total_score >= pass_threshold else PhaseVerdict.FAIL
        details = {"score": total_score, "max_score": max_total}
        rationale_text = f"{total_score}/{max_total}점"

    # 동일 rationale_category RETIRED 전략 경고
    if rationale_category:
        _warn_low_category_success(store, rationale_category)

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
            rationale_category=rationale_category,
        ),
        phases={
            PhaseId.P1: PhaseResult(
                status=verdict,
                date=today,
                details=details,
            ),
        },
        decisions=[
            Decision(
                date=today,
                phase=PhaseId.P1,
                verdict=verdict,
                rationale=rationale_text,
            ),
        ],
    )
    store.save(record_obj)

    if verdict == PhaseVerdict.FAIL:
        console.print(
            f"[yellow]Created: strategies/{name}.yaml (CANDIDATE) — P1 FAIL ({total_score}/{max_total})[/yellow]"
        )
    else:
        console.print(f"[green]Created: strategies/{name}.yaml (CANDIDATE)[/green]")


def _warn_low_category_success(store: StrategyStore, category: str) -> None:
    """동일 rationale_category의 RETIRED 전략이 있으면 경고."""
    all_records = store.load_all()
    retired_same = [
        r
        for r in all_records
        if r.meta.status == StrategyStatus.RETIRED and r.meta.rationale_category == category
    ]
    active_same = [
        r
        for r in all_records
        if r.meta.status == StrategyStatus.ACTIVE and r.meta.rationale_category == category
    ]
    if retired_same:
        total = len(retired_same) + len(active_same)
        success_rate = len(active_same) / total * 100 if total > 0 else 0
        names = ", ".join(r.meta.name for r in retired_same[:3])
        msg = (
            f"[yellow]WARNING: '{category}' 카테고리 성공률 {success_rate:.0f}% "
            + f"({len(retired_same)}개 RETIRED: {names})[/yellow]"
        )
        console.print(msg)


@app.command()
def record(
    name: Annotated[str, typer.Argument(help="Strategy name")],
    phase: Annotated[str, typer.Option("--phase", "-g", help="Phase ID (P1, P2, ...)")],
    verdict: Annotated[str, typer.Option("--verdict", "-v", help="PASS or FAIL")],
    rationale: Annotated[str, typer.Option("--rationale", "-r", help="판정 사유")] = "",
    detail: Annotated[
        list[str] | None, typer.Option("--detail", "-d", help="key=value pairs")
    ] = None,
    no_retire: Annotated[
        bool, typer.Option("--no-retire", help="FAIL 시 자동 RETIRED 방지")
    ] = False,
) -> None:
    """Phase 결과 기록."""
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

    pid = PhaseId(phase)
    pv = PhaseVerdict(verdict)
    store.record_phase(name, pid, pv, details=details, rationale=rationale)
    console.print(f"[green]Recorded {phase} {verdict} for {name}[/green]")

    if pv == PhaseVerdict.FAIL and not no_retire:
        store.update_status(name, StrategyStatus.RETIRED)
        console.print("[yellow]Status → RETIRED[/yellow]")
    elif pv == PhaseVerdict.FAIL and no_retire:
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
def report(
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Save markdown to file instead of console"),
    ] = None,
) -> None:
    """전략 상황판 출력. --output FILE 시 markdown 저장."""
    from pathlib import Path

    from src.pipeline.lesson_store import LessonStore
    from src.pipeline.phase_criteria_store import PhaseCriteriaStore

    store = StrategyStore()
    lesson_store = LessonStore()
    phase_store = PhaseCriteriaStore()

    if output:
        from src.pipeline.report import DashboardGenerator

        generator = DashboardGenerator(store, lesson_store=lesson_store, phase_store=phase_store)
        content = generator.generate()
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        console.print(f"[green]Dashboard saved: {out_path}[/green]")
    else:
        from src.pipeline.report import ConsoleRenderer

        renderer = ConsoleRenderer(
            store, lesson_store=lesson_store, phase_store=phase_store, console=console
        )
        renderer.render()


@app.command(name="table")
def full_table() -> None:
    """모든 전략의 현황을 Phase 진행도 표로 출력."""
    store = StrategyStore()
    records = store.load_all()

    if not records:
        console.print("[yellow]No strategies found.[/yellow]")
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
    for pname in _PHASE_DISPLAY:
        table.add_column(pname, justify="center", width=3)
    table.add_column("Next", justify="center", width=4)

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

        phase_cells = [_phase_badge_colored(r, PhaseId(p)) for p in _PHASE_DISPLAY]
        next_phase_str = r.next_phase if r.next_phase else "-"

        table.add_row(
            str(i),
            r.meta.display_name,
            r.meta.timeframe,
            f"[{color}]{r.meta.status}[/{color}]",
            sharpe_str,
            cagr_str,
            mdd_str,
            best_asset_str,
            *phase_cells,
            next_phase_str,
        )

    console.print()
    console.print(table)

    # Summary
    active = sum(1 for r in records if r.meta.status == StrategyStatus.ACTIVE)
    retired = sum(1 for r in records if r.meta.status == StrategyStatus.RETIRED)
    summary = f"\n  [green]ACTIVE: {active}[/green] | [red]RETIRED: {retired}[/red] | Total: {len(records)}"
    console.print(summary)


@app.command(name="retired-analysis")
def retired_analysis() -> None:
    """RETIRED 전략 실패 패턴 분석."""
    store = StrategyStore()
    all_records = store.load_all()
    retired = [r for r in all_records if r.meta.status == StrategyStatus.RETIRED]
    active = [r for r in all_records if r.meta.status == StrategyStatus.ACTIVE]

    if not retired:
        console.print("[yellow]No RETIRED strategies found.[/yellow]")
        return

    # 1) Phase별 FAIL 분포
    phase_fail_counts: dict[str, int] = {}
    for r in retired:
        fp = r.fail_phase
        if fp:
            phase_fail_counts[fp] = phase_fail_counts.get(fp, 0) + 1
        else:
            phase_fail_counts["N/A"] = phase_fail_counts.get("N/A", 0) + 1

    total_retired = len(retired)
    phase_table = Table(title=f"Phase별 FAIL 분포 ({total_retired} RETIRED)")
    phase_table.add_column("Phase", style="bold")
    phase_table.add_column("Count", justify="right")
    phase_table.add_column("%", justify="right")

    for phase, count in sorted(phase_fail_counts.items(), key=lambda x: -x[1]):
        pct = count / total_retired * 100
        phase_table.add_row(phase, str(count), f"{pct:.0f}%")

    console.print(phase_table)

    # 2) 카테고리별 성공률
    cat_retired: dict[str, int] = {}
    cat_active: dict[str, int] = {}
    for r in retired:
        cat = r.meta.rationale_category or "N/A"
        cat_retired[cat] = cat_retired.get(cat, 0) + 1
    for r in active:
        cat = r.meta.rationale_category or "N/A"
        cat_active[cat] = cat_active.get(cat, 0) + 1

    all_cats = sorted(set(cat_retired) | set(cat_active))
    if all_cats:
        cat_table = Table(title="카테고리별 성공률")
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Active", justify="right", style="green")
        cat_table.add_column("Retired", justify="right", style="red")
        cat_table.add_column("Rate", justify="right")

        for cat in all_cats:
            n_active = cat_active.get(cat, 0)
            n_retired = cat_retired.get(cat, 0)
            total = n_active + n_retired
            rate = n_active / total * 100 if total > 0 else 0
            cat_table.add_row(cat, str(n_active), str(n_retired), f"{rate:.0f}%")

        console.print(cat_table)

    # 3) 최근 실패 Top 10
    by_date = sorted(retired, key=lambda r: r.meta.retired_at or r.meta.created_at, reverse=True)
    recent = by_date[:10]
    recent_table = Table(title="최근 RETIRED Top 10")
    recent_table.add_column("Name", style="bold")
    recent_table.add_column("Phase", justify="center")
    recent_table.add_column("Category")
    recent_table.add_column("Date")

    for r in recent:
        recent_table.add_row(
            r.meta.name,
            r.fail_phase or "-",
            r.meta.rationale_category or "-",
            str(r.meta.retired_at or r.meta.created_at),
        )

    console.print(recent_table)


# ─── Lessons commands ────────────────────────────────────────────────


@app.command(name="lessons-list")
def lessons_list(
    category: Annotated[
        str | None, typer.Option("--category", "-c", help="Filter by category")
    ] = None,
    tag: Annotated[str | None, typer.Option("--tag", "-t", help="Filter by tag")] = None,
    strategy: Annotated[
        str | None, typer.Option("--strategy", "-s", help="Filter by strategy")
    ] = None,
    timeframe: Annotated[str | None, typer.Option("--tf", help="Filter by timeframe")] = None,
) -> None:
    """교훈 목록 (필터링 가능)."""
    from src.pipeline.lesson_models import LessonCategory
    from src.pipeline.lesson_store import LessonStore

    store = LessonStore()

    if category:
        try:
            cat = LessonCategory(category)
        except ValueError:
            console.print(f"[red]Invalid category: {category}[/red]")
            console.print(f"Valid: {', '.join(c.value for c in LessonCategory)}")
            raise typer.Exit(code=1) from None
        records = store.filter_by_category(cat)
    elif tag:
        records = store.filter_by_tag(tag)
    elif strategy:
        records = store.filter_by_strategy(strategy)
    elif timeframe:
        records = store.filter_by_timeframe(timeframe)
    else:
        records = store.load_all()

    if not records:
        console.print("[yellow]No lessons match the given filters.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold", title=f"Lessons ({len(records)})")
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="bold", min_width=20)
    table.add_column("Category", width=18)
    table.add_column("Tags")

    for r in records:
        table.add_row(
            str(r.id),
            r.title,
            r.category.value,
            ", ".join(r.tags),
        )

    console.print(table)


@app.command(name="lessons-show")
def lessons_show(
    lesson_id: Annotated[int, typer.Argument(help="Lesson ID")],
) -> None:
    """교훈 상세 정보."""
    from src.pipeline.lesson_store import LessonStore

    store = LessonStore()
    try:
        record = store.load(lesson_id)
    except FileNotFoundError:
        console.print(f"[red]Lesson not found: {lesson_id}[/red]")
        raise typer.Exit(code=1) from None

    lines = [
        f"[bold]#{record.id}[/bold] {record.title}",
        "",
        record.body,
        "",
        f"[bold]Category:[/bold] {record.category.value}",
        f"[bold]Tags:[/bold] {', '.join(record.tags) if record.tags else '-'}",
        f"[bold]Strategies:[/bold] {', '.join(record.strategies) if record.strategies else '-'}",
        f"[bold]Timeframes:[/bold] {', '.join(record.timeframes) if record.timeframes else '-'}",
        f"[bold]Added:[/bold] {record.added_at}",
    ]
    console.print(Panel("\n".join(lines), title=f"Lesson #{record.id}"))


@app.command(name="lessons-add")
def lessons_add(
    title: Annotated[str, typer.Option("--title", help="교훈 제목")],
    body: Annotated[str, typer.Option("--body", help="상세 설명")],
    category: Annotated[str, typer.Option("--category", "-c", help="카테고리")],
    tag: Annotated[list[str] | None, typer.Option("--tag", "-t", help="태그 (복수 가능)")] = None,
    strategy: Annotated[
        list[str] | None, typer.Option("--strategy", "-s", help="관련 전략 (복수 가능)")
    ] = None,
    timeframe: Annotated[list[str] | None, typer.Option("--tf", help="관련 TF (복수 가능)")] = None,
) -> None:
    """새 교훈 추가 (next_id 자동)."""
    from src.pipeline.lesson_models import LessonCategory, LessonRecord
    from src.pipeline.lesson_store import LessonStore

    try:
        cat = LessonCategory(category)
    except ValueError:
        console.print(f"[red]Invalid category: {category}[/red]")
        console.print(f"Valid: {', '.join(c.value for c in LessonCategory)}")
        raise typer.Exit(code=1) from None

    store = LessonStore()
    new_id = store.next_id()

    record = LessonRecord(
        id=new_id,
        title=title,
        body=body,
        category=cat,
        tags=tag or [],
        strategies=strategy or [],
        timeframes=timeframe or [],
        added_at=date.today(),
    )
    store.save(record)
    console.print(f"[green]Created: lessons/{new_id:03d}.yaml — {title}[/green]")


# ─── Phase criteria commands ─────────────────────────────────────────


@app.command(name="phases-list")
def phases_list() -> None:
    """Phase 평가 기준 요약 테이블."""
    from src.pipeline.phase_criteria_store import PhaseCriteriaStore

    store = PhaseCriteriaStore()
    try:
        phases = store.load_all()
    except FileNotFoundError:
        console.print("[red]gates/phase-criteria.yaml not found.[/red]")
        raise typer.Exit(code=1) from None

    table = Table(
        show_header=True,
        header_style="bold",
        title=f"Phase Criteria ({len(phases)} phases)",
    )
    table.add_column("Phase", style="bold", width=6)
    table.add_column("Name", min_width=14)
    table.add_column("Type", width=10)
    table.add_column("CLI")

    for p in phases:
        cli = p.cli_command if p.cli_command else "-"
        table.add_row(p.phase_id, p.name, p.phase_type.value, cli)

    console.print(table)


@app.command(name="phases-show")
def phases_show(
    phase_id: Annotated[str, typer.Argument(help="Phase ID (P1, P2, ...)")],
) -> None:
    """Phase 상세 기준 표시."""
    from src.pipeline.phase_criteria_models import PhaseType, Severity
    from src.pipeline.phase_criteria_store import PhaseCriteriaStore

    store = PhaseCriteriaStore()
    try:
        p = store.load(phase_id)
    except (FileNotFoundError, KeyError):
        console.print(f"[red]Phase not found: {phase_id}[/red]")
        raise typer.Exit(code=1) from None

    lines = [
        f"[bold]{p.phase_id}[/bold] — {p.name}",
        "",
        p.description,
        "",
        f"[bold]Type:[/bold] {p.phase_type.value}",
    ]
    if p.cli_command:
        lines.append(f"[bold]CLI:[/bold] {p.cli_command}")

    if p.phase_type == PhaseType.SCORING and p.scoring:
        lines.append(f"\n[bold]PASS:[/bold] >= {p.scoring.pass_threshold}/{p.scoring.max_total}")
        lines.extend(f"  - {item.name}: {item.description}" for item in p.scoring.items)

    elif p.phase_type == PhaseType.CHECKLIST and p.checklist:
        lines.append(f"\n[bold]PASS:[/bold] {p.checklist.pass_rule}")
        for item in p.checklist.items:
            c = "red" if item.severity == Severity.CRITICAL else "yellow"
            lines.append(f"  [{c}]{item.code}[/{c}] {item.name}: {item.description}")

    elif p.phase_type == PhaseType.THRESHOLD and p.threshold:
        lines.append("\n[bold]PASS Metrics:[/bold]")
        lines.extend(
            f"  - {m.name} {m.operator} {int(m.value) if m.value == int(m.value) else m.value}{m.unit}"
            for m in p.threshold.pass_metrics
        )
        if p.threshold.immediate_fail:
            lines.append("\n[bold]Immediate FAIL:[/bold]")
            lines.extend(
                f"  - {rule.condition} → {rule.reason}" for rule in p.threshold.immediate_fail
            )

    console.print(Panel("\n".join(lines), title=f"Phase {p.phase_id}"))


# ─── P1 check command ────────────────────────────────────────────────


@app.command(name="p1-check")
def p1_check(
    indicator_name: Annotated[
        str,
        typer.Argument(help="Indicator name (e.g., rsi, momentum, efficiency_ratio)"),
    ],
    symbol: Annotated[
        str,
        typer.Argument(help="Trading symbol (e.g., BTC/USDT)"),
    ] = "BTC/USDT",
    timeframe: Annotated[str, typer.Option("--tf", help="Timeframe")] = "1D",
    year: Annotated[
        list[int],
        typer.Option("--year", "-y", help="Year(s)"),
    ] = [2020, 2021, 2022, 2023, 2024, 2025],  # noqa: B006
    param: Annotated[
        list[str] | None,
        typer.Option("--param", "-p", help="Indicator param key=value (e.g., period=14)"),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", "-c", help="Rationale category for success rate scoring"),
    ] = None,
) -> None:
    """P1 데이터 기반 항목 자동 점수 계산 도우미.

    IC 사전 검증, 카테고리 성공률, 레짐 독립성 3개 항목의
    추천 점수를 자동 계산합니다.

    Example:
        uv run mcbot pipeline p1-check rsi BTC/USDT --tf 1D -p period=14 --category momentum
    """
    import json

    from src.pipeline.p1_helpers import (
        compute_category_success_score,
        compute_ic_score,
        compute_regime_independence_score,
    )

    # 파라미터 파싱
    params: dict[str, object] = {}
    for p in param or []:
        key, _, val = p.partition("=")
        try:
            params[key] = int(val)
        except ValueError:
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val

    console.print(
        Panel(
            (
                f"[bold]P1 Data-Based Score Check[/bold]\n"
                f"Indicator: {indicator_name}\n"
                f"Symbol: {symbol} | TF: {timeframe}\n"
                f"Params: {params or 'default'}"
            ),
            border_style="cyan",
        )
    )

    # 데이터 로드
    from datetime import UTC, datetime

    from src.config.settings import get_settings
    from src.core.exceptions import DataNotFoundError
    from src.core.logger import setup_logger
    from src.data.market_data import MarketDataRequest
    from src.data.service import MarketDataService

    setup_logger(console_level="WARNING")

    try:
        settings = get_settings()
        data_service = MarketDataService(settings)
        start_date = datetime(min(year), 1, 1, tzinfo=UTC)
        end_date = datetime(max(year), 12, 31, 23, 59, 59, tzinfo=UTC)
        data = data_service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            ),
        )
    except DataNotFoundError as e:
        console.print(f"[red]Data load failed: {e}[/red]")
        raise typer.Exit(code=1) from e

    # 지표 계산
    from src.market.feature_store import IndicatorSpec, compute_indicator

    spec = IndicatorSpec(name=indicator_name, params=dict(params))
    try:
        indicator_series = compute_indicator(spec, data.ohlcv)
    except AttributeError:
        console.print(f"[red]Indicator not found: {indicator_name}[/red]")
        raise typer.Exit(code=1) from None

    # Forward returns
    close_series = data.ohlcv["close"]
    forward_returns = close_series.pct_change().shift(-1)

    # 1) IC 점수
    from src.backtest.ic_analyzer import ICAnalyzer

    ic_corr, _ = ICAnalyzer.rank_ic(indicator_series, forward_returns)  # type: ignore[arg-type]
    ic_score = compute_ic_score(ic_corr)

    scores = [ic_score]

    # 2) 카테고리 성공률
    if category:
        cat_score = compute_category_success_score(category, StrategyStore())
        scores.append(cat_score)

    # 3) 레짐 독립성
    from src.regime.config import RegimeLabel
    from src.regime.detector import RegimeDetector

    detector = RegimeDetector()
    regime_df = detector.classify_series(close_series)  # type: ignore[arg-type]

    from src.pipeline.p1_helpers import MIN_REGIME_SAMPLES

    regime_ics: dict[str, float] = {}
    for label in RegimeLabel:
        mask = regime_df["regime_label"] == label
        if mask.sum() > MIN_REGIME_SAMPLES:
            subset_ind = indicator_series[mask]
            subset_ret = forward_returns[mask]
            regime_ic, _ = ICAnalyzer.rank_ic(subset_ind, subset_ret)  # type: ignore[arg-type]
            regime_ics[label.value] = regime_ic

    if regime_ics:
        regime_score = compute_regime_independence_score(regime_ics)
        scores.append(regime_score)

    # 결과 테이블
    result_table = Table(title="P1 Data-Based Scores", show_header=True, header_style="bold")
    result_table.add_column("항목", min_width=16)
    result_table.add_column("점수", justify="right", width=6)
    result_table.add_column("근거")

    items_json: dict[str, int] = {}
    for s in scores:
        result_table.add_row(s.item_name, f"{s.score}/5", s.reason)
        items_json[s.item_name] = s.score

    console.print(result_table)

    # --p1-items 복사 가능 JSON
    console.print(
        f"\n[bold]--p1-items JSON:[/bold]\n'{json.dumps(items_json, ensure_ascii=False)}'"
    )


# ─── Phase runner commands ───────────────────────────────────────────


@app.command(name="phase4-run")
def phase4_run(
    strategies: Annotated[list[str], typer.Argument(help="전략 이름 (복수)")],
    symbols: Annotated[
        str, typer.Option("--symbols", help="쉼표 구분 심볼")
    ] = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,DOGE/USDT",
    start: Annotated[str, typer.Option("--start", help="시작일 (YYYY-MM-DD)")] = "2020-01-01",
    end: Annotated[str, typer.Option("--end", help="종료일 (YYYY-MM-DD)")] = "2025-12-31",
    capital: Annotated[int, typer.Option("--capital", help="초기 자본")] = 100_000,
    save_json: Annotated[bool, typer.Option("--json/--no-json", help="JSON 결과 저장")] = True,
    parallel: Annotated[
        bool, typer.Option("--parallel/--no-parallel", help="심볼 간 병렬 실행")
    ] = True,
) -> None:
    """Phase 4: 5-coin x 6-year 단일에셋 백테스트 + YAML 자동 갱신."""
    from datetime import UTC, datetime

    from src.cli._phase_runners import run_phase4

    symbol_list = [s.strip() for s in symbols.split(",")]
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

    run_phase4(
        strategies=strategies,
        symbols=symbol_list,
        start=start_dt,
        end=end_dt,
        capital=capital,
        save_json=save_json,
        console=console,
        parallel=parallel,
    )


@app.command(name="phase5-run")
def phase5_run(
    strategies: Annotated[list[str], typer.Argument(help="전략 이름 (복수)")],
    n_trials: Annotated[int, typer.Option("--n-trials", "-n", help="Optuna trial 수")] = 100,
    seed: Annotated[int, typer.Option("--seed", help="재현성 seed")] = 42,
    save_json: Annotated[bool, typer.Option("--json/--no-json", help="JSON 결과 저장")] = True,
) -> None:
    """Phase 5: Optuna TPE 파라미터 최적화 (IS only, Always PASS)."""
    from src.cli._phase_runners_p5 import run_phase5

    run_phase5(
        strategies=strategies,
        n_trials=n_trials,
        seed=seed,
        save_json=save_json,
        console=console,
    )


@app.command(name="phase5-stability")
def phase5_stability(
    strategies: Annotated[
        list[str] | None, typer.Argument(help="전략 이름 (미지정 시 전체)")
    ] = None,
    save_json: Annotated[bool, typer.Option("--json/--no-json", help="JSON 결과 저장")] = True,
) -> None:
    """Phase 5: 파라미터 안정성 검증 (plateau + ±20% stability) + YAML 자동 갱신."""
    from src.cli._phase_runners import run_phase5_stability

    run_phase5_stability(strategies=strategies, save_json=save_json, console=console)


# ─── Display helpers ─────────────────────────────────────────────────


def _print_strategy_table(records: list[StrategyRecord]) -> None:
    """전략 목록을 Rich Table로 출력."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="bold")
    table.add_column("TF")
    table.add_column("Status")
    table.add_column("Phase")
    table.add_column("Sharpe", justify="right")
    table.add_column("Best Asset")

    for r in records:
        color = _STATUS_COLORS.get(r.meta.status, "white")
        sharpe = f"{r.best_sharpe:.2f}" if r.best_sharpe is not None else "-"
        phase = str(r.current_phase) if r.current_phase else "-"
        if r.fail_phase:
            phase = f"{r.fail_phase} [red]FAIL[/red]"
        elif r.current_phase and r.next_phase:
            phase = f"{r.current_phase} → {r.next_phase}"
        table.add_row(
            r.meta.display_name,
            r.meta.timeframe,
            f"[{color}]{r.meta.status}[/{color}]",
            phase,
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

    # Phase progress
    phase_line = "  ".join(f"{p}: {_phase_badge_colored(r, PhaseId(p))}" for p in _PHASE_DISPLAY)
    console.print(f"\n[bold]Phase Progress:[/bold] {phase_line}")

    # Next phase info
    if r.fail_phase:
        console.print(f"[red]Pipeline: BLOCKED at {r.fail_phase}[/red]\n")
    elif r.next_phase:
        console.print(f"[yellow]Next Phase: {r.next_phase}[/yellow]\n")
    else:
        console.print("[green]Pipeline: COMPLETE[/green]\n")

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
        dt.add_column("Phase")
        dt.add_column("Verdict")
        dt.add_column("Rationale")
        for d in r.decisions:
            v_color = "green" if d.verdict == PhaseVerdict.PASS else "red"
            dt.add_row(str(d.date), str(d.phase), f"[{v_color}]{d.verdict}[/{v_color}]", d.rationale)
        console.print(dt)
