"""Dashboard markdown generator from YAML data.

StrategyStore에서 전략 데이터를 읽어 dashboard.md를 자동 생성.
정적 콘텐츠(파이프라인 다이어그램, Gate 기준, 비용 모델)와
동적 콘텐츠(활성/폐기 전략 테이블)를 결합.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.models import (
    GateId,
    GateVerdict,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.store import StrategyStore

_LESSONS_PATH = Path("docs/strategy/lessons.md")


class DashboardGenerator:
    """YAML 데이터 → dashboard.md 생성."""

    def __init__(self, store: StrategyStore) -> None:
        self.store = store

    def generate(self) -> str:
        """전체 dashboard markdown 생성."""
        records = self.store.load_all()
        active = [r for r in records if r.meta.status == StrategyStatus.ACTIVE]
        retired = [r for r in records if r.meta.status == StrategyStatus.RETIRED]

        sections = [
            self._header(len(records), len(active), len(retired)),
            self._pipeline_diagram(),
            self._gate_criteria_table(),
            self._cost_model_table(),
            self._strategy_summary(len(records), len(active), len(retired)),
            self._active_table(active),
            self._retired_sections(retired),
            self._lessons(),
        ]
        return "\n\n---\n\n".join(s for s in sections if s)

    # ─── Static sections ──────────────────────────────────────────

    def _header(self, total: int, active: int, retired: int) -> str:
        return (
            "# 전략 상황판 (Strategy Dashboard)\n\n"
            f"> {total}개 전략의 평가 현황과 검증 기준을 한눈에 파악하는 문서. "
            f"(활성 {active} + 폐기 {retired})\n"
            "> 개별 스코어카드는 [docs/scorecard/](../scorecard/)에, "
            "상세 평가 기준은 [전략 평가 표준](evaluation-standard.md)에 있다."
        )

    def _pipeline_diagram(self) -> str:
        return (
            "## 평가 파이프라인\n\n"
            "```\n"
            "Gate 0A → Gate 0B → Gate 1 → Gate 2 → Gate 3 → Gate 4 → Gate 5 → Gate 6 → Gate 7\n"
            "아이디어   코드검증  백테스트  IS/OOS   파라미터  심층검증   EDA     Paper   실전배포\n"
            "```"
        )

    def _gate_criteria_table(self) -> str:
        return (
            "### Gate별 통과 기준\n\n"
            "| Gate | 검증 | 핵심 기준 | CLI |\n"
            "|:----:|------|----------|-----|\n"
            "| **0A** | 아이디어 검증 | 6항목 합계 >= 18/30 | — |\n"
            "| **0B** | 코드 품질 검증 | Critical 7항목 결함 0개 | `/quant-code-audit` |\n"
            "| **1** | 단일에셋 백테스트 (5코인 x 6년) | Sharpe > 1.0, CAGR > 20%, MDD < 40%, Trades > 50 | `run {config}` |\n"
            "| **2** | IS/OOS 70/30 | OOS Sharpe >= 0.3, Decay < 50% | `validate -m quick` |\n"
            "| **3** | 파라미터 안정성 | 고원 존재, ±20% Sharpe 부호 유지 | `sweep {config}` |\n"
            "| **4** | WFA + CPCV + PBO + DSR | WFA OOS >= 0.5, PBO 이중 경로, DSR > 0.95 | `validate -m milestone/final` |\n"
            "| **5** | EDA Parity | VBT vs EDA 수익 부호 일치, 편차 < 20% | `eda run {config}` |\n"
            "| **6** | Paper Trading (2주+) | 시그널 일치 > 90%, 무중단 | `eda run-live` |\n"
            "| **7** | 실전 배포 | 3개월 이동 Sharpe > 0.3 | — |"
        )

    def _cost_model_table(self) -> str:
        return (
            "### 비용 모델\n\n"
            "| 항목 | 값 | 항목 | 값 |\n"
            "|------|---:|------|---:|\n"
            "| Maker Fee | 0.02% | Slippage | 0.05% |\n"
            "| Taker Fee | 0.04% | Funding (8h) | 0.01% |\n"
            "| Market Impact | 0.02% | **편도 합계** | **~0.11%** |"
        )

    # ─── Dynamic sections ─────────────────────────────────────────

    def _strategy_summary(self, total: int, active: int, retired: int) -> str:
        return f"## 현재 전략 현황 ({total}개 = 활성 {active} + 폐기 {retired})"

    def _active_table(self, active: list[StrategyRecord]) -> str:
        if not active:
            return "### 활성 전략 (0개)"

        lines = [
            f"### 활성 전략 ({len(active)}개, Gate 5 완료)\n",
            "| 전략 | Best Asset | TF | Sharpe | CAGR | MDD | Trades | G0 | G1 | G2 | G3 | G4 | G5 | 비고 |",
            "|------|-----------|-----|--------|------|-----|--------|:--:|:--:|:--:|:--:|:--:|:--:|------|",
        ]

        for r in sorted(active, key=lambda x: -(x.best_sharpe or 0)):
            best = max(r.asset_performance, key=lambda a: a.sharpe) if r.asset_performance else None
            scorecard_link = f"[**{r.meta.display_name}**](../scorecard/{r.meta.name}.md)"

            if best:
                gates_str = " | ".join(
                    _gate_letter(r, gid)
                    for gid in [GateId.G0A, GateId.G1, GateId.G2, GateId.G3, GateId.G4, GateId.G5]
                )
                note = _extract_note(r)
                cagr = f"+{best.cagr:.1f}%" if best.cagr > 0 else f"{best.cagr:.1f}%"
                row = (
                    f"| {scorecard_link} | {best.symbol} | {r.meta.timeframe} | "
                    + f"{best.sharpe:.2f} | {cagr} | -{best.mdd:.1f}% | {best.trades} | "
                    + f"{gates_str} | {note} |"
                )
                lines.append(row)

        return "\n".join(lines)

    def _retired_sections(self, retired: list[StrategyRecord]) -> str:
        """폐기 전략을 fail gate별로 분류."""
        sections: list[str] = [f"### 폐기 전략 ({len(retired)}개)\n"]
        gate_groups = self._group_retired(retired)
        self._append_group_tables(gate_groups, sections)
        return "\n\n".join(sections)

    def _group_retired(
        self,
        retired: list[StrategyRecord],
    ) -> dict[str, list[StrategyRecord]]:
        """폐기 전략을 fail gate별 그룹으로 분류."""
        groups: dict[str, list[StrategyRecord]] = {
            "G4": [],
            "G3": [],
            "G2": [],
            "G1_sharpe": [],
            "G1_negative": [],
            "G1_data": [],
            "G1_structural": [],
        }
        gate_map = {"G4": "G4", "G3": "G3", "G2": "G2"}
        for r in retired:
            fg = r.fail_gate
            if fg in gate_map:
                groups[gate_map[fg]].append(r)
            elif fg == "G1":
                self._classify_g1_fail(r, groups)
            else:
                groups["G1_structural"].append(r)
        return groups

    def _append_group_tables(
        self,
        groups: dict[str, list[StrategyRecord]],
        sections: list[str],
    ) -> None:
        """그룹별 테이블을 sections에 추가."""
        renderers = [
            ("G4", self._g4_fail_table),
            ("G3", self._g3_fail_table),
            ("G2", self._g2_fail_table),
            ("G1_sharpe", self._g1_sharpe_fail_table),
            ("G1_negative", self._g1_negative_fail_table),
            ("G1_data", self._g1_data_fail_table),
            ("G1_structural", self._g1_structural_fail_table),
        ]
        for key, renderer in renderers:
            if groups[key]:
                sections.append(renderer(groups[key]))

    def _classify_g1_fail(
        self,
        r: StrategyRecord,
        groups: dict[str, list[StrategyRecord]],
    ) -> None:
        """G1 FAIL 전략을 세부 카테고리로 분류."""
        if not r.asset_performance:
            groups["G1_data"].append(r)
            return

        best = max(r.asset_performance, key=lambda a: a.sharpe)

        # All negative or near-zero
        all_negative = all(a.sharpe <= 0 for a in r.asset_performance)
        if all_negative or best.sharpe < 0:
            groups["G1_negative"].append(r)
        elif best.sharpe < 1.0 and best.cagr > 0:
            groups["G1_sharpe"].append(r)
        else:
            groups["G1_structural"].append(r)

    def _g4_fail_table(self, records: list[StrategyRecord]) -> str:
        lines = [
            "#### Gate 4 실패 -- WFA 심층검증\n",
            "| 전략 | Sharpe | 사유 |",
            "|------|--------|------|",
        ]
        for r in sorted(records, key=lambda x: -(x.best_sharpe or 0)):
            link = f"[{r.meta.display_name}](../scorecard/fail/{r.meta.name}.md)"
            sharpe = f"{r.best_sharpe:.2f}" if r.best_sharpe else "-"
            note = _fail_rationale(r, GateId.G4)
            lines.append(f"| {link} | {sharpe} | {note} |")
        return "\n".join(lines)

    def _g3_fail_table(self, records: list[StrategyRecord]) -> str:
        lines = [
            "#### Gate 3 실패 -- 파라미터 불안정\n",
            "| 전략 | Sharpe | 사유 |",
            "|------|--------|------|",
        ]
        for r in records:
            link = f"[{r.meta.display_name}](../scorecard/fail/{r.meta.name}.md)"
            sharpe = f"{r.best_sharpe:.2f}" if r.best_sharpe else "-"
            note = _fail_rationale(r, GateId.G3)
            lines.append(f"| {link} | {sharpe} | {note} |")
        return "\n".join(lines)

    def _g2_fail_table(self, records: list[StrategyRecord]) -> str:
        lines = [
            f"#### Gate 2 실패 -- IS/OOS 과적합 ({len(records)}개)\n",
            "| 전략 | Sharpe | OOS Sharpe | Decay |",
            "|------|--------|-----------|-------|",
        ]
        for r in sorted(records, key=lambda x: -(x.best_sharpe or 0)):
            link = f"[{r.meta.display_name}](../scorecard/fail/{r.meta.name}.md)"
            sharpe = f"{r.best_sharpe:.2f}" if r.best_sharpe else "-"
            g2 = r.gates.get(GateId.G2)
            oos = f"{g2.details.get('oos_sharpe', '-')}" if g2 else "-"
            decay = f"{g2.details.get('decay', '-')}%" if g2 and "decay" in g2.details else "-"
            lines.append(f"| {link} | {sharpe} | {oos} | {decay} |")
        return "\n".join(lines)

    def _g1_sharpe_fail_table(self, records: list[StrategyRecord]) -> str:
        lines = [
            "#### Gate 1 실패 -- Sharpe/CAGR 미달\n",
            "| 전략 | Sharpe | CAGR | 사유 |",
            "|------|--------|------|------|",
        ]
        for r in sorted(records, key=lambda x: -(x.best_sharpe or 0)):
            link = f"[{r.meta.display_name}](../scorecard/fail/{r.meta.name}.md)"
            best = max(r.asset_performance, key=lambda a: a.sharpe) if r.asset_performance else None
            sharpe = f"{best.sharpe:.2f}" if best else "-"
            cagr = (
                f"+{best.cagr:.1f}%"
                if best and best.cagr > 0
                else (f"{best.cagr:.1f}%" if best else "-")
            )
            note = _fail_rationale(r, GateId.G1)
            lines.append(f"| {link} | {sharpe} | {cagr} | {note} |")
        return "\n".join(lines)

    def _g1_negative_fail_table(self, records: list[StrategyRecord]) -> str:
        lines = [
            "#### Gate 1 실패 -- 전 에셋 Sharpe 음수/0 근접\n",
            "| 전략 | Sharpe | CAGR | 사유 |",
            "|------|--------|------|------|",
        ]
        for r in sorted(records, key=lambda x: -(x.best_sharpe or 0)):
            link = f"[{r.meta.display_name}](../scorecard/fail/{r.meta.name}.md)"
            best = max(r.asset_performance, key=lambda a: a.sharpe) if r.asset_performance else None
            sharpe = f"{best.sharpe:.2f}" if best else "-"
            cagr = (
                f"+{best.cagr:.1f}%"
                if best and best.cagr > 0
                else (f"{best.cagr:.1f}%" if best else "-")
            )
            note = _fail_rationale(r, GateId.G1)
            lines.append(f"| {link} | {sharpe} | {cagr} | {note} |")
        return "\n".join(lines)

    def _g1_data_fail_table(self, records: list[StrategyRecord]) -> str:
        lines = [
            "#### Gate 1 실패 -- 데이터 부재 / 인프라 미구축\n",
            "| 전략 | G0 점수 | 사유 |",
            "|------|---------|------|",
        ]
        for r in records:
            link = f"[{r.meta.display_name}](../scorecard/fail/{r.meta.name}.md)"
            g0 = r.gates.get(GateId.G0A)
            score = f"{g0.details.get('score', '?')}/30" if g0 else "-"
            note = _fail_rationale(r, GateId.G1)
            lines.append(f"| {link} | {score} | {note} |")
        return "\n".join(lines)

    def _g1_structural_fail_table(self, records: list[StrategyRecord]) -> str:
        lines = [
            "#### Gate 1 실패 -- 구조적 결함\n",
            "| 전략 | Sharpe | 사유 |",
            "|------|--------|------|",
        ]
        for r in records:
            link = f"[{r.meta.display_name}](../scorecard/fail/{r.meta.name}.md)"
            sharpe = f"{r.best_sharpe:.2f}" if r.best_sharpe else "-"
            note = _fail_rationale(r, GateId.G1)
            lines.append(f"| {link} | {sharpe} | {note} |")
        return "\n".join(lines)

    def _lessons(self) -> str:
        """핵심 교훈 (별도 파일에서 로드)."""
        if _LESSONS_PATH.exists():
            return _LESSONS_PATH.read_text(encoding="utf-8").strip()
        return ""


# ─── Helpers ─────────────────────────────────────────────────────────


def _gate_letter(record: StrategyRecord, gid: GateId) -> str:
    """Gate 결과를 P/F/- 단일 문자로."""
    result = record.gates.get(gid)
    if result is None:
        return " "
    return "P" if result.status == GateVerdict.PASS else "F"


def _extract_note(record: StrategyRecord) -> str:
    """활성 전략의 비고."""
    g4 = record.gates.get(GateId.G4)
    if g4 and "note" in g4.details:
        return str(g4.details["note"])[:60]
    return ""


def _fail_rationale(record: StrategyRecord, gate: GateId) -> str:
    """FAIL 사유 추출."""
    result = record.gates.get(gate)
    if not result:
        return ""
    note = result.details.get("note", "")
    if note:
        return str(note)[:80]
    # Build from structured details
    parts = []
    if "sharpe" in result.details:
        parts.append(f"Sharpe {result.details['sharpe']}")
    if "cagr" in result.details:
        parts.append(f"CAGR {result.details['cagr']}%")
    return ", ".join(parts)[:80] if parts else ""
