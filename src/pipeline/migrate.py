"""Migrate scorecard markdown + results JSON → YAML.

데이터 소스 우선순위:
1. docs/scorecard/*.md, docs/scorecard/fail/*.md → meta, gates, decisions
2. results/gate1_pipeline_results.json → asset_performance (보강)
3. results/gate2_validation_results.json → G2 details (보강)
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from src.pipeline.models import (
    AssetMetrics,
    Decision,
    GateId,
    GateResult,
    GateVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.store import StrategyStore

console = Console()

# ─── Paths ───────────────────────────────────────────────────────────

_SCORECARD_DIR = Path("docs/scorecard")
_FAIL_DIR = _SCORECARD_DIR / "fail"
_RESULTS_DIR = Path("results")
_G1_RESULTS = _RESULTS_DIR / "gate1_pipeline_results.json"
_G2_RESULTS = _RESULTS_DIR / "gate2_validation_results.json"


def run_migration(base_dir: Path = Path("strategies")) -> list[StrategyRecord]:
    """전체 마이그레이션 실행."""
    store = StrategyStore(base_dir=base_dir)

    # 1. Load supplemental JSON data
    g1_data = _load_json(_G1_RESULTS)
    g2_data = _load_json(_G2_RESULTS)

    records: list[StrategyRecord] = []

    # 2. Process active scorecards
    for md_path in sorted(_SCORECARD_DIR.glob("*.md")):
        if md_path.name == "template.md":
            continue
        record = _parse_scorecard(md_path, is_retired=False, g1_data=g1_data, g2_data=g2_data)
        if record:
            store.save(record)
            records.append(record)
            logger.info(f"Migrated: {record.meta.name} (ACTIVE)")

    # 3. Process fail scorecards
    for md_path in sorted(_FAIL_DIR.glob("*.md")):
        record = _parse_scorecard(md_path, is_retired=True, g1_data=g1_data, g2_data=g2_data)
        if record:
            store.save(record)
            records.append(record)
            logger.debug(f"Migrated: {record.meta.name} (RETIRED)")

    console.print(f"[green]Migrated {len(records)} strategies to {base_dir}/[/green]")
    return records


# ─── Scorecard parsing ───────────────────────────────────────────────

_RE_STRATEGY_NAME = re.compile(r"\*\*전략명\*\*\s*\|\s*(.+?)\s*\|")
_RE_KEBAB_NAME = re.compile(r"\(`([a-z0-9-]+)`\)")
_RE_CATEGORY = re.compile(r"\*\*유형\*\*\s*\|\s*(.+?)\s*\|")
_RE_TIMEFRAME = re.compile(r"\*\*타임프레임\*\*\s*\|\s*(.+?)\s*\|")
_RE_RATIONALE = re.compile(r"\*\*경제적 논거\*\*\s*\|\s*(.+?)\s*\|")
_RE_GATE_LINE = re.compile(r"(G\d[AB]?)\s+\S+\s+\[(PASS|FAIL| {4})\]\s*(.*)")
_RE_ASSET_ROW = re.compile(
    r"\|\s*\*?\*?\d+\*?\*?\s*\|\s*\*?\*?([A-Z/]+)\*?\*?\s*\|\s*\*?\*?([+-]?[\d.]+)\*?\*?\s*\|"
    + r"\s*([+-]?[\d.]+)%?\s*\|\s*-?([\d.]+)%?\s*\|\s*([\d,]+)\s*\|\s*([+-]?[\d.]+)\s*\|"
)
_RE_DECISION_ROW = re.compile(
    r"\|\s*([\d-]+)\s*\|\s*(G\d[AB]?)\s*\|\s*(PASS|FAIL)\s*\|\s*(.+?)\s*\|"
)


def _parse_scorecard(
    path: Path,
    *,
    is_retired: bool,
    g1_data: dict[str, Any],
    g2_data: dict[str, Any],
) -> StrategyRecord | None:
    """스코어카드 마크다운 파싱 → StrategyRecord."""
    text = path.read_text(encoding="utf-8")

    # Extract name
    kebab_match = _RE_KEBAB_NAME.search(text)
    name = kebab_match.group(1) if kebab_match else path.stem

    name_match = _RE_STRATEGY_NAME.search(text)
    display_name = name_match.group(1).strip() if name_match else name
    # Clean display_name: remove registry key part
    if "(" in display_name:
        display_name = display_name.split("(")[0].strip()
    # Remove markdown bold
    display_name = display_name.replace("**", "").strip()

    # Extract category
    cat_match = _RE_CATEGORY.search(text)
    category = cat_match.group(1).strip() if cat_match else "Unknown"

    # Extract timeframe
    tf_match = _RE_TIMEFRAME.search(text)
    timeframe = tf_match.group(1).strip() if tf_match else "1D"

    # Extract rationale
    rat_match = _RE_RATIONALE.search(text)
    rationale = rat_match.group(1).strip() if rat_match else ""

    # Determine short_mode from strategy knowledge
    short_mode = _infer_short_mode(name)

    # Parse gates
    gates, _fail_gate = _parse_gates(text)

    # Determine status
    status = StrategyStatus.RETIRED if is_retired else StrategyStatus.ACTIVE

    # Parse asset performance from markdown
    assets = _parse_asset_table(text)

    # Supplement from G1 JSON
    if not assets and name in g1_data.get("results", {}):
        assets = _assets_from_g1(g1_data["results"][name])

    # Supplement G2 details from JSON
    if GateId.G2 in gates and name in g2_data.get("results", {}):
        g2_info = g2_data["results"][name]
        gates[GateId.G2] = GateResult(
            status=gates[GateId.G2].status,
            date=gates[GateId.G2].date,
            details={
                "is_sharpe": g2_info.get("avg_train_sharpe", 0),
                "oos_sharpe": g2_info.get("avg_test_sharpe", 0),
                "decay": round(g2_info.get("sharpe_decay", 0) * 100, 1),
            },
        )

    # Parse decisions
    decisions = _parse_decisions(text)

    meta = StrategyMeta(
        name=name,
        display_name=display_name,
        category=category,
        timeframe=timeframe,
        short_mode=short_mode,
        status=status,
        created_at=date(2026, 2, 10),
        retired_at=date(2026, 2, 10) if is_retired else None,
        economic_rationale=rationale,
    )

    return StrategyRecord(
        meta=meta,
        gates=gates,
        asset_performance=assets,
        decisions=decisions,
    )


def _parse_gates(text: str) -> tuple[dict[GateId, GateResult], GateId | None]:
    """Gate 진행 현황 코드블록 파싱."""
    gates: dict[GateId, GateResult] = {}
    fail_gate: GateId | None = None

    # Normalize gate IDs: G0 아이디어 → G0A, G0B 코드검증 → G0B
    for match in _RE_GATE_LINE.finditer(text):
        raw_gid = match.group(1)
        verdict_str = match.group(2).strip()
        detail_text = match.group(3).strip()

        if verdict_str not in ("PASS", "FAIL"):
            continue

        # Map G0 → G0A
        gid_str = "G0A" if raw_gid == "G0" else raw_gid
        try:
            gid = GateId(gid_str)
        except ValueError:
            continue

        verdict = GateVerdict(verdict_str)
        details = _parse_gate_details(detail_text, gid)

        gates[gid] = GateResult(status=verdict, date=date(2026, 2, 10), details=details)

        if verdict == GateVerdict.FAIL:
            fail_gate = gid

    return gates, fail_gate


def _parse_gate_details(text: str, gate: GateId) -> dict[str, Any]:
    """Gate detail 텍스트에서 수치 추출."""
    details: dict[str, Any] = {}

    # G0A: "22/30점" → score
    score_match = re.search(r"(\d+)/(\d+)", text)
    if score_match and gate == GateId.G0A:
        details["score"] = int(score_match.group(1))
        details["max_score"] = int(score_match.group(2))

    # Sharpe extraction
    sharpe_match = re.search(r"Sharpe\s+([+-]?[\d.]+)", text)
    if sharpe_match:
        details["sharpe"] = float(sharpe_match.group(1))

    # CAGR extraction
    cagr_match = re.search(r"CAGR\s+([+-]?[\d.]+)%", text)
    if cagr_match:
        details["cagr"] = float(cagr_match.group(1))

    # MDD extraction
    mdd_match = re.search(r"MDD\s+-?([\d.]+)%", text)
    if mdd_match:
        details["mdd"] = float(mdd_match.group(1))

    # OOS Sharpe
    oos_match = re.search(r"OOS\s+Sharpe\s+([+-]?[\d.]+)", text)
    if oos_match:
        details["oos_sharpe"] = float(oos_match.group(1))

    # Decay
    decay_match = re.search(r"Decay\s+([+-]?[\d.]+)%", text)
    if decay_match:
        details["decay"] = float(decay_match.group(1))

    # Fallback: store raw text if no structured data found
    if not details:
        details["note"] = text

    return details


def _parse_asset_table(text: str) -> list[AssetMetrics]:
    """에셋별 비교 테이블 파싱."""
    assets: list[AssetMetrics] = []

    # More flexible pattern: look for rows in the asset comparison table
    # Pattern: | rank | symbol | sharpe | cagr | mdd | trades | pf | ...
    for line in text.split("\n"):
        # Skip header/separator rows
        if "순위" in line or "---" in line:
            continue

        match = _RE_ASSET_ROW.match(line)
        if match:
            symbol = match.group(1).replace("**", "")
            sharpe = float(match.group(2).replace("**", ""))
            cagr = float(match.group(3))
            mdd = abs(float(match.group(4)))
            trades = int(match.group(5).replace(",", ""))
            pf = float(match.group(6))

            assets.append(
                AssetMetrics(
                    symbol=symbol,
                    sharpe=sharpe,
                    cagr=cagr,
                    mdd=mdd,
                    trades=trades,
                    profit_factor=pf if pf > 0 else None,
                )
            )

    return assets


def _parse_decisions(text: str) -> list[Decision]:
    """의사결정 기록 테이블 파싱."""
    decisions: list[Decision] = []
    for match in _RE_DECISION_ROW.finditer(text):
        date_str = match.group(1)
        raw_gid = match.group(2)
        verdict = match.group(3)
        rationale = match.group(4).strip()

        gid_str = "G0A" if raw_gid == "G0" else raw_gid
        try:
            gid = GateId(gid_str)
        except ValueError:
            continue

        decisions.append(
            Decision(
                date=date.fromisoformat(date_str),
                gate=gid,
                verdict=GateVerdict(verdict),
                rationale=rationale,
            )
        )
    return decisions


def _assets_from_g1(results: list[dict[str, Any]]) -> list[AssetMetrics]:
    """G1 JSON 결과에서 AssetMetrics 생성."""
    return [
        AssetMetrics(
            symbol=r["symbol"],
            sharpe=round(r["sharpe_ratio"], 2),
            cagr=round(r["cagr"] * 100, 1),
            mdd=round(abs(r["max_drawdown"]), 1),
            trades=r["total_trades"],
            profit_factor=round(r["profit_factor"], 2) if r.get("profit_factor") else None,
            win_rate=round(r["win_rate"], 1) if r.get("win_rate") else None,
            sortino=round(r["sortino_ratio"], 2) if r.get("sortino_ratio") else None,
            calmar=round(r["calmar_ratio"], 2) if r.get("calmar_ratio") else None,
            alpha=round(r["alpha"], 1) if r.get("alpha") else None,
            beta=round(r["beta"], 2) if r.get("beta") else None,
        )
        for r in results
    ]


# ─── Helpers ─────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any]:
    """JSON 파일 로드 (없으면 빈 dict)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


# Short mode 추론 (전략별 기본값)
_SHORT_MODE_MAP: dict[str, str] = {
    "ou-meanrev": "FULL",
    "ctrend": "FULL",
    "tsmom": "HEDGE_ONLY",
    "enhanced-tsmom": "HEDGE_ONLY",
    "xsmom": "HEDGE_ONLY",
    "vw-tsmom": "HEDGE_ONLY",
    "vwap-disposition": "FULL",
    "anchor-mom": "HEDGE_ONLY",
    "accel-conv": "FULL",
    "qd-mom": "FULL",
}


def _infer_short_mode(name: str) -> str:
    """전략 이름에서 short_mode 추론."""
    return _SHORT_MODE_MAP.get(name, "HEDGE_ONLY")
