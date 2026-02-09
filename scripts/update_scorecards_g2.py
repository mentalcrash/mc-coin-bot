#!/usr/bin/env python3
"""Gate 2 결과를 스코어카드에 반영하는 스크립트."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORECARD_DIR = ROOT / "docs" / "scorecard"

# Gate 2 결과 (strategy_cli_name -> results)
G2_RESULTS: dict[str, dict[str, object]] = {
    "vol-regime": {
        "asset": "ETH/USDT",
        "is_sharpe": 1.65,
        "oos_sharpe": 0.37,
        "decay": 77.3,
        "oos_return": 13.8,
        "verdict": "FAIL",
    },
    "tsmom": {
        "asset": "SOL/USDT",
        "is_sharpe": 1.49,
        "oos_sharpe": 0.19,
        "decay": 87.2,
        "oos_return": 2.6,
        "verdict": "FAIL",
    },
    "enhanced-tsmom": {
        "asset": "BTC/USDT",
        "is_sharpe": 1.67,
        "oos_sharpe": 0.25,
        "decay": 85.2,
        "oos_return": 5.7,
        "verdict": "FAIL",
    },
    "vol-structure": {
        "asset": "SOL/USDT",
        "is_sharpe": 1.39,
        "oos_sharpe": 0.59,
        "decay": 57.2,
        "oos_return": 22.3,
        "verdict": "FAIL",
    },
    "kama": {
        "asset": "DOGE/USDT",
        "is_sharpe": 1.30,
        "oos_sharpe": 1.01,
        "decay": 21.8,
        "oos_return": 31.8,
        "verdict": "PASS",
    },
    "vol-adaptive": {
        "asset": "SOL/USDT",
        "is_sharpe": 1.74,
        "oos_sharpe": -0.97,
        "decay": 155.9,
        "oos_return": -32.7,
        "verdict": "FAIL",
    },
    "donchian": {
        "asset": "SOL/USDT",
        "is_sharpe": 1.30,
        "oos_sharpe": 0.12,
        "decay": 91.1,
        "oos_return": -0.6,
        "verdict": "FAIL",
    },
    "donchian-ensemble": {
        "asset": "ETH/USDT",
        "is_sharpe": 1.01,
        "oos_sharpe": 0.99,
        "decay": 1.7,
        "oos_return": 13.4,
        "verdict": "PASS",
    },
    "adx-regime": {
        "asset": "SOL/USDT",
        "is_sharpe": 1.48,
        "oos_sharpe": -0.68,
        "decay": 146.3,
        "oos_return": -26.8,
        "verdict": "FAIL",
    },
    "ttm-squeeze": {
        "asset": "BTC/USDT",
        "is_sharpe": 1.06,
        "oos_sharpe": 0.58,
        "decay": 45.6,
        "oos_return": 12.1,
        "verdict": "PASS",
    },
    "stoch-mom": {
        "asset": "SOL/USDT",
        "is_sharpe": 1.37,
        "oos_sharpe": -0.34,
        "decay": 124.9,
        "oos_return": -3.4,
        "verdict": "FAIL",
    },
    "max-min": {
        "asset": "DOGE/USDT",
        "is_sharpe": 0.85,
        "oos_sharpe": 0.80,
        "decay": 6.2,
        "oos_return": 10.4,
        "verdict": "PASS",
    },
    "gk-breakout": {
        "asset": "DOGE/USDT",
        "is_sharpe": 0.96,
        "oos_sharpe": 0.39,
        "decay": 59.0,
        "oos_return": 6.4,
        "verdict": "FAIL",
    },
    "mtf-macd": {
        "asset": "SOL/USDT",
        "is_sharpe": 0.96,
        "oos_sharpe": 0.21,
        "decay": 78.1,
        "oos_return": 2.4,
        "verdict": "FAIL",
    },
    "hmm-regime": {
        "asset": "BTC/USDT",
        "is_sharpe": 1.04,
        "oos_sharpe": -0.66,
        "decay": 162.9,
        "oos_return": -7.0,
        "verdict": "FAIL",
    },
    "adaptive-breakout": {
        "asset": "SOL/USDT",
        "is_sharpe": 0.67,
        "oos_sharpe": -0.68,
        "decay": 201.1,
        "oos_return": -10.5,
        "verdict": "FAIL",
    },
    "bb-rsi": {
        "asset": "SOL/USDT",
        "is_sharpe": 0.61,
        "oos_sharpe": 0.59,
        "decay": 2.2,
        "oos_return": 5.3,
        "verdict": "PASS",
    },
    "mom-mr-blend": {
        "asset": "ETH/USDT",
        "is_sharpe": 1.12,
        "oos_sharpe": -0.10,
        "decay": 109.1,
        "oos_return": -5.9,
        "verdict": "FAIL",
    },
}

# Gate 2 기준 (전략 평가 표준)
G2_CRITERIA = "OOS Sharpe >= 0.3, Decay < 50%, OOS Return > 0%"


def get_failure_reasons(r: dict[str, object]) -> list[str]:
    reasons = []
    oos_sh = float(r["oos_sharpe"])  # type: ignore[arg-type]
    decay = float(r["decay"])  # type: ignore[arg-type]
    oos_ret = float(r["oos_return"])  # type: ignore[arg-type]
    if oos_sh < 0.3:
        reasons.append(f"OOS Sharpe ({oos_sh:.2f}) < 0.3")
    if decay >= 50.0:
        reasons.append(f"Sharpe Decay ({decay:.1f}%) >= 50%")
    if oos_ret <= 0.0:
        reasons.append(f"OOS Return ({oos_ret:+.1f}%) <= 0%")
    return reasons


def update_scorecard(name: str, r: dict[str, object]) -> bool:
    """스코어카드 파일 업데이트. 성공 시 True 반환."""
    filepath = SCORECARD_DIR / f"{name}.md"
    if not filepath.exists():
        print(f"  SKIP: {filepath} not found")
        return False

    content = filepath.read_text()
    verdict = r["verdict"]
    oos_sh = float(r["oos_sharpe"])  # type: ignore[arg-type]
    is_sh = float(r["is_sharpe"])  # type: ignore[arg-type]
    decay = float(r["decay"])  # type: ignore[arg-type]
    oos_ret = float(r["oos_return"])  # type: ignore[arg-type]

    # G2 라인 업데이트 (Gate 진행 현황 블록 내)
    g2_new = f"G2 IS/OOS    [{verdict}] OOS Sharpe {oos_sh:.2f}, Decay {decay:.1f}%"
    content = re.sub(
        r"G2 IS/OOS\s+\[.*?\].*",
        g2_new,
        content,
    )

    # Gate 상세 업데이트/추가
    if verdict == "PASS":
        gate2_detail = (
            f"**Gate 2** (PASS): IS Sharpe {is_sh:.2f}, OOS Sharpe {oos_sh:.2f}, "
            f"Decay {decay:.1f}%, OOS Return {oos_ret:+.1f}%"
        )
    else:
        reasons = get_failure_reasons(r)
        gate2_detail = (
            f"**Gate 2** (FAIL): IS Sharpe {is_sh:.2f}, OOS Sharpe {oos_sh:.2f}, "
            f"Decay {decay:.1f}%\n"
            f"  - 실패 사유: {'; '.join(reasons)}"
        )

    # 기존 Gate 2 상세 교체 또는 추가
    gate2_pattern = r"\*\*Gate 2\*\*.*?(?=\n\n|\n---|\n\*\*Gate [3-7]|\Z)"
    if re.search(gate2_pattern, content, re.DOTALL):
        content = re.sub(gate2_pattern, gate2_detail, content, flags=re.DOTALL)
    else:
        # Gate 상세 섹션 끝에 추가
        marker = "> Gate 2 이후의 상세 결과는 해당 Gate 완료 시 추가한다."
        if marker in content:
            content = content.replace(marker, gate2_detail)

    # 의사결정 기록에 G2 행 추가 (이미 있으면 교체)
    g2_record = f"| 2026-02-09 | G2 | {verdict} | OOS Sharpe {oos_sh:.2f}, Decay {decay:.1f}% |"
    if re.search(r"\|\s*\d{4}-\d{2}-\d{2}\s*\|\s*G2\s*\|", content):
        content = re.sub(
            r"\|.*?\|\s*G2\s*\|.*?\|.*?\|",
            g2_record,
            content,
        )
    else:
        # 마지막 의사결정 기록 행 다음에 추가
        lines = content.split("\n")
        last_record_idx = -1
        for i, line in enumerate(lines):
            if re.match(r"\|\s*\d{4}-\d{2}-\d{2}\s*\|\s*G[01]\s*\|", line):
                last_record_idx = i
        if last_record_idx > 0:
            lines.insert(last_record_idx + 1, g2_record)
            content = "\n".join(lines)

    filepath.write_text(content)
    return True


def main() -> None:
    print("Updating scorecards with Gate 2 results...")
    print(f"Criteria: {G2_CRITERIA}")
    print()

    updated = 0
    for name, result in G2_RESULTS.items():
        if name == "tsmom":
            print(f"  SKIP: {name} (manually managed)")
            continue
        ok = update_scorecard(name, result)
        if ok:
            v = result["verdict"]
            print(f"  {'✓' if v == 'PASS' else '✗'} {name:<22} G2={v}")
            updated += 1

    print(f"\nUpdated {updated} scorecards.")
    print("Note: tsmom.md is manually managed and was skipped.")


if __name__ == "__main__":
    main()
