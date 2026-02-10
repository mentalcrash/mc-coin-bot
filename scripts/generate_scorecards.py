#!/usr/bin/env python3
"""Generate Strategy Scorecards from bulk backtest results.

bulk_backtest_results.json을 읽어서 전략별 간결한 마크다운 스코어카드를 생성한다.
Gate 0-1 자동 평가를 수행하고, PASS/WATCH/FAIL 판정을 내린다.

- 활성 전략 → docs/scorecard/{name}.md
- 폐기 전략 → docs/scorecard/fail/{name}.md
- TSMOM은 수동 관리 (자동 생성 제외)

Usage:
    uv run python scripts/generate_scorecards.py

Input:
    results/bulk_backtest_results.json
    results/gate2_validation_results.json (optional)

Output:
    docs/scorecard/{name}.md
    docs/scorecard/fail/{name}.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# =============================================================================
# Constants
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "bulk_backtest_results.json"
GATE2_RESULTS_PATH = ROOT / "results" / "gate2_validation_results.json"
SCORECARD_DIR = ROOT / "docs" / "scorecard"
FAIL_DIR = SCORECARD_DIR / "fail"

# TSMOM 스코어카드는 수동 작성 — 자동 생성 대상에서 제외
MANUAL_SCORECARDS = {"tsmom"}

# 코드 삭제된 폐기 전략 (Gate 1 FAIL)
DEPRECATED_STRATEGIES = {
    "larry-vb",
    "overnight",
    "zscore-mr",
    "rsi-crossover",
    "hurst-regime",
    "risk-mom",
}

# Strategy metadata: name → (display_name, strategy_type, timeframe, rationale)
STRATEGY_META: dict[str, tuple[str, str, str, str]] = {
    "adaptive-breakout": (
        "Adaptive Breakout",
        "추세추종",
        "1D",
        "변동성 압축 후 확장 시 돌파 모멘텀이 발생하며, 적응형 채널이 시장 상태에 맞춰 필터링한다.",
    ),
    "adx-regime": (
        "ADX Regime",
        "하이브리드",
        "1D",
        "ADX가 높은 구간(강추세)에서는 모멘텀이, 낮은 구간(횡보)에서는 평균회귀가 유효하다.",
    ),
    "bb-rsi": (
        "BB-RSI",
        "평균회귀",
        "1D",
        "볼린저밴드 상/하한에서의 가격 복귀는 과매수/과매도 심리의 회복에 의해 발생한다.",
    ),
    "donchian": (
        "Donchian Channel",
        "추세추종",
        "1D",
        "신고가/신저가 돌파는 추세 시작의 강한 시그널이며, 터틀 트레이딩으로 검증되었다.",
    ),
    "donchian-ensemble": (
        "Donchian Ensemble",
        "추세추종",
        "1D",
        "다양한 lookback의 평균은 특정 기간에 대한 과적합을 방지하고, 앙상블 효과로 안정성을 높인다.",
    ),
    "enhanced-tsmom": (
        "Enhanced VW-TSMOM",
        "추세추종",
        "1D",
        "거래량 증가와 함께하는 가격 움직임은 더 높은 지속성을 가지며, 허위 돌파를 필터링한다.",
    ),
    "gk-breakout": (
        "GK Breakout",
        "변동성 돌파",
        "1D",
        "변동성 압축(squeeze)은 에너지 축적을 의미하며, 해소 시 강한 방향성 움직임이 나타난다.",
    ),
    "hmm-regime": (
        "HMM Regime",
        "레짐 전환",
        "1D",
        "시장은 관찰 불가한 상태(레짐)에 의해 지배되며, HMM이 이를 확률적으로 추정한다.",
    ),
    "hurst-regime": (
        "Hurst/ER Regime",
        "레짐 전환",
        "1D",
        "Hurst > 0.5는 추세 지속, < 0.5는 평균회귀 성향을 나타내며, 이에 맞는 전략을 선택한다.",
    ),
    "kama": (
        "KAMA",
        "추세추종",
        "1D",
        "적응형 이동평균은 추세 시장에서는 빠르게, 횡보 시장에서는 느리게 반응하여 whipsaw를 줄인다.",
    ),
    "larry-vb": (
        "Larry VB",
        "변동성 돌파",
        "1D",
        "전일 변동폭 대비 일정 비율 이상 상승 시 강한 매수세가 존재한다고 판단한다.",
    ),
    "max-min": (
        "MAX-MIN",
        "하이브리드",
        "1D",
        "신고가 돌파는 추세 지속을, 신저가 매수는 과매도 반등을 포착하여 시장 레짐에 적응한다.",
    ),
    "mom-mr-blend": (
        "Mom-MR Blend",
        "하이브리드",
        "1D",
        "모멘텀과 평균회귀를 동일 비중으로 혼합하면 레짐 변화에 대한 로버스트성이 향상된다는 가설.",
    ),
    "mtf-macd": (
        "MTF-MACD",
        "추세추종",
        "1D",
        "상위 타임프레임의 추세 방향과 하위 타임프레임의 진입 타이밍을 결합하여 거래 신뢰도를 높인다.",
    ),
    "overnight": (
        "Overnight",
        "계절성",
        "1H",
        "기관 투자자의 영업시간과 아시아 시간대의 유동성 차이가 가격 패턴을 만든다.",
    ),
    "risk-mom": (
        "Risk Momentum",
        "추세추종",
        "1D",
        "변동성이 높을 때 포지션을 줄이고, 낮을 때 늘려서 위험 조정 수익률을 개선한다.",
    ),
    "rsi-crossover": (
        "RSI Crossover",
        "평균회귀",
        "1D",
        "RSI 극단값은 가격의 과도한 이동을 나타내며, 평균으로의 복귀 확률이 높다.",
    ),
    "stoch-mom": (
        "Stochastic Momentum",
        "하이브리드",
        "1D",
        "스토캐스틱 크로스오버는 단기 모멘텀 변화를, SMA 필터는 전체 추세 방향을 확인한다.",
    ),
    "ttm-squeeze": (
        "TTM Squeeze",
        "변동성 돌파",
        "1D",
        "BB가 KC 안에 들어가면 변동성 압축, 밖으로 나오면 폭발적 움직임이 시작된다.",
    ),
    "vol-adaptive": (
        "Vol-Adaptive",
        "추세추종",
        "1D",
        "여러 지표의 동시 확인(confluence)은 허위 시그널을 줄이고, ATR 사이징은 변동성에 적응한다.",
    ),
    "vol-regime": (
        "Vol Regime",
        "레짐 전환",
        "1D",
        "시장 변동성 수준에 따라 최적 전략 파라미터가 다르며, 자동 전환이 성과를 개선한다.",
    ),
    "vol-structure": (
        "Vol Structure",
        "레짐 전환",
        "1D",
        "단기/장기 변동성 비율은 시장 레짐 전환의 선행 지표로, 구조적 변화를 조기에 포착한다.",
    ),
    "zscore-mr": (
        "Z-Score MR",
        "평균회귀",
        "1D",
        "가격의 통계적 이상값(z-score > 2)은 평균으로의 회귀 압력이 높음을 시사한다.",
    ),
    # --- 신규 7개 전략 (2026-02-10) ---
    "xsmom": (
        "XSMOM",
        "크로스섹셔널",
        "1D",
        "코인 간 상대 강도(herding, attention bias)를 활용한 market-neutral long-short 전략",
    ),
    "funding-carry": (
        "Funding Carry",
        "캐리",
        "1D",
        "Perpetual futures funding rate의 구조적 risk premium 수취 (FX carry trade 원리)",
    ),
    "ctrend": (
        "CTREND",
        "ML 앙상블",
        "1D",
        "28개 기술지표를 elastic net으로 결합한 regularized cross-sectional trend factor",
    ),
    "multi-factor": (
        "Multi-Factor",
        "멀티팩터",
        "1D",
        "직교 alpha source 3개를 균등가중 결합하여 factor diversification 달성",
    ),
    "copula-pairs": (
        "Copula Pairs",
        "통계적 차익",
        "1D",
        "코인 쌍의 공적분 관계 이탈 시 spread mean-reversion으로 수익",
    ),
    "vw-tsmom": (
        "VW-TSMOM",
        "추세추종",
        "1D",
        "거래량 가중 수익률로 informed trading 구간의 모멘텀에 집중",
    ),
    "har-vol": (
        "HAR Vol",
        "변동성 오버레이",
        "1D",
        "HAR-RV 모델의 변동성 예측 오차로 포지션 사이징 조절",
    ),
}

# Gate 0 scores (idea evaluation)
GATE0_SCORES: dict[str, dict[str, int]] = {
    "adaptive-breakout": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "adx-regime": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 4,
    },
    "bb-rsi": {"economic": 4, "novelty": 2, "data": 5, "complexity": 5, "capacity": 4, "regime": 3},
    "donchian": {
        "economic": 5,
        "novelty": 2,
        "data": 5,
        "complexity": 5,
        "capacity": 5,
        "regime": 3,
    },
    "donchian-ensemble": {
        "economic": 5,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 5,
        "regime": 3,
    },
    "enhanced-tsmom": {
        "economic": 5,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "gk-breakout": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "hmm-regime": {
        "economic": 3,
        "novelty": 4,
        "data": 5,
        "complexity": 3,
        "capacity": 4,
        "regime": 4,
    },
    "hurst-regime": {
        "economic": 4,
        "novelty": 4,
        "data": 5,
        "complexity": 3,
        "capacity": 4,
        "regime": 4,
    },
    "kama": {"economic": 4, "novelty": 2, "data": 5, "complexity": 4, "capacity": 4, "regime": 3},
    "larry-vb": {
        "economic": 3,
        "novelty": 2,
        "data": 5,
        "complexity": 5,
        "capacity": 3,
        "regime": 2,
    },
    "max-min": {
        "economic": 3,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "mom-mr-blend": {
        "economic": 3,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 4,
    },
    "mtf-macd": {
        "economic": 4,
        "novelty": 2,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "overnight": {
        "economic": 3,
        "novelty": 4,
        "data": 5,
        "complexity": 4,
        "capacity": 3,
        "regime": 2,
    },
    "risk-mom": {
        "economic": 5,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "rsi-crossover": {
        "economic": 3,
        "novelty": 1,
        "data": 5,
        "complexity": 5,
        "capacity": 4,
        "regime": 3,
    },
    "stoch-mom": {
        "economic": 3,
        "novelty": 2,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "ttm-squeeze": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    "vol-adaptive": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 3,
        "capacity": 4,
        "regime": 3,
    },
    "vol-regime": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 3,
        "capacity": 4,
        "regime": 4,
    },
    "vol-structure": {
        "economic": 4,
        "novelty": 4,
        "data": 5,
        "complexity": 3,
        "capacity": 4,
        "regime": 4,
    },
    "zscore-mr": {
        "economic": 4,
        "novelty": 2,
        "data": 5,
        "complexity": 4,
        "capacity": 4,
        "regime": 3,
    },
    # --- 신규 7개 전략 (2026-02-10) ---
    "xsmom": {"economic": 5, "novelty": 3, "data": 5, "complexity": 4, "capacity": 4, "regime": 3},
    "funding-carry": {
        "economic": 5,
        "novelty": 4,
        "data": 4,
        "complexity": 4,
        "capacity": 4,
        "regime": 4,
    },
    "ctrend": {"economic": 4, "novelty": 4, "data": 5, "complexity": 3, "capacity": 3, "regime": 3},
    "multi-factor": {
        "economic": 5,
        "novelty": 3,
        "data": 4,
        "complexity": 3,
        "capacity": 4,
        "regime": 4,
    },
    "copula-pairs": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 2,
        "capacity": 3,
        "regime": 3,
    },
    "vw-tsmom": {
        "economic": 4,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 3,
        "regime": 2,
    },
    "har-vol": {
        "economic": 3,
        "novelty": 3,
        "data": 5,
        "complexity": 4,
        "capacity": 2,
        "regime": 2,
    },
}

# 폐기 사유 (코드 삭제된 전략)
DEPRECATION_REASONS: dict[str, str] = {
    "larry-vb": "1-bar hold 비용 구조적 문제 (연 125건 x 0.1% = 12.5% drag)",
    "overnight": "1H TF 데이터 부족 + 계절성 불안정",
    "zscore-mr": "단일 z-score 평균회귀, 낮은 Sharpe",
    "rsi-crossover": "RSI 단순 크로스오버, 통계적 무의미",
    "hurst-regime": "Hurst exponent 추정 노이즈, 실용성 부족",
    "risk-mom": "TSMOM과 높은 상관, 차별화 부족",
}


# =============================================================================
# Helpers
# =============================================================================


def fmt(val: float | None, decimals: int = 2, suffix: str = "") -> str:
    """Format a number, returning '-' for None/NaN."""
    if val is None:
        return "—"
    try:
        if isinstance(val, float) and math.isnan(val):
            return "—"
        return f"{val:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def gate1_verdict(best: dict[str, Any]) -> str:
    """Gate 1 자동 판정."""
    m = best["metrics"]
    sharpe = m["sharpe_ratio"]
    mdd = m["max_drawdown"]
    trades = m["total_trades"]
    total_return = m["total_return"]

    if (total_return < 0 and trades < 20) or (sharpe < 0.5 and trades < 20):
        return "FAIL"
    if sharpe > 1.0:
        return "PASS"
    if 0.5 <= sharpe <= 1.0 or 25 <= mdd <= 40:
        return "WATCH"
    return "FAIL"


def find_best(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Sharpe 기준 최적 자산 선택."""
    return max(entries, key=lambda e: e["metrics"]["sharpe_ratio"])


def load_gate2_results() -> dict[str, Any]:
    """Gate 2 검증 결과를 로드. 파일이 없으면 빈 dict 반환."""
    if not GATE2_RESULTS_PATH.exists():
        return {}
    data = json.loads(GATE2_RESULTS_PATH.read_text(encoding="utf-8"))
    return data.get("results", {})


# =============================================================================
# Scorecard Generation (Simplified)
# =============================================================================


def generate_scorecard(
    name: str,
    entries: list[dict[str, Any]],
    gate2: dict[str, Any] | None = None,
) -> str:
    """간결한 전략 스코어카드 생성 (template.md 형식)."""
    meta = STRATEGY_META.get(name, (name, "Unknown", "1D", ""))
    display_name, strategy_type, timeframe, rationale = meta
    g0 = GATE0_SCORES.get(
        name, {"economic": 3, "novelty": 3, "data": 5, "complexity": 3, "capacity": 3, "regime": 3}
    )
    g0_total = sum(g0.values())
    g0_verdict = "PASS" if g0_total >= 18 else "FAIL"

    sorted_entries = sorted(entries, key=lambda e: e["metrics"]["sharpe_ratio"], reverse=True)
    best = sorted_entries[0]
    bm = best["metrics"]
    best_sym = best["symbol"]
    verdict = gate1_verdict(best)

    is_deprecated = name in DEPRECATED_STRATEGIES
    status = "`폐기`" if is_deprecated else "`검증중`"

    lines: list[str] = []
    a = lines.append

    # Header
    a(f"# 전략 스코어카드: {display_name}")
    a("")
    a("> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)")
    a("")

    # 기본 정보
    a("## 기본 정보")
    a("")
    a("| 항목 | 값 |")
    a("|------|---|")
    a(f"| **전략명** | {display_name} (`{name}`) |")
    a(f"| **유형** | {strategy_type} |")
    a(f"| **타임프레임** | {timeframe} |")
    a(f"| **상태** | {status} |")
    a(f"| **Best Asset** | {best_sym} (Sharpe {fmt(bm['sharpe_ratio'])}) |")
    if len(sorted_entries) > 1:
        s2 = sorted_entries[1]
        a(f"| **2nd Asset** | {s2['symbol']} (Sharpe {fmt(s2['metrics']['sharpe_ratio'])}) |")
    a(f"| **경제적 논거** | {rationale} |")

    if is_deprecated:
        reason = DEPRECATION_REASONS.get(name, "Gate 1 FAIL")
        a(f"| **폐기 사유** | {reason} |")
        a("| **폐기일** | 2026-02-09 |")

    a("")
    a("---")
    a("")

    # 성과 요약
    a("## 성과 요약 (6년, 2020-2025)")
    a("")

    # 에셋별 비교
    a("### 에셋별 비교")
    a("")
    a("| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |")
    a("|------|------|--------|------|-----|--------|------|")
    for i, entry in enumerate(sorted_entries, 1):
        m = entry["metrics"]
        sym = entry["symbol"]
        bold = "**" if i == 1 else ""
        a(
            f"| {bold}{i}{bold} | {bold}{sym}{bold} | {bold}{fmt(m['sharpe_ratio'])}{bold} | {fmt(m['cagr'])}% | -{fmt(m['max_drawdown'])}% | {m['total_trades']} | {fmt(m['profit_factor'])} |"
        )
    a("")

    # Best Asset 핵심 지표
    a("### Best Asset 핵심 지표")
    a("")
    a("| 지표 | 값 | 기준 | 판정 |")
    a("|------|---|------|------|")
    sharpe = bm["sharpe_ratio"]
    mdd = bm["max_drawdown"]
    trades = bm["total_trades"]
    win_rate = bm["win_rate"]
    sortino = bm["sortino_ratio"]
    a(f"| Sharpe | {fmt(sharpe)} | > 1.0 | {'PASS' if sharpe > 1.0 else 'FAIL'} |")
    a(f"| MDD | -{fmt(mdd)}% | < 40% | {'PASS' if mdd < 40 else 'FAIL'} |")
    a(f"| Trades | {trades} | > 50 | {'PASS' if trades > 50 else 'FAIL'} |")
    a(f"| Win Rate | {fmt(win_rate)}% | > 45% | — |")
    a(f"| Sortino | {fmt(sortino)} | > 1.5 | — |")
    a("")
    a("---")
    a("")

    # Gate 진행 현황
    a("## Gate 진행 현황")
    a("")
    a("```")
    a(f"G0 아이디어  [{g0_verdict:^4}] {g0_total}/30점")
    a(f"G1 백테스트  [{verdict:^4}] Sharpe {fmt(bm['sharpe_ratio'])}, MDD {fmt(mdd)}%")

    if gate2:
        g2v = gate2["verdict"]
        a(
            f"G2 IS/OOS    [{g2v:^4}] OOS Sharpe {fmt(gate2['avg_test_sharpe'])}, Decay {fmt(gate2['sharpe_decay'] * 100, decimals=1)}%"
        )
    else:
        a("G2 IS/OOS    [    ]")

    a("G3 파라미터  [    ]")
    a("G4 심층검증  [    ]")
    a("G5 EDA검증   [    ]")
    a("G6 모의거래  [    ]")
    a("G7 실전배포  [    ]")
    a("```")
    a("")

    # Gate 상세 (완료된 Gate만 기록)
    a("### Gate 상세 (완료된 Gate만 기록)")
    a("")

    if gate2:
        g2v = gate2["verdict"]
        a(
            f"**Gate 2** ({g2v}): IS Sharpe {fmt(gate2['avg_train_sharpe'])}, OOS Sharpe {fmt(gate2['avg_test_sharpe'])}, Decay {fmt(gate2['sharpe_decay'] * 100, decimals=1)}%"
        )
        if gate2.get("failure_reasons"):
            a(f"  - 실패 사유: {'; '.join(gate2['failure_reasons'])}")
        a("")
    else:
        a("> Gate 2 이후의 상세 결과는 해당 Gate 완료 시 추가한다.")
        a("")

    a("---")
    a("")

    # 의사결정 기록
    a("## 의사결정 기록")
    a("")
    a("| 날짜 | Gate | 판정 | 근거 |")
    a("|------|------|------|------|")
    a(f"| 2026-02-09 | G0 | {g0_verdict} | {g0_total}/30점 |")
    a(f"| 2026-02-09 | G1 | {verdict} | {best_sym} Sharpe {fmt(bm['sharpe_ratio'])} |")

    if gate2:
        g2v = gate2["verdict"]
        a(
            f"| 2026-02-09 | G2 | {g2v} | OOS Sharpe {fmt(gate2['avg_test_sharpe'])}, Decay {fmt(gate2['sharpe_decay'] * 100, decimals=1)}% |"
        )

    if is_deprecated:
        reason = DEPRECATION_REASONS.get(name, "Gate 1 FAIL")
        a(f"| 2026-02-09 | 폐기 | — | {reason} |")

    a("")

    return "\n".join(lines)


def print_readme_table(
    results: dict[str, list[dict[str, Any]]],
    gate2_results: dict[str, Any],
) -> None:
    """README용 전략 테이블을 stdout에 출력."""
    active: list[tuple[float, str, str, float, float, float, str, str, str]] = []

    for name, entries in sorted(results.items()):
        if not entries or name in DEPRECATED_STRATEGIES:
            continue
        best = find_best(entries)
        m = best["metrics"]
        g1 = gate1_verdict(best)

        g2_data = gate2_results.get(name)
        g2_label = g2_data["verdict"][0] if g2_data else "—"

        g0 = GATE0_SCORES.get(name, {})
        g0_total = sum(g0.values())
        g0_label = "P" if g0_total >= 18 else "F"
        g1_label = g1[0]

        active.append(
            (
                m["sharpe_ratio"],
                name,
                best["symbol"],
                m["sharpe_ratio"],
                m["cagr"],
                m["max_drawdown"],
                g0_label,
                g1_label,
                g2_label,
            )
        )

    active.sort(key=lambda x: x[0], reverse=True)
    display_names = {k: v[0] for k, v in STRATEGY_META.items()}

    print("\n### README 활성 전략 테이블:\n")
    print("| # | 전략 | Best Asset | Sharpe | CAGR | MDD | G0 | G1 | G2 | 스코어카드 |")
    print("|---|------|-----------|--------|------|-----|:--:|:--:|:--:|-----------|")

    for i, (_, name, sym, sharpe_v, cagr, mdd_v, g0l, g1l, g2l) in enumerate(active, 1):
        dn = display_names.get(name, name)
        bold = "**" if g1l == "P" else ""
        link = f"[scorecard](docs/scorecard/{name}.md)"
        print(
            f"| {i} | {bold}{dn}{bold} | {sym} | {sharpe_v:.2f} | +{cagr:.1f}% | -{mdd_v:.1f}% | {g0l} | {g1l} | {g2l} | {link} |"
        )

    # 폐기 전략 테이블
    deprecated: list[tuple[str, str, float, str, str]] = []
    for name in sorted(DEPRECATED_STRATEGIES):
        entries = results.get(name, [])
        if not entries:
            continue
        best = find_best(entries)
        m = best["metrics"]
        reason = DEPRECATION_REASONS.get(name, "Gate 1 FAIL")
        dn = display_names.get(name, name)
        deprecated.append((dn, name, m["sharpe_ratio"], reason, "G1"))

    if deprecated:
        print("\n### README 폐기 전략 테이블:\n")
        print("| 전략 | Sharpe | 실패 Gate | 폐기 사유 | 스코어카드 |")
        print("|------|--------|:-------:|----------|-----------|")
        for dn, name, sharpe_v, reason, gate in deprecated:
            link = f"[scorecard](docs/scorecard/fail/{name}.md)"
            print(f"| {dn} | {sharpe_v:.2f} | {gate} | {reason} | {link} |")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results: dict[str, list[dict[str, Any]]] = data["results"]

    gate2_results = load_gate2_results()
    if gate2_results:
        print(f"Loaded Gate 2 results for {len(gate2_results)} strategies")

    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    for name, entries in sorted(results.items()):
        if not entries:
            print(f"SKIP: {name} — no results")
            continue

        if name in MANUAL_SCORECARDS:
            print(f"SKIP: {name} — manual scorecard (protected)")
            skipped += 1
            continue

        gate2 = gate2_results.get(name)
        md = generate_scorecard(name, entries, gate2=gate2)

        if name in DEPRECATED_STRATEGIES:
            output_path = FAIL_DIR / f"{name}.md"
        else:
            output_path = SCORECARD_DIR / f"{name}.md"

        output_path.write_text(md, encoding="utf-8")

        tag = " [DEPRECATED]" if name in DEPRECATED_STRATEGIES else ""
        g2_tag = f" [G2: {gate2['verdict']}]" if gate2 else ""
        print(f"OK: {output_path.relative_to(ROOT)}{tag}{g2_tag}")
        generated += 1

    print(f"\nGenerated {generated} scorecards ({skipped} skipped)")
    print(f"  Active: {SCORECARD_DIR.relative_to(ROOT)}")
    print(f"  Failed: {FAIL_DIR.relative_to(ROOT)}")

    print_readme_table(results, gate2_results)


if __name__ == "__main__":
    main()
