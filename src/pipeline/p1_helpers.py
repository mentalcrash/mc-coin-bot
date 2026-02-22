"""P1 (Alpha Research) 데이터 기반 점수 계산 헬퍼.

항목 2(IC 사전 검증), 3(카테고리 성공률), 4(레짐 독립성)의
자동 점수 매핑을 수행하는 순수 함수 모듈입니다.

설계 원칙: I/O 없음 — 데이터 로딩은 CLI 책임, 여기는 점수 매핑만.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.pipeline.models import StrategyStatus

if TYPE_CHECKING:
    import pandas as pd

    from src.pipeline.store import StrategyStore

# ─── Scoring Thresholds ─────────────────────────────────────────────

IC_HIGH_THRESHOLD = 0.05
IC_MEDIUM_THRESHOLD = 0.02
CATEGORY_SUCCESS_HIGH_PCT = 20
CATEGORY_RETIRED_MIN_FOR_FAIL = 3
REGIME_POSITIVE_HIGH = 3
REGIME_POSITIVE_MEDIUM = 2
MIN_REGIME_SAMPLES = 10
_ITEM_MAX_SCORE = 5


@dataclass(frozen=True)
class P1ItemScore:
    """P1 항목별 점수 결과.

    Attributes:
        item_name: 항목명
        score: 점수 (1-5)
        evidence: 근거 데이터 (예: {"rank_ic": 0.06})
        reason: 판정 사유
    """

    item_name: str
    score: int
    evidence: dict[str, float]
    reason: str


def compute_ic_score(rank_ic: float) -> P1ItemScore:
    """IC 사전 검증 점수 계산.

    ICAnalyzer의 rank_ic 결과를 받아 P1 항목 2 점수로 매핑합니다.

    Args:
        rank_ic: Spearman rank correlation (ICResult.rank_ic)

    Returns:
        P1ItemScore with score 1/3/5
    """
    abs_ic = abs(rank_ic)
    if abs_ic > IC_HIGH_THRESHOLD:
        score, reason = 5, f"|Rank IC|={abs_ic:.4f} > {IC_HIGH_THRESHOLD}"
    elif abs_ic > IC_MEDIUM_THRESHOLD:
        score, reason = 3, f"|Rank IC|={abs_ic:.4f} > {IC_MEDIUM_THRESHOLD}"
    else:
        score, reason = 1, f"|Rank IC|={abs_ic:.4f} < {IC_MEDIUM_THRESHOLD}"

    return P1ItemScore(
        item_name="IC 사전 검증",
        score=score,
        evidence={"rank_ic": rank_ic, "abs_rank_ic": abs_ic},
        reason=reason,
    )


def compute_category_success_score(category: str, store: StrategyStore) -> P1ItemScore:
    """카테고리 성공률 점수 계산.

    동일 rationale_category의 ACTIVE/RETIRED 비율로 점수를 매깁니다.

    Args:
        category: rationale_category 문자열
        store: StrategyStore 인스턴스

    Returns:
        P1ItemScore with score 1/3/5
    """
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

    n_retired = len(retired_same)
    n_active = len(active_same)
    total = n_retired + n_active
    success_rate = n_active / total * 100 if total > 0 else 100.0

    if success_rate > CATEGORY_SUCCESS_HIGH_PCT:
        score = 5
        reason = f"성공률 {success_rate:.0f}% > {CATEGORY_SUCCESS_HIGH_PCT}% (ACTIVE {n_active}, RETIRED {n_retired})"
    elif success_rate > 0 or n_retired < CATEGORY_RETIRED_MIN_FOR_FAIL:
        score = 3
        reason = f"성공률 {success_rate:.0f}%, 차별화 가능 (ACTIVE {n_active}, RETIRED {n_retired})"
    else:
        score = 1
        reason = f"성공률 0% + RETIRED {n_retired}개 (동일 카테고리 반복 실패)"

    return P1ItemScore(
        item_name="카테고리 성공률",
        score=score,
        evidence={"success_rate": success_rate, "n_active": n_active, "n_retired": n_retired},
        reason=reason,
    )


def compute_regime_independence_score(regime_ics: dict[str, float]) -> P1ItemScore:
    """레짐 독립성 점수 계산.

    레짐별 IC가 양수인 레짐 수로 점수를 매깁니다.

    Args:
        regime_ics: 레짐별 IC 딕셔너리 (예: {"trending": 0.03, "ranging": -0.01, "volatile": 0.02})

    Returns:
        P1ItemScore with score 1/3/5
    """
    positive_regimes = [name for name, ic in regime_ics.items() if ic > 0]
    n_positive = len(positive_regimes)

    if n_positive >= REGIME_POSITIVE_HIGH:
        score = 5
        reason = f"{n_positive}개 레짐에서 IC 양수: {', '.join(positive_regimes)}"
    elif n_positive >= REGIME_POSITIVE_MEDIUM:
        score = 3
        reason = f"2개 레짐에서 IC 양수: {', '.join(positive_regimes)}"
    else:
        score = 1
        if n_positive == 1:
            reason = f"단일 레짐만 IC 양수: {positive_regimes[0]}"
        else:
            reason = "모든 레짐에서 IC 음수 또는 0"

    return P1ItemScore(
        item_name="레짐 독립성",
        score=score,
        evidence=dict(regime_ics),
        reason=reason,
    )


# ─── Extended Scoring Thresholds ────────────────────────────────────

DECAY_QUARTERS_STABLE = 4
DECAY_QUARTERS_PARTIAL = 3
TURNOVER_LOW = 50
TURNOVER_HIGH = 200
COST_RATIO_EXCELLENT = 0.15
COST_RATIO_RISKY = 0.30
CORRELATION_LOW = 0.2
CORRELATION_HIGH = 0.5
DATA_SOURCE_HIGH = 3
DATA_SOURCE_MEDIUM = 2


def compute_signal_decay_score(rolling_ic: pd.Series) -> P1ItemScore:
    """시그널 감쇠 안정성 점수 계산.

    Rolling IC의 분기별 일관성을 평가합니다.

    Args:
        rolling_ic: ICAnalyzer.rolling_ic() 결과 시리즈

    Returns:
        P1ItemScore with score 1/3/5
    """
    clean = rolling_ic.dropna()
    quarter_size = 63
    n_quarters = max(1, len(clean) // quarter_size)
    n_quarters = min(n_quarters, 8)  # 최대 2년

    # 분기별 평균 IC 부호 일관성
    positive_quarters = 0
    for q in range(n_quarters):
        start_idx = len(clean) - (n_quarters - q) * quarter_size
        end_idx = start_idx + quarter_size
        if start_idx < 0:
            continue
        chunk = clean.iloc[start_idx:end_idx]
        if len(chunk) > 0 and float(chunk.mean()) > 0:
            positive_quarters += 1

    if positive_quarters >= DECAY_QUARTERS_STABLE:
        score = 5
        reason = f"{positive_quarters}분기 연속 IC 양수 (>= {DECAY_QUARTERS_STABLE})"
    elif positive_quarters >= DECAY_QUARTERS_PARTIAL:
        score = 3
        reason = f"{positive_quarters}분기 IC 양수 (>= {DECAY_QUARTERS_PARTIAL})"
    else:
        score = 1
        reason = f"{positive_quarters}분기만 IC 양수 (< {DECAY_QUARTERS_PARTIAL})"

    return P1ItemScore(
        item_name="시그널 감쇠 안정성",
        score=score,
        evidence={"positive_quarters": positive_quarters, "total_quarters": n_quarters},
        reason=reason,
    )


def compute_turnover_score(
    estimated_trades_per_year: float,
    cost_ratio: float,
) -> P1ItemScore:
    """거래 비용 효율 점수 계산.

    연간 거래 빈도와 비용 비율로 효율성을 평가합니다.

    Args:
        estimated_trades_per_year: 연간 예상 거래 횟수
        cost_ratio: 연간 거래비용 / 예상 총수익 비율 (0~1)

    Returns:
        P1ItemScore with score 1/3/5
    """
    in_range = TURNOVER_LOW <= estimated_trades_per_year <= TURNOVER_HIGH
    low_cost = cost_ratio < COST_RATIO_EXCELLENT
    mid_cost = cost_ratio < COST_RATIO_RISKY

    if in_range and low_cost:
        score = 5
        reason = (
            f"거래 {estimated_trades_per_year:.0f}건/yr ({TURNOVER_LOW}~{TURNOVER_HIGH}) "
            f"+ 비용 {cost_ratio:.1%} < {COST_RATIO_EXCELLENT:.0%}"
        )
    elif mid_cost:
        score = 3
        reason = (
            f"거래 {estimated_trades_per_year:.0f}건/yr, "
            f"비용 {cost_ratio:.1%} < {COST_RATIO_RISKY:.0%}"
        )
    else:
        score = 1
        reason = f"비용 비율 {cost_ratio:.1%} >= {COST_RATIO_RISKY:.0%} (비효율적)"

    return P1ItemScore(
        item_name="거래 비용 효율",
        score=score,
        evidence={
            "trades_per_year": estimated_trades_per_year,
            "cost_ratio": cost_ratio,
        },
        reason=reason,
    )


def compute_data_source_diversity_score(data_sources: list[str]) -> P1ItemScore:
    """데이터 소스 다양성 점수 계산.

    사용 데이터 소스 수로 다양성을 평가합니다.

    Args:
        data_sources: 사용 데이터 소스 리스트 (예: ["ohlcv", "onchain", "derivatives"])

    Returns:
        P1ItemScore with score 1/3/5
    """
    unique = list(dict.fromkeys(data_sources))  # 순서 유지 중복 제거
    n_sources = len(unique)

    if n_sources >= DATA_SOURCE_HIGH:
        score = 5
        reason = f"{n_sources}개 소스 사용: {', '.join(unique)} (>= {DATA_SOURCE_HIGH})"
    elif n_sources >= DATA_SOURCE_MEDIUM:
        score = 3
        reason = f"{n_sources}개 소스 사용: {', '.join(unique)}"
    else:
        score = 1
        reason = (
            f"OHLCV-only ({n_sources}개 소스)" if n_sources <= 1 else f"{n_sources}개 소스만 사용"
        )

    return P1ItemScore(
        item_name="데이터 소스 다양성",
        score=score,
        evidence={"n_sources": float(n_sources), "sources": 0.0},
        reason=reason,
    )


def compute_active_correlation_score(expected_correlation: float) -> P1ItemScore:
    """활성 전략 독립성 점수 계산.

    활성 전략과의 예상 상관으로 독립성을 평가합니다.

    Args:
        expected_correlation: 활성 전략(Anchor-Mom 등)과의 예상 상관 (-1~1)

    Returns:
        P1ItemScore with score 1/3/5
    """
    abs_corr = abs(expected_correlation)

    if abs_corr < CORRELATION_LOW:
        score = 5
        reason = f"|상관|={abs_corr:.2f} < {CORRELATION_LOW} (높은 독립성)"
    elif abs_corr < CORRELATION_HIGH:
        score = 3
        reason = f"|상관|={abs_corr:.2f}, {CORRELATION_LOW}~{CORRELATION_HIGH} 범위"
    else:
        score = 1
        reason = f"|상관|={abs_corr:.2f} >= {CORRELATION_HIGH} (포트폴리오 가치 제한)"

    return P1ItemScore(
        item_name="활성 전략 독립성",
        score=score,
        evidence={"expected_correlation": expected_correlation, "abs_correlation": abs_corr},
        reason=reason,
    )
