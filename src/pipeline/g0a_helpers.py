"""G0A v2 데이터 기반 점수 계산 헬퍼.

항목 2(IC 사전 검증), 3(카테고리 성공률), 4(레짐 독립성)의
자동 점수 매핑을 수행하는 순수 함수 모듈입니다.

설계 원칙: I/O 없음 — 데이터 로딩은 CLI 책임, 여기는 점수 매핑만.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.pipeline.models import StrategyStatus

if TYPE_CHECKING:
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
class G0AItemScore:
    """G0A 항목별 점수 결과.

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


def compute_ic_score(rank_ic: float) -> G0AItemScore:
    """IC 사전 검증 점수 계산.

    ICAnalyzer의 rank_ic 결과를 받아 G0A 항목 2 점수로 매핑합니다.

    Args:
        rank_ic: Spearman rank correlation (ICResult.rank_ic)

    Returns:
        G0AItemScore with score 1/3/5
    """
    abs_ic = abs(rank_ic)
    if abs_ic > IC_HIGH_THRESHOLD:
        score, reason = 5, f"|Rank IC|={abs_ic:.4f} > {IC_HIGH_THRESHOLD}"
    elif abs_ic > IC_MEDIUM_THRESHOLD:
        score, reason = 3, f"|Rank IC|={abs_ic:.4f} > {IC_MEDIUM_THRESHOLD}"
    else:
        score, reason = 1, f"|Rank IC|={abs_ic:.4f} < {IC_MEDIUM_THRESHOLD}"

    return G0AItemScore(
        item_name="IC 사전 검증",
        score=score,
        evidence={"rank_ic": rank_ic, "abs_rank_ic": abs_ic},
        reason=reason,
    )


def compute_category_success_score(category: str, store: StrategyStore) -> G0AItemScore:
    """카테고리 성공률 점수 계산.

    동일 rationale_category의 ACTIVE/RETIRED 비율로 점수를 매깁니다.

    Args:
        category: rationale_category 문자열
        store: StrategyStore 인스턴스

    Returns:
        G0AItemScore with score 1/3/5
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

    return G0AItemScore(
        item_name="카테고리 성공률",
        score=score,
        evidence={"success_rate": success_rate, "n_active": n_active, "n_retired": n_retired},
        reason=reason,
    )


def compute_regime_independence_score(regime_ics: dict[str, float]) -> G0AItemScore:
    """레짐 독립성 점수 계산.

    레짐별 IC가 양수인 레짐 수로 점수를 매깁니다.

    Args:
        regime_ics: 레짐별 IC 딕셔너리 (예: {"trending": 0.03, "ranging": -0.01, "volatile": 0.02})

    Returns:
        G0AItemScore with score 1/3/5
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

    return G0AItemScore(
        item_name="레짐 독립성",
        score=score,
        evidence=dict(regime_ics),
        reason=reason,
    )
