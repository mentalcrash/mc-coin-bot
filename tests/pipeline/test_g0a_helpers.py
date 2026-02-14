"""Tests for src/pipeline/g0a_helpers.py — G0A v2 데이터 기반 점수 계산."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.g0a_helpers import (
    G0AItemScore,
    compute_category_success_score,
    compute_ic_score,
    compute_regime_independence_score,
)
from src.pipeline.models import (
    GateId,
    GateResult,
    GateVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.store import StrategyStore

# ─── compute_ic_score ────────────────────────────────────────────────


class TestComputeICScore:
    def test_high_ic_returns_5(self) -> None:
        """|IC| > 0.05 → 5점."""
        result = compute_ic_score(0.07)
        assert result.score == 5
        assert result.item_name == "IC 사전 검증"
        assert result.evidence["rank_ic"] == 0.07
        assert "0.05" in result.reason

    def test_medium_ic_returns_3(self) -> None:
        """|IC| > 0.02 → 3점."""
        result = compute_ic_score(0.03)
        assert result.score == 3
        assert "0.02" in result.reason

    def test_low_ic_returns_1(self) -> None:
        """|IC| < 0.02 → 1점."""
        result = compute_ic_score(0.01)
        assert result.score == 1

    def test_negative_ic_uses_absolute(self) -> None:
        """음수 IC는 절대값으로 평가."""
        result = compute_ic_score(-0.06)
        assert result.score == 5
        assert result.evidence["abs_rank_ic"] == pytest.approx(0.06)

    def test_zero_ic_returns_1(self) -> None:
        """IC=0 → 1점."""
        result = compute_ic_score(0.0)
        assert result.score == 1

    def test_boundary_0_05_returns_3(self) -> None:
        """IC=0.05 정확히 → 3점 (> 0.05이어야 5점)."""
        result = compute_ic_score(0.05)
        assert result.score == 3

    def test_boundary_0_02_returns_1(self) -> None:
        """IC=0.02 정확히 → 1점 (> 0.02이어야 3점)."""
        result = compute_ic_score(0.02)
        assert result.score == 1

    def test_returns_frozen_dataclass(self) -> None:
        result = compute_ic_score(0.03)
        assert isinstance(result, G0AItemScore)
        with pytest.raises(AttributeError):
            result.score = 1  # type: ignore[misc]


# ─── compute_category_success_score ─────────────────────────────────


def _make_record(
    name: str,
    status: StrategyStatus,
    rationale_category: str | None = None,
) -> StrategyRecord:
    """테스트용 최소 StrategyRecord 생성."""
    from datetime import date

    return StrategyRecord(
        meta=StrategyMeta(
            name=name,
            display_name=name,
            category="Test",
            timeframe="1D",
            short_mode="DISABLED",
            status=status,
            created_at=date(2026, 1, 1),
            rationale_category=rationale_category,
        ),
        gates={
            GateId.G0A: GateResult(
                status=GateVerdict.PASS,
                date=date(2026, 1, 1),
                details={"score": 22, "max_score": 30},
            )
        },
    )


class TestComputeCategorySuccessScore:
    @pytest.fixture()
    def strategies_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "strategies"
        d.mkdir()
        return d

    def _make_store(self, strategies_dir: Path, records: list[StrategyRecord]) -> StrategyStore:
        store = StrategyStore(base_dir=strategies_dir)
        for r in records:
            store.save(r)
        return store

    def test_high_success_rate_returns_5(self, strategies_dir: Path) -> None:
        """성공률 > 20% → 5점."""
        records = [
            _make_record("active-1", StrategyStatus.ACTIVE, "momentum"),
            _make_record("retired-1", StrategyStatus.RETIRED, "momentum"),
            _make_record("retired-2", StrategyStatus.RETIRED, "momentum"),
        ]
        store = self._make_store(strategies_dir, records)
        result = compute_category_success_score("momentum", store)
        assert result.score == 5
        assert result.evidence["success_rate"] == pytest.approx(100 / 3)

    def test_low_success_rate_returns_3(self, strategies_dir: Path) -> None:
        """0~20% 성공률이지만 차별화 → 3점."""
        records = [
            _make_record("active-1", StrategyStatus.ACTIVE, "momentum"),
            *[_make_record(f"retired-{i}", StrategyStatus.RETIRED, "momentum") for i in range(10)],
        ]
        store = self._make_store(strategies_dir, records)
        result = compute_category_success_score("momentum", store)
        # 1/11 ≈ 9.09%
        assert result.score == 3

    def test_zero_success_many_retired_returns_1(self, strategies_dir: Path) -> None:
        """성공률 0% + RETIRED 3개+ → 1점."""
        records = [
            _make_record(f"retired-{i}", StrategyStatus.RETIRED, "vol-premium") for i in range(5)
        ]
        store = self._make_store(strategies_dir, records)
        result = compute_category_success_score("vol-premium", store)
        assert result.score == 1

    def test_no_records_returns_5(self, strategies_dir: Path) -> None:
        """기록 없음 → 성공률 100% → 5점."""
        store = StrategyStore(base_dir=strategies_dir)
        result = compute_category_success_score("new-category", store)
        assert result.score == 5

    def test_few_retired_returns_3(self, strategies_dir: Path) -> None:
        """RETIRED 2개, ACTIVE 0 → 성공률 0% 이지만 RETIRED < 3 → 3점."""
        records = [
            _make_record("retired-0", StrategyStatus.RETIRED, "mean-reversion"),
            _make_record("retired-1", StrategyStatus.RETIRED, "mean-reversion"),
        ]
        store = self._make_store(strategies_dir, records)
        result = compute_category_success_score("mean-reversion", store)
        assert result.score == 3


# ─── compute_regime_independence_score ──────────────────────────────


class TestComputeRegimeIndependenceScore:
    def test_three_positive_regimes_returns_5(self) -> None:
        """3개 레짐 IC 양수 → 5점."""
        regime_ics = {"trending": 0.03, "ranging": 0.01, "volatile": 0.02}
        result = compute_regime_independence_score(regime_ics)
        assert result.score == 5

    def test_two_positive_regimes_returns_3(self) -> None:
        """2개 레짐 IC 양수 → 3점."""
        regime_ics = {"trending": 0.03, "ranging": -0.01, "volatile": 0.02}
        result = compute_regime_independence_score(regime_ics)
        assert result.score == 3

    def test_one_positive_regime_returns_1(self) -> None:
        """1개 레짐만 IC 양수 → 1점."""
        regime_ics = {"trending": 0.03, "ranging": -0.01, "volatile": -0.02}
        result = compute_regime_independence_score(regime_ics)
        assert result.score == 1
        assert "단일 레짐" in result.reason

    def test_no_positive_regimes_returns_1(self) -> None:
        """모든 레짐 IC 음수 → 1점."""
        regime_ics = {"trending": -0.03, "ranging": -0.01, "volatile": -0.02}
        result = compute_regime_independence_score(regime_ics)
        assert result.score == 1
        assert "모든 레짐" in result.reason

    def test_zero_ic_not_counted(self) -> None:
        """IC=0은 양수로 카운트 안됨."""
        regime_ics = {"trending": 0.0, "ranging": 0.0, "volatile": 0.01}
        result = compute_regime_independence_score(regime_ics)
        assert result.score == 1

    def test_evidence_contains_all_regimes(self) -> None:
        regime_ics = {"trending": 0.03, "ranging": 0.01, "volatile": 0.02}
        result = compute_regime_independence_score(regime_ics)
        assert "trending" in result.evidence
        assert "ranging" in result.evidence
        assert "volatile" in result.evidence
