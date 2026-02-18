"""Tests for src/pipeline/models.py."""

from __future__ import annotations

from datetime import date

import pytest

from src.pipeline.models import (
    PHASE_ORDER,
    AssetMetrics,
    Decision,
    PhaseId,
    PhaseResult,
    PhaseVerdict,
    RationaleReference,
    RationaleRefType,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)


class TestEnums:
    def test_strategy_status_values(self) -> None:
        assert StrategyStatus.CANDIDATE == "CANDIDATE"
        assert StrategyStatus.ACTIVE == "ACTIVE"
        assert StrategyStatus.RETIRED == "RETIRED"

    def test_phase_id_values(self) -> None:
        assert PhaseId.P1 == "P1"
        assert PhaseId.P7 == "P7"

    def test_phase_verdict_values(self) -> None:
        assert PhaseVerdict.PASS == "PASS"
        assert PhaseVerdict.FAIL == "FAIL"

    def test_phase_order_length(self) -> None:
        assert len(PHASE_ORDER) == 7
        assert PHASE_ORDER[0] == PhaseId.P1
        assert PHASE_ORDER[-1] == PhaseId.P7

    def test_phase_order_sequential(self) -> None:
        for i, pid in enumerate(PHASE_ORDER):
            assert pid == f"P{i + 1}"


class TestStrategyMeta:
    def test_creation(self, sample_meta: StrategyMeta) -> None:
        assert sample_meta.name == "ctrend"
        assert sample_meta.timeframe == "1D"
        assert sample_meta.status == StrategyStatus.ACTIVE

    def test_frozen(self, sample_meta: StrategyMeta) -> None:
        with pytest.raises(Exception):  # noqa: B017
            sample_meta.name = "changed"  # type: ignore[misc]


class TestRationaleReference:
    def test_creation(self) -> None:
        ref = RationaleReference(
            type=RationaleRefType.PAPER,
            title="Moskowitz et al. (2012)",
            source="JFE",
            url="https://example.com",
            relevance="TSMOM 원본 논문",
        )
        assert ref.type == RationaleRefType.PAPER
        assert ref.title == "Moskowitz et al. (2012)"

    def test_frozen(self) -> None:
        ref = RationaleReference(type=RationaleRefType.LESSON, title="Test")
        with pytest.raises(Exception):  # noqa: B017
            ref.title = "Changed"  # type: ignore[misc]

    def test_default_fields(self) -> None:
        ref = RationaleReference(type=RationaleRefType.PRIOR_STRATEGY)
        assert ref.title == ""
        assert ref.source == ""
        assert ref.url == ""
        assert ref.relevance == ""

    def test_ref_type_values(self) -> None:
        assert RationaleRefType.PAPER == "paper"
        assert RationaleRefType.LESSON == "lesson"
        assert RationaleRefType.PRIOR_STRATEGY == "prior_strategy"


class TestStrategyMetaRationale:
    def test_default_rationale_fields(self) -> None:
        meta = StrategyMeta(
            name="test",
            display_name="Test",
            category="Test",
            timeframe="1D",
            short_mode="DISABLED",
            status=StrategyStatus.CANDIDATE,
            created_at=date(2026, 1, 1),
        )
        assert meta.rationale_references == []
        assert meta.rationale_category is None

    def test_with_rationale_fields(self) -> None:
        ref = RationaleReference(type=RationaleRefType.PAPER, title="Test Paper")
        meta = StrategyMeta(
            name="test",
            display_name="Test",
            category="Test",
            timeframe="1D",
            short_mode="DISABLED",
            status=StrategyStatus.CANDIDATE,
            created_at=date(2026, 1, 1),
            rationale_references=[ref],
            rationale_category="momentum",
        )
        assert len(meta.rationale_references) == 1
        assert meta.rationale_references[0].type == RationaleRefType.PAPER
        assert meta.rationale_category == "momentum"


class TestAssetMetrics:
    def test_creation_minimal(self) -> None:
        m = AssetMetrics(symbol="BTC/USDT", sharpe=1.5, cagr=50.0, mdd=20.0, trades=100)
        assert m.profit_factor is None
        assert m.win_rate is None

    def test_creation_full(self, sample_assets: list[AssetMetrics]) -> None:
        sol = sample_assets[0]
        assert sol.symbol == "SOL/USDT"
        assert sol.sharpe == 2.05
        assert sol.profit_factor == 1.60


class TestPhaseResult:
    def test_creation(self) -> None:
        r = PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 2, 10), details={"sharpe": 2.05})
        assert r.status == PhaseVerdict.PASS
        assert r.details["sharpe"] == 2.05

    def test_empty_details(self) -> None:
        r = PhaseResult(status=PhaseVerdict.FAIL, date=date(2026, 2, 10))
        assert r.details == {}


class TestDecision:
    def test_creation(self) -> None:
        d = Decision(
            date=date(2026, 2, 10), phase=PhaseId.P4, verdict=PhaseVerdict.PASS, rationale="Good"
        )
        assert d.phase == PhaseId.P4


class TestStrategyRecord:
    def test_best_asset(self, sample_record: StrategyRecord) -> None:
        assert sample_record.best_asset == "SOL/USDT"

    def test_best_asset_none(self) -> None:
        record = StrategyRecord(
            meta=StrategyMeta(
                name="empty",
                display_name="Empty",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.CANDIDATE,
                created_at=date(2026, 1, 1),
            ),
        )
        assert record.best_asset is None

    def test_current_phase(self, sample_record: StrategyRecord) -> None:
        # P1 PASS, P4 PASS → current = P4
        assert sample_record.current_phase == "P4"

    def test_current_phase_with_gap(self) -> None:
        """P1 PASS, P2 없음 → current_phase = P1 (gap에서 중단)."""
        record = StrategyRecord(
            meta=StrategyMeta(
                name="test",
                display_name="Test",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.TESTING,
                created_at=date(2026, 1, 1),
            ),
            phases={
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            },
        )
        assert record.current_phase == "P1"

    def test_fail_phase(self, retired_record: StrategyRecord) -> None:
        assert retired_record.fail_phase == "P4"

    def test_fail_phase_none(self, sample_record: StrategyRecord) -> None:
        assert sample_record.fail_phase is None

    def test_best_sharpe(self, sample_record: StrategyRecord) -> None:
        assert sample_record.best_sharpe == 2.05

    def test_best_sharpe_none(self) -> None:
        record = StrategyRecord(
            meta=StrategyMeta(
                name="empty",
                display_name="Empty",
                category="Test",
                timeframe="1D",
                short_mode="DISABLED",
                status=StrategyStatus.CANDIDATE,
                created_at=date(2026, 1, 1),
            ),
        )
        assert record.best_sharpe is None

    def test_version_default(self, sample_record: StrategyRecord) -> None:
        assert sample_record.version == 2


def _make_record(
    phases: dict[PhaseId, PhaseResult] | None = None,
) -> StrategyRecord:
    """next_phase 테스트용 헬퍼."""
    return StrategyRecord(
        meta=StrategyMeta(
            name="test",
            display_name="Test",
            category="Test",
            timeframe="1D",
            short_mode="DISABLED",
            status=StrategyStatus.TESTING,
            created_at=date(2026, 1, 1),
        ),
        phases=phases or {},
    )


class TestNextPhase:
    def test_no_phases_returns_p1(self) -> None:
        record = _make_record()
        assert record.next_phase == "P1"

    def test_p1_pass_returns_p2(self) -> None:
        record = _make_record(
            {
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            }
        )
        assert record.next_phase == "P2"

    def test_p1_p2_p3_pass_returns_p4(self) -> None:
        record = _make_record(
            {
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P2: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P3: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            }
        )
        assert record.next_phase == "P4"

    def test_next_phase_after_p4_is_p5(self) -> None:
        """P4 PASS 후 next_phase = P5."""
        record = _make_record(
            {
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P2: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P3: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P4: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            }
        )
        assert record.next_phase == "P5"

    def test_next_phase_after_p5_is_p6(self) -> None:
        """P5 PASS 후 next_phase = P6."""
        record = _make_record(
            {
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P2: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P3: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P4: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P5: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            }
        )
        assert record.next_phase == "P6"

    def test_fail_returns_none(self) -> None:
        record = _make_record(
            {
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P4: PhaseResult(status=PhaseVerdict.FAIL, date=date(2026, 1, 1)),
            }
        )
        assert record.next_phase is None

    def test_all_pass_returns_none(self) -> None:
        all_pass = {
            pid: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)) for pid in PHASE_ORDER
        }
        record = _make_record(all_pass)
        assert record.next_phase is None

    def test_gap_returns_missing_phase(self) -> None:
        """P1 PASS, P2 없음, P3 PASS → P2 (gap 위치)."""
        record = _make_record(
            {
                PhaseId.P1: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
                PhaseId.P3: PhaseResult(status=PhaseVerdict.PASS, date=date(2026, 1, 1)),
            }
        )
        assert record.next_phase == "P2"
