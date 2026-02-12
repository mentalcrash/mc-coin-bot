"""Tests for src/pipeline/models.py."""

from __future__ import annotations

from datetime import date

import pytest

from src.pipeline.models import (
    GATE_ORDER,
    AssetMetrics,
    Decision,
    GateId,
    GateResult,
    GateVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)


class TestEnums:
    def test_strategy_status_values(self) -> None:
        assert StrategyStatus.CANDIDATE == "CANDIDATE"
        assert StrategyStatus.ACTIVE == "ACTIVE"
        assert StrategyStatus.RETIRED == "RETIRED"

    def test_gate_id_values(self) -> None:
        assert GateId.G0A == "G0A"
        assert GateId.G5 == "G5"
        assert GateId.G7 == "G7"

    def test_gate_verdict_values(self) -> None:
        assert GateVerdict.PASS == "PASS"
        assert GateVerdict.FAIL == "FAIL"

    def test_gate_order_length(self) -> None:
        assert len(GATE_ORDER) == 9
        assert GATE_ORDER[0] == GateId.G0A
        assert GATE_ORDER[-1] == GateId.G7


class TestStrategyMeta:
    def test_creation(self, sample_meta: StrategyMeta) -> None:
        assert sample_meta.name == "ctrend"
        assert sample_meta.timeframe == "1D"
        assert sample_meta.status == StrategyStatus.ACTIVE

    def test_frozen(self, sample_meta: StrategyMeta) -> None:
        with pytest.raises(Exception):  # noqa: B017
            sample_meta.name = "changed"  # type: ignore[misc]


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


class TestGateResult:
    def test_creation(self) -> None:
        r = GateResult(status=GateVerdict.PASS, date=date(2026, 2, 10), details={"sharpe": 2.05})
        assert r.status == GateVerdict.PASS
        assert r.details["sharpe"] == 2.05

    def test_empty_details(self) -> None:
        r = GateResult(status=GateVerdict.FAIL, date=date(2026, 2, 10))
        assert r.details == {}


class TestDecision:
    def test_creation(self) -> None:
        d = Decision(
            date=date(2026, 2, 10), gate=GateId.G1, verdict=GateVerdict.PASS, rationale="Good"
        )
        assert d.gate == GateId.G1


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

    def test_current_gate(self, sample_record: StrategyRecord) -> None:
        # G0A PASS, G1 PASS, G2 PASS → current = G2
        assert sample_record.current_gate == "G2"

    def test_current_gate_with_gap(self) -> None:
        """G0A PASS, G0B 없음 → current_gate = G0A (gap에서 중단)."""
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
            gates={
                GateId.G0A: GateResult(status=GateVerdict.PASS, date=date(2026, 1, 1)),
            },
        )
        assert record.current_gate == "G0A"

    def test_fail_gate(self, retired_record: StrategyRecord) -> None:
        assert retired_record.fail_gate == "G1"

    def test_fail_gate_none(self, sample_record: StrategyRecord) -> None:
        assert sample_record.fail_gate is None

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
