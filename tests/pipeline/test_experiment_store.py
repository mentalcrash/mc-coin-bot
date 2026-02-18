"""ExperimentStore 테스트.

Models + YAML CRUD + analyze 검증.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pipeline.experiment_models import AssetResult, ExperimentAnalysis, ExperimentRecord
from src.pipeline.experiment_store import ExperimentStore

# ─── Fixtures ────────────────────────────────────────────────────────


def _make_asset_result(symbol: str = "BTC/USDT", sharpe: float = 1.5) -> AssetResult:
    return AssetResult(
        symbol=symbol,
        sharpe=sharpe,
        cagr=0.30,
        mdd=0.15,
        trades=120,
        win_rate=0.55,
        profit_factor=1.8,
    )


def _make_record(
    strategy: str = "ctrend",
    gate: str = "G2",
    ts: datetime | None = None,
    passed: bool = True,
    sharpe: float = 1.5,
) -> ExperimentRecord:
    return ExperimentRecord(
        strategy_name=strategy,
        gate_id=gate,
        timestamp=ts or datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC),
        params={"lookback": 60, "threshold": 0.5},
        asset_results=[_make_asset_result(sharpe=sharpe)],
        passed=passed,
        rationale="OOS Sharpe > 1.0",
        duration_seconds=12.5,
    )


# ─── TestAssetResult ─────────────────────────────────────────────────


class TestAssetResult:
    def test_creation(self) -> None:
        ar = _make_asset_result()
        assert ar.symbol == "BTC/USDT"
        assert ar.sharpe == 1.5
        assert ar.trades == 120

    def test_frozen(self) -> None:
        ar = _make_asset_result()
        with pytest.raises(ValidationError):
            ar.symbol = "ETH/USDT"  # type: ignore[misc]


# ─── TestExperimentRecord ────────────────────────────────────────────


class TestExperimentRecord:
    def test_creation_with_asset_results(self) -> None:
        record = _make_record()
        assert record.strategy_name == "ctrend"
        assert record.gate_id == "G2"
        assert len(record.asset_results) == 1
        assert record.passed is True

    def test_model_dump_json(self) -> None:
        record = _make_record()
        data = record.model_dump(mode="json")
        assert isinstance(data, dict)
        assert isinstance(data["timestamp"], str)
        assert data["strategy_name"] == "ctrend"
        assert len(data["asset_results"]) == 1


# ─── TestExperimentStore ─────────────────────────────────────────────


class TestExperimentStore:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        store = ExperimentStore(base_dir=tmp_path)
        record = _make_record()
        path = store.save(record)
        assert path.exists()
        assert path.suffix == ".yaml"
        assert "ctrend" in str(path)

    def test_load_all_for_strategy(self, tmp_path: Path) -> None:
        store = ExperimentStore(base_dir=tmp_path)
        r1 = _make_record(ts=datetime(2026, 1, 10, 8, 0, 0, tzinfo=UTC))
        r2 = _make_record(ts=datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC))
        store.save(r1)
        store.save(r2)

        records = store.load_all_for_strategy("ctrend")
        assert len(records) == 2
        assert records[0].timestamp < records[1].timestamp

    def test_get_latest(self, tmp_path: Path) -> None:
        store = ExperimentStore(base_dir=tmp_path)
        r1 = _make_record(ts=datetime(2026, 1, 10, 8, 0, 0, tzinfo=UTC))
        r2 = _make_record(ts=datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC))
        store.save(r1)
        store.save(r2)

        latest = store.get_latest("ctrend")
        assert latest is not None
        assert latest.timestamp == r2.timestamp

    def test_get_latest_filtered(self, tmp_path: Path) -> None:
        store = ExperimentStore(base_dir=tmp_path)
        r1 = _make_record(
            gate="G1", ts=datetime(2026, 1, 10, 8, 0, 0, tzinfo=UTC)
        )
        r2 = _make_record(
            gate="G2", ts=datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        )
        r3 = _make_record(
            gate="G1", ts=datetime(2026, 1, 20, 12, 0, 0, tzinfo=UTC)
        )
        store.save(r1)
        store.save(r2)
        store.save(r3)

        latest_g1 = store.get_latest("ctrend", gate_id="G1")
        assert latest_g1 is not None
        assert latest_g1.gate_id == "G1"
        assert latest_g1.timestamp == r3.timestamp

        latest_g2 = store.get_latest("ctrend", gate_id="G2")
        assert latest_g2 is not None
        assert latest_g2.gate_id == "G2"

    def test_analyze(self, tmp_path: Path) -> None:
        store = ExperimentStore(base_dir=tmp_path)
        r1 = _make_record(
            gate="G1",
            passed=True,
            sharpe=1.2,
            ts=datetime(2026, 1, 10, 8, 0, 0, tzinfo=UTC),
        )
        r2 = _make_record(
            gate="G2",
            passed=False,
            sharpe=0.8,
            ts=datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC),
        )
        r3 = _make_record(
            gate="G3",
            passed=True,
            sharpe=2.0,
            ts=datetime(2026, 1, 20, 12, 0, 0, tzinfo=UTC),
        )
        store.save(r1)
        store.save(r2)
        store.save(r3)

        analysis = store.analyze("ctrend")
        assert analysis is not None
        assert isinstance(analysis, ExperimentAnalysis)
        assert analysis.total_experiments == 3
        assert analysis.pass_rate == pytest.approx(2 / 3)
        assert analysis.best_gate == "G3"
        assert analysis.best_sharpe == pytest.approx(2.0)
        assert analysis.avg_mdd == pytest.approx(0.15)

    def test_empty_strategy(self, tmp_path: Path) -> None:
        store = ExperimentStore(base_dir=tmp_path)
        records = store.load_all_for_strategy("nonexistent")
        assert records == []

    def test_get_latest_none(self, tmp_path: Path) -> None:
        store = ExperimentStore(base_dir=tmp_path)
        result = store.get_latest("nonexistent")
        assert result is None
