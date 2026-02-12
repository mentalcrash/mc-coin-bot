"""Pipeline test fixtures."""

from __future__ import annotations

from datetime import date

import pytest

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


@pytest.fixture
def sample_meta() -> StrategyMeta:
    return StrategyMeta(
        name="ctrend",
        display_name="CTREND",
        category="ML Ensemble",
        timeframe="1D",
        short_mode="FULL",
        status=StrategyStatus.ACTIVE,
        created_at=date(2026, 2, 10),
        economic_rationale="다수 기술적 지표의 앙상블 합산으로 개별 지표 노이즈를 상쇄",
    )


@pytest.fixture
def sample_assets() -> list[AssetMetrics]:
    return [
        AssetMetrics(
            symbol="SOL/USDT",
            sharpe=2.05,
            cagr=97.8,
            mdd=27.7,
            trades=288,
            profit_factor=1.60,
            win_rate=57.1,
            alpha=3072.9,
            beta=0.11,
        ),
        AssetMetrics(
            symbol="BTC/USDT",
            sharpe=1.70,
            cagr=72.5,
            mdd=31.7,
            trades=304,
            profit_factor=1.71,
        ),
        AssetMetrics(
            symbol="ETH/USDT",
            sharpe=1.53,
            cagr=64.1,
            mdd=41.5,
            trades=321,
            profit_factor=1.37,
        ),
    ]


@pytest.fixture
def sample_gates() -> dict[GateId, GateResult]:
    return {
        GateId.G0A: GateResult(
            status=GateVerdict.PASS,
            date=date(2026, 2, 10),
            details={"score": 22, "max_score": 30},
        ),
        GateId.G1: GateResult(
            status=GateVerdict.PASS,
            date=date(2026, 2, 10),
            details={"best_asset": "SOL/USDT", "sharpe": 2.05, "cagr": 97.8},
        ),
        GateId.G2: GateResult(
            status=GateVerdict.PASS,
            date=date(2026, 2, 10),
            details={"is_sharpe": 2.69, "oos_sharpe": 1.78, "decay": 33.7},
        ),
    }


@pytest.fixture
def sample_decisions() -> list[Decision]:
    return [
        Decision(
            date=date(2026, 2, 10), gate=GateId.G0A, verdict=GateVerdict.PASS, rationale="22/30점"
        ),
        Decision(
            date=date(2026, 2, 10),
            gate=GateId.G1,
            verdict=GateVerdict.PASS,
            rationale="SOL Sharpe 2.05",
        ),
    ]


@pytest.fixture
def sample_record(
    sample_meta: StrategyMeta,
    sample_assets: list[AssetMetrics],
    sample_gates: dict[GateId, GateResult],
    sample_decisions: list[Decision],
) -> StrategyRecord:
    return StrategyRecord(
        meta=sample_meta,
        parameters={"training_window": 252, "vol_target": 0.35},
        gates=sample_gates,
        asset_performance=sample_assets,
        decisions=sample_decisions,
    )


@pytest.fixture
def retired_record() -> StrategyRecord:
    """G1 FAIL 폐기 전략."""
    return StrategyRecord(
        meta=StrategyMeta(
            name="bb-rsi",
            display_name="BB-RSI",
            category="Mean Reversion",
            timeframe="1D",
            short_mode="HEDGE_ONLY",
            status=StrategyStatus.RETIRED,
            created_at=date(2026, 1, 15),
            retired_at=date(2026, 2, 10),
            economic_rationale="BB + RSI 과매도 반전",
        ),
        gates={
            GateId.G0A: GateResult(
                status=GateVerdict.PASS,
                date=date(2026, 1, 15),
                details={"score": 20},
            ),
            GateId.G1: GateResult(
                status=GateVerdict.FAIL,
                date=date(2026, 2, 10),
                details={"sharpe": 0.59, "cagr": 4.6},
            ),
        },
        asset_performance=[
            AssetMetrics(symbol="SOL/USDT", sharpe=0.59, cagr=4.6, mdd=35.0, trades=120),
        ],
        decisions=[
            Decision(
                date=date(2026, 2, 10),
                gate=GateId.G1,
                verdict=GateVerdict.FAIL,
                rationale="CAGR < 20%",
            ),
        ],
    )
