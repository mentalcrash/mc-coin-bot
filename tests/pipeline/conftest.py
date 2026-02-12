"""Pipeline test fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from src.pipeline.gate_store import GateCriteriaStore
from src.pipeline.lesson_models import LessonCategory, LessonRecord
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


@pytest.fixture
def sample_lesson() -> LessonRecord:
    """strategy-design 카테고리 교훈."""
    return LessonRecord(
        id=1,
        title="앙상블 > 단일지표",
        body="ML 앙상블(CTREND)의 낮은 Decay(33.7%)가 단일 팩터 전략 대비 일반화 우수",
        category=LessonCategory.STRATEGY_DESIGN,
        tags=["ML", "ensemble", "decay"],
        strategies=["ctrend"],
        timeframes=["1D"],
        added_at=date(2026, 2, 10),
    )


@pytest.fixture
def sample_lesson_market() -> LessonRecord:
    """market-structure 카테고리 교훈."""
    return LessonRecord(
        id=13,
        title="FX Session Edge ≠ Crypto Edge",
        body="Asian session breakout은 FX 시장의 institutional flow 시간대 분리에 기반. 크립토 24/7 시장에서 session 분리는 구조적으로 무효",
        category=LessonCategory.MARKET_STRUCTURE,
        tags=["session", "FX", "crypto"],
        strategies=["session-breakout"],
        timeframes=["1H"],
        added_at=date(2026, 2, 10),
    )


_SAMPLE_GATE_YAML = {
    "gates": [
        {
            "gate_id": "G0A",
            "name": "아이디어 검증",
            "gate_type": "scoring",
            "scoring": {
                "pass_threshold": 18,
                "max_total": 30,
                "items": [
                    {"name": "경제적 논거", "description": "5=행동편향"},
                    {"name": "참신성", "description": "5=미공개"},
                ],
            },
        },
        {
            "gate_id": "G1",
            "name": "단일에셋 백테스트",
            "gate_type": "threshold",
            "cli_command": "run {config}",
            "threshold": {
                "pass_metrics": [
                    {"name": "Sharpe", "operator": ">", "value": 1.0},
                    {"name": "CAGR", "operator": ">", "value": 20.0, "unit": "%"},
                    {"name": "MDD", "operator": "<", "value": 40.0, "unit": "%"},
                    {"name": "Trades", "operator": ">", "value": 50},
                ],
            },
        },
    ],
}


@pytest.fixture
def gate_yaml_path(tmp_path: Path) -> Path:
    """임시 criteria.yaml 작성."""
    gate_dir = tmp_path / "gates"
    gate_dir.mkdir(exist_ok=True)
    path = gate_dir / "criteria.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_GATE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def gate_store(gate_yaml_path: Path) -> GateCriteriaStore:
    """임시 GateCriteriaStore."""
    return GateCriteriaStore(path=gate_yaml_path)
