"""Pipeline test fixtures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from src.pipeline.lesson_models import LessonCategory, LessonRecord
from src.pipeline.models import (
    AssetMetrics,
    Decision,
    PhaseId,
    PhaseResult,
    PhaseVerdict,
    StrategyMeta,
    StrategyRecord,
    StrategyStatus,
)
from src.pipeline.phase_criteria_store import PhaseCriteriaStore


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
def sample_phases() -> dict[PhaseId, PhaseResult]:
    return {
        PhaseId.P1: PhaseResult(
            status=PhaseVerdict.PASS,
            date=date(2026, 2, 10),
            details={"score": 22, "max_score": 30},
        ),
        PhaseId.P4: PhaseResult(
            status=PhaseVerdict.PASS,
            date=date(2026, 2, 10),
            details={"best_asset": "SOL/USDT", "sharpe": 2.05, "cagr": 97.8},
        ),
    }


@pytest.fixture
def sample_decisions() -> list[Decision]:
    return [
        Decision(
            date=date(2026, 2, 10), phase=PhaseId.P1, verdict=PhaseVerdict.PASS, rationale="22/30점"
        ),
        Decision(
            date=date(2026, 2, 10),
            phase=PhaseId.P4,
            verdict=PhaseVerdict.PASS,
            rationale="SOL Sharpe 2.05",
        ),
    ]


@pytest.fixture
def sample_record(
    sample_meta: StrategyMeta,
    sample_assets: list[AssetMetrics],
    sample_phases: dict[PhaseId, PhaseResult],
    sample_decisions: list[Decision],
) -> StrategyRecord:
    return StrategyRecord(
        meta=sample_meta,
        parameters={"training_window": 252, "vol_target": 0.35},
        phases=sample_phases,
        asset_performance=sample_assets,
        decisions=sample_decisions,
    )


@pytest.fixture
def retired_record() -> StrategyRecord:
    """P4 FAIL 폐기 전략."""
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
        phases={
            PhaseId.P1: PhaseResult(
                status=PhaseVerdict.PASS,
                date=date(2026, 1, 15),
                details={"score": 20},
            ),
            PhaseId.P4: PhaseResult(
                status=PhaseVerdict.FAIL,
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
                phase=PhaseId.P4,
                verdict=PhaseVerdict.FAIL,
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


_SAMPLE_PHASE_YAML = {
    "phases": [
        {
            "phase_id": "P1",
            "name": "아이디어 검증",
            "phase_type": "scoring",
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
            "phase_id": "P4",
            "name": "단일에셋 백테스트",
            "phase_type": "threshold",
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
def phase_yaml_path(tmp_path: Path) -> Path:
    """임시 phase-criteria.yaml 작성."""
    phase_dir = tmp_path / "gates"
    phase_dir.mkdir(exist_ok=True)
    path = phase_dir / "phase-criteria.yaml"
    path.write_text(
        yaml.dump(_SAMPLE_PHASE_YAML, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def phase_store(phase_yaml_path: Path) -> PhaseCriteriaStore:
    """임시 PhaseCriteriaStore."""
    return PhaseCriteriaStore(path=phase_yaml_path)
