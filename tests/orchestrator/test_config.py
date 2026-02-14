"""Tests for orchestrator configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.orchestrator.config import (
    GraduationCriteria,
    OrchestratorConfig,
    PodConfig,
    RetirementCriteria,
)
from src.orchestrator.models import AllocationMethod, RebalanceTrigger

# ── Helpers ─────────────────────────────────────────────────────


def _make_pod(**overrides: object) -> PodConfig:
    defaults: dict[str, object] = {
        "pod_id": "pod-tsmom",
        "strategy_name": "tsmom",
        "symbols": ("BTC/USDT",),
    }
    defaults.update(overrides)
    return PodConfig(**defaults)  # type: ignore[arg-type]


def _make_two_pods() -> tuple[PodConfig, PodConfig]:
    p1 = _make_pod(pod_id="pod-a", initial_fraction=0.30)
    p2 = _make_pod(pod_id="pod-b", initial_fraction=0.30, symbols=("ETH/USDT",))
    return p1, p2


# ── PodConfig ───────────────────────────────────────────────────


class TestPodConfig:
    def test_defaults(self) -> None:
        pod = _make_pod()
        assert pod.pod_id == "pod-tsmom"
        assert pod.strategy_name == "tsmom"
        assert pod.symbols == ("BTC/USDT",)
        assert pod.timeframe == "1D"
        assert pod.initial_fraction == 0.10
        assert pod.max_fraction == 0.40
        assert pod.min_fraction == 0.02
        assert pod.max_drawdown == 0.15
        assert pod.drawdown_warning == 0.10
        assert pod.max_leverage == 2.0
        assert pod.system_stop_loss == 0.10
        assert pod.use_trailing_stop is False
        assert pod.trailing_stop_atr_multiplier == 3.0
        assert pod.rebalance_threshold == 0.05

    def test_frozen(self) -> None:
        pod = _make_pod()
        with pytest.raises(ValidationError):
            pod.pod_id = "changed"  # type: ignore[misc]

    def test_min_greater_than_initial_rejected(self) -> None:
        with pytest.raises(ValidationError, match="min_fraction"):
            _make_pod(min_fraction=0.20, initial_fraction=0.10)

    def test_initial_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="initial_fraction"):
            _make_pod(initial_fraction=0.50, max_fraction=0.40)

    def test_warning_ge_max_drawdown_rejected(self) -> None:
        with pytest.raises(ValidationError, match="drawdown_warning"):
            _make_pod(drawdown_warning=0.15, max_drawdown=0.15)

    def test_symbols_min_length(self) -> None:
        with pytest.raises(ValidationError):
            _make_pod(symbols=())

    def test_custom_params(self) -> None:
        pod = _make_pod(strategy_params={"lookback": 63, "vol_target": 0.35})
        assert pod.strategy_params["lookback"] == 63
        assert pod.strategy_params["vol_target"] == 0.35

    def test_json_serialization(self) -> None:
        pod = _make_pod()
        data = pod.model_dump(mode="json")
        restored = PodConfig.model_validate(data)
        assert restored == pod

    def test_system_stop_loss_none(self) -> None:
        pod = _make_pod(system_stop_loss=None)
        assert pod.system_stop_loss is None


# ── GraduationCriteria ──────────────────────────────────────────


class TestGraduationCriteria:
    def test_defaults(self) -> None:
        g = GraduationCriteria()
        assert g.min_live_days == 90
        assert g.min_sharpe == 1.0
        assert g.max_drawdown == 0.15
        assert g.min_trade_count == 30
        assert g.min_calmar == 0.8
        assert g.max_backtest_live_gap == 0.30
        assert g.max_portfolio_correlation == 0.50

    def test_frozen(self) -> None:
        g = GraduationCriteria()
        with pytest.raises(ValidationError):
            g.min_live_days = 120  # type: ignore[misc]

    def test_custom_values(self) -> None:
        g = GraduationCriteria(min_live_days=60, min_sharpe=1.5, min_trade_count=50)
        assert g.min_live_days == 60
        assert g.min_sharpe == 1.5
        assert g.min_trade_count == 50


# ── RetirementCriteria ──────────────────────────────────────────


class TestRetirementCriteria:
    def test_defaults(self) -> None:
        r = RetirementCriteria()
        assert r.max_drawdown_breach == 0.25
        assert r.consecutive_loss_months == 6
        assert r.rolling_sharpe_floor == 0.3
        assert r.probation_days == 30

    def test_frozen(self) -> None:
        r = RetirementCriteria()
        with pytest.raises(ValidationError):
            r.max_drawdown_breach = 0.50  # type: ignore[misc]


# ── OrchestratorConfig ──────────────────────────────────────────


class TestOrchestratorConfig:
    def test_defaults(self) -> None:
        cfg = OrchestratorConfig(pods=(_make_pod(),))
        assert cfg.allocation_method == AllocationMethod.RISK_PARITY
        assert cfg.rebalance_trigger == RebalanceTrigger.HYBRID
        assert cfg.kelly_fraction == 0.25
        assert cfg.max_portfolio_volatility == 0.20
        assert cfg.max_portfolio_drawdown == 0.15
        assert cfg.max_gross_leverage == 3.0
        assert cfg.daily_loss_limit == 0.03
        assert cfg.cost_bps == 4.0
        assert cfg.correlation_lookback == 90

    def test_frozen(self) -> None:
        cfg = OrchestratorConfig(pods=(_make_pod(),))
        with pytest.raises(ValidationError):
            cfg.cost_bps = 5.0  # type: ignore[misc]

    def test_empty_pods_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrchestratorConfig(pods=())

    def test_duplicate_pod_ids_rejected(self) -> None:
        p1 = _make_pod(pod_id="same")
        p2 = _make_pod(pod_id="same", symbols=("ETH/USDT",))
        with pytest.raises(ValidationError, match="Duplicate pod_id"):
            OrchestratorConfig(pods=(p1, p2))

    def test_initial_sum_exceeds_one_rejected(self) -> None:
        p1 = _make_pod(pod_id="a", initial_fraction=0.60, max_fraction=0.80)
        p2 = _make_pod(pod_id="b", initial_fraction=0.50, max_fraction=0.80, symbols=("ETH/USDT",))
        with pytest.raises(ValidationError, match=r"exceeds 1\.0"):
            OrchestratorConfig(pods=(p1, p2))

    def test_two_pods_valid(self) -> None:
        p1, p2 = _make_two_pods()
        cfg = OrchestratorConfig(pods=(p1, p2))
        assert cfg.n_pods == 2

    def test_all_symbols_dedup(self) -> None:
        p1 = _make_pod(pod_id="a", symbols=("BTC/USDT", "ETH/USDT"))
        p2 = _make_pod(pod_id="b", symbols=("ETH/USDT", "SOL/USDT"))
        cfg = OrchestratorConfig(pods=(p1, p2))
        assert cfg.all_symbols == ("BTC/USDT", "ETH/USDT", "SOL/USDT")

    def test_json_serialization(self) -> None:
        cfg = OrchestratorConfig(pods=(_make_pod(),))
        data = cfg.model_dump(mode="json")
        restored = OrchestratorConfig.model_validate(data)
        assert restored == cfg

    def test_custom_allocation(self) -> None:
        cfg = OrchestratorConfig(
            pods=(_make_pod(),),
            allocation_method=AllocationMethod.ADAPTIVE_KELLY,
            kelly_fraction=0.50,
        )
        assert cfg.allocation_method == AllocationMethod.ADAPTIVE_KELLY
        assert cfg.kelly_fraction == 0.50

    def test_graduation_retirement_defaults(self) -> None:
        cfg = OrchestratorConfig(pods=(_make_pod(),))
        assert isinstance(cfg.graduation, GraduationCriteria)
        assert isinstance(cfg.retirement, RetirementCriteria)
        assert cfg.graduation.min_live_days == 90
        assert cfg.retirement.probation_days == 30
