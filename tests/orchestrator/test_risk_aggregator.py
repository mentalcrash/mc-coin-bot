"""Tests for Risk Aggregator — 포트폴리오 리스크 집계·검사."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod, PodPerformance, RiskAlert
from src.orchestrator.risk_aggregator import (
    RiskAggregator,
    check_asset_correlation_stress,
    check_correlation_stress,
    compute_effective_n,
    compute_portfolio_drawdown,
    compute_risk_contributions,
)

# ── Helpers ──────────────────────────────────────────────────────


def _make_pod_config(pod_id: str, symbols: tuple[str, ...] = ("BTC/USDT",)) -> PodConfig:
    return PodConfig(
        pod_id=pod_id,
        strategy_name="tsmom",
        symbols=symbols,
        initial_fraction=0.10,
        max_fraction=0.40,
        min_fraction=0.02,
    )


def _make_config(**overrides: object) -> OrchestratorConfig:
    defaults: dict[str, object] = {
        "pods": (
            _make_pod_config("pod-a"),
            _make_pod_config("pod-b"),
        ),
        "allocation_method": AllocationMethod.EQUAL_WEIGHT,
        "max_gross_leverage": 3.0,
        "max_portfolio_drawdown": 0.15,
        "daily_loss_limit": 0.03,
        "max_single_pod_risk_pct": 0.40,
        "correlation_stress_threshold": 0.70,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_pod_returns(n_days: int = 60, n_pods: int = 2) -> pd.DataFrame:
    """랜덤 Pod 수익률 DataFrame."""
    rng = np.random.default_rng(42)
    data = rng.normal(0.001, 0.02, (n_days, n_pods))
    return pd.DataFrame(data, columns=[f"pod-{chr(97 + i)}" for i in range(n_pods)])


# ── TestComputeRiskContributions ─────────────────────────────────


class TestComputeRiskContributions:
    def test_equal_weights_equal_vol(self) -> None:
        """동일 가중·동일 변동성 → PRC ≈ 균등."""
        returns = _make_pod_returns(100, 2)
        prc = compute_risk_contributions(returns, {"pod-a": 0.5, "pod-b": 0.5})
        assert len(prc) == 2
        assert sum(prc.values()) == pytest.approx(1.0, abs=0.01)

    def test_single_pod_100pct(self) -> None:
        returns = _make_pod_returns(100, 2)
        prc = compute_risk_contributions(returns, {"pod-a": 1.0, "pod-b": 0.0})
        # pod-a가 대부분의 리스크
        assert prc["pod-a"] > 0.9

    def test_empty_weights(self) -> None:
        returns = _make_pod_returns(100, 2)
        prc = compute_risk_contributions(returns, {})
        assert prc == {}

    def test_missing_columns(self) -> None:
        """weights에 있지만 returns에 없는 pod → 무시."""
        returns = _make_pod_returns(100, 2)
        prc = compute_risk_contributions(returns, {"pod-a": 0.5, "pod-c": 0.5})
        assert "pod-c" not in prc
        assert "pod-a" in prc


# ── TestComputeEffectiveN ────────────────────────────────────────


class TestComputeEffectiveN:
    def test_two_equal(self) -> None:
        """균등 PRC → N_eff = 2."""
        n_eff = compute_effective_n({"pod-a": 0.5, "pod-b": 0.5})
        assert n_eff == pytest.approx(2.0)

    def test_three_equal(self) -> None:
        n_eff = compute_effective_n({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
        assert n_eff == pytest.approx(3.0)

    def test_single_pod(self) -> None:
        n_eff = compute_effective_n({"pod-a": 1.0})
        assert n_eff == pytest.approx(1.0)

    def test_concentrated(self) -> None:
        """90/10 집중 → N_eff ≈ 1.22."""
        n_eff = compute_effective_n({"a": 0.9, "b": 0.1})
        assert 1.0 < n_eff < 2.0

    def test_empty(self) -> None:
        assert compute_effective_n({}) == 0.0


# ── TestCheckCorrelationStress ───────────────────────────────────


class TestCheckCorrelationStress:
    def test_uncorrelated_no_stress(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-a": rng.normal(0, 0.01, 200),
                "pod-b": rng.normal(0, 0.01, 200),
            }
        )
        is_stressed, avg_corr = check_correlation_stress(returns, 0.70)
        assert is_stressed is False
        assert avg_corr < 0.70

    def test_highly_correlated_stress(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, 200)
        returns = pd.DataFrame(
            {
                "pod-a": base,
                "pod-b": base + rng.normal(0, 0.001, 200),  # 높은 상관
            }
        )
        is_stressed, avg_corr = check_correlation_stress(returns, 0.70)
        assert is_stressed is True
        assert avg_corr > 0.70

    def test_single_pod_no_stress(self) -> None:
        returns = pd.DataFrame({"pod-a": np.random.default_rng(42).normal(0, 0.01, 100)})
        is_stressed, avg_corr = check_correlation_stress(returns, 0.70)
        assert is_stressed is False
        assert avg_corr == 0.0


# ── TestComputePortfolioDrawdown ─────────────────────────────────


class TestComputePortfolioDrawdown:
    def test_weighted_average(self) -> None:
        perfs = {
            "pod-a": PodPerformance(pod_id="pod-a", current_drawdown=0.10),
            "pod-b": PodPerformance(pod_id="pod-b", current_drawdown=0.20),
        }
        dd = compute_portfolio_drawdown(perfs, {"pod-a": 0.6, "pod-b": 0.4})
        # (0.10*0.6 + 0.20*0.4) / (0.6+0.4) = 0.14
        assert dd == pytest.approx(0.14)

    def test_zero_weights(self) -> None:
        perfs = {"pod-a": PodPerformance(pod_id="pod-a", current_drawdown=0.10)}
        dd = compute_portfolio_drawdown(perfs, {"pod-a": 0.0})
        assert dd == pytest.approx(0.0)

    def test_empty(self) -> None:
        dd = compute_portfolio_drawdown({}, {})
        assert dd == pytest.approx(0.0)


# ── TestRiskAggregatorLeverage ───────────────────────────────────


class TestRiskAggregatorLeverage:
    def test_within_limit_no_alert(self) -> None:
        config = _make_config(max_gross_leverage=3.0)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={"BTC/USDT": 1.0},
            pod_performances={},
            pod_weights={},
        )
        leverage_alerts = [a for a in alerts if a.alert_type == "gross_leverage"]
        assert len(leverage_alerts) == 0

    def test_warning_at_80pct(self) -> None:
        config = _make_config(max_gross_leverage=3.0)
        ra = RiskAggregator(config)
        # 2.5x → 80%+ of 3.0
        alerts = ra.check_portfolio_limits(
            net_weights={"BTC/USDT": 1.5, "ETH/USDT": 1.0},
            pod_performances={},
            pod_weights={},
        )
        leverage_alerts = [a for a in alerts if a.alert_type == "gross_leverage"]
        assert len(leverage_alerts) == 1
        assert leverage_alerts[0].severity == "warning"

    def test_critical_at_limit(self) -> None:
        config = _make_config(max_gross_leverage=3.0)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={"BTC/USDT": 2.0, "ETH/USDT": 1.5},
            pod_performances={},
            pod_weights={},
        )
        leverage_alerts = [a for a in alerts if a.alert_type == "gross_leverage"]
        assert len(leverage_alerts) == 1
        assert leverage_alerts[0].severity == "critical"


# ── TestRiskAggregatorDrawdown ───────────────────────────────────


class TestRiskAggregatorDrawdown:
    def test_no_drawdown_no_alert(self) -> None:
        config = _make_config(max_portfolio_drawdown=0.15)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={"pod-a": PodPerformance(pod_id="pod-a", current_drawdown=0.0)},
            pod_weights={"pod-a": 1.0},
        )
        dd_alerts = [a for a in alerts if a.alert_type == "portfolio_drawdown"]
        assert len(dd_alerts) == 0

    def test_critical_drawdown(self) -> None:
        config = _make_config(max_portfolio_drawdown=0.15)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={"pod-a": PodPerformance(pod_id="pod-a", current_drawdown=0.16)},
            pod_weights={"pod-a": 1.0},
        )
        dd_alerts = [a for a in alerts if a.alert_type == "portfolio_drawdown"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0].severity == "critical"


# ── TestRiskAggregatorDailyLoss ──────────────────────────────────


class TestRiskAggregatorDailyLoss:
    def test_no_loss_no_alert(self) -> None:
        config = _make_config(daily_loss_limit=0.03)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={},
            daily_pnl_pct=0.01,
        )
        loss_alerts = [a for a in alerts if a.alert_type == "daily_loss"]
        assert len(loss_alerts) == 0

    def test_critical_daily_loss(self) -> None:
        config = _make_config(daily_loss_limit=0.03)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={},
            daily_pnl_pct=-0.04,
        )
        loss_alerts = [a for a in alerts if a.alert_type == "daily_loss"]
        assert len(loss_alerts) == 1
        assert loss_alerts[0].severity == "critical"

    def test_warning_daily_loss(self) -> None:
        config = _make_config(daily_loss_limit=0.03)
        ra = RiskAggregator(config)
        # -0.025 → abs=0.025, threshold=0.03, 80%=0.024 → warning
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={},
            daily_pnl_pct=-0.025,
        )
        loss_alerts = [a for a in alerts if a.alert_type == "daily_loss"]
        assert len(loss_alerts) == 1
        assert loss_alerts[0].severity == "warning"

    def test_positive_pnl_no_alert(self) -> None:
        config = _make_config(daily_loss_limit=0.03)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={},
            daily_pnl_pct=0.05,
        )
        loss_alerts = [a for a in alerts if a.alert_type == "daily_loss"]
        assert len(loss_alerts) == 0


# ── TestRiskAggregatorPodRisk ────────────────────────────────────


class TestRiskAggregatorPodRisk:
    def test_concentrated_pod_critical(self) -> None:
        config = _make_config(max_single_pod_risk_pct=0.40)
        ra = RiskAggregator(config)
        returns = _make_pod_returns(100, 2)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={"pod-a": 0.95, "pod-b": 0.05},
            pod_returns=returns,
        )
        pod_alerts = [a for a in alerts if a.alert_type == "single_pod_risk"]
        # pod-a의 PRC가 높아서 critical 또는 warning
        assert len(pod_alerts) >= 1

    def test_balanced_pods_no_alert(self) -> None:
        config = _make_config(max_single_pod_risk_pct=0.70)
        ra = RiskAggregator(config)
        returns = _make_pod_returns(100, 2)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={"pod-a": 0.5, "pod-b": 0.5},
            pod_returns=returns,
        )
        pod_alerts = [a for a in alerts if a.alert_type == "single_pod_risk"]
        assert len(pod_alerts) == 0


# ── TestRiskAggregatorCorrelation ────────────────────────────────


class TestRiskAggregatorCorrelation:
    def test_high_correlation_critical(self) -> None:
        config = _make_config(correlation_stress_threshold=0.70)
        ra = RiskAggregator(config)
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, 200)
        returns = pd.DataFrame(
            {
                "pod-a": base,
                "pod-b": base + rng.normal(0, 0.001, 200),
            }
        )
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={"pod-a": 0.5, "pod-b": 0.5},
            pod_returns=returns,
        )
        corr_alerts = [a for a in alerts if a.alert_type == "correlation_stress"]
        assert len(corr_alerts) == 1
        assert corr_alerts[0].severity == "critical"

    def test_low_correlation_no_alert(self) -> None:
        config = _make_config(correlation_stress_threshold=0.70)
        ra = RiskAggregator(config)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-a": rng.normal(0, 0.01, 200),
                "pod-b": rng.normal(0, 0.01, 200),
            }
        )
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={"pod-a": 0.5, "pod-b": 0.5},
            pod_returns=returns,
        )
        corr_alerts = [a for a in alerts if a.alert_type == "correlation_stress"]
        assert len(corr_alerts) == 0

    def test_no_returns_no_correlation_check(self) -> None:
        config = _make_config()
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={},
            pod_returns=None,
        )
        corr_alerts = [a for a in alerts if a.alert_type == "correlation_stress"]
        assert len(corr_alerts) == 0


# ── TestRiskAggregatorHasCritical ────────────────────────────────


class TestRiskAggregatorHasCritical:
    def test_multiple_alerts_has_critical(self) -> None:
        config = _make_config(max_gross_leverage=3.0, daily_loss_limit=0.03)
        ra = RiskAggregator(config)
        alerts = ra.check_portfolio_limits(
            net_weights={"BTC/USDT": 2.0, "ETH/USDT": 2.0},
            pod_performances={},
            pod_weights={},
            daily_pnl_pct=-0.05,
        )
        assert RiskAlert.has_critical(alerts) is True


# ── TestM5SignedCorrelation ──────────────────────────────────────


class TestM5SignedCorrelation:
    """M-5: check_correlation_stress()가 signed mean 사용."""

    def test_negative_correlation_no_stress(self) -> None:
        """음의 상관 = 분산 이득 → stress 아님."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, 200)
        returns = pd.DataFrame(
            {
                "pod-a": base,
                "pod-b": -base + rng.normal(0, 0.001, 200),  # 음의 상관
            }
        )
        is_stressed, avg_corr = check_correlation_stress(returns, 0.70)
        assert is_stressed is False
        assert avg_corr < 0  # signed mean이 음수

    def test_positive_high_correlation_stress(self) -> None:
        """강한 양의 상관 → stress."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, 200)
        returns = pd.DataFrame(
            {
                "pod-a": base,
                "pod-b": base + rng.normal(0, 0.001, 200),
            }
        )
        is_stressed, avg_corr = check_correlation_stress(returns, 0.70)
        assert is_stressed is True
        assert avg_corr > 0.70


# ── TestAssetCorrelationStress ──────────────────────────────────


class TestAssetCorrelationStress:
    """Step 5: 에셋 레벨 상관 스트레스 테스트."""

    def test_asset_correlation_stress_high(self) -> None:
        """상관 높은 에셋 → stressed."""
        rng = np.random.default_rng(42)
        base_prices = np.cumsum(rng.normal(0, 1, 100)) + 10000
        noise = rng.normal(0, 0.1, 100)
        price_history = {
            "BTC/USDT": base_prices.tolist(),
            "ETH/USDT": (base_prices + noise).tolist(),
        }
        is_stressed, avg_corr = check_asset_correlation_stress(price_history, 0.70)
        assert is_stressed is True
        assert avg_corr > 0.70

    def test_asset_correlation_stress_low(self) -> None:
        """독립 에셋 → not stressed."""
        rng = np.random.default_rng(42)
        price_history = {
            "BTC/USDT": (np.cumsum(rng.normal(0, 1, 100)) + 10000).tolist(),
            "ETH/USDT": (np.cumsum(rng.normal(0, 1, 100)) + 3000).tolist(),
        }
        is_stressed, _avg_corr = check_asset_correlation_stress(price_history, 0.70)
        assert is_stressed is False

    def test_asset_correlation_stress_insufficient_data(self) -> None:
        """데이터 부족 → skip."""
        price_history = {
            "BTC/USDT": [50000.0, 51000.0],
            "ETH/USDT": [3000.0, 3100.0],
        }
        is_stressed, avg_corr = check_asset_correlation_stress(price_history, 0.70)
        assert is_stressed is False
        assert avg_corr == pytest.approx(0.0)

    def test_asset_correlation_single_asset(self) -> None:
        """에셋 1개 → skip."""
        price_history = {"BTC/USDT": [50000.0] * 10}
        is_stressed, _avg_corr = check_asset_correlation_stress(price_history, 0.70)
        assert is_stressed is False

    def test_portfolio_limits_2pod_uses_asset_check(self) -> None:
        """2-pod + asset data → 에셋 체크 발동."""
        config = _make_config(correlation_stress_threshold=0.70)
        ra = RiskAggregator(config)
        rng = np.random.default_rng(42)

        # 2-pod 수익률 (corr check skips with < 3 rows using pod_returns)
        pod_returns = pd.DataFrame(
            {
                "pod-a": rng.normal(0, 0.01, 100),
                "pod-b": rng.normal(0, 0.01, 100),
            }
        )

        # 높은 상관 에셋 가격
        base = np.cumsum(rng.normal(0, 1, 100)) + 10000
        asset_price_history = {
            "BTC/USDT": base.tolist(),
            "ETH/USDT": (base + rng.normal(0, 0.01, 100)).tolist(),
        }

        alerts = ra.check_portfolio_limits(
            net_weights={},
            pod_performances={},
            pod_weights={"pod-a": 0.5, "pod-b": 0.5},
            pod_returns=pod_returns,
            asset_price_history=asset_price_history,
        )
        asset_alerts = [a for a in alerts if a.alert_type == "asset_correlation_stress"]
        assert len(asset_alerts) >= 1
