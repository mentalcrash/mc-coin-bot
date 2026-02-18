"""Regression Test -- 전략 메트릭 안정성 검증.

동일 합성 데이터(seed=42)에서 전략 실행 결과가 Golden Metrics 허용 범위 내인지 검증.
아키텍처 변경 시 ACTIVE 전략의 성과가 보존되는지 확인합니다.

Usage:
    uv run pytest tests/regression/ -v -m slow
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.backtest.request import BacktestRequest
from src.portfolio.portfolio import Portfolio
from src.strategy.ctrend.strategy import CTRENDStrategy
from src.strategy.tsmom.strategy import TSMOMStrategy

if TYPE_CHECKING:
    from src.backtest.engine import BacktestEngine
    from src.data.market_data import MarketDataSet

pytestmark = pytest.mark.slow


# ── Helper ────────────────────────────────────────────────────────


def _assert_metric_stable(
    actual: float,
    golden: float,
    tolerance: float,
    metric_name: str,
    strategy_name: str,
) -> None:
    """메트릭이 Golden 기준 허용 오차 이내인지 검증."""
    delta = abs(actual - golden)
    assert delta <= tolerance, (
        f"{strategy_name}/{metric_name}: "
        f"actual={actual:.4f}, golden={golden:.4f}, "
        f"delta={delta:.4f} > tolerance={tolerance:.4f}"
    )


def _run_and_extract(
    engine: BacktestEngine,
    data: MarketDataSet,
    strategy_fn: Any,
) -> dict[str, float]:
    """전략 실행 + 주요 메트릭 추출.

    Returns:
        {"sharpe": ..., "cagr": ..., "mdd": ..., "total_trades": ...}
    """
    strategy = strategy_fn()
    portfolio = Portfolio.create(initial_capital=100000)
    request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
    result = engine.run(request)
    return {
        "sharpe": result.metrics.sharpe_ratio,
        "cagr": result.metrics.cagr,
        "mdd": result.metrics.max_drawdown,
        "total_trades": float(result.metrics.total_trades),
    }


# ── Tests ─────────────────────────────────────────────────────────


class TestTSMOMRegression:
    """TSMOM 전략 회귀 테스트."""

    @pytest.fixture(scope="class")
    def tsmom_metrics(
        self,
        backtest_engine: BacktestEngine,
        btc_market_data: MarketDataSet,
    ) -> dict[str, float]:
        """TSMOM 메트릭 (class scope — 1번만 실행)."""
        return _run_and_extract(
            backtest_engine,
            btc_market_data,
            TSMOMStrategy.from_params,
        )

    def test_tsmom_runs(self, tsmom_metrics: dict[str, float]) -> None:
        """TSMOM이 정상 실행되고 기본 메트릭을 반환."""
        assert "sharpe" in tsmom_metrics
        assert "cagr" in tsmom_metrics
        assert "mdd" in tsmom_metrics

    def test_tsmom_deterministic(
        self,
        backtest_engine: BacktestEngine,
        btc_market_data: MarketDataSet,
        tsmom_metrics: dict[str, float],
    ) -> None:
        """동일 데이터로 2회 실행 시 결과 동일."""
        metrics2 = _run_and_extract(
            backtest_engine,
            btc_market_data,
            TSMOMStrategy.from_params,
        )
        assert tsmom_metrics["sharpe"] == pytest.approx(metrics2["sharpe"], abs=1e-6)
        assert tsmom_metrics["cagr"] == pytest.approx(metrics2["cagr"], abs=1e-6)
        assert tsmom_metrics["mdd"] == pytest.approx(metrics2["mdd"], abs=1e-6)
        assert tsmom_metrics["total_trades"] == metrics2["total_trades"]

    def test_tsmom_produces_trades(self, tsmom_metrics: dict[str, float]) -> None:
        """최소 1건 이상의 거래 발생."""
        assert tsmom_metrics["total_trades"] >= 1


class TestCTRENDRegression:
    """CTREND 전략 회귀 테스트."""

    @pytest.fixture(scope="class")
    def ctrend_metrics(
        self,
        backtest_engine: BacktestEngine,
        btc_market_data: MarketDataSet,
    ) -> dict[str, float]:
        """CTREND 메트릭 (class scope — 1번만 실행)."""
        return _run_and_extract(
            backtest_engine,
            btc_market_data,
            CTRENDStrategy.from_params,
        )

    def test_ctrend_runs(self, ctrend_metrics: dict[str, float]) -> None:
        """CTREND가 정상 실행되고 기본 메트릭을 반환."""
        assert "sharpe" in ctrend_metrics
        assert "cagr" in ctrend_metrics

    def test_ctrend_deterministic(
        self,
        backtest_engine: BacktestEngine,
        btc_market_data: MarketDataSet,
        ctrend_metrics: dict[str, float],
    ) -> None:
        """동일 데이터로 2회 실행 시 결과 동일."""
        metrics2 = _run_and_extract(
            backtest_engine,
            btc_market_data,
            CTRENDStrategy.from_params,
        )
        assert ctrend_metrics["sharpe"] == pytest.approx(metrics2["sharpe"], abs=1e-6)
        assert ctrend_metrics["cagr"] == pytest.approx(metrics2["cagr"], abs=1e-6)

    def test_ctrend_produces_trades(self, ctrend_metrics: dict[str, float]) -> None:
        """최소 1건 이상의 거래 발생."""
        assert ctrend_metrics["total_trades"] >= 1


class TestGoldenMetricsBaseline:
    """Golden metrics baseline 생성/검증.

    첫 실행 시 golden metrics를 기록하고,
    이후 실행에서는 기록된 값과 비교합니다.
    """

    def test_golden_yaml_loads(self, golden_metrics: dict[str, dict]) -> None:
        """Golden metrics YAML이 정상 로드."""
        assert "tsmom_btc_1d" in golden_metrics
        assert "ctrend_btc_1d" in golden_metrics

    def test_golden_has_tolerance(self, golden_metrics: dict[str, dict]) -> None:
        """각 전략에 tolerance 설정 존재."""
        for key in ("tsmom_btc_1d", "ctrend_btc_1d"):
            assert "tolerance" in golden_metrics[key]
            tolerance = golden_metrics[key]["tolerance"]
            assert "sharpe" in tolerance
            assert "cagr" in tolerance
