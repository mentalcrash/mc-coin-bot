"""Tests for Allocation Comparator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.orchestrator.allocation_comparator import (
    AllocationComparisonResult,
    compare_allocations,
)
from src.orchestrator.models import AllocationMethod

# ── AllocationComparisonResult ────────────────────────────────────


class TestAllocationComparisonResult:
    """AllocationComparisonResult dataclass 테스트."""

    def test_creation(self) -> None:
        """정상 생성."""
        result = AllocationComparisonResult(
            method="risk_parity",
            sharpe=1.5,
            mdd=-15.0,
            calmar=1.0,
            cagr=15.0,
            total_return=30.0,
        )
        assert result.method == "risk_parity"
        assert result.sharpe == 1.5
        assert result.mdd == -15.0

    def test_sorting(self) -> None:
        """Sharpe 기준 정렬."""
        results = [
            AllocationComparisonResult("a", sharpe=0.5, mdd=-10, calmar=0.5, cagr=5, total_return=10),
            AllocationComparisonResult("b", sharpe=1.5, mdd=-8, calmar=1.8, cagr=15, total_return=30),
            AllocationComparisonResult("c", sharpe=1.0, mdd=-12, calmar=0.8, cagr=10, total_return=20),
        ]
        results.sort(key=lambda r: r.sharpe, reverse=True)
        assert results[0].method == "b"
        assert results[1].method == "c"
        assert results[2].method == "a"


# ── compare_allocations (mocked) ──────────────────────────────────


class TestCompareAllocations:
    """compare_allocations 함수 mock 테스트."""

    def _make_mock_config(self) -> MagicMock:
        """Minimal OrchestratorConfig mock."""
        config = MagicMock()
        config.allocation_method = AllocationMethod.INVERSE_VOLATILITY
        config.model_copy.return_value = config
        return config

    def _make_mock_result(self, sharpe: float, mdd: float, cagr: float) -> MagicMock:
        """Mock OrchestratedResult."""
        result = MagicMock()
        result.portfolio_metrics.sharpe_ratio = sharpe
        result.portfolio_metrics.max_drawdown = mdd
        result.portfolio_metrics.cagr = cagr
        result.portfolio_metrics.total_return = cagr * 2
        return result

    @patch("src.eda.orchestrated_runner.OrchestratedRunner")
    def test_runs_four_backtests(
        self,
        mock_runner_cls: MagicMock,
    ) -> None:
        """4가지 method로 백테스트 실행."""
        config = self._make_mock_config()
        data = MagicMock()

        # 각 method별 다른 결과
        results = [
            self._make_mock_result(0.8, -10, 8),
            self._make_mock_result(1.2, -12, 12),
            self._make_mock_result(1.5, -8, 15),
            self._make_mock_result(1.0, -15, 10),
        ]
        mock_runner = MagicMock()
        mock_runner.run = MagicMock(side_effect=results)
        mock_runner_cls.backtest.return_value = mock_runner

        # asyncio.run은 coroutine을 실행하지만 mock은 동기이므로 직접 patch
        with patch("src.orchestrator.allocation_comparator.asyncio.run", side_effect=results):
            comparison = compare_allocations(config, data, "12H", 10000.0)

        assert len(comparison) == 4
        # Sharpe 내림차순 정렬 확인
        assert comparison[0].sharpe >= comparison[1].sharpe
        assert comparison[1].sharpe >= comparison[2].sharpe

    def test_calmar_zero_mdd(self) -> None:
        """MDD=0 → calmar=0."""
        config = self._make_mock_config()
        data = MagicMock()

        result = self._make_mock_result(1.0, 0.0, 10.0)

        with (
            patch("src.eda.orchestrated_runner.OrchestratedRunner") as mock_cls,
            patch("src.orchestrator.allocation_comparator.asyncio.run", return_value=result),
        ):
            mock_cls.backtest.return_value = MagicMock()
            comparison = compare_allocations(config, data, "12H")

        for r in comparison:
            assert r.calmar == 0.0

    def test_config_variant_created(self) -> None:
        """model_copy로 config variant가 4번 생성됨."""
        config = self._make_mock_config()
        data = MagicMock()
        result = self._make_mock_result(1.0, -10, 10)

        with (
            patch("src.eda.orchestrated_runner.OrchestratedRunner") as mock_cls,
            patch("src.orchestrator.allocation_comparator.asyncio.run", return_value=result),
        ):
            mock_cls.backtest.return_value = MagicMock()
            compare_allocations(config, data, "12H")

        assert config.model_copy.call_count == 4
        # 각 method로 호출 확인
        called_methods = {
            call.kwargs["update"]["allocation_method"]
            for call in config.model_copy.call_args_list
        }
        assert called_methods == set(AllocationMethod)
