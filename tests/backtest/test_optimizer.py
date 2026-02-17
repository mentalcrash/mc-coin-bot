"""Tests for src/backtest/optimizer.py — G2H parameter optimizer."""

from __future__ import annotations

from enum import IntEnum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

from src.backtest.optimizer import (
    OptimizationResult,
    ParamSpec,
    extract_search_space,
    generate_g3_sweeps,
)

_OPTUNA_AVAILABLE = True
try:
    import optuna  # noqa: F401
except ImportError:
    _OPTUNA_AVAILABLE = False

# ─── Test Config fixtures ──────────────────────────────────────────


class _ShortMode(IntEnum):
    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class _SampleConfig(BaseModel):
    """Test config mimicking KAMAConfig structure."""

    model_config = ConfigDict(frozen=True)

    er_lookback: int = Field(default=10, ge=5, le=100)
    fast_period: int = Field(default=2, ge=2, le=10)
    slow_period: int = Field(default=30, ge=10, le=100)
    atr_multiplier: float = Field(default=1.5, ge=0.5, le=5.0)
    vol_target: float = Field(default=0.30, ge=0.05, le=1.0)

    # Should be skipped
    annualization_factor: float = Field(default=365.0, gt=0)
    use_log_returns: bool = Field(default=True)
    min_volatility: float = Field(default=0.05, ge=0.01, le=0.20)
    short_mode: _ShortMode = Field(default=_ShortMode.HEDGE_ONLY)
    hedge_threshold: float = Field(default=-0.07, ge=-0.30, le=-0.05)
    hedge_strength_ratio: float = Field(default=0.8, ge=0.1, le=1.0)


class _WeightConfig(BaseModel):
    """Test config with weight pairs."""

    model_config = ConfigDict(frozen=True)

    bb_period: int = Field(default=20, ge=5, le=100)
    bb_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    rsi_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    vol_target: float = Field(default=0.20, ge=0.05, le=1.0)


# ─── TestExtractSearchSpace ────────────────────────────────────────


class TestExtractSearchSpace:
    def test_basic_fields_extracted(self) -> None:
        """KAMAConfig-like config에서 optimizable 필드를 추출."""
        specs = extract_search_space(_SampleConfig)
        names = {s.name for s in specs}
        assert "er_lookback" in names
        assert "fast_period" in names
        assert "slow_period" in names
        assert "atr_multiplier" in names
        assert "vol_target" in names

    def test_skips_bool_and_enum(self) -> None:
        """use_log_returns(bool), short_mode(IntEnum) skip."""
        specs = extract_search_space(_SampleConfig)
        names = {s.name for s in specs}
        assert "use_log_returns" not in names
        assert "short_mode" not in names

    def test_skips_named_fields(self) -> None:
        """annualization_factor, min_volatility, hedge_* skip."""
        specs = extract_search_space(_SampleConfig)
        names = {s.name for s in specs}
        assert "annualization_factor" not in names
        assert "min_volatility" not in names
        assert "hedge_threshold" not in names
        assert "hedge_strength_ratio" not in names

    def test_skips_complement_weights(self) -> None:
        """rsi_weight (complement of bb_weight) skip."""
        specs = extract_search_space(_WeightConfig)
        names = {s.name for s in specs}
        assert "bb_weight" in names
        assert "rsi_weight" not in names

    def test_int_float_distinction(self) -> None:
        """er_lookback=int, vol_target=float 구분."""
        specs = extract_search_space(_SampleConfig)
        spec_map = {s.name: s for s in specs}

        assert spec_map["er_lookback"].param_type == "int"
        assert spec_map["vol_target"].param_type == "float"

    def test_bounds_correct(self) -> None:
        """ge/le 값이 정확히 추출되는지 확인."""
        specs = extract_search_space(_SampleConfig)
        spec_map = {s.name: s for s in specs}

        assert spec_map["er_lookback"].low == 5
        assert spec_map["er_lookback"].high == 100
        assert spec_map["vol_target"].low == 0.05
        assert spec_map["vol_target"].high == 1.0

    def test_no_bounds_skipped(self) -> None:
        """gt만 있는 annualization_factor는 skip (ge/le 둘 다 필요)."""
        specs = extract_search_space(_SampleConfig)
        names = {s.name for s in specs}
        assert "annualization_factor" not in names

    def test_defaults_captured(self) -> None:
        """default 값이 ParamSpec에 저장."""
        specs = extract_search_space(_SampleConfig)
        spec_map = {s.name: s for s in specs}
        assert spec_map["er_lookback"].default == 10
        assert spec_map["vol_target"].default == 0.30


# ─── TestOptimizeStrategy ──────────────────────────────────────────


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
class TestOptimizeStrategy:
    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        mock_result = MagicMock()
        mock_result.metrics.sharpe_ratio = 1.5
        engine.run.return_value = mock_result
        return engine

    @patch("src.backtest.engine.BacktestEngine")
    @patch("src.strategy.registry.get_strategy")
    def test_returns_result(
        self,
        mock_get_strategy: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        """OptimizationResult 정상 반환."""
        mock_strategy_cls = MagicMock()
        mock_strategy_cls.from_params.return_value = MagicMock()
        mock_config = _SampleConfig()
        mock_strategy_cls.return_value.config = mock_config
        mock_get_strategy.return_value = mock_strategy_cls

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.metrics.sharpe_ratio = 1.5
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_data = MagicMock()
        mock_portfolio = MagicMock()

        from src.backtest.optimizer import optimize_strategy

        result = optimize_strategy(
            "kama",
            mock_data,
            mock_portfolio,
            n_trials=5,
            seed=42,
        )

        assert isinstance(result, OptimizationResult)
        assert result.n_trials == 5
        assert (
            result.best_sharpe >= result.default_sharpe
            or result.best_sharpe == result.default_sharpe
        )

    @patch("src.backtest.engine.BacktestEngine")
    @patch("src.strategy.registry.get_strategy")
    def test_seed_reproducibility(
        self,
        mock_get_strategy: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        """동일 seed → 동일 결과."""
        call_count = 0

        mock_strategy_cls = MagicMock()
        mock_config = _SampleConfig()
        mock_strategy_cls.return_value.config = mock_config
        mock_strategy_cls.from_params.return_value = MagicMock()
        mock_get_strategy.return_value = mock_strategy_cls

        mock_engine = MagicMock()

        def side_effect(request: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            # Deterministic: same call number → same sharpe
            result.metrics.sharpe_ratio = 1.0 + (call_count % 3) * 0.1
            return result

        mock_engine.run.side_effect = side_effect
        mock_engine_cls.return_value = mock_engine

        mock_data = MagicMock()
        mock_portfolio = MagicMock()

        from src.backtest.optimizer import optimize_strategy

        # First run
        call_count = 0
        r1 = optimize_strategy("kama", mock_data, mock_portfolio, n_trials=3, seed=42)

        # Reset and run again
        call_count = 0
        r2 = optimize_strategy("kama", mock_data, mock_portfolio, n_trials=3, seed=42)

        assert r1.best_sharpe == r2.best_sharpe

    @patch("src.backtest.engine.BacktestEngine")
    @patch("src.strategy.registry.get_strategy")
    def test_failed_trials_handled(
        self,
        mock_get_strategy: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        """from_params 실패해도 계속 진행."""
        mock_strategy_cls = MagicMock()
        mock_config = _SampleConfig()
        mock_strategy_cls.return_value.config = mock_config
        # All from_params calls raise
        mock_strategy_cls.from_params.side_effect = ValueError("bad params")
        mock_get_strategy.return_value = mock_strategy_cls

        # Default run (no params → strategy_cls()) should work
        mock_engine = MagicMock()
        mock_bt_result = MagicMock()
        mock_bt_result.metrics.sharpe_ratio = 1.0
        mock_engine.run.return_value = mock_bt_result
        mock_engine_cls.return_value = mock_engine

        mock_data = MagicMock()
        mock_portfolio = MagicMock()

        from src.backtest.optimizer import optimize_strategy

        # Should not raise even though all from_params fail
        result = optimize_strategy("kama", mock_data, mock_portfolio, n_trials=3, seed=42)
        assert isinstance(result, OptimizationResult)

    @patch("src.backtest.engine.BacktestEngine")
    @patch("src.strategy.registry.get_strategy")
    def test_weight_pair_complement(
        self,
        mock_get_strategy: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        """bb_weight optimize → rsi_weight = 1 - bb_weight."""
        captured_params: list[dict[str, Any]] = []

        mock_strategy_cls = MagicMock()
        mock_config = _WeightConfig()
        mock_strategy_cls.return_value.config = mock_config

        def capture_from_params(**params: Any) -> MagicMock:
            captured_params.append(dict(params))
            return MagicMock()

        mock_strategy_cls.from_params.side_effect = capture_from_params
        mock_get_strategy.return_value = mock_strategy_cls

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.metrics.sharpe_ratio = 1.2
        mock_engine.run.return_value = mock_result
        mock_engine_cls.return_value = mock_engine

        mock_data = MagicMock()
        mock_portfolio = MagicMock()

        from src.backtest.optimizer import optimize_strategy

        result = optimize_strategy("bb-rsi", mock_data, mock_portfolio, n_trials=5, seed=42)

        # Check that best_params has complement
        if "bb_weight" in result.best_params:
            bb = result.best_params["bb_weight"]
            rsi = result.best_params.get("rsi_weight")
            assert rsi is not None
            assert abs(bb + rsi - 1.0) < 1e-4


# ─── TestGenerateG3Sweeps ─────────────────────────────────────────


class TestGenerateG3Sweeps:
    @pytest.fixture
    def sample_result(self) -> OptimizationResult:
        return OptimizationResult(
            best_params={"er_lookback": 12, "vol_target": 0.35},
            best_sharpe=2.0,
            default_sharpe=1.5,
            improvement_pct=33.3,
            n_trials=100,
            n_completed=95,
            search_space=[
                ParamSpec(name="er_lookback", param_type="int", low=5, high=100, default=10),
                ParamSpec(name="vol_target", param_type="float", low=0.05, high=1.0, default=0.30),
            ],
        )

    def test_centered_on_best(self, sample_result: OptimizationResult) -> None:
        """sweep이 best param 중심."""
        sweeps = generate_g3_sweeps(sample_result, _SampleConfig)
        # er_lookback sweep should be roughly centered on 12
        er_values = sweeps["er_lookback"]
        assert min(er_values) <= 12 <= max(er_values)

    def test_within_bounds(self, sample_result: OptimizationResult) -> None:
        """Config ge/le 범위 내."""
        sweeps = generate_g3_sweeps(sample_result, _SampleConfig)
        for v in sweeps["er_lookback"]:
            assert 5 <= v <= 100
        for v in sweeps["vol_target"]:
            assert 0.05 <= v <= 1.0

    def test_best_included(self, sample_result: OptimizationResult) -> None:
        """best value가 sweep에 포함."""
        sweeps = generate_g3_sweeps(sample_result, _SampleConfig)
        assert 12 in sweeps["er_lookback"]
        # Float: check approximate inclusion
        vol_values = sweeps["vol_target"]
        assert any(abs(v - 0.35) < 1e-4 for v in vol_values)

    def test_int_values(self, sample_result: OptimizationResult) -> None:
        """int 파라미터 sweep이 정수."""
        sweeps = generate_g3_sweeps(sample_result, _SampleConfig)
        for v in sweeps["er_lookback"]:
            assert isinstance(v, int)

    def test_respects_n_points(self, sample_result: OptimizationResult) -> None:
        """n_points 파라미터 적용."""
        sweeps = generate_g3_sweeps(sample_result, _SampleConfig, n_points=5)
        # Should have around 5 points (+1 if best is added)
        for values in sweeps.values():
            assert len(values) <= 10  # reasonable upper bound

    def test_edge_case_zero_best(self) -> None:
        """best_val=0일 때도 동작."""
        result = OptimizationResult(
            best_params={"vol_target": 0.05},
            best_sharpe=1.0,
            default_sharpe=0.5,
            improvement_pct=100.0,
            n_trials=10,
            n_completed=10,
            search_space=[
                ParamSpec(name="vol_target", param_type="float", low=0.05, high=1.0, default=0.30),
            ],
        )
        sweeps = generate_g3_sweeps(result, _SampleConfig)
        assert "vol_target" in sweeps
        assert len(sweeps["vol_target"]) > 0
