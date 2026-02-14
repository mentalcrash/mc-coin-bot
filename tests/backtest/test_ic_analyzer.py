"""Tests for src/backtest/ic_analyzer.py — IC Quick Check module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.ic_analyzer import ICAnalyzer, ICVerdict


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def n() -> int:
    return 500


class TestRankIC:
    def test_perfect_correlation(self, n: int) -> None:
        """indicator와 forward_returns 동일 → IC ≈ 1.0."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(n)
        indicator = pd.Series(base)
        forward_returns = pd.Series(base)  # 동일 시리즈
        ic, pvalue = ICAnalyzer.rank_ic(indicator, forward_returns)
        assert ic > 0.99
        assert pvalue < 0.01

    def test_random_signal(self, rng: np.random.Generator, n: int) -> None:
        """랜덤 지표 → IC ≈ 0."""
        indicator = pd.Series(rng.standard_normal(n))
        forward_returns = pd.Series(rng.standard_normal(n))
        ic, _ = ICAnalyzer.rank_ic(indicator, forward_returns)
        assert abs(ic) < 0.15

    def test_insufficient_data(self) -> None:
        """데이터 부족 시 (0.0, 1.0) 반환."""
        indicator = pd.Series([1.0, 2.0])
        forward_returns = pd.Series([0.1, 0.2])
        ic, pvalue = ICAnalyzer.rank_ic(indicator, forward_returns)
        assert ic == 0.0
        assert pvalue == 1.0

    def test_nan_handling(self, rng: np.random.Generator) -> None:
        """NaN 포함 시리즈 처리."""
        n = 200
        indicator = pd.Series(rng.standard_normal(n))
        forward_returns = pd.Series(rng.standard_normal(n))
        # NaN 삽입
        indicator.iloc[10:20] = np.nan
        forward_returns.iloc[50:60] = np.nan

        ic, pvalue = ICAnalyzer.rank_ic(indicator, forward_returns)
        assert isinstance(ic, float)
        assert isinstance(pvalue, float)
        assert not np.isnan(ic)


class TestRollingIC:
    def test_output_shape(self, rng: np.random.Generator, n: int) -> None:
        """출력 길이 = n - window (NaN 제거 후)."""
        window = 60
        indicator = pd.Series(rng.standard_normal(n))
        forward_returns = pd.Series(rng.standard_normal(n))
        rolling = ICAnalyzer.rolling_ic(indicator, forward_returns, window=window)
        # dropna applied to input: all valid, so output = n - window + 1
        assert len(rolling) == n - window + 1

    def test_consistent_signal_has_positive_rolling(self) -> None:
        """일관된 양의 상관 → rolling IC 대부분 양수."""
        n = 300
        rng = np.random.default_rng(42)
        base = rng.standard_normal(n)
        noise = rng.standard_normal(n) * 0.3
        indicator = pd.Series(base)
        forward_returns = pd.Series(base + noise)

        rolling = ICAnalyzer.rolling_ic(indicator, forward_returns, window=60)
        assert (rolling > 0).mean() > 0.8


class TestICIR:
    def test_consistent_ic_has_high_ir(self) -> None:
        """약간의 변동 있는 양수 rolling IC → 높은 IR."""
        rng = np.random.default_rng(42)
        rolling_ic_var = pd.Series(0.05 + rng.standard_normal(100) * 0.01)
        ir_var = ICAnalyzer.ic_ir(rolling_ic_var)
        assert ir_var > 1.0

    def test_random_ic_has_low_ir(self, rng: np.random.Generator) -> None:
        """랜덤 rolling IC → 낮은 IR."""
        rolling_ic = pd.Series(rng.standard_normal(100))
        ir = ICAnalyzer.ic_ir(rolling_ic)
        assert abs(ir) < 1.0

    def test_empty_series(self) -> None:
        """빈 시리즈 → IR=0."""
        ir = ICAnalyzer.ic_ir(pd.Series(dtype=float))
        assert ir == 0.0


class TestDecayStable:
    def test_stable_positive(self) -> None:
        """모든 분기 양수 → stable."""
        rng = np.random.default_rng(42)
        # 4분기 * 63 = 252 bars, 양수 기본값
        rolling_ic = pd.Series(0.05 + rng.standard_normal(252) * 0.02)
        assert ICAnalyzer.ic_decay_stable(rolling_ic) is True

    def test_sign_reversal(self) -> None:
        """분기 부호 반전 → unstable."""
        # 첫 2분기 양수, 나머지 2분기 음수
        vals = [0.05] * 126 + [-0.05] * 126
        rolling_ic = pd.Series(vals)
        assert ICAnalyzer.ic_decay_stable(rolling_ic) is False

    def test_short_series_passes(self) -> None:
        """짧은 시리즈(< 63 bars) → True (판단 불가)."""
        rolling_ic = pd.Series([0.05] * 30)
        assert ICAnalyzer.ic_decay_stable(rolling_ic) is True


class TestHitRate:
    def test_perfect_hit(self) -> None:
        """100% 방향 일치 → 100%."""
        indicator = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0])
        forward_returns = pd.Series([0.1, -0.1, 0.1, -0.1, 0.1])
        hr = ICAnalyzer.hit_rate(indicator, forward_returns)
        assert hr == pytest.approx(100.0)

    def test_random_hit(self, rng: np.random.Generator) -> None:
        """랜덤 → ~50%."""
        n = 10000
        indicator = pd.Series(rng.standard_normal(n))
        forward_returns = pd.Series(rng.standard_normal(n))
        hr = ICAnalyzer.hit_rate(indicator, forward_returns)
        assert 45 < hr < 55

    def test_zero_indicator_excluded(self) -> None:
        """indicator=0 또는 return=0은 제외."""
        indicator = pd.Series([0.0, 1.0, -1.0])
        forward_returns = pd.Series([0.1, 0.0, -0.1])
        # Only (indicator=-1, return=-0.1) qualifies
        hr = ICAnalyzer.hit_rate(indicator, forward_returns)
        assert hr == pytest.approx(100.0)


class TestAnalyze:
    def test_strong_indicator_passes(self) -> None:
        """강한 예측력 지표 → PASS."""
        rng = np.random.default_rng(42)
        n = 500
        base = rng.standard_normal(n)
        noise = rng.standard_normal(n) * 0.3
        indicator = pd.Series(base)
        forward_returns = pd.Series(base + noise)

        result = ICAnalyzer.analyze(indicator, forward_returns)
        assert result.verdict == ICVerdict.PASS
        assert abs(result.rank_ic) > 0.02
        assert result.hit_rate > 52

    def test_random_indicator_fails(self, rng: np.random.Generator) -> None:
        """랜덤 지표 → FAIL."""
        n = 500
        indicator = pd.Series(rng.standard_normal(n))
        forward_returns = pd.Series(rng.standard_normal(n))

        result = ICAnalyzer.analyze(indicator, forward_returns)
        assert result.verdict == ICVerdict.FAIL

    def test_result_fields_populated(self, rng: np.random.Generator) -> None:
        """모든 결과 필드가 유효한 값."""
        n = 300
        indicator = pd.Series(rng.standard_normal(n))
        forward_returns = pd.Series(rng.standard_normal(n))

        result = ICAnalyzer.analyze(indicator, forward_returns)
        assert isinstance(result.rank_ic, float)
        assert isinstance(result.rank_ic_pvalue, float)
        assert isinstance(result.ic_ir, float)
        assert isinstance(result.ic_decay_stable, bool)
        assert isinstance(result.hit_rate, float)
        assert result.verdict in (ICVerdict.PASS, ICVerdict.FAIL)

    def test_nan_handling(self, rng: np.random.Generator) -> None:
        """NaN이 포함된 시리즈도 정상 처리."""
        n = 300
        indicator = pd.Series(rng.standard_normal(n))
        forward_returns = pd.Series(rng.standard_normal(n))
        indicator.iloc[:30] = np.nan
        forward_returns.iloc[100:110] = np.nan

        result = ICAnalyzer.analyze(indicator, forward_returns)
        assert not np.isnan(result.rank_ic)
        assert not np.isnan(result.hit_rate)
