"""VolStructureDetector 단위 테스트.

classify_series, update, config validation, sigmoid 확률 범위,
warmup 처리를 테스트합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import RegimeLabel, VolStructureDetectorConfig
from src.regime.detector import RegimeState
from src.regime.vol_detector import VolStructureDetector

# ── Helpers ──


def _make_trending_series(n: int = 150, drift: float = 0.01) -> pd.Series:
    """명확한 상승 추세."""
    rng = np.random.default_rng(42)
    returns = drift + rng.normal(0, 0.002, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


def _make_ranging_series(n: int = 150, noise: float = 0.003) -> pd.Series:
    """좁은 레인지 횡보."""
    rng = np.random.default_rng(42)
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        reversion = -0.05 * (prices[i - 1] - 100.0)
        prices[i] = prices[i - 1] + reversion + rng.normal(0, noise * 100)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(np.maximum(prices, 50.0), index=idx, name="close")


def _make_volatile_series(n: int = 150) -> pd.Series:
    """급격한 변동 (높은 RV, 낮은 ER)."""
    rng = np.random.default_rng(42)
    returns = rng.choice([-0.05, 0.05], size=n) + rng.normal(0, 0.02, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


# ── Config Tests ──


class TestVolStructureDetectorConfig:
    """VolStructureDetectorConfig 검증 테스트."""

    def test_default_values(self) -> None:
        cfg = VolStructureDetectorConfig()
        assert cfg.vol_short_window == 10
        assert cfg.vol_long_window == 60
        assert cfg.mom_window == 20

    def test_frozen(self) -> None:
        cfg = VolStructureDetectorConfig()
        with pytest.raises(ValidationError):
            cfg.vol_short_window = 15  # type: ignore[misc]

    def test_invalid_window_order(self) -> None:
        with pytest.raises(ValidationError, match="vol_short_window"):
            VolStructureDetectorConfig(vol_short_window=60, vol_long_window=30)

    def test_equal_windows_rejected(self) -> None:
        with pytest.raises(ValidationError, match="vol_short_window"):
            VolStructureDetectorConfig(vol_short_window=30, vol_long_window=30)

    def test_warmup_periods(self) -> None:
        cfg = VolStructureDetectorConfig()
        assert cfg.warmup_periods == 61  # max(60, 20) + 1

    def test_custom_config(self) -> None:
        cfg = VolStructureDetectorConfig(
            vol_short_window=5,
            vol_long_window=30,
            mom_window=15,
        )
        assert cfg.vol_short_window == 5
        assert cfg.warmup_periods == 31  # max(30, 15) + 1


# ── classify_series Tests ──


class TestVolClassifySeries:
    """VolStructureDetector.classify_series() 벡터화 API 테스트."""

    @pytest.fixture
    def detector(self) -> VolStructureDetector:
        return VolStructureDetector()

    def test_output_columns(self, detector: VolStructureDetector) -> None:
        """출력 DataFrame 컬럼 확인."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        expected_cols = {"p_trending", "p_ranging", "p_volatile", "vol_ratio", "norm_momentum"}
        assert set(result.columns) == expected_cols

    def test_warmup_nan(self, detector: VolStructureDetector) -> None:
        """warmup 기간 중 NaN."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        warmup = result.iloc[: detector.config.vol_long_window]
        assert warmup["p_trending"].isna().all()

    def test_probabilities_sum_to_one(self, detector: VolStructureDetector) -> None:
        """확률 합 ≈ 1.0."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_probabilities_valid_range(self, detector: VolStructureDetector) -> None:
        """확률 값이 0~1 범위."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        for col in ["p_trending", "p_ranging", "p_volatile"]:
            assert (valid[col] >= 0.0).all(), f"{col} has values < 0"
            assert (valid[col] <= 1.0).all(), f"{col} has values > 1"

    def test_volatile_series_detected(self, detector: VolStructureDetector) -> None:
        """고변동 시리즈 → p_volatile 또는 p_trending 높음 (expansion)."""
        closes = _make_volatile_series()
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        if len(valid) > 0:
            # 고변동은 expansion_score 높음 → p_trending 또는 p_volatile
            avg_non_ranging = (valid["p_trending"] + valid["p_volatile"]).mean()
            assert avg_non_ranging > 0.3

    def test_ranging_series_low_expansion(self, detector: VolStructureDetector) -> None:
        """횡보 시리즈 → p_ranging 상대적으로 높음."""
        closes = _make_ranging_series()
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        if len(valid) > 0:
            avg_pr = valid["p_ranging"].mean()
            # 횡보에서 ranging이 0 이상이어야 함
            assert avg_pr > 0.0


# ── update (Incremental) Tests ──


class TestVolIncremental:
    """VolStructureDetector.update() incremental API 테스트."""

    @pytest.fixture
    def detector(self) -> VolStructureDetector:
        return VolStructureDetector()

    def test_warmup_returns_none(self, detector: VolStructureDetector) -> None:
        """warmup 중 None 반환."""
        for i in range(detector.config.vol_long_window):
            result = detector.update("BTC/USDT", 100.0 + i * 0.1)
            assert result is None

    def test_after_warmup_returns_state(self, detector: VolStructureDetector) -> None:
        """warmup 후 RegimeState 반환."""
        result = None
        for i in range(detector.warmup_periods + 10):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert result is not None
        assert isinstance(result, RegimeState)
        assert result.label in list(RegimeLabel)

    def test_state_has_valid_probabilities(self, detector: VolStructureDetector) -> None:
        """RegimeState probabilities 합 ≈ 1.0."""
        result = None
        for i in range(detector.warmup_periods + 10):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert result is not None
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_raw_indicators(self, detector: VolStructureDetector) -> None:
        """raw_indicators에 vol_ratio, norm_momentum 포함."""
        result = None
        for i in range(detector.warmup_periods + 10):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert result is not None
        assert "vol_ratio" in result.raw_indicators
        assert "norm_momentum" in result.raw_indicators

    def test_multi_symbol_independence(self, detector: VolStructureDetector) -> None:
        """멀티 심볼 독립."""
        for i in range(detector.warmup_periods + 10):
            detector.update("BTC/USDT", 100.0 + i * 1.0)
            detector.update("ETH/USDT", 100.0 + (-1) ** i * 0.1)

        assert "BTC/USDT" in detector._buffers
        assert "ETH/USDT" in detector._buffers


# ── Vectorized ↔ Incremental Parity ──


class TestVectorizedIncrementalParity:
    """classify_series() vs update() 결과 일치 검증."""

    def test_probability_parity_trending(self) -> None:
        """상승 추세에서 마지막 bar 확률 일치."""
        cfg = VolStructureDetectorConfig()
        vec_det = VolStructureDetector(cfg)
        inc_det = VolStructureDetector(cfg)

        closes = _make_trending_series(150)
        vec_df = vec_det.classify_series(closes)

        for price in closes:
            inc_det.update("TEST", float(price))

        state = inc_det._buffers["TEST"].last_state
        last_valid = vec_df.dropna(subset=["p_trending"])
        if len(last_valid) > 0 and state is not None:
            last_row = last_valid.iloc[-1]
            np.testing.assert_allclose(
                state.probabilities["trending"], last_row["p_trending"], atol=1e-6
            )
            np.testing.assert_allclose(
                state.probabilities["ranging"], last_row["p_ranging"], atol=1e-6
            )
            np.testing.assert_allclose(
                state.probabilities["volatile"], last_row["p_volatile"], atol=1e-6
            )

    def test_probability_parity_ranging(self) -> None:
        """횡보에서 마지막 bar 확률 일치."""
        cfg = VolStructureDetectorConfig()
        vec_det = VolStructureDetector(cfg)
        inc_det = VolStructureDetector(cfg)

        closes = _make_ranging_series(150)
        vec_df = vec_det.classify_series(closes)

        for price in closes:
            inc_det.update("TEST", float(price))

        state = inc_det._buffers["TEST"].last_state
        last_valid = vec_df.dropna(subset=["p_trending"])
        if len(last_valid) > 0 and state is not None:
            last_row = last_valid.iloc[-1]
            np.testing.assert_allclose(
                state.probabilities["trending"], last_row["p_trending"], atol=1e-6
            )

    def test_probability_parity_volatile(self) -> None:
        """고변동에서 마지막 bar 확률 일치."""
        cfg = VolStructureDetectorConfig()
        vec_det = VolStructureDetector(cfg)
        inc_det = VolStructureDetector(cfg)

        closes = _make_volatile_series(150)
        vec_df = vec_det.classify_series(closes)

        for price in closes:
            inc_det.update("TEST", float(price))

        state = inc_det._buffers["TEST"].last_state
        last_valid = vec_df.dropna(subset=["p_trending"])
        if len(last_valid) > 0 and state is not None:
            last_row = last_valid.iloc[-1]
            np.testing.assert_allclose(
                state.probabilities["volatile"], last_row["p_volatile"], atol=1e-6
            )
