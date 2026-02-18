"""MSARDetector 단위 테스트.

classify_series, update, config validation, probability 검증,
warmup 처리를 테스트합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import MSARDetectorConfig, RegimeLabel
from src.regime.msar_detector import MSAR_AVAILABLE, MSARDetector

pytestmark = pytest.mark.skipif(not MSAR_AVAILABLE, reason="statsmodels not installed")


# ── Helpers ──


def _make_trending_series(n: int = 400, drift: float = 0.01) -> pd.Series:
    """명확한 상승 추세."""
    rng = np.random.default_rng(42)
    returns = drift + rng.normal(0, 0.002, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


def _make_volatile_series(n: int = 400) -> pd.Series:
    """급격한 변동 시리즈."""
    rng = np.random.default_rng(42)
    returns = rng.choice([-0.05, 0.05], size=n) + rng.normal(0, 0.02, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


# ── Config Tests ──


class TestMSARDetectorConfig:
    """MSARDetectorConfig 검증 테스트."""

    def test_default_values(self) -> None:
        cfg = MSARDetectorConfig()
        assert cfg.k_regimes == 3
        assert cfg.order == 2
        assert cfg.switching_ar is True
        assert cfg.switching_variance is True
        assert cfg.sliding_window == 504
        assert cfg.retrain_interval == 21
        assert cfg.min_train_window == 252
        assert cfg.use_log_returns is True

    def test_frozen(self) -> None:
        cfg = MSARDetectorConfig()
        with pytest.raises(ValidationError):
            cfg.k_regimes = 5  # type: ignore[misc]

    def test_invalid_k_regimes(self) -> None:
        with pytest.raises(ValidationError):
            MSARDetectorConfig(k_regimes=1)

    def test_invalid_min_train_window(self) -> None:
        with pytest.raises(ValidationError):
            MSARDetectorConfig(min_train_window=50)

    def test_warmup_periods(self) -> None:
        cfg = MSARDetectorConfig(min_train_window=200)
        assert cfg.warmup_periods == 201

    def test_custom_config(self) -> None:
        cfg = MSARDetectorConfig(
            k_regimes=2,
            order=1,
            sliding_window=252,
            retrain_interval=10,
        )
        assert cfg.k_regimes == 2
        assert cfg.order == 1
        assert cfg.sliding_window == 252


# ── classify_series Tests ──


class TestMSARClassifySeries:
    """MSARDetector.classify_series() 벡터화 API 테스트."""

    @pytest.fixture
    def fast_config(self) -> MSARDetectorConfig:
        """테스트용 빠른 설정."""
        return MSARDetectorConfig(
            k_regimes=2,
            order=1,
            min_train_window=120,
            retrain_interval=50,
            sliding_window=0,
            switching_ar=False,
        )

    def test_output_columns(self, fast_config: MSARDetectorConfig) -> None:
        """출력 DataFrame 컬럼 확인."""
        detector = MSARDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        expected_cols = {"p_trending", "p_ranging", "p_volatile", "msar_state"}
        assert set(result.columns) == expected_cols

    def test_warmup_nan(self, fast_config: MSARDetectorConfig) -> None:
        """warmup 기간 중 NaN."""
        detector = MSARDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)

        warmup = result.iloc[: fast_config.min_train_window]
        assert warmup["p_trending"].isna().all()

    def test_probabilities_valid_range(self, fast_config: MSARDetectorConfig) -> None:
        """확률 값이 0~1 범위."""
        detector = MSARDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        if len(valid) > 0:
            assert (valid["p_trending"] >= -1e-10).all()
            assert (valid["p_trending"] <= 1.0 + 1e-10).all()
            assert (valid["p_ranging"] >= -1e-10).all()
            assert (valid["p_volatile"] >= -1e-10).all()

    def test_probabilities_sum_to_one(self, fast_config: MSARDetectorConfig) -> None:
        """확률 합 ≈ 1.0."""
        detector = MSARDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        if len(valid) > 0:
            prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
            np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-6)

    def test_index_preserved(self, fast_config: MSARDetectorConfig) -> None:
        """출력 인덱스가 입력과 동일."""
        detector = MSARDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        pd.testing.assert_index_equal(result.index, closes.index)


# ── update (Incremental) Tests ──


class TestMSARIncremental:
    """MSARDetector.update() incremental API 테스트."""

    @pytest.fixture
    def fast_config(self) -> MSARDetectorConfig:
        return MSARDetectorConfig(
            k_regimes=2,
            order=1,
            min_train_window=120,
            retrain_interval=50,
            sliding_window=0,
            switching_ar=False,
        )

    def test_warmup_returns_none(self, fast_config: MSARDetectorConfig) -> None:
        """warmup 중 None 반환."""
        detector = MSARDetector(fast_config)
        for i in range(fast_config.min_train_window):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)
            assert result is None

    def test_after_warmup_may_return_state(self, fast_config: MSARDetectorConfig) -> None:
        """warmup 후 RegimeState 반환 가능."""
        from src.regime.detector import RegimeState

        detector = MSARDetector(fast_config)
        result = None
        for i in range(fast_config.min_train_window + 30):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        # MSAR 수렴 실패 가능 → result가 None일 수 있음
        if result is not None:
            assert isinstance(result, RegimeState)
            assert result.label in list(RegimeLabel)

    def test_multi_symbol_independence(self, fast_config: MSARDetectorConfig) -> None:
        """멀티 심볼 독립 버퍼."""
        detector = MSARDetector(fast_config)
        for i in range(fast_config.min_train_window + 10):
            detector.update("BTC/USDT", 100.0 + i * 0.5)
            detector.update("ETH/USDT", 100.0 + (-1) ** i * 0.1)

        assert "BTC/USDT" in detector._buffers
        assert "ETH/USDT" in detector._buffers


# ── Convergence Tracking Tests ──


@pytest.mark.skipif(not MSAR_AVAILABLE, reason="statsmodels not installed")
class TestConvergenceTracking:
    """MSARDetector convergence tracking 검증."""

    def test_convergence_rate_initial(self) -> None:
        """학습 전 convergence_rate = 1.0."""
        cfg = MSARDetectorConfig(
            k_regimes=2, order=1, min_train_window=120,
            sliding_window=0, switching_ar=False,
        )
        detector = MSARDetector(cfg)
        assert detector.convergence_rate == 1.0
        assert detector._fit_attempts == 0

    def test_convergence_rate_after_training(self) -> None:
        """학습 후 convergence_rate가 0~1 범위."""
        cfg = MSARDetectorConfig(
            k_regimes=2, order=1, min_train_window=120,
            retrain_interval=50, sliding_window=0, switching_ar=False,
        )
        detector = MSARDetector(cfg)
        closes = _make_trending_series(300)
        detector.classify_series(closes)

        assert detector._fit_attempts > 0
        assert 0.0 <= detector.convergence_rate <= 1.0
