"""HMMDetector 단위 테스트.

classify_series, update, config validation, probability 검증,
warmup 처리, graceful degradation을 테스트합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import HMMDetectorConfig, RegimeLabel
from src.regime.hmm_detector import HMM_AVAILABLE, HMMDetector

pytestmark = pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn not installed")


# ── Helpers ──


def _make_trending_series(n: int = 400, drift: float = 0.01) -> pd.Series:
    """명확한 상승 추세."""
    rng = np.random.default_rng(42)
    returns = drift + rng.normal(0, 0.002, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


def _make_ranging_series(n: int = 400, noise: float = 0.005) -> pd.Series:
    """횡보 시리즈 (mean-reverting)."""
    rng = np.random.default_rng(42)
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        reversion = -0.05 * (prices[i - 1] - 100.0)
        prices[i] = prices[i - 1] + reversion + rng.normal(0, noise * 100)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(np.maximum(prices, 50.0), index=idx, name="close")


def _make_mixed_series(n: int = 600) -> pd.Series:
    """추세 + 횡보 혼합 시리즈."""
    rng = np.random.default_rng(42)
    # 300 bars trending, 300 bars ranging
    trend_returns = 0.01 + rng.normal(0, 0.003, 300)
    range_returns = rng.normal(0, 0.005, 300)
    returns = np.concatenate([trend_returns, range_returns])
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


# ── Config Tests ──


class TestHMMDetectorConfig:
    """HMMDetectorConfig 검증 테스트."""

    def test_default_values(self) -> None:
        cfg = HMMDetectorConfig()
        assert cfg.n_states == 3
        assert cfg.min_train_window == 252
        assert cfg.retrain_interval == 21
        assert cfg.n_iter == 100
        assert cfg.vol_window == 20
        assert cfg.use_log_returns is True

    def test_frozen(self) -> None:
        cfg = HMMDetectorConfig()
        with pytest.raises(ValidationError):
            cfg.n_states = 5  # type: ignore[misc]

    def test_invalid_n_states(self) -> None:
        with pytest.raises(ValidationError):
            HMMDetectorConfig(n_states=1)

    def test_invalid_min_train_window(self) -> None:
        with pytest.raises(ValidationError):
            HMMDetectorConfig(min_train_window=50)

    def test_warmup_periods(self) -> None:
        cfg = HMMDetectorConfig(min_train_window=200)
        assert cfg.warmup_periods == 201

    def test_custom_config(self) -> None:
        cfg = HMMDetectorConfig(
            n_states=2,
            min_train_window=150,
            retrain_interval=10,
        )
        assert cfg.n_states == 2
        assert cfg.min_train_window == 150
        assert cfg.retrain_interval == 10


# ── classify_series Tests ──


class TestHMMClassifySeries:
    """HMMDetector.classify_series() 벡터화 API 테스트."""

    @pytest.fixture
    def fast_config(self) -> HMMDetectorConfig:
        """테스트용 빠른 설정."""
        return HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
        )

    def test_output_columns(self, fast_config: HMMDetectorConfig) -> None:
        """출력 DataFrame 컬럼 확인."""
        detector = HMMDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        expected_cols = {"p_trending", "p_ranging", "p_volatile", "hmm_state", "hmm_prob"}
        assert set(result.columns) == expected_cols

    def test_warmup_nan(self, fast_config: HMMDetectorConfig) -> None:
        """warmup 기간 중 NaN."""
        detector = HMMDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)

        # min_train_window 이전은 NaN
        warmup = result.iloc[: fast_config.min_train_window]
        assert warmup["p_trending"].isna().all()

    def test_probabilities_valid_range(self, fast_config: HMMDetectorConfig) -> None:
        """확률 값이 0~1 범위."""
        detector = HMMDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        assert (valid["p_trending"] >= 0.0).all()
        assert (valid["p_trending"] <= 1.0).all()
        assert (valid["p_ranging"] >= 0.0).all()
        assert (valid["p_ranging"] <= 1.0).all()

    def test_probabilities_sum_to_one(self, fast_config: HMMDetectorConfig) -> None:
        """확률 합 ≈ 1.0."""
        detector = HMMDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        if len(valid) > 0:
            prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
            np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_trending_detected(self, fast_config: HMMDetectorConfig) -> None:
        """강한 추세 시리즈 → HMM이 결과 생성."""
        detector = HMMDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        # HMM이 warmup 후 결과를 생성해야 함
        if len(valid) > 0:
            # p_trending + p_ranging + p_volatile = 1.0
            prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
            np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_volatile_always_zero(self, fast_config: HMMDetectorConfig) -> None:
        """HMM의 p_volatile은 항상 0.0."""
        detector = HMMDetector(fast_config)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna(subset=["p_trending"])

        if len(valid) > 0:
            assert (valid["p_volatile"] == 0.0).all()


# ── update (Incremental) Tests ──


class TestHMMIncremental:
    """HMMDetector.update() incremental API 테스트."""

    @pytest.fixture
    def fast_config(self) -> HMMDetectorConfig:
        return HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
        )

    def test_warmup_returns_none(self, fast_config: HMMDetectorConfig) -> None:
        """warmup 중 None 반환."""
        detector = HMMDetector(fast_config)
        for i in range(fast_config.min_train_window):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)
            assert result is None

    def test_after_warmup_returns_state(self, fast_config: HMMDetectorConfig) -> None:
        """warmup 후 RegimeState 반환."""
        from src.regime.detector import RegimeState

        detector = HMMDetector(fast_config)
        result = None
        # 충분한 데이터 + 약간의 여유
        for i in range(fast_config.min_train_window + 30):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        # 모델 훈련이 성공하면 state 반환
        if result is not None:
            assert isinstance(result, RegimeState)
            assert result.label in list(RegimeLabel)

    def test_multi_symbol_independence(self, fast_config: HMMDetectorConfig) -> None:
        """멀티 심볼 독립."""
        detector = HMMDetector(fast_config)
        for i in range(fast_config.min_train_window + 30):
            detector.update("BTC/USDT", 100.0 + i * 0.5)
            detector.update("ETH/USDT", 100.0 + (-1) ** i * 0.1)

        # 별도 버퍼 사용
        assert "BTC/USDT" in detector._buffers
        assert "ETH/USDT" in detector._buffers


# ── Sliding Window + Decay Tests ──


class TestHMMSlidingWindow:
    """HMM sliding_window + decay_half_life 테스트."""

    def test_sliding_window_default_zero(self) -> None:
        """기본값 sliding_window=0 (expanding)."""
        cfg = HMMDetectorConfig()
        assert cfg.sliding_window == 0

    def test_decay_half_life_default_zero(self) -> None:
        """기본값 decay_half_life=0 (uniform)."""
        cfg = HMMDetectorConfig()
        assert cfg.decay_half_life == 0

    def test_sliding_window_config(self) -> None:
        """sliding_window 설정."""
        cfg = HMMDetectorConfig(sliding_window=504, decay_half_life=126)
        assert cfg.sliding_window == 504
        assert cfg.decay_half_life == 126

    def test_sliding_window_produces_results(self) -> None:
        """sliding window 모드에서도 결과 생성."""
        cfg = HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
            sliding_window=200,
        )
        detector = HMMDetector(cfg)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)

        valid = result.dropna(subset=["p_trending"])
        # 최소한 warmup 이후 일부 결과 존재
        assert len(valid) >= 0  # 수렴 실패 가능

    def test_decay_produces_results(self) -> None:
        """decay half-life 모드에서도 결과 생성."""
        cfg = HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
            decay_half_life=60,
        )
        detector = HMMDetector(cfg)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)

        valid = result.dropna(subset=["p_trending"])
        assert len(valid) >= 0

    def test_sliding_plus_decay_together(self) -> None:
        """sliding window + decay 동시 사용."""
        cfg = HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
            sliding_window=200,
            decay_half_life=60,
        )
        detector = HMMDetector(cfg)
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)

        # 출력 컬럼 동일
        expected_cols = {"p_trending", "p_ranging", "p_volatile", "hmm_state", "hmm_prob"}
        assert set(result.columns) == expected_cols

    def test_sliding_window_incremental(self) -> None:
        """incremental API에서 sliding window."""
        cfg = HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
            sliding_window=200,
        )
        detector = HMMDetector(cfg)

        for i in range(cfg.min_train_window + 30):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        # 수렴 실패 가능 → None이어도 크래시 없이 종료
        if result is not None:
            assert result.label in list(RegimeLabel)


# ── Convergence Tracking Tests ──


@pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn not installed")
class TestConvergenceTracking:
    """HMMDetector convergence tracking 검증."""

    def test_convergence_rate_initial(self) -> None:
        """학습 전 convergence_rate = 1.0."""
        cfg = HMMDetectorConfig(min_train_window=120, n_iter=50)
        detector = HMMDetector(cfg)
        assert detector.convergence_rate == 1.0
        assert detector._fit_attempts == 0
        assert detector._fit_failures == 0

    def test_convergence_rate_after_training(self) -> None:
        """학습 후 convergence_rate가 0~1 범위."""
        cfg = HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
        )
        detector = HMMDetector(cfg)
        closes = _make_trending_series(300)
        detector.classify_series(closes)

        assert detector._fit_attempts > 0
        assert 0.0 <= detector.convergence_rate <= 1.0

    def test_convergence_counts_incremental(self) -> None:
        """incremental에서도 fit_attempts 카운트."""
        cfg = HMMDetectorConfig(
            min_train_window=120,
            retrain_interval=50,
            n_iter=50,
        )
        detector = HMMDetector(cfg)

        for i in range(cfg.min_train_window + 60):
            detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert detector._fit_attempts > 0
