"""DerivativesDetector 단위 테스트.

Vectorized/Incremental API, cascade risk, classification을 검증합니다.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import DerivativesDetectorConfig, RegimeLabel
from src.regime.derivatives_detector import DerivativesDetector

# ── Helpers ──


def _make_deriv_df(
    n: int = 100,
    fr_mean: float = 0.0001,
    fr_std: float = 0.0005,
    oi_base: float = 1e9,
    oi_growth: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Derivatives 테스트 DataFrame 생성."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz=UTC)
    funding_rate = fr_mean + rng.normal(0, fr_std, n)
    oi = oi_base * np.exp(np.cumsum(rng.normal(oi_growth, 0.02, n)))
    return pd.DataFrame(
        {"funding_rate": funding_rate, "oi": oi},
        index=idx,
    )


def _make_extreme_funding_df(n: int = 100) -> pd.DataFrame:
    """극단적 funding rate DataFrame (높은 cascade risk 유도)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz=UTC)
    # 전반은 정상, 후반은 극단적 양수 funding
    funding_rate = np.zeros(n)
    funding_rate[:50] = 0.0001
    funding_rate[50:] = 0.005  # 매우 높은 funding
    oi = np.ones(n) * 1e9
    oi[50:] = 2e9  # OI 급증
    return pd.DataFrame(
        {"funding_rate": funding_rate, "oi": oi},
        index=idx,
    )


# ── Config Tests ──


class TestDerivativesDetectorConfig:
    """DerivativesDetectorConfig 검증."""

    def test_defaults(self) -> None:
        cfg = DerivativesDetectorConfig()
        assert cfg.funding_zscore_window == 7
        assert cfg.oi_change_window == 1
        assert cfg.funding_persistence_window == 14
        assert cfg.cascade_risk_threshold == 0.7

    def test_warmup_periods(self) -> None:
        cfg = DerivativesDetectorConfig()
        assert cfg.warmup_periods == 15  # max(7, 14) + 1

    def test_frozen(self) -> None:
        cfg = DerivativesDetectorConfig()
        with pytest.raises(ValidationError):
            cfg.funding_zscore_window = 10  # type: ignore[misc]


# ── Vectorized API Tests ──


class TestClassifySeries:
    """classify_series() vectorized API 검증."""

    def test_output_columns(self) -> None:
        """출력 DataFrame에 필수 컬럼 존재."""
        detector = DerivativesDetector()
        df = _make_deriv_df()
        result = detector.classify_series(df)
        expected = {
            "regime_label", "p_trending", "p_ranging", "p_volatile",
            "rv_ratio", "efficiency_ratio", "confidence",
        }
        assert expected <= set(result.columns)

    def test_same_length(self) -> None:
        """입출력 길이 동일."""
        detector = DerivativesDetector()
        df = _make_deriv_df(n=50)
        result = detector.classify_series(df)
        assert len(result) == 50

    def test_probabilities_sum(self) -> None:
        """확률 합 = 1.0."""
        detector = DerivativesDetector()
        df = _make_deriv_df()
        result = detector.classify_series(df)
        valid = result.dropna(subset=["p_trending"])
        if len(valid) > 0:
            prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
            np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_warmup_nan(self) -> None:
        """warmup 기간에 NaN 출력."""
        detector = DerivativesDetector()
        df = _make_deriv_df()
        result = detector.classify_series(df)
        # 처음 funding_zscore_window(7) bar는 NaN
        assert result.iloc[:6]["p_trending"].isna().all()

    def test_valid_labels(self) -> None:
        """label이 RegimeLabel 값 중 하나."""
        detector = DerivativesDetector()
        df = _make_deriv_df()
        result = detector.classify_series(df)
        valid_labels = result["regime_label"].dropna().unique()
        allowed = {label.value for label in RegimeLabel}
        for label in valid_labels:
            assert label in allowed

    def test_missing_funding_rate_returns_nan(self) -> None:
        """funding_rate 컬럼 누락 시 NaN."""
        detector = DerivativesDetector()
        df = pd.DataFrame({"oi": [1e9] * 10})
        result = detector.classify_series(df)
        assert result["p_trending"].isna().all()


# ── Cascade Risk Tests ──


class TestCascadeRisk:
    """cascade risk 계산 검증."""

    def test_cascade_risk_series_range(self) -> None:
        """cascade risk는 0~1 범위."""
        detector = DerivativesDetector()
        df = _make_deriv_df()
        risk = detector.get_cascade_risk_series(df)
        valid = risk.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_extreme_funding_high_cascade(self) -> None:
        """극단적 funding rate → cascade risk 상승."""
        detector = DerivativesDetector()
        df = _make_extreme_funding_df()
        risk = detector.get_cascade_risk_series(df)

        # 후반부(극단적 funding)의 cascade risk가 전반부보다 높아야 함
        early_risk = risk.iloc[10:40].mean()
        late_risk = risk.iloc[70:].mean()
        assert late_risk > early_risk


# ── Incremental API Tests ──


class TestIncrementalUpdate:
    """update() incremental API 검증."""

    def test_warmup_returns_none(self) -> None:
        """warmup 미완료 시 None."""
        detector = DerivativesDetector()
        for _i in range(5):
            result = detector.update("BTC/USDT", funding_rate=0.0001, oi=1e9)
        assert result is None

    def test_after_warmup_returns_state(self) -> None:
        """warmup 완료 후 RegimeState 반환."""
        detector = DerivativesDetector()
        result = None
        for i in range(20):
            result = detector.update("BTC/USDT", funding_rate=0.0001 + i * 0.00001, oi=1e9)
        assert result is not None
        assert isinstance(result.label, RegimeLabel)

    def test_probabilities_sum(self) -> None:
        """incremental 확률 합 = 1.0."""
        detector = DerivativesDetector()
        result = None
        for i in range(20):
            result = detector.update("BTC/USDT", funding_rate=0.0001 + i * 0.00001, oi=1e9)
        assert result is not None
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_confidence_always_one(self) -> None:
        """단일 detector → confidence = 1.0."""
        detector = DerivativesDetector()
        result = None
        for _i in range(20):
            result = detector.update("BTC/USDT", funding_rate=0.0001, oi=1e9)
        assert result is not None
        assert result.confidence == 1.0

    def test_cascade_risk_incremental(self) -> None:
        """incremental cascade risk 조회."""
        detector = DerivativesDetector()
        for _i in range(20):
            detector.update("BTC/USDT", funding_rate=0.0001, oi=1e9)
        risk = detector.get_cascade_risk("BTC/USDT")
        assert 0.0 <= risk <= 1.0

    def test_multi_symbol(self) -> None:
        """여러 심볼 독립 추적."""
        detector = DerivativesDetector()
        for _i in range(20):
            detector.update("BTC/USDT", funding_rate=0.0001, oi=1e9)
            detector.update("ETH/USDT", funding_rate=0.001, oi=5e8)

        btc = detector.get_regime("BTC/USDT")
        eth = detector.get_regime("ETH/USDT")
        assert btc is not None
        assert eth is not None

    def test_unknown_symbol(self) -> None:
        """미등록 심볼 → None."""
        detector = DerivativesDetector()
        assert detector.get_regime("UNKNOWN") is None
        assert detector.get_cascade_risk("UNKNOWN") == 0.0


# ── Vectorized ↔ Incremental Parity ──


class TestVectorizedIncrementalParity:
    """classify_series() vs update() 결과 일치 검증."""

    def test_probability_parity(self) -> None:
        """마지막 bar 확률이 vectorized와 incremental에서 일치."""
        detector_vec = DerivativesDetector()
        detector_inc = DerivativesDetector()

        df = _make_deriv_df(n=50, seed=42)
        vec_result = detector_vec.classify_series(df)

        # Incremental
        for _, row in df.iterrows():
            detector_inc.update(
                "BTC/USDT",
                funding_rate=float(row["funding_rate"]),
                oi=float(row["oi"]),
            )

        state = detector_inc.get_regime("BTC/USDT")
        valid = vec_result.dropna(subset=["p_trending"])

        if len(valid) > 0 and state is not None:
            last_row = valid.iloc[-1]
            np.testing.assert_allclose(
                state.probabilities["trending"], last_row["p_trending"], atol=0.05
            )
            np.testing.assert_allclose(
                state.probabilities["volatile"], last_row["p_volatile"], atol=0.05
            )

    def test_cascade_risk_parity(self) -> None:
        """cascade risk가 vectorized와 incremental에서 유사."""
        detector_vec = DerivativesDetector()
        detector_inc = DerivativesDetector()

        df = _make_extreme_funding_df(n=60)
        vec_risk = detector_vec.get_cascade_risk_series(df)

        for _, row in df.iterrows():
            detector_inc.update(
                "BTC/USDT",
                funding_rate=float(row["funding_rate"]),
                oi=float(row["oi"]),
            )

        inc_risk = detector_inc.get_cascade_risk("BTC/USDT")
        vec_last = vec_risk.dropna().iloc[-1] if not vec_risk.dropna().empty else 0.0

        # Cascade risk 방향 일치 (절대값 비교는 rolling window 차이로 느슨)
        assert abs(inc_risk - vec_last) < 0.3


# ── Edge Case Tests ──


class TestEdgeCases:
    """극단 입력 시 crash 없는 graceful handling 검증."""

    def test_constant_funding_rate(self) -> None:
        """일정 funding rate → crash 없이 유효 결과."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D", tz=UTC)
        df = pd.DataFrame(
            {"funding_rate": [0.0001] * 30, "oi": [1e9] * 30},
            index=idx,
        )
        detector = DerivativesDetector()
        result = detector.classify_series(df)
        assert len(result) == 30

    def test_nan_funding_rate(self) -> None:
        """중간 NaN funding rate → graceful handling."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D", tz=UTC)
        fr = np.full(30, 0.0001)
        fr[10:15] = np.nan
        df = pd.DataFrame(
            {"funding_rate": fr, "oi": [1e9] * 30},
            index=idx,
        )
        detector = DerivativesDetector()
        result = detector.classify_series(df)
        assert len(result) == 30  # crash 없음

    def test_oi_near_zero(self) -> None:
        """OI가 극히 작은 값 → 0 나눗셈 없음."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D", tz=UTC)
        oi = np.full(30, 0.001)  # 극히 작은 OI
        df = pd.DataFrame(
            {"funding_rate": [0.0001] * 30, "oi": oi},
            index=idx,
        )
        detector = DerivativesDetector()
        result = detector.classify_series(df)
        assert len(result) == 30
        # NaN이나 inf 없이 cascade risk 유효
        risk = detector.get_cascade_risk_series(df)
        valid = risk.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_very_short_series(self) -> None:
        """warmup 미만 시리즈 → 전체 NaN."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz=UTC)
        df = pd.DataFrame(
            {"funding_rate": [0.0001, 0.0002, 0.0001], "oi": [1e9, 1e9, 1e9]},
            index=idx,
        )
        detector = DerivativesDetector()
        result = detector.classify_series(df)
        assert result["p_trending"].isna().all()

    def test_incremental_oi_zero(self) -> None:
        """Incremental OI=0 → 0 나눗셈 없음."""
        detector = DerivativesDetector()
        for _i in range(20):
            result = detector.update("BTC/USDT", funding_rate=0.0001, oi=0.0)
        # crash 없이 결과 반환
        if result is not None:
            assert 0.0 <= result.probabilities["trending"] <= 1.0
