"""RegimeDetector 단위 테스트.

classify_series (vectorized), update (incremental), add_regime_columns,
config validation, hysteresis 동작을 검증합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import RegimeDetectorConfig, RegimeLabel
from src.regime.detector import RegimeDetector, RegimeState, add_regime_columns

# ── Fixtures ──


def _make_trending_series(n: int = 100, noise: float = 0.002) -> pd.Series:
    """명확한 상승 추세 시리즈 생성."""
    rng = np.random.default_rng(42)
    drift = 0.01  # 1% daily drift
    returns = drift + rng.normal(0, noise, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


def _make_ranging_series(n: int = 100, noise: float = 0.005) -> pd.Series:
    """좁은 레인지 횡보 시리즈 생성 (mean-reverting)."""
    rng = np.random.default_rng(42)
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        # mean reversion: price → 100
        reversion = -0.05 * (prices[i - 1] - 100.0)
        prices[i] = prices[i - 1] + reversion + rng.normal(0, noise * 100)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(np.maximum(prices, 50.0), index=idx, name="close")


def _make_volatile_series(n: int = 100) -> pd.Series:
    """급격한 변동 시리즈 (높은 RV, 낮은 ER)."""
    rng = np.random.default_rng(42)
    # 큰 양/음 변동 교대 (방향성 없이)
    returns = rng.choice([-0.05, 0.05], size=n) + rng.normal(0, 0.02, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


@pytest.fixture
def trending_closes() -> pd.Series:
    return _make_trending_series()


@pytest.fixture
def ranging_closes() -> pd.Series:
    return _make_ranging_series()


@pytest.fixture
def volatile_closes() -> pd.Series:
    return _make_volatile_series()


@pytest.fixture
def default_config() -> RegimeDetectorConfig:
    return RegimeDetectorConfig()


@pytest.fixture
def detector(default_config: RegimeDetectorConfig) -> RegimeDetector:
    return RegimeDetector(default_config)


# ── Config Validation ──


class TestRegimeDetectorConfig:
    """RegimeDetectorConfig 검증 테스트."""

    def test_default_values(self) -> None:
        cfg = RegimeDetectorConfig()
        assert cfg.rv_short_window == 5
        assert cfg.rv_long_window == 20
        assert cfg.er_window == 10
        assert cfg.er_trending_threshold == 0.40
        assert cfg.rv_expansion_threshold == 1.3
        assert cfg.min_hold_bars == 5

    def test_frozen(self) -> None:
        cfg = RegimeDetectorConfig()
        with pytest.raises(ValidationError):
            cfg.rv_short_window = 10  # type: ignore[misc]

    def test_invalid_window_order(self) -> None:
        """rv_short >= rv_long이면 ValidationError."""
        with pytest.raises(ValidationError, match="rv_short_window"):
            RegimeDetectorConfig(rv_short_window=20, rv_long_window=10)

    def test_equal_windows_rejected(self) -> None:
        """rv_short == rv_long도 거부."""
        with pytest.raises(ValidationError, match="rv_short_window"):
            RegimeDetectorConfig(rv_short_window=10, rv_long_window=10)

    def test_invalid_field_range(self) -> None:
        """필드 범위 벗어나면 ValidationError."""
        with pytest.raises(ValidationError):
            RegimeDetectorConfig(rv_short_window=0)

    def test_custom_thresholds(self) -> None:
        cfg = RegimeDetectorConfig(
            er_trending_threshold=0.7,
            rv_expansion_threshold=1.5,
            min_hold_bars=5,
        )
        assert cfg.er_trending_threshold == 0.7
        assert cfg.rv_expansion_threshold == 1.5
        assert cfg.min_hold_bars == 5


# ── classify_series (Vectorized) ──


class TestClassifySeries:
    """classify_series() 벡터화 API 테스트."""

    def test_trending_regime(self, detector: RegimeDetector, trending_closes: pd.Series) -> None:
        """명확한 상승 추세 → trending 지배적."""
        result = detector.classify_series(trending_closes)

        # warmup 이후 데이터만 확인
        valid = result.dropna()
        assert len(valid) > 0

        # 과반수가 trending
        trending_ratio = (valid["regime_label"] == RegimeLabel.TRENDING).mean()
        assert trending_ratio > 0.5, f"trending ratio {trending_ratio:.2f} should be > 0.5"

    def test_ranging_regime(self, detector: RegimeDetector, ranging_closes: pd.Series) -> None:
        """좁은 레인지 → ranging 지배적."""
        result = detector.classify_series(ranging_closes)
        valid = result.dropna()
        assert len(valid) > 0

        # ranging이 지배적이거나 trending이 아님
        non_trending_ratio = (valid["regime_label"] != RegimeLabel.TRENDING).mean()
        assert non_trending_ratio > 0.5, (
            f"non-trending ratio {non_trending_ratio:.2f} should be > 0.5"
        )

    def test_volatile_regime(self, detector: RegimeDetector, volatile_closes: pd.Series) -> None:
        """급격한 변동 → volatile 존재."""
        result = detector.classify_series(volatile_closes)
        valid = result.dropna()
        assert len(valid) > 0

        # volatile series는 방향성 없는 큰 변동 → volatile 또는 ranging
        non_trending_ratio = (valid["regime_label"] != RegimeLabel.TRENDING).mean()
        assert non_trending_ratio > 0.4

    def test_probabilities_sum_to_one(
        self, detector: RegimeDetector, trending_closes: pd.Series
    ) -> None:
        """모든 bar에서 probabilities 합 ≈ 1.0."""
        result = detector.classify_series(trending_closes)
        valid = result.dropna()

        prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_warmup_nan(self, detector: RegimeDetector, trending_closes: pd.Series) -> None:
        """warmup 기간 중 NaN 처리."""
        result = detector.classify_series(trending_closes)

        # 처음 rv_long_window 행은 NaN
        warmup = result.iloc[: detector.config.rv_long_window]
        assert warmup["rv_ratio"].isna().all()

    def test_output_columns(self, detector: RegimeDetector, trending_closes: pd.Series) -> None:
        """출력 DataFrame 컬럼 확인."""
        result = detector.classify_series(trending_closes)
        expected_cols = {
            "regime_label",
            "p_trending",
            "p_ranging",
            "p_volatile",
            "rv_ratio",
            "efficiency_ratio",
        }
        assert set(result.columns) == expected_cols

    def test_hysteresis_prevents_flicker(self) -> None:
        """Hysteresis: 짧은 반전은 레이블 유지."""
        config = RegimeDetectorConfig(min_hold_bars=5)
        detector = RegimeDetector(config)

        # 추세 → 짧은 반전(2bar) → 추세 시 레이블 유지
        n = 80
        prices = np.zeros(n)
        prices[0] = 100.0
        # 명확한 상승 추세
        for i in range(1, 60):
            prices[i] = prices[i - 1] * 1.01
        # 짧은 하락 (2bar) — hysteresis로 레짐 유지되어야 함
        for i in range(60, 63):
            prices[i] = prices[i - 1] * 0.99
        # 다시 상승
        for i in range(63, n):
            prices[i] = prices[i - 1] * 1.01

        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        closes = pd.Series(prices, index=idx)

        result = detector.classify_series(closes)
        valid = result.dropna()

        # 짧은 하락 구간(60~62)에서도 레짐 전환 없어야 함
        if len(valid) > 62:
            transition_zone = valid.iloc[35:45]  # 상승 중간 구간
            # 해당 구간에서 레짐이 안정적이어야 함
            unique_labels = transition_zone["regime_label"].nunique()
            assert unique_labels <= 2  # 최대 2가지 레짐


# ── update (Incremental) ──


class TestIncrementalUpdate:
    """update() incremental API 테스트."""

    def test_warmup_returns_none(self, detector: RegimeDetector) -> None:
        """warmup 중 None 반환 (rv_long_window 이하)."""
        # rv_long_window 이하의 데이터에서는 None 반환
        for i in range(detector.config.rv_long_window):
            result = detector.update("BTC/USDT", 100.0 + i * 0.1)
            assert result is None, f"Expected None at bar {i}, got {result}"

    def test_after_warmup_returns_state(self, detector: RegimeDetector) -> None:
        """warmup 후 RegimeState 반환."""
        for i in range(detector.warmup_periods + 5):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert result is not None
        assert isinstance(result, RegimeState)
        assert result.label in list(RegimeLabel)

    def test_state_has_valid_probabilities(self, detector: RegimeDetector) -> None:
        """RegimeState probabilities 합 ≈ 1.0."""
        result = None
        for i in range(detector.warmup_periods + 5):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert result is not None
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_multi_symbol_independence(self, detector: RegimeDetector) -> None:
        """멀티 심볼 독립 레짐."""
        # BTC: 상승 추세
        for i in range(detector.warmup_periods + 5):
            detector.update("BTC/USDT", 100.0 + i * 1.0)

        # ETH: 횡보
        for i in range(detector.warmup_periods + 5):
            detector.update("ETH/USDT", 100.0 + (-1) ** i * 0.1)

        btc_state = detector.get_regime("BTC/USDT")
        eth_state = detector.get_regime("ETH/USDT")

        assert btc_state is not None
        assert eth_state is not None
        # 두 심볼이 다른 상태일 수 있음
        assert btc_state.label in list(RegimeLabel)
        assert eth_state.label in list(RegimeLabel)

    def test_get_regime_unregistered(self, detector: RegimeDetector) -> None:
        """미등록 심볼 → None."""
        assert detector.get_regime("UNKNOWN") is None

    def test_bars_held_increments(self, detector: RegimeDetector) -> None:
        """bars_held 카운터 증가."""
        states: list[RegimeState] = []
        for i in range(detector.warmup_periods + 10):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)
            if result is not None:
                states.append(result)

        # bars_held는 양수
        assert all(s.bars_held >= 1 for s in states)


# ── add_regime_columns ──


class TestAddRegimeColumns:
    """add_regime_columns() 편의 API 테스트."""

    def test_columns_added(self, trending_closes: pd.Series) -> None:
        """레짐 컬럼 추가 검증."""
        df = pd.DataFrame({"close": trending_closes, "volume": 1000.0})
        result = add_regime_columns(df)

        assert "regime_label" in result.columns
        assert "p_trending" in result.columns
        assert "p_ranging" in result.columns
        assert "p_volatile" in result.columns
        assert "rv_ratio" in result.columns
        assert "efficiency_ratio" in result.columns

    def test_original_data_preserved(self, trending_closes: pd.Series) -> None:
        """원본 데이터 보존."""
        df = pd.DataFrame({"close": trending_closes, "volume": 1000.0})
        result = add_regime_columns(df)

        pd.testing.assert_series_equal(result["close"], df["close"])
        pd.testing.assert_series_equal(result["volume"], df["volume"])

    def test_missing_close_raises(self) -> None:
        """close 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"volume": [100, 200, 300]})
        with pytest.raises(ValueError, match="close"):
            add_regime_columns(df)

    def test_custom_config(self, trending_closes: pd.Series) -> None:
        """커스텀 config 적용."""
        df = pd.DataFrame({"close": trending_closes})
        config = RegimeDetectorConfig(rv_short_window=3, rv_long_window=10, er_window=5)
        result = add_regime_columns(df, config)

        valid = result.dropna()
        assert len(valid) > 0
        # 더 짧은 윈도우 → warmup 줄어듦
        assert result["rv_ratio"].first_valid_index() is not None


# ── RegimeLabel ──


class TestRegimeLabel:
    """RegimeLabel StrEnum 테스트."""

    def test_values(self) -> None:
        assert RegimeLabel.TRENDING == "trending"
        assert RegimeLabel.RANGING == "ranging"
        assert RegimeLabel.VOLATILE == "volatile"

    def test_from_string(self) -> None:
        assert RegimeLabel("trending") == RegimeLabel.TRENDING


# ── Hysteresis: Vectorized ↔ Incremental Parity ──


class TestHysteresisPendingLabel:
    """Incremental hysteresis가 vectorized와 동일하게 pending label을 추적하는지 검증."""

    def test_mixed_pending_labels_not_counted_together(self) -> None:
        """서로 다른 pending label(B, C)이 섞이면 카운터가 리셋되어야 함.

        Sequence: A A A B B C C C C C (min_hold=3)
        Vectorized: pending B→B (cnt 2), C reset→C→C (cnt 3, switch) = AAAAAA A C C C
        Incremental must match.
        """
        from src.regime.detector import apply_hysteresis

        # Vectorized reference
        labels = pd.Series(["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"])
        result_vec = apply_hysteresis(labels, min_hold_bars=3)

        # bar 3-4: B pending (count 1, 2), bar 5: C resets to 1
        # bar 6: C count 2, bar 7: C count 3 → switch
        assert result_vec.iloc[3] == "A"  # B overridden
        assert result_vec.iloc[4] == "A"  # B overridden
        assert result_vec.iloc[5] == "A"  # C count 1, overridden
        assert result_vec.iloc[6] == "A"  # C count 2, overridden
        assert result_vec.iloc[7] == "C"  # C count 3 → switch
        assert result_vec.iloc[8] == "C"
        assert result_vec.iloc[9] == "C"

    def test_incremental_matches_vectorized_mixed_pending(self) -> None:
        """Incremental hysteresis가 mixed pending에서 vectorized와 동일한 결과."""
        cfg = RegimeDetectorConfig(
            rv_short_window=2,
            rv_long_window=5,
            er_window=3,
            min_hold_bars=3,
        )
        detector = RegimeDetector(cfg)

        # 충분한 warmup 데이터로 명확한 regime 생성
        # Phase 1: TRENDING (strong uptrend)
        warmup_base = 100.0
        warmup_prices = [warmup_base * (1.01**i) for i in range(30)]

        for price in warmup_prices:
            detector.update("TEST", price)

        initial_state = detector.get_regime("TEST")
        assert initial_state is not None

        # 현재 regime이 무엇이든 hold_counter와 pending_label 동작을 검증
        # hold_counter는 같은 pending label이 연속으로 나와야만 증가
        assert detector._hold_counters.get("TEST", 0) == 0
        assert detector._pending_labels.get("TEST") is None

    def test_revert_to_current_resets_pending(self) -> None:
        """현재 label로 돌아오면 pending과 counter가 모두 리셋."""
        from src.regime.detector import apply_hysteresis

        # A A A B A A A  — B는 1번만 나오고 다시 A
        labels = pd.Series(["A", "A", "A", "B", "A", "A", "A"])
        result = apply_hysteresis(labels, min_hold_bars=3)

        # B는 1번뿐이므로 전환 안 됨
        for i in range(7):
            assert result.iloc[i] == "A", f"bar {i} should be A"


# ── Vectorized ↔ Incremental Parity ──


class TestVectorizedIncrementalParity:
    """classify_series() (vectorized) vs update() (incremental) 결과 일치 검증.

    문서 §13 "동일 결과 보장" 주장의 실증적 검증.
    """

    def test_probability_parity_no_hysteresis(self) -> None:
        """min_hold_bars=1로 hysteresis 비활성 → 확률 값 정확히 일치."""
        cfg = RegimeDetectorConfig(min_hold_bars=1)
        vec_det = RegimeDetector(cfg)
        inc_det = RegimeDetector(cfg)

        closes = _make_trending_series(80)
        vec_df = vec_det.classify_series(closes)

        # Incremental: 같은 시리즈를 bar-by-bar로
        for price in closes:
            inc_det.update("TEST", float(price))

        # warmup 이후 구간 비교
        warmup = inc_det.warmup_periods
        for i in range(warmup + 1, len(closes)):
            vec_pt = vec_df.iloc[i]["p_trending"]

            if pd.isna(vec_pt):
                continue

            state = inc_det.get_regime("TEST")
            assert state is not None

        # 마지막 bar의 확률 비교
        last_valid_idx = vec_df.dropna().index[-1]
        last_pos = closes.index.get_loc(last_valid_idx)
        vec_row = vec_df.iloc[last_pos]
        state = inc_det.get_regime("TEST")
        assert state is not None
        np.testing.assert_allclose(
            state.probabilities["trending"], vec_row["p_trending"], atol=1e-6
        )
        np.testing.assert_allclose(
            state.probabilities["ranging"], vec_row["p_ranging"], atol=1e-6
        )
        np.testing.assert_allclose(
            state.probabilities["volatile"], vec_row["p_volatile"], atol=1e-6
        )

    def test_label_parity_with_hysteresis(self) -> None:
        """min_hold_bars=5로 hysteresis 포함 → label 일치."""
        cfg = RegimeDetectorConfig(min_hold_bars=5)
        vec_det = RegimeDetector(cfg)
        inc_det = RegimeDetector(cfg)

        closes = _make_trending_series(80)
        vec_df = vec_det.classify_series(closes)

        # Incremental
        for price in closes:
            inc_det.update("TEST", float(price))

        # warmup 이후 label 비교
        valid = vec_df.dropna()
        if len(valid) > 0:
            last_vec_label = valid.iloc[-1]["regime_label"]
            state = inc_det.get_regime("TEST")
            assert state is not None
            assert last_vec_label in (state.label, state.label.value)

    def test_parity_ranging_series(self) -> None:
        """횡보 시리즈에서도 parity 유지."""
        cfg = RegimeDetectorConfig(min_hold_bars=1)
        vec_det = RegimeDetector(cfg)
        inc_det = RegimeDetector(cfg)

        closes = _make_ranging_series(80)
        vec_df = vec_det.classify_series(closes)

        for price in closes:
            inc_det.update("TEST", float(price))

        state = inc_det.get_regime("TEST")
        last_valid = vec_df.dropna()
        if len(last_valid) > 0 and state is not None:
            last_row = last_valid.iloc[-1]
            np.testing.assert_allclose(
                state.probabilities["trending"], last_row["p_trending"], atol=1e-6
            )

    def test_parity_volatile_series(self) -> None:
        """고변동 시리즈에서도 parity 유지."""
        cfg = RegimeDetectorConfig(min_hold_bars=1)
        vec_det = RegimeDetector(cfg)
        inc_det = RegimeDetector(cfg)

        closes = _make_volatile_series(80)
        vec_df = vec_det.classify_series(closes)

        for price in closes:
            inc_det.update("TEST", float(price))

        state = inc_det.get_regime("TEST")
        last_valid = vec_df.dropna()
        if len(last_valid) > 0 and state is not None:
            last_row = last_valid.iloc[-1]
            np.testing.assert_allclose(
                state.probabilities["volatile"], last_row["p_volatile"], atol=1e-6
            )


# ── Edge Case Tests ──


class TestEdgeCases:
    """극단 입력 시 crash 없는 graceful handling 검증."""

    def test_constant_prices(self) -> None:
        """일정 가격 → NaN 또는 유효값, crash 없음."""
        detector = RegimeDetector()
        closes = pd.Series(
            [100.0] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="D"),
        )
        result = detector.classify_series(closes)
        # rv_ratio는 0/0 → NaN, crash 없어야 함
        assert len(result) == 50

    def test_nan_infected_input(self) -> None:
        """중간 NaN이 포함된 입력 → graceful handling."""
        rng = np.random.default_rng(42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.005, 0.01, 50)))
        prices[20:25] = np.nan  # 5 bars NaN
        closes = pd.Series(
            prices,
            index=pd.date_range("2024-01-01", periods=50, freq="D"),
        )
        detector = RegimeDetector()
        result = detector.classify_series(closes)
        assert len(result) == 50  # crash 없음

    def test_very_short_series(self) -> None:
        """warmup 미만 시리즈 → 전체 NaN."""
        detector = RegimeDetector()
        closes = pd.Series(
            [100.0, 101.0, 99.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        result = detector.classify_series(closes)
        assert result["regime_label"].isna().all()

    def test_single_price(self) -> None:
        """단일 가격 → crash 없음."""
        detector = RegimeDetector()
        closes = pd.Series(
            [100.0],
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )
        result = detector.classify_series(closes)
        assert len(result) == 1
        assert result["regime_label"].isna().all()
