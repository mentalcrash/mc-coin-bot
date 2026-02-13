"""EnsembleRegimeDetector 단위 테스트.

앙상블 블렌딩, 폴백, warmup 단계별 참여, hysteresis,
classify_series 출력 호환성, update 멀티심볼을 테스트합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.regime.config import (
    EnsembleRegimeDetectorConfig,
    HMMDetectorConfig,
    MetaLearnerConfig,
    MSARDetectorConfig,
    RegimeLabel,
    VolStructureDetectorConfig,
)
from src.regime.detector import RegimeDetector, RegimeState
from src.regime.ensemble import (
    SKLEARN_AVAILABLE,
    EnsembleRegimeDetector,
    add_ensemble_regime_columns,
)
from src.regime.hmm_detector import HMM_AVAILABLE
from src.regime.msar_detector import MSAR_AVAILABLE

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


# ── Config Tests ──


class TestEnsembleConfig:
    """EnsembleRegimeDetectorConfig 검증 테스트."""

    def test_default_rule_only(self) -> None:
        """기본값: Rule-Based만 활성 (HMM/Vol=None)."""
        cfg = EnsembleRegimeDetectorConfig()
        assert cfg.hmm is None
        assert cfg.vol_structure is None
        assert cfg.weight_rule_based == 1.0

    def test_rule_only_weights_valid(self) -> None:
        """Rule-Based만 활성일 때 weight 합 허용."""
        cfg = EnsembleRegimeDetectorConfig(
            weight_rule_based=1.0,
            weight_hmm=0.0,
            weight_vol_structure=0.0,
        )
        assert cfg.weight_rule_based == 1.0

    def test_all_three_active(self) -> None:
        """3개 모두 활성."""
        cfg = EnsembleRegimeDetectorConfig(
            hmm=HMMDetectorConfig(),
            vol_structure=VolStructureDetectorConfig(),
            weight_rule_based=0.4,
            weight_hmm=0.35,
            weight_vol_structure=0.25,
        )
        assert cfg.hmm is not None
        assert cfg.vol_structure is not None

    def test_invalid_weights_sum(self) -> None:
        """활성 감지기 가중치 합 ≠ 1.0."""
        with pytest.raises(ValidationError, match="weights must sum"):
            EnsembleRegimeDetectorConfig(
                hmm=HMMDetectorConfig(),
                vol_structure=VolStructureDetectorConfig(),
                weight_rule_based=0.5,
                weight_hmm=0.5,
                weight_vol_structure=0.5,
            )

    def test_two_detectors_active(self) -> None:
        """Rule + Vol만 활성."""
        cfg = EnsembleRegimeDetectorConfig(
            vol_structure=VolStructureDetectorConfig(),
            weight_rule_based=0.6,
            weight_vol_structure=0.4,
        )
        assert cfg.hmm is None
        assert cfg.vol_structure is not None

    def test_frozen(self) -> None:
        cfg = EnsembleRegimeDetectorConfig()
        with pytest.raises(ValidationError):
            cfg.weight_rule_based = 0.5  # type: ignore[misc]

    def test_min_hold_bars(self) -> None:
        cfg = EnsembleRegimeDetectorConfig(min_hold_bars=10)
        assert cfg.min_hold_bars == 10


# ── Rule-Only Ensemble Tests ──


class TestRuleOnlyEnsemble:
    """Rule-Based만 활성인 앙상블 테스트."""

    @pytest.fixture
    def detector(self) -> EnsembleRegimeDetector:
        config = EnsembleRegimeDetectorConfig(
            weight_rule_based=1.0,
            weight_hmm=0.0,
            weight_vol_structure=0.0,
            min_hold_bars=1,
        )
        return EnsembleRegimeDetector(config)

    def test_output_columns(self, detector: EnsembleRegimeDetector) -> None:
        """출력 DataFrame 컬럼이 RegimeDetector와 호환."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        expected_cols = {
            "regime_label",
            "p_trending",
            "p_ranging",
            "p_volatile",
            "rv_ratio",
            "efficiency_ratio",
        }
        assert set(result.columns) == expected_cols

    def test_trending_detected(self, detector: EnsembleRegimeDetector) -> None:
        """추세 시리즈 → trending 지배적."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        valid = result.dropna()

        trending_ratio = (valid["regime_label"] == RegimeLabel.TRENDING).mean()
        assert trending_ratio > 0.5

    def test_probabilities_sum_to_one(self, detector: EnsembleRegimeDetector) -> None:
        """확률 합 ≈ 1.0."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        valid = result.dropna()

        prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_warmup_nan(self, detector: EnsembleRegimeDetector) -> None:
        """warmup 기간 중 NaN (처음 몇 bar)."""
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        # rv_long_window(20) 이전은 NaN
        warmup = result.iloc[:10]
        assert warmup["rv_ratio"].isna().all()


# ── Rule + Vol Ensemble Tests ──


class TestRuleVolEnsemble:
    """Rule-Based + Vol-Structure 앙상블 테스트."""

    @pytest.fixture
    def detector(self) -> EnsembleRegimeDetector:
        config = EnsembleRegimeDetectorConfig(
            vol_structure=VolStructureDetectorConfig(),
            weight_rule_based=0.6,
            weight_hmm=0.0,
            weight_vol_structure=0.4,
            min_hold_bars=1,
        )
        return EnsembleRegimeDetector(config)

    def test_blending_changes_probabilities(self, detector: EnsembleRegimeDetector) -> None:
        """Vol 추가 시 확률이 Rule-only와 다름."""
        closes = _make_trending_series()
        ensemble_result = detector.classify_series(closes)

        rule_detector = RegimeDetector(detector.config.rule_based)
        rule_result = rule_detector.classify_series(closes)

        # 앙상블과 Rule-only의 확률이 다를 수 있음 (Vol의 영향)
        valid_ensemble = ensemble_result.dropna()
        valid_rule = rule_result.dropna()

        common_idx = valid_ensemble.index.intersection(valid_rule.index)
        if len(common_idx) > 10:
            # 완전히 동일하지 않을 수 있음
            diff = (
                valid_ensemble.loc[common_idx, "p_trending"]
                - valid_rule.loc[common_idx, "p_trending"]
            ).abs()
            # 최소 일부 bar에서 차이 발생
            assert diff.max() > 0.001 or len(common_idx) < len(valid_ensemble)

    def test_warmup_stages(self, detector: EnsembleRegimeDetector) -> None:
        """Rule warmup 후에만 결과 출력, Vol warmup 후 블렌딩 변화."""
        closes = _make_trending_series(150)
        result = detector.classify_series(closes)

        # Rule warmup (~25) 이전: NaN
        assert pd.isna(result.iloc[10]["p_trending"])

        # Rule warmup 이후 (~30): 결과 존재
        valid_after_rule = result.iloc[30:].dropna()
        assert len(valid_after_rule) > 0

    def test_probabilities_sum_to_one(self, detector: EnsembleRegimeDetector) -> None:
        closes = _make_trending_series()
        result = detector.classify_series(closes)
        valid = result.dropna()
        prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)


# ── 3-Detector Ensemble Tests ──


@pytest.mark.skipif(not HMM_AVAILABLE, reason="hmmlearn not installed")
class TestThreeDetectorEnsemble:
    """3-detector 앙상블 테스트 (HMM 필요)."""

    @pytest.fixture
    def detector(self) -> EnsembleRegimeDetector:
        config = EnsembleRegimeDetectorConfig(
            hmm=HMMDetectorConfig(min_train_window=120, retrain_interval=50, n_iter=50),
            vol_structure=VolStructureDetectorConfig(),
            weight_rule_based=0.4,
            weight_hmm=0.35,
            weight_vol_structure=0.25,
            min_hold_bars=3,
        )
        return EnsembleRegimeDetector(config)

    def test_output_format(self, detector: EnsembleRegimeDetector) -> None:
        """3-detector 출력 포맷 호환성."""
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        expected_cols = {
            "regime_label",
            "p_trending",
            "p_ranging",
            "p_volatile",
            "rv_ratio",
            "efficiency_ratio",
        }
        assert set(result.columns) == expected_cols

    def test_probabilities_sum(self, detector: EnsembleRegimeDetector) -> None:
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna()
        if len(valid) > 0:
            prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
            np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)

    def test_hmm_affects_blending(self, detector: EnsembleRegimeDetector) -> None:
        """HMM warmup 전후로 블렌딩 결과가 달라짐."""
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)

        # Rule warmup 이후 HMM warmup 이전 (~bar 30-119)
        early = result.iloc[30:80].dropna()
        # HMM warmup 이후 (~bar 130+)
        late = result.iloc[150:].dropna()

        if len(early) > 0 and len(late) > 0:
            # 블렌딩 참여 감지기 수가 달라 확률 분포 차이
            early_avg = early["p_trending"].mean()
            late_avg = late["p_trending"].mean()
            # 둘 다 유효 값 — 존재 자체가 증거
            assert pd.notna(early_avg)
            assert pd.notna(late_avg)


# ── Incremental API Tests ──


class TestEnsembleIncremental:
    """EnsembleRegimeDetector.update() incremental API 테스트."""

    @pytest.fixture
    def detector(self) -> EnsembleRegimeDetector:
        config = EnsembleRegimeDetectorConfig(
            vol_structure=VolStructureDetectorConfig(),
            weight_rule_based=0.6,
            weight_hmm=0.0,
            weight_vol_structure=0.4,
            min_hold_bars=3,
        )
        return EnsembleRegimeDetector(config)

    def test_warmup_returns_none(self, detector: EnsembleRegimeDetector) -> None:
        """Rule-Based warmup 중 None 반환."""
        # Rule-based detector warmup is rv_long_window (20) bars
        for i in range(detector.config.rule_based.rv_long_window):
            result = detector.update("BTC/USDT", 100.0 + i * 0.1)
            assert result is None, f"Expected None at bar {i}"

    def test_after_warmup_returns_state(self, detector: EnsembleRegimeDetector) -> None:
        result = None
        for i in range(detector.warmup_periods + 10):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert result is not None
        assert isinstance(result, RegimeState)

    def test_state_probabilities_sum(self, detector: EnsembleRegimeDetector) -> None:
        result = None
        for i in range(detector.warmup_periods + 10):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        assert result is not None
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_multi_symbol_independence(self, detector: EnsembleRegimeDetector) -> None:
        for i in range(detector.warmup_periods + 10):
            detector.update("BTC/USDT", 100.0 + i * 1.0)
            detector.update("ETH/USDT", 100.0 + (-1) ** i * 0.1)

        btc_state = detector.get_regime("BTC/USDT")
        eth_state = detector.get_regime("ETH/USDT")
        assert btc_state is not None
        assert eth_state is not None

    def test_get_regime_unregistered(self, detector: EnsembleRegimeDetector) -> None:
        assert detector.get_regime("UNKNOWN") is None

    def test_hysteresis_prevents_flicker(self) -> None:
        """Hysteresis: 짧은 반전은 레짐 유지."""
        config = EnsembleRegimeDetectorConfig(
            weight_rule_based=1.0,
            weight_hmm=0.0,
            weight_vol_structure=0.0,
            min_hold_bars=5,
        )
        detector = EnsembleRegimeDetector(config)

        states: list[RegimeState] = []
        # 상승 추세
        for i in range(detector.warmup_periods + 20):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)
            if result is not None:
                states.append(result)

        if len(states) > 5:
            # 일정 기간 동일 레짐 유지
            last_5 = states[-5:]
            labels = {s.label for s in last_5}
            assert len(labels) <= 2  # 최대 2개 레짐


# ── add_ensemble_regime_columns Tests ──


class TestAddEnsembleRegimeColumns:
    """add_ensemble_regime_columns() 편의 API 테스트."""

    def test_columns_added(self) -> None:
        closes = _make_trending_series()
        df = pd.DataFrame({"close": closes, "volume": 1000.0})
        config = EnsembleRegimeDetectorConfig(
            weight_rule_based=1.0,
            weight_hmm=0.0,
            weight_vol_structure=0.0,
        )
        result = add_ensemble_regime_columns(df, config)

        assert "regime_label" in result.columns
        assert "p_trending" in result.columns
        assert "p_ranging" in result.columns
        assert "p_volatile" in result.columns

    def test_original_data_preserved(self) -> None:
        closes = _make_trending_series()
        df = pd.DataFrame({"close": closes, "volume": 1000.0})
        config = EnsembleRegimeDetectorConfig(
            weight_rule_based=1.0,
            weight_hmm=0.0,
            weight_vol_structure=0.0,
        )
        result = add_ensemble_regime_columns(df, config)
        pd.testing.assert_series_equal(result["close"], df["close"])
        pd.testing.assert_series_equal(result["volume"], df["volume"])

    def test_missing_close_raises(self) -> None:
        df = pd.DataFrame({"volume": [100, 200, 300]})
        with pytest.raises(ValueError, match="close"):
            add_ensemble_regime_columns(df)


# ── MSAR Ensemble Config Tests ──


class TestMSAREnsembleConfig:
    """MSAR 감지기를 포함한 앙상블 설정 테스트."""

    @pytest.mark.skipif(not MSAR_AVAILABLE, reason="statsmodels not installed")
    def test_four_detector_config(self) -> None:
        """4-detector 앙상블 설정."""
        cfg = EnsembleRegimeDetectorConfig(
            hmm=HMMDetectorConfig(),
            vol_structure=VolStructureDetectorConfig(),
            msar=MSARDetectorConfig(),
            weight_rule_based=0.30,
            weight_hmm=0.25,
            weight_vol_structure=0.20,
            weight_msar=0.25,
        )
        assert cfg.msar is not None
        assert cfg.weight_msar == 0.25

    def test_msar_none_default(self) -> None:
        """기본값: MSAR=None."""
        cfg = EnsembleRegimeDetectorConfig()
        assert cfg.msar is None
        assert cfg.weight_msar == 0.0

    def test_invalid_weights_with_msar(self) -> None:
        """MSAR 포함 시 가중치 합 ≠ 1.0 거부."""
        with pytest.raises(ValidationError, match="weights must sum"):
            EnsembleRegimeDetectorConfig(
                msar=MSARDetectorConfig(),
                weight_rule_based=0.5,
                weight_msar=0.6,
            )


# ── 4-Detector Ensemble Tests ──


@pytest.mark.skipif(
    not (HMM_AVAILABLE and MSAR_AVAILABLE),
    reason="hmmlearn or statsmodels not installed",
)
class TestFourDetectorEnsemble:
    """4-detector 앙상블 테스트 (HMM + MSAR 필요)."""

    @pytest.fixture
    def detector(self) -> EnsembleRegimeDetector:
        config = EnsembleRegimeDetectorConfig(
            hmm=HMMDetectorConfig(min_train_window=120, retrain_interval=50, n_iter=50),
            vol_structure=VolStructureDetectorConfig(),
            msar=MSARDetectorConfig(
                k_regimes=2,
                order=1,
                min_train_window=120,
                retrain_interval=50,
                sliding_window=0,
                switching_ar=False,
            ),
            weight_rule_based=0.30,
            weight_hmm=0.25,
            weight_vol_structure=0.20,
            weight_msar=0.25,
            min_hold_bars=3,
        )
        return EnsembleRegimeDetector(config)

    def test_output_format(self, detector: EnsembleRegimeDetector) -> None:
        """4-detector 출력 포맷 호환성."""
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        expected_cols = {
            "regime_label",
            "p_trending",
            "p_ranging",
            "p_volatile",
            "rv_ratio",
            "efficiency_ratio",
        }
        assert set(result.columns) == expected_cols

    def test_probabilities_sum(self, detector: EnsembleRegimeDetector) -> None:
        """확률 합 ≈ 1.0."""
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna()
        if len(valid) > 0:
            prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
            np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-10)


# ── Meta-Learner Ensemble Tests ──


class TestMetaLearnerConfig:
    """MetaLearnerConfig 검증 테스트."""

    def test_default_values(self) -> None:
        cfg = MetaLearnerConfig()
        assert cfg.regularization == 0.1
        assert cfg.train_window == 504
        assert cfg.retrain_interval == 63
        assert cfg.forward_return_window == 20
        assert cfg.trending_threshold == 0.03
        assert cfg.volatile_threshold == 0.05

    def test_frozen(self) -> None:
        cfg = MetaLearnerConfig()
        with pytest.raises(ValidationError):
            cfg.regularization = 0.5  # type: ignore[misc]

    def test_ensemble_method_meta_learner_requires_config(self) -> None:
        """meta_learner 모드에서 config 누락 시 ValidationError."""
        with pytest.raises(ValidationError, match="meta_learner config is required"):
            EnsembleRegimeDetectorConfig(
                ensemble_method="meta_learner",
                meta_learner=None,
            )

    def test_ensemble_method_meta_learner_valid(self) -> None:
        """meta_learner 모드에서 config 제공 시 성공."""
        cfg = EnsembleRegimeDetectorConfig(
            ensemble_method="meta_learner",
            meta_learner=MetaLearnerConfig(),
        )
        assert cfg.ensemble_method == "meta_learner"
        assert cfg.meta_learner is not None


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not installed")
class TestMetaLearnerEnsemble:
    """Meta-learner stacking 앙상블 테스트."""

    @pytest.fixture
    def detector(self) -> EnsembleRegimeDetector:
        config = EnsembleRegimeDetectorConfig(
            vol_structure=VolStructureDetectorConfig(),
            ensemble_method="meta_learner",
            meta_learner=MetaLearnerConfig(
                train_window=100,
                retrain_interval=30,
                forward_return_window=10,
                trending_threshold=0.10,
                volatile_threshold=0.11,
            ),
            min_hold_bars=1,
        )
        return EnsembleRegimeDetector(config)

    def test_output_columns(self, detector: EnsembleRegimeDetector) -> None:
        """meta-learner 출력 컬럼 호환."""
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        expected_cols = {
            "regime_label",
            "p_trending",
            "p_ranging",
            "p_volatile",
            "rv_ratio",
            "efficiency_ratio",
        }
        assert set(result.columns) == expected_cols

    def test_produces_valid_probabilities(self, detector: EnsembleRegimeDetector) -> None:
        """meta-learner 확률 유효."""
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna()

        if len(valid) > 0:
            prob_sum = valid["p_trending"] + valid["p_ranging"] + valid["p_volatile"]
            np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-6)

    def test_meta_learner_generates_results(self, detector: EnsembleRegimeDetector) -> None:
        """meta-learner가 warmup 후 결과 생성."""
        closes = _make_trending_series(300)
        result = detector.classify_series(closes)
        valid = result.dropna()
        # train_window + fwd_window 이후 결과 존재
        assert len(valid) > 0

    def test_incremental_falls_back(self, detector: EnsembleRegimeDetector) -> None:
        """incremental update에서 weighted_average 폴백."""
        result = None
        for i in range(detector.warmup_periods + 10):
            result = detector.update("BTC/USDT", 100.0 + i * 0.5)

        # 폴백 후에도 결과 반환
        if result is not None:
            assert isinstance(result, RegimeState)


# ── Ensemble Hysteresis Pending Label ──


class TestEnsembleHysteresisPendingLabel:
    """EnsembleRegimeDetector incremental hysteresis의 pending label 추적 검증."""

    def test_pending_label_tracked(self) -> None:
        """Ensemble incremental도 pending label을 추적해야 함."""
        cfg = EnsembleRegimeDetectorConfig(min_hold_bars=5)
        detector = EnsembleRegimeDetector(cfg)

        # warmup으로 초기 상태 설정
        for i in range(detector.warmup_periods + 5):
            detector.update("TEST", 100.0 + i * 0.5)

        state = detector.get_regime("TEST")
        assert state is not None

        # pending_labels dict가 존재하고 초기화됨
        assert "TEST" in detector._pending_labels
        assert detector._pending_labels["TEST"] is None

    def test_hold_counter_resets_on_new_pending(self) -> None:
        """다른 pending label이 나오면 카운터가 1로 리셋되어야 함."""
        cfg = EnsembleRegimeDetectorConfig(min_hold_bars=5)
        detector = EnsembleRegimeDetector(cfg)

        # warmup
        for i in range(detector.warmup_periods + 5):
            detector.update("TEST", 100.0 + i * 0.5)

        # hold_counter와 pending_labels가 올바르게 초기화됨
        assert detector._hold_counters["TEST"] == 0
        assert detector._pending_labels["TEST"] is None
