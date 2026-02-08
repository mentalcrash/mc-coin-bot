"""Unit tests for Mom-MR Blend preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.mom_mr_blend.config import MomMrBlendConfig
from src.strategy.mom_mr_blend.preprocessor import preprocess


class TestPreprocessColumns:
    """전처리 결과 컬럼 검증."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 모두 추가되는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = {
            "mom_returns",
            "mom_zscore",
            "mr_deviation",
            "mr_zscore",
            "realized_vol",
            "vol_scalar",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_preprocess_preserves_original_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 원본 OHLCV 컬럼이 유지되는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_preprocess_preserves_index(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 인덱스가 유지되는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)
        pd.testing.assert_index_equal(result.index, sample_ohlcv.index)

    def test_preprocess_row_count(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 행 수가 동일한지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)

    def test_preprocess_does_not_mutate_input(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 변경되지 않는지 확인."""
        config = MomMrBlendConfig()
        original = sample_ohlcv.copy()
        preprocess(sample_ohlcv, config)
        pd.testing.assert_frame_equal(sample_ohlcv, original)


class TestMomentumIndicators:
    """모멘텀 지표 검증."""

    def test_mom_returns_has_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """모멘텀 수익률이 계산되는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        mom_valid = result["mom_returns"].dropna()
        assert len(mom_valid) > 0

    def test_mom_zscore_has_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """모멘텀 z-score가 계산되는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        mom_z_valid = result["mom_zscore"].dropna()
        assert len(mom_z_valid) > 0

    def test_mom_zscore_centered(self, sample_ohlcv: pd.DataFrame) -> None:
        """모멘텀 z-score가 대체로 0 중심인지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        mom_z_valid = result["mom_zscore"].dropna()
        if len(mom_z_valid) > 0:
            # z-score는 0 주변에 분포해야 함 (허용 범위 +/- 2)
            assert abs(mom_z_valid.mean()) < 2.0


class TestMeanReversionIndicators:
    """평균회귀 지표 검증."""

    def test_mr_deviation_has_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """평균회귀 편차가 계산되는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        mr_dev_valid = result["mr_deviation"].dropna()
        assert len(mr_dev_valid) > 0

    def test_mr_zscore_has_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """평균회귀 z-score가 계산되는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        mr_z_valid = result["mr_zscore"].dropna()
        assert len(mr_z_valid) > 0


class TestVolatilityIndicators:
    """변동성 지표 검증."""

    def test_realized_vol_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """실현 변동성은 양수여야 한다."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        vol_valid = result["realized_vol"].dropna()
        assert len(vol_valid) > 0
        assert (vol_valid > 0).all()

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """변동성 스케일러는 양수여야 한다."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        vs_valid = result["vol_scalar"].dropna()
        assert len(vs_valid) > 0
        assert (vs_valid > 0).all()

    def test_vol_scalar_shift1(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar에 shift(1)이 적용되었는지 확인."""
        config = MomMrBlendConfig()
        result = preprocess(sample_ohlcv, config)

        # shift(1)이므로 첫 번째 값은 NaN이어야 함
        # realized_vol의 첫 유효값 위치 + 1이 vol_scalar의 첫 유효값 위치여야 함
        assert pd.isna(result["vol_scalar"].iloc[0])

    def test_higher_vol_target_higher_scalar(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_target이 높을수록 vol_scalar도 높아야 한다."""
        config_low = MomMrBlendConfig(vol_target=0.10, min_volatility=0.05)
        config_high = MomMrBlendConfig(vol_target=0.80, min_volatility=0.05)

        result_low = preprocess(sample_ohlcv, config_low)
        result_high = preprocess(sample_ohlcv, config_high)

        vs_low = result_low["vol_scalar"].dropna().mean()
        vs_high = result_high["vol_scalar"].dropna().mean()

        assert vs_high > vs_low


class TestPreprocessValidation:
    """전처리 입력 검증 테스트."""

    def test_missing_close_raises(self) -> None:
        """close 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"open": [1, 2, 3], "volume": [100, 200, 300]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = MomMrBlendConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_close_only_works(self) -> None:
        """close 컬럼만 있어도 전처리가 동작해야 한다."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {"close": 50000 + np.cumsum(np.random.randn(n) * 500)},
            index=dates,
        )
        config = MomMrBlendConfig()
        result = preprocess(df, config)
        assert "mom_returns" in result.columns
        assert "mr_zscore" in result.columns
        assert "vol_scalar" in result.columns


class TestPreprocessWithDifferentConfigs:
    """다양한 설정으로 전처리 검증."""

    def test_preprocess_with_conservative(self, sample_ohlcv: pd.DataFrame) -> None:
        """보수적 설정으로 전처리 실행."""
        config = MomMrBlendConfig.conservative()
        result = preprocess(sample_ohlcv, config)
        assert "mom_zscore" in result.columns
        assert "mr_zscore" in result.columns

    def test_preprocess_with_aggressive(self, sample_ohlcv: pd.DataFrame) -> None:
        """공격적 설정으로 전처리 실행."""
        config = MomMrBlendConfig.aggressive()
        result = preprocess(sample_ohlcv, config)
        assert "mom_zscore" in result.columns
        assert "mr_zscore" in result.columns
