"""Unit tests for Risk-Mom preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.risk_mom.config import RiskMomConfig
from src.strategy.risk_mom.preprocessor import preprocess


class TestPreprocessColumns:
    """전처리 결과 컬럼 검증."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 모두 추가되는지 확인."""
        config = RiskMomConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = {
            "returns",
            "vw_momentum",
            "realized_vol",
            "realized_var",
            "bsc_scaling",
            "drawdown",
            "atr",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_preprocess_preserves_original_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 원본 OHLCV 컬럼이 유지되는지 확인."""
        config = RiskMomConfig()
        result = preprocess(sample_ohlcv, config)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_preprocess_preserves_index(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 인덱스가 유지되는지 확인."""
        config = RiskMomConfig()
        result = preprocess(sample_ohlcv, config)
        pd.testing.assert_index_equal(result.index, sample_ohlcv.index)

    def test_preprocess_row_count(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 행 수가 동일한지 확인."""
        config = RiskMomConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)

    def test_preprocess_does_not_mutate_input(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 변경되지 않는지 확인."""
        config = RiskMomConfig()
        original = sample_ohlcv.copy()
        preprocess(sample_ohlcv, config)
        pd.testing.assert_frame_equal(sample_ohlcv, original)


class TestBSCScaling:
    """BSC (Barroso-Santa-Clara) variance scaling 검증."""

    def test_bsc_scaling_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """BSC scaling은 양수여야 한다 (vol_target^2 / variance)."""
        config = RiskMomConfig()
        result = preprocess(sample_ohlcv, config)

        # NaN 제거 후 검증
        bsc_valid = result["bsc_scaling"].dropna()
        assert len(bsc_valid) > 0
        assert (bsc_valid > 0).all()

    def test_bsc_scaling_inverse_var(self, sample_ohlcv: pd.DataFrame) -> None:
        """분산이 높을수록 BSC scaling이 낮아야 한다."""
        config = RiskMomConfig()
        result = preprocess(sample_ohlcv, config)

        # NaN 제거
        valid_mask = result["realized_var"].notna() & result["bsc_scaling"].notna()
        valid = result[valid_mask]

        if len(valid) > 10:
            # 상위/하위 분산 그룹 비교
            median_var = valid["realized_var"].median()
            high_var_group = valid[valid["realized_var"] > median_var]
            low_var_group = valid[valid["realized_var"] <= median_var]

            if len(high_var_group) > 0 and len(low_var_group) > 0:
                assert high_var_group["bsc_scaling"].mean() < low_var_group["bsc_scaling"].mean()

    def test_bsc_scaling_formula(self, sample_ohlcv: pd.DataFrame) -> None:
        """BSC scaling = vol_target^2 / clipped(realized_var) 공식 검증."""
        config = RiskMomConfig()
        result = preprocess(sample_ohlcv, config)

        valid_mask = result["realized_var"].notna()
        valid = result[valid_mask]

        if len(valid) > 0:
            expected_bsc = config.vol_target**2 / valid["realized_var"].clip(
                lower=config.min_variance
            )
            pd.testing.assert_series_equal(
                valid["bsc_scaling"],
                expected_bsc,
                check_names=False,
            )

    def test_bsc_scaling_with_different_vol_targets(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_target이 높을수록 BSC scaling도 높아야 한다."""
        config_low = RiskMomConfig(vol_target=0.10, min_volatility=0.05)
        config_high = RiskMomConfig(vol_target=0.50, min_volatility=0.05)

        result_low = preprocess(sample_ohlcv, config_low)
        result_high = preprocess(sample_ohlcv, config_high)

        bsc_low = result_low["bsc_scaling"].dropna().mean()
        bsc_high = result_high["bsc_scaling"].dropna().mean()

        assert bsc_high > bsc_low


class TestPreprocessValidation:
    """전처리 입력 검증 테스트."""

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼(close, volume) 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"open": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = RiskMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_close_raises(self) -> None:
        """close 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"volume": [100, 200, 300]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = RiskMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_volume_raises(self) -> None:
        """volume 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"close": [100, 200, 300]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = RiskMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_no_atr_without_high_low(self) -> None:
        """high/low 컬럼 없으면 ATR이 계산되지 않는다."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "close": 50000 + np.cumsum(np.random.randn(n) * 500),
                "volume": np.random.uniform(100, 10000, n),
            },
            index=dates,
        )
        config = RiskMomConfig()
        result = preprocess(df, config)
        assert "atr" not in result.columns


class TestPreprocessWithDifferentConfigs:
    """다양한 설정으로 전처리 검증."""

    def test_preprocess_with_conservative(self, sample_ohlcv: pd.DataFrame) -> None:
        """보수적 설정으로 전처리 실행."""
        config = RiskMomConfig.conservative()
        result = preprocess(sample_ohlcv, config)
        assert "bsc_scaling" in result.columns
        assert result["bsc_scaling"].dropna().shape[0] > 0

    def test_preprocess_with_aggressive(self, sample_ohlcv: pd.DataFrame) -> None:
        """공격적 설정으로 전처리 실행."""
        config = RiskMomConfig.aggressive()
        result = preprocess(sample_ohlcv, config)
        assert "bsc_scaling" in result.columns
        assert result["bsc_scaling"].dropna().shape[0] > 0
