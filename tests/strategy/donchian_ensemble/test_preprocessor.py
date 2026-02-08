"""Unit tests for Donchian Ensemble preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig
from src.strategy.donchian_ensemble.preprocessor import (
    calculate_donchian_channel,
    preprocess,
)


class TestPreprocessColumns:
    """전처리 결과 컬럼 검증."""

    def test_preprocess_adds_dc_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 모든 dc_upper/dc_lower 컬럼이 추가되는지 확인."""
        config = DonchianEnsembleConfig()
        result = preprocess(sample_ohlcv, config)

        for lb in config.lookbacks:
            assert f"dc_upper_{lb}" in result.columns, f"dc_upper_{lb} missing"
            assert f"dc_lower_{lb}" in result.columns, f"dc_lower_{lb} missing"

    def test_preprocess_adds_vol_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 realized_vol, vol_scalar 컬럼이 추가되는지 확인."""
        config = DonchianEnsembleConfig()
        result = preprocess(sample_ohlcv, config)

        assert "realized_vol" in result.columns
        assert "vol_scalar" in result.columns

    def test_preprocess_preserves_original_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 원본 OHLCV 컬럼이 유지되는지 확인."""
        config = DonchianEnsembleConfig()
        result = preprocess(sample_ohlcv, config)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_preprocess_preserves_index(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 인덱스가 유지되는지 확인."""
        config = DonchianEnsembleConfig()
        result = preprocess(sample_ohlcv, config)
        pd.testing.assert_index_equal(result.index, sample_ohlcv.index)

    def test_preprocess_row_count(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 행 수가 동일한지 확인."""
        config = DonchianEnsembleConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)

    def test_preprocess_does_not_mutate_input(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 변경되지 않는지 확인."""
        config = DonchianEnsembleConfig()
        original = sample_ohlcv.copy()
        preprocess(sample_ohlcv, config)
        pd.testing.assert_frame_equal(sample_ohlcv, original)

    def test_preprocess_custom_lookbacks(self, sample_ohlcv: pd.DataFrame) -> None:
        """커스텀 lookback으로 전처리 시 올바른 컬럼 추가."""
        config = DonchianEnsembleConfig(lookbacks=(10, 30, 60))
        result = preprocess(sample_ohlcv, config)

        assert "dc_upper_10" in result.columns
        assert "dc_lower_30" in result.columns
        assert "dc_upper_60" in result.columns
        # 기본 lookback 컬럼은 없어야 함
        assert "dc_upper_5" not in result.columns
        assert "dc_upper_360" not in result.columns


class TestDonchianChannel:
    """Donchian Channel 계산 검증."""

    def test_channel_upper_is_max(self) -> None:
        """upper channel이 rolling max인지 확인."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1D")
        high = pd.Series([10, 12, 11, 13, 9, 15, 14, 16, 12, 11], index=dates, dtype=float)
        low = pd.Series([8, 9, 8, 10, 7, 12, 11, 13, 10, 9], index=dates, dtype=float)

        upper, _lower = calculate_donchian_channel(high, low, period=3)

        # index 2 (3rd element): max of [10, 12, 11] = 12
        assert upper.iloc[2] == 12.0
        # index 4 (5th element): max of [11, 13, 9] = 13
        assert upper.iloc[4] == 13.0

    def test_channel_lower_is_min(self) -> None:
        """lower channel이 rolling min인지 확인."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1D")
        high = pd.Series([10, 12, 11, 13, 9, 15, 14, 16, 12, 11], index=dates, dtype=float)
        low = pd.Series([8, 9, 8, 10, 7, 12, 11, 13, 10, 9], index=dates, dtype=float)

        _upper, lower = calculate_donchian_channel(high, low, period=3)

        # index 2: min of [8, 9, 8] = 8
        assert lower.iloc[2] == 8.0
        # index 4: min of [8, 10, 7] = 7
        assert lower.iloc[4] == 7.0

    def test_channel_nan_before_period(self) -> None:
        """lookback 기간 전에는 NaN이어야 한다."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1D")
        high = pd.Series(range(10, 20), index=dates, dtype=float)
        low = pd.Series(range(5, 15), index=dates, dtype=float)

        upper, lower = calculate_donchian_channel(high, low, period=5)

        # index 0~3 (first 4) should be NaN
        assert upper.iloc[:4].isna().all()
        assert lower.iloc[:4].isna().all()
        # index 4 should have valid value
        assert not np.isnan(upper.iloc[4])
        assert not np.isnan(lower.iloc[4])

    def test_upper_always_ge_lower(self, sample_ohlcv: pd.DataFrame) -> None:
        """upper channel >= lower channel."""
        high: pd.Series = sample_ohlcv["high"]  # type: ignore[assignment]
        low: pd.Series = sample_ohlcv["low"]  # type: ignore[assignment]

        upper, lower = calculate_donchian_channel(high, low, period=20)
        valid_mask = upper.notna() & lower.notna()
        assert (upper[valid_mask] >= lower[valid_mask]).all()


class TestPreprocessValidation:
    """전처리 입력 검증 테스트."""

    def test_missing_high_raises(self) -> None:
        """high 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"low": [1, 2, 3], "close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = DonchianEnsembleConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_low_raises(self) -> None:
        """low 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"high": [1, 2, 3], "close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = DonchianEnsembleConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_close_raises(self) -> None:
        """close 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"high": [1, 2, 3], "low": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = DonchianEnsembleConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_all_required_raises(self) -> None:
        """모든 필수 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"volume": [100, 200, 300]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = DonchianEnsembleConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)
