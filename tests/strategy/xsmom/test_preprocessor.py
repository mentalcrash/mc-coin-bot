"""Unit tests for XSMOM preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.xsmom.config import XSMOMConfig
from src.strategy.xsmom.preprocessor import (
    calculate_holding_signal,
    calculate_rolling_return,
    preprocess,
)


class TestPreprocessColumns:
    """전처리 결과 컬럼 검증."""

    def test_preprocess_adds_required_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 추가되는지 확인."""
        config = XSMOMConfig()
        result = preprocess(sample_ohlcv, config)

        assert "returns" in result.columns
        assert "realized_vol" in result.columns
        assert "rolling_return" in result.columns
        assert "vol_scalar" in result.columns
        assert "atr" in result.columns

    def test_preprocess_preserves_original_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 원본 OHLCV 컬럼이 유지되는지 확인."""
        config = XSMOMConfig()
        result = preprocess(sample_ohlcv, config)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_preprocess_preserves_index(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 인덱스가 유지되는지 확인."""
        config = XSMOMConfig()
        result = preprocess(sample_ohlcv, config)
        pd.testing.assert_index_equal(result.index, sample_ohlcv.index)

    def test_preprocess_row_count(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 행 수가 동일한지 확인."""
        config = XSMOMConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)

    def test_preprocess_does_not_mutate_input(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 변경되지 않는지 확인."""
        config = XSMOMConfig()
        original = sample_ohlcv.copy()
        preprocess(sample_ohlcv, config)
        pd.testing.assert_frame_equal(sample_ohlcv, original)


class TestRollingReturn:
    """Rolling return 계산 검증."""

    def test_rolling_return_log(self) -> None:
        """로그 수익률 계산 검증."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1D")
        close = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41], index=dates)

        result = calculate_rolling_return(close, lookback=2, use_log=True)

        # index 2: ln(121/100) = ln(1.21)
        expected = np.log(121.0 / 100.0)
        assert result.iloc[2] == pytest.approx(expected)

    def test_rolling_return_simple(self) -> None:
        """단순 수익률 계산 검증."""
        dates = pd.date_range("2024-01-01", periods=5, freq="1D")
        close = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=dates)

        result = calculate_rolling_return(close, lookback=2, use_log=False)

        # index 2: (120 - 100) / 100 = 0.20
        assert result.iloc[2] == pytest.approx(0.20)

    def test_rolling_return_log_vs_simple_direction(self) -> None:
        """로그와 단순 수익률의 방향이 동일한지 확인."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1D")
        close = pd.Series(
            [100, 105, 103, 108, 106, 110, 107, 112, 109, 115],
            index=dates,
            dtype=float,
        )

        log_ret = calculate_rolling_return(close, lookback=3, use_log=True)
        simple_ret = calculate_rolling_return(close, lookback=3, use_log=False)

        # 유효한 값에서 방향이 동일해야 함
        valid = log_ret.notna() & simple_ret.notna()
        assert (np.sign(log_ret[valid]) == np.sign(simple_ret[valid])).all()

    def test_rolling_return_nan_before_lookback(self) -> None:
        """lookback 기간 전에는 NaN이어야 한다."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1D")
        close = pd.Series(range(100, 110), index=dates, dtype=float)

        result = calculate_rolling_return(close, lookback=5, use_log=True)

        # index 0~4 should be NaN
        assert result.iloc[:5].isna().all()
        assert not np.isnan(result.iloc[5])

    def test_rolling_return_empty_series_raises(self) -> None:
        """빈 시리즈에서 ValueError 발생."""
        empty = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Empty Series"):
            calculate_rolling_return(empty, lookback=5)


class TestHoldingSignal:
    """Holding signal 필터링 검증."""

    def test_holding_signal_period_1(self) -> None:
        """holding_period=1이면 원본과 동일."""
        signal = pd.Series([1.0, -1.0, 0.5, -0.5, 1.0])
        result = calculate_holding_signal(signal, holding_period=1)
        pd.testing.assert_series_equal(result, signal)

    def test_holding_signal_period_3(self) -> None:
        """holding_period=3에서 시그널이 유지되는지 확인."""
        signal = pd.Series([1.0, -1.0, 0.5, -0.5, 1.0, -0.3])
        result = calculate_holding_signal(signal, holding_period=3)

        # index 0: 갱신 (1.0), index 1-2: ffill (1.0)
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 1.0
        # index 3: 갱신 (-0.5), index 4-5: ffill (-0.5)
        assert result.iloc[3] == -0.5
        assert result.iloc[4] == -0.5
        assert result.iloc[5] == -0.5

    def test_holding_signal_preserves_length(self) -> None:
        """출력 길이가 입력과 동일한지 확인."""
        signal = pd.Series(np.random.randn(50))
        result = calculate_holding_signal(signal, holding_period=7)
        assert len(result) == len(signal)


class TestPreprocessValidation:
    """전처리 입력 검증 테스트."""

    def test_missing_close_raises(self) -> None:
        """close 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"high": [1, 2, 3], "low": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = XSMOMConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_high_raises(self) -> None:
        """high 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"close": [1, 2, 3], "low": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = XSMOMConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_low_raises(self) -> None:
        """low 컬럼 누락 시 ValueError 발생."""
        df = pd.DataFrame(
            {"close": [1, 2, 3], "high": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = XSMOMConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestNumericConversion:
    """숫자형 변환 테스트."""

    def test_numeric_conversion(self, sample_ohlcv: pd.DataFrame) -> None:
        """OHLCV 컬럼이 float64로 변환되는지 확인."""
        config = XSMOMConfig()
        result = preprocess(sample_ohlcv, config)

        assert result["close"].dtype == np.float64
        assert result["high"].dtype == np.float64
        assert result["low"].dtype == np.float64


class TestVolScalar:
    """Vol scalar 계산 검증."""

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar가 양수인지 확인 (NaN 제외)."""
        config = XSMOMConfig()
        result = preprocess(sample_ohlcv, config)

        valid_vs = result["vol_scalar"].dropna()
        assert (valid_vs > 0).all()

    def test_atr_calculated(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR이 올바르게 계산되는지 확인."""
        config = XSMOMConfig()
        result = preprocess(sample_ohlcv, config)

        valid_atr = result["atr"].dropna()
        assert len(valid_atr) > 0
        assert (valid_atr > 0).all()
