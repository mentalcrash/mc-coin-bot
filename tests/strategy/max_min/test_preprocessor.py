"""Tests for MAX/MIN Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.max_min.config import MaxMinConfig
from src.strategy.max_min.preprocessor import preprocess


class TestPreprocess:
    """preprocess() 메인 함수 테스트."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 모두 존재."""
        config = MaxMinConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "rolling_max",
            "rolling_min",
            "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rolling_max_shift(self, sample_ohlcv: pd.DataFrame) -> None:
        """rolling_max는 shift(1) 적용 -- 현재 봉의 high가 포함되지 않아야 함.

        rolling_max[i]는 high[i-lookback:i] (현재 봉 제외) 기준이어야 함.
        rolling(10, min_periods=10)은 인덱스 0-8이 NaN (9개), shift(1)로 인덱스 9도 NaN.
        따라서 인덱스 0-9 (10개)가 NaN.
        """
        config = MaxMinConfig(lookback=10)
        result = preprocess(sample_ohlcv, config)

        # lookback=10: rolling(min_periods=10)은 인덱스 9부터 유효, shift(1)로 인덱스 10부터 유효
        # 즉, 인덱스 0-9 (10개)가 NaN
        assert result["rolling_max"].iloc[:10].isna().all()
        assert result["rolling_max"].iloc[10:].notna().all()

        # shift(1) 검증: result["rolling_max"][i]는 high[i-10:i]의 max여야 함
        # (shift(1) 때문에 i시점에서는 i-1까지의 rolling max)
        idx = 50  # 충분히 큰 인덱스
        expected_max = sample_ohlcv["high"].iloc[idx - 10 : idx].max()
        actual_max = result["rolling_max"].iloc[idx]
        assert abs(actual_max - expected_max) < 1e-6

    def test_rolling_min_shift(self, sample_ohlcv: pd.DataFrame) -> None:
        """rolling_min는 shift(1) 적용 -- 현재 봉의 low가 포함되지 않아야 함.

        rolling_min[i]는 low[i-lookback:i] (현재 봉 제외) 기준이어야 함.
        rolling(10, min_periods=10)은 인덱스 0-8이 NaN (9개), shift(1)로 인덱스 9도 NaN.
        따라서 인덱스 0-9 (10개)가 NaN.
        """
        config = MaxMinConfig(lookback=10)
        result = preprocess(sample_ohlcv, config)

        # lookback=10: rolling(min_periods=10)은 인덱스 9부터 유효, shift(1)로 인덱스 10부터 유효
        # 즉, 인덱스 0-9 (10개)가 NaN
        assert result["rolling_min"].iloc[:10].isna().all()
        assert result["rolling_min"].iloc[10:].notna().all()

        # shift(1) 검증
        idx = 50
        expected_min = sample_ohlcv["low"].iloc[idx - 10 : idx].min()
        actual_min = result["rolling_min"].iloc[idx]
        assert abs(actual_min - expected_min) < 1e-6

    def test_missing_columns(self) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        config = MaxMinConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 수정되지 않음."""
        config = MaxMinConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar는 워밍업 이후 항상 양수."""
        config = MaxMinConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR은 워밍업 이후 항상 양수."""
        config = MaxMinConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["atr"].dropna()
        assert (valid > 0).all()

    def test_returns_log(self, sample_ohlcv: pd.DataFrame) -> None:
        """returns는 로그 수익률."""
        config = MaxMinConfig()
        result = preprocess(sample_ohlcv, config)

        # 로그 수익률 수동 계산과 비교
        expected = np.log(sample_ohlcv["close"] / sample_ohlcv["close"].shift(1))
        pd.testing.assert_series_equal(
            result["returns"].dropna(),
            expected.dropna(),
            check_names=False,
        )

    def test_output_length_matches_input(self, sample_ohlcv: pd.DataFrame) -> None:
        """출력 DataFrame 길이가 입력과 동일."""
        config = MaxMinConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)
