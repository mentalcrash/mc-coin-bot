"""Tests for GK Breakout Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import donchian_channel, garman_klass_volatility
from src.strategy.gk_breakout.config import GKBreakoutConfig
from src.strategy.gk_breakout.preprocessor import (
    calculate_vol_ratio,
    preprocess,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """추세 + 횡보 패턴의 샘플 OHLCV DataFrame (200일)."""
    np.random.seed(42)
    n = 200

    base_price = 50000.0
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + noise - noise.mean()
    close = np.maximum(close, base_price * 0.8)

    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestGKVariance:
    """Garman-Klass variance 계산 테스트."""

    def test_gk_var_positive_for_most(self, sample_ohlcv: pd.DataFrame):
        """GK variance는 대부분 양수 (음수도 가능하지만 대부분 양수)."""
        gk_var = garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        positive_ratio = (gk_var > 0).mean()
        assert positive_ratio > 0.5  # 대부분 양수

    def test_gk_var_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        gk_var = garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        assert len(gk_var) == len(sample_ohlcv)

    def test_gk_var_name(self, sample_ohlcv: pd.DataFrame):
        """시리즈 이름이 'gk_var'."""
        gk_var = garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        assert gk_var.name == "gk_var"

    def test_gk_var_no_nan(self, sample_ohlcv: pd.DataFrame):
        """유효한 OHLC 데이터에서 NaN 없음."""
        gk_var = garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        assert gk_var.notna().all()


class TestVolRatio:
    """Vol ratio 계산 테스트."""

    def test_ratio_around_one_for_stationary(self):
        """정상적인 변동성에서 ratio는 대략 1.0 근처."""
        np.random.seed(42)
        n = 200
        # 일정한 변동성을 가진 가격 시리즈
        prices = 50000 + np.cumsum(np.random.randn(n) * 100)
        high = prices + np.abs(np.random.randn(n) * 50)
        low = prices - np.abs(np.random.randn(n) * 50)
        close = pd.Series(prices)
        open_ = close + np.random.randn(n) * 30

        gk_var = garman_klass_volatility(open_, pd.Series(high), pd.Series(low), close)
        vol_ratio = calculate_vol_ratio(gk_var, lookback=20)

        valid = vol_ratio.dropna()
        # 정상적인 변동성에서 평균 ratio는 1.0 근처 (0.5~1.5 범위 내)
        assert 0.5 < valid.mean() < 1.5

    def test_vol_ratio_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        gk_var = garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        vol_ratio = calculate_vol_ratio(gk_var, lookback=20)
        assert len(vol_ratio) == len(sample_ohlcv)

    def test_vol_ratio_name(self, sample_ohlcv: pd.DataFrame):
        """시리즈 이름이 'vol_ratio'."""
        gk_var = garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        vol_ratio = calculate_vol_ratio(gk_var, lookback=20)
        assert vol_ratio.name == "vol_ratio"

    def test_vol_ratio_warmup_nan(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간에는 NaN."""
        gk_var = garman_klass_volatility(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        lookback = 20
        vol_ratio = calculate_vol_ratio(gk_var, lookback=lookback)
        # long window = lookback * 2 = 40, 이전은 NaN
        assert vol_ratio.iloc[: lookback * 2 - 1].isna().all()


class TestDonchianChannel:
    """Donchian Channel 계산 테스트."""

    def test_upper_gte_lower(self, sample_ohlcv: pd.DataFrame):
        """upper >= lower 항상 성립."""
        upper, _middle, lower = donchian_channel(
            sample_ohlcv["high"], sample_ohlcv["low"], period=20
        )
        valid_mask = upper.notna() & lower.notna()
        assert (upper[valid_mask] >= lower[valid_mask]).all()

    def test_channel_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        upper, _middle, lower = donchian_channel(
            sample_ohlcv["high"], sample_ohlcv["low"], period=20
        )
        assert len(upper) == len(sample_ohlcv)
        assert len(lower) == len(sample_ohlcv)

    def test_channel_names(self, sample_ohlcv: pd.DataFrame):
        """시리즈 이름 확인."""
        upper, _middle, lower = donchian_channel(
            sample_ohlcv["high"], sample_ohlcv["low"], period=20
        )
        assert upper.name == "dc_upper"
        assert lower.name == "dc_lower"

    def test_channel_warmup_nan(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간에는 NaN."""
        lookback = 20
        upper, _middle, _lower = donchian_channel(
            sample_ohlcv["high"], sample_ohlcv["low"], period=lookback
        )
        assert upper.iloc[: lookback - 1].isna().all()
        assert upper.iloc[lookback - 1 :].notna().all()


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_all_columns_present(self, sample_ohlcv: pd.DataFrame):
        """필수 출력 컬럼이 모두 존재."""
        config = GKBreakoutConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "gk_var",
            "vol_ratio",
            "dc_upper",
            "dc_lower",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간 이후 NaN 없음."""
        config = GKBreakoutConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["vol_ratio", "dc_upper", "dc_lower", "vol_scalar", "atr"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame):
        """원본 DataFrame이 수정되지 않음."""
        config = GKBreakoutConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        config = GKBreakoutConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame):
        """vol_scalar는 항상 양수."""
        config = GKBreakoutConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(self, sample_ohlcv: pd.DataFrame):
        """drawdown은 항상 0 이하."""
        config = GKBreakoutConfig()
        result = preprocess(sample_ohlcv, config)
        assert (result["drawdown"] <= 0).all()
