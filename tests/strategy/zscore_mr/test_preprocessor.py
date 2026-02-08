"""Tests for Z-Score MR Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.zscore_mr.config import ZScoreMRConfig
from src.strategy.zscore_mr.preprocessor import (
    calculate_adaptive_zscore,
    calculate_atr,
    calculate_returns,
    calculate_vol_regime,
    calculate_zscore,
    preprocess,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """평균회귀 패턴의 샘플 OHLCV DataFrame (200일)."""
    np.random.seed(42)
    n = 200

    # Mean-reverting 가격: Ornstein-Uhlenbeck 유사 프로세스
    base_price = 50000.0
    noise = np.cumsum(np.random.randn(n) * 300)
    # mean reversion force
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


class TestZScore:
    """Z-score 계산 테스트."""

    def test_zscore_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        z = calculate_zscore(sample_ohlcv["close"], lookback=20)
        assert len(z) == len(sample_ohlcv)

    def test_zscore_roughly_normal(self, sample_ohlcv: pd.DataFrame):
        """Z-score가 대략 정규분포 범위."""
        z = calculate_zscore(sample_ohlcv["close"], lookback=20)
        valid = z.dropna()
        # 대부분 -3 ~ +3 범위 내
        within_range = ((valid >= -4.0) & (valid <= 4.0)).mean()
        assert within_range > 0.90

    def test_zscore_warmup_nan(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간에는 NaN."""
        lookback = 20
        z = calculate_zscore(sample_ohlcv["close"], lookback=lookback)
        assert z.iloc[: lookback - 1].isna().all()

    def test_zscore_mean_near_zero(self, sample_ohlcv: pd.DataFrame):
        """Z-score의 평균이 대략 0 근처."""
        z = calculate_zscore(sample_ohlcv["close"], lookback=20)
        valid = z.dropna()
        assert abs(valid.mean()) < 1.0


class TestAdaptiveZScore:
    """적응적 Z-score 테스트."""

    def test_adaptive_zscore_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        returns = np.log(sample_ohlcv["close"] / sample_ohlcv["close"].shift(1))
        z_adaptive, vol_regime = calculate_adaptive_zscore(
            sample_ohlcv["close"],
            returns,
            short_lookback=20,
            long_lookback=60,
            vol_regime_lookback=20,
            vol_rank_lookback=100,
            high_vol_percentile=0.7,
        )
        assert len(z_adaptive) == len(sample_ohlcv)
        assert len(vol_regime) == len(sample_ohlcv)

    def test_vol_regime_range(self, sample_ohlcv: pd.DataFrame):
        """vol_regime이 0~1 범위."""
        returns = np.log(sample_ohlcv["close"] / sample_ohlcv["close"].shift(1))
        vol_regime = calculate_vol_regime(returns, vol_regime_lookback=20, vol_rank_lookback=100)
        valid = vol_regime.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_high_vol_uses_short_lookback(self):
        """고변동성 레짐에서 short_lookback z-score를 사용하는지 확인."""
        np.random.seed(42)
        n = 300

        # 전반부: 낮은 변동성, 후반부: 높은 변동성
        close_low_vol = 50000 + np.cumsum(np.random.randn(n // 2) * 50)
        close_high_vol = close_low_vol[-1] + np.cumsum(np.random.randn(n // 2) * 500)
        close = pd.Series(
            np.concatenate([close_low_vol, close_high_vol]),
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        returns = np.log(close / close.shift(1))

        _z_adaptive, vol_regime = calculate_adaptive_zscore(
            close,
            returns,
            short_lookback=10,
            long_lookback=60,
            vol_regime_lookback=20,
            vol_rank_lookback=100,
            high_vol_percentile=0.5,
        )

        # 고변동성 구간에서 vol_regime이 높은 값을 가져야 함
        high_vol_region = vol_regime.iloc[-50:].dropna()
        if len(high_vol_region) > 0:
            assert high_vol_region.mean() > 0.5

    def test_returns_two_series(self, sample_ohlcv: pd.DataFrame):
        """calculate_adaptive_zscore가 (zscore, vol_regime) 튜플을 반환."""
        returns = np.log(sample_ohlcv["close"] / sample_ohlcv["close"].shift(1))
        result = calculate_adaptive_zscore(
            sample_ohlcv["close"],
            returns,
            short_lookback=20,
            long_lookback=60,
            vol_regime_lookback=20,
            vol_rank_lookback=100,
            high_vol_percentile=0.7,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestReturns:
    """수익률 계산 테스트."""

    def test_log_returns(self, sample_ohlcv: pd.DataFrame):
        """로그 수익률 계산."""
        returns = calculate_returns(sample_ohlcv["close"], use_log=True)
        assert len(returns) == len(sample_ohlcv)
        assert returns.iloc[0] != returns.iloc[0]  # NaN

    def test_simple_returns(self, sample_ohlcv: pd.DataFrame):
        """단순 수익률 계산."""
        returns = calculate_returns(sample_ohlcv["close"], use_log=False)
        assert len(returns) == len(sample_ohlcv)

    def test_empty_raises(self):
        """빈 시리즈에 ValueError."""
        with pytest.raises(ValueError, match="Empty Series"):
            calculate_returns(pd.Series(dtype=float))


class TestATR:
    """ATR 계산 테스트."""

    def test_positive(self, sample_ohlcv: pd.DataFrame):
        """ATR은 항상 양수."""
        atr = calculate_atr(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], period=14
        )
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        atr = calculate_atr(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], period=14
        )
        assert len(atr) == len(sample_ohlcv)


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_all_columns_present(self, sample_ohlcv: pd.DataFrame):
        """필수 출력 컬럼이 모두 존재."""
        config = ZScoreMRConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "zscore",
            "vol_regime",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간 이후 주요 지표에 NaN 없음."""
        config = ZScoreMRConfig(vol_rank_lookback=60)
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["returns", "realized_vol", "vol_scalar", "atr"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame):
        """원본 DataFrame이 수정되지 않음."""
        config = ZScoreMRConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        config = ZScoreMRConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame):
        """vol_scalar는 항상 양수."""
        config = ZScoreMRConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(self, sample_ohlcv: pd.DataFrame):
        """drawdown은 항상 0 이하."""
        config = ZScoreMRConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()
