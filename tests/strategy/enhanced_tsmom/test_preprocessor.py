"""Tests for Enhanced VW-TSMOM Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.enhanced_tsmom.config import EnhancedTSMOMConfig
from src.strategy.enhanced_tsmom.preprocessor import (
    calculate_enhanced_vw_momentum,
    preprocess,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """트렌딩 패턴의 샘플 OHLCV DataFrame (200일)."""
    np.random.seed(42)
    n = 200

    # 상승 트렌드 패턴
    base_price = 50000.0
    trend = np.linspace(0, 5000, n)
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + trend + noise

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


class TestEnhancedVWMomentum:
    """볼륨 비율 정규화 모멘텀 테스트."""

    def test_volume_normalization(self, sample_ohlcv: pd.DataFrame):
        """볼륨 비율 정규화가 올바르게 적용되는지 확인."""
        close = sample_ohlcv["close"]
        volume = sample_ohlcv["volume"]
        returns = np.log(close / close.shift(1))

        momentum = calculate_enhanced_vw_momentum(
            returns, volume, lookback=30, volume_lookback=20, volume_clip_max=5.0
        )

        assert isinstance(momentum, pd.Series)
        assert momentum.name == "evw_momentum"
        assert len(momentum) == len(sample_ohlcv)

        # 워밍업 기간 이후 유효 데이터 존재
        valid = momentum.dropna()
        assert len(valid) > 0

    def test_clip_max_effect(self, sample_ohlcv: pd.DataFrame):
        """clip_max가 극단적 볼륨 비율을 제한하는지 확인."""
        close = sample_ohlcv["close"]
        volume = sample_ohlcv["volume"].copy()
        returns = np.log(close / close.shift(1))

        # 일부 거래량을 극단적으로 증가
        volume.iloc[100:105] = volume.iloc[100:105] * 100

        # clip_max가 낮을 때 vs 높을 때
        momentum_low_clip = calculate_enhanced_vw_momentum(
            returns, volume, lookback=30, volume_lookback=20, volume_clip_max=2.0
        )
        momentum_high_clip = calculate_enhanced_vw_momentum(
            returns, volume, lookback=30, volume_lookback=20, volume_clip_max=20.0
        )

        # 극단적 볼륨이 있는 구간 이후에서 비교
        valid_low = momentum_low_clip.dropna()
        valid_high = momentum_high_clip.dropna()

        # 낮은 clip에서는 극단값 영향이 제한되어 abs 합이 작아야 함
        assert valid_low.abs().sum() <= valid_high.abs().sum()

    def test_output_shape(self, sample_ohlcv: pd.DataFrame):
        """출력 길이가 입력과 동일."""
        close = sample_ohlcv["close"]
        volume = sample_ohlcv["volume"]
        returns = np.log(close / close.shift(1))

        momentum = calculate_enhanced_vw_momentum(returns, volume, lookback=30, volume_lookback=20)
        assert len(momentum) == len(sample_ohlcv)


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_all_columns_present(self, sample_ohlcv: pd.DataFrame):
        """필수 출력 컬럼이 모두 존재."""
        config = EnhancedTSMOMConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "evw_momentum",
            "vol_scalar",
            "returns",
            "realized_vol",
            "drawdown",
            "atr",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간 이후 NaN 없음."""
        config = EnhancedTSMOMConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["evw_momentum", "vol_scalar", "returns", "realized_vol"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame):
        """원본 DataFrame이 수정되지 않음."""
        config = EnhancedTSMOMConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        config = EnhancedTSMOMConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame):
        """vol_scalar는 항상 양수."""
        config = EnhancedTSMOMConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(self, sample_ohlcv: pd.DataFrame):
        """drawdown은 항상 0 이하."""
        config = EnhancedTSMOMConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame):
        """ATR은 항상 양수."""
        config = EnhancedTSMOMConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["atr"].dropna()
        assert (valid > 0).all()

    def test_numeric_conversion(self):
        """Decimal 타입 컬럼이 pd.to_numeric으로 변환됨."""
        from decimal import Decimal

        n = 50
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "open": [Decimal(50000)] * n,
                "high": [Decimal(51000)] * n,
                "low": [Decimal(49000)] * n,
                "close": [Decimal(str(50000 + i * 10)) for i in range(n)],
                "volume": [Decimal(1000000)] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        config = EnhancedTSMOMConfig()
        result = preprocess(df, config)
        # Decimal -> float 변환 확인
        assert result["close"].dtype == np.float64
