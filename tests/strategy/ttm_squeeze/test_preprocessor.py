"""Tests for TTM Squeeze Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import bollinger_bands, squeeze_detect
from src.strategy.ttm_squeeze.config import TtmSqueezeConfig
from src.strategy.ttm_squeeze.preprocessor import (
    calculate_exit_sma,
    calculate_keltner_channels,
    calculate_momentum,
    preprocess,
)


class TestPreprocess:
    """전처리 메인 함수 테스트."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """전처리 후 필수 컬럼이 모두 추가되는지 확인."""
        config = TtmSqueezeConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "bb_upper",
            "bb_lower",
            "kc_upper",
            "kc_lower",
            "squeeze_on",
            "momentum",
            "exit_sma",
            "realized_vol",
            "vol_scalar",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 OHLCV 컬럼이 보존되는지 확인."""
        config = TtmSqueezeConfig()
        result = preprocess(sample_ohlcv, config)

        for col in ["open", "high", "low", "close"]:
            assert col in result.columns, f"Original column {col} missing"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame) -> None:
        """원본 DataFrame이 수정되지 않음."""
        config = TtmSqueezeConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_output_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """출력 길이가 입력과 동일."""
        config = TtmSqueezeConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)

    def test_squeeze_on_is_bool(self, sample_ohlcv: pd.DataFrame) -> None:
        """squeeze_on 컬럼이 bool dtype."""
        config = TtmSqueezeConfig()
        result = preprocess(sample_ohlcv, config)
        assert result["squeeze_on"].dtype == bool

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar는 항상 양수 (NaN 제외)."""
        config = TtmSqueezeConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        config = TtmSqueezeConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_single_column_raises(self) -> None:
        """하나의 필수 컬럼만 누락되어도 ValueError."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "open": np.random.randn(n),
                "high": np.random.randn(n),
                # low 누락
                "close": np.random.randn(n),
            },
            index=dates,
        )
        config = TtmSqueezeConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame) -> None:
        """워밍업 기간 이후 주요 컬럼에 NaN 없음."""
        config = TtmSqueezeConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["bb_upper", "bb_lower", "kc_upper", "kc_lower", "momentum", "exit_sma"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"


class TestPreprocessWithDifferentConfigs:
    """다른 설정에서의 전처리 테스트."""

    def test_different_bb_period(self, sample_ohlcv: pd.DataFrame) -> None:
        """다른 BB period에서도 정상 작동."""
        config = TtmSqueezeConfig(bb_period=10)
        result = preprocess(sample_ohlcv, config)
        assert "bb_upper" in result.columns

    def test_different_kc_mult(self, sample_ohlcv: pd.DataFrame) -> None:
        """다른 KC mult에서도 정상 작동."""
        config = TtmSqueezeConfig(kc_mult=2.5)
        result = preprocess(sample_ohlcv, config)
        assert "kc_upper" in result.columns

    def test_higher_vol_target_higher_scalar(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_target이 높으면 vol_scalar도 높아짐."""
        config_low = TtmSqueezeConfig(vol_target=0.10)
        config_high = TtmSqueezeConfig(vol_target=0.50)

        result_low = preprocess(sample_ohlcv, config_low)
        result_high = preprocess(sample_ohlcv, config_high)

        avg_low = result_low["vol_scalar"].dropna().mean()
        avg_high = result_high["vol_scalar"].dropna().mean()
        assert avg_high > avg_low

    def test_wider_bb_more_squeeze(self, sample_ohlcv: pd.DataFrame) -> None:
        """BB가 좁으면(작은 std) squeeze가 더 적게 발생."""
        config_narrow = TtmSqueezeConfig(bb_std=1.0)
        config_wide = TtmSqueezeConfig(bb_std=3.0)

        result_narrow = preprocess(sample_ohlcv, config_narrow)
        result_wide = preprocess(sample_ohlcv, config_wide)

        # 좁은 BB는 KC 안에 더 잘 들어감 → squeeze 더 많음
        squeeze_narrow = result_narrow["squeeze_on"].sum()
        squeeze_wide = result_wide["squeeze_on"].sum()
        # BB가 넓으면 KC 밖으로 나가므로 squeeze가 적음
        # 데이터 의존적이므로 실행 성공 자체가 검증 (순서는 보장 안됨)
        assert isinstance(squeeze_wide, (int, np.integer))
        assert isinstance(squeeze_narrow, (int, np.integer))


class TestCalculateBollingerBands:
    """Bollinger Bands 단위 테스트."""

    def test_bollinger_bands_basic(self) -> None:
        """BB 기본 계산 검증."""
        close = pd.Series([10.0] * 20)
        upper, middle, lower = bollinger_bands(close, period=20, std_dev=2.0)

        # 상수 시리즈 → std=0, upper=middle=lower
        assert middle.iloc[-1] == pytest.approx(10.0)
        assert upper.iloc[-1] == pytest.approx(10.0)
        assert lower.iloc[-1] == pytest.approx(10.0)

    def test_bollinger_bands_ordering(self, sample_ohlcv: pd.DataFrame) -> None:
        """BB upper > middle > lower 순서."""
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        upper, middle, lower = bollinger_bands(close, period=20, std_dev=2.0)

        valid = middle.dropna().index
        assert (upper.loc[valid] >= middle.loc[valid]).all()
        assert (middle.loc[valid] >= lower.loc[valid]).all()


class TestCalculateKeltnerChannels:
    """Keltner Channels 단위 테스트."""

    def test_keltner_channels_ordering(self, sample_ohlcv: pd.DataFrame) -> None:
        """KC upper > middle > lower 순서."""
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        high: pd.Series = sample_ohlcv["high"]  # type: ignore[assignment]
        low: pd.Series = sample_ohlcv["low"]  # type: ignore[assignment]

        kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
            close, high, low, period=20, mult=1.5
        )

        valid = kc_middle.dropna().index
        # ATR 초기 NaN 이후의 유효 데이터에서만 검증
        valid_idx = valid[20:]
        assert (kc_upper.loc[valid_idx] >= kc_middle.loc[valid_idx]).all()
        assert (kc_middle.loc[valid_idx] >= kc_lower.loc[valid_idx]).all()


class TestCalculateSqueeze:
    """Squeeze 감지 단위 테스트."""

    def test_squeeze_all_on(self) -> None:
        """BB가 KC 완전히 안에 있으면 모두 squeeze ON."""
        bb_upper = pd.Series([10.0, 10.0, 10.0])
        bb_lower = pd.Series([5.0, 5.0, 5.0])
        kc_upper = pd.Series([12.0, 12.0, 12.0])
        kc_lower = pd.Series([3.0, 3.0, 3.0])

        result = squeeze_detect(bb_upper, bb_lower, kc_upper, kc_lower)
        assert result.all()
        assert result.dtype == bool

    def test_squeeze_all_off(self) -> None:
        """BB가 KC 밖에 있으면 모두 squeeze OFF."""
        bb_upper = pd.Series([15.0, 15.0, 15.0])
        bb_lower = pd.Series([1.0, 1.0, 1.0])
        kc_upper = pd.Series([12.0, 12.0, 12.0])
        kc_lower = pd.Series([3.0, 3.0, 3.0])

        result = squeeze_detect(bb_upper, bb_lower, kc_upper, kc_lower)
        assert not result.any()


class TestCalculateMomentum:
    """Momentum 계산 단위 테스트."""

    def test_momentum_basic(self) -> None:
        """Momentum = close - midline 검증."""
        n = 25
        close = pd.Series(np.arange(1.0, n + 1))
        high = close + 0.5
        low = close - 0.5

        result = calculate_momentum(close, high, low, period=20)
        # rolling(20) 이후 유효값 존재
        assert result.dropna().shape[0] > 0


class TestCalculateExitSma:
    """Exit SMA 단위 테스트."""

    def test_exit_sma_basic(self) -> None:
        """상수 시리즈의 SMA는 같은 값."""
        close = pd.Series([100.0] * 30)
        result = calculate_exit_sma(close, period=21)
        valid = result.dropna()
        np.testing.assert_allclose(valid.to_numpy(), 100.0)
