"""Tests for Vol-Regime Adaptive preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_regime.config import VolRegimeConfig
from src.strategy.vol_regime.preprocessor import (
    calculate_regime_strength,
    calculate_vol_regime,
    preprocess,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (350일).

    vol_rank_lookback=252 기본값에서 non-NaN 값을 보장하기 위해
    최소 253일 이상의 데이터가 필요합니다. 350일로 설정.
    """
    np.random.seed(42)
    n = 350

    # 상승 추세 + 노이즈
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )

    return df


class TestVolRegime:
    """변동성 regime 계산 테스트."""

    def test_vol_pct_range(self, sample_ohlcv: pd.DataFrame):
        """vol_pct는 0~1 범위 내 값."""
        config = VolRegimeConfig()
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        returns = np.log(close / close.shift(1))

        vol_pct = calculate_vol_regime(
            returns,
            vol_lookback=config.vol_lookback,
            vol_rank_lookback=config.vol_rank_lookback,
            annualization_factor=config.annualization_factor,
        )

        # NaN이 아닌 값들은 0~1 범위
        valid = vol_pct.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_vol_pct_name(self, sample_ohlcv: pd.DataFrame):
        """vol_pct 시리즈 이름 확인."""
        config = VolRegimeConfig()
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        returns = np.log(close / close.shift(1))

        vol_pct = calculate_vol_regime(
            returns,
            vol_lookback=config.vol_lookback,
            vol_rank_lookback=config.vol_rank_lookback,
            annualization_factor=config.annualization_factor,
        )

        assert vol_pct.name == "vol_regime"


class TestRegimeStrength:
    """Regime별 강도 계산 테스트."""

    def test_output_has_values(self, sample_ohlcv: pd.DataFrame):
        """regime_strength에 유효한 값이 존재."""
        config = VolRegimeConfig()
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        volume: pd.Series = sample_ohlcv["volume"]  # type: ignore[assignment]
        returns = np.log(close / close.shift(1))

        vol_pct = calculate_vol_regime(
            returns,
            vol_lookback=config.vol_lookback,
            vol_rank_lookback=config.vol_rank_lookback,
            annualization_factor=config.annualization_factor,
        )

        strength = calculate_regime_strength(returns, volume, close, vol_pct, config)

        valid = strength.dropna()
        assert len(valid) > 0
        assert strength.name == "regime_strength"

    def test_different_regimes_produce_different_strengths(self, sample_ohlcv: pd.DataFrame):
        """서로 다른 regime 파라미터는 다른 강도를 생성."""
        close: pd.Series = sample_ohlcv["close"]  # type: ignore[assignment]
        volume: pd.Series = sample_ohlcv["volume"]  # type: ignore[assignment]
        returns = np.log(close / close.shift(1))

        # 두 가지 다른 설정
        config_a = VolRegimeConfig(
            high_vol_target=0.10,
            normal_vol_target=0.20,
            low_vol_target=0.30,
        )
        config_b = VolRegimeConfig(
            high_vol_target=0.30,
            normal_vol_target=0.50,
            low_vol_target=0.80,
        )

        vol_pct = calculate_vol_regime(
            returns,
            vol_lookback=20,
            vol_rank_lookback=252,
            annualization_factor=365.0,
        )

        strength_a = calculate_regime_strength(returns, volume, close, vol_pct, config_a)
        strength_b = calculate_regime_strength(returns, volume, close, vol_pct, config_b)

        # 서로 다른 설정이면 다른 결과
        valid_a = strength_a.dropna()
        valid_b = strength_b.dropna()
        common_idx = valid_a.index.intersection(valid_b.index)
        assert len(common_idx) > 0
        # vol_target이 다르므로 절대값 평균이 다를 것
        assert not np.allclose(
            strength_a.loc[common_idx].values,
            strength_b.loc[common_idx].values,
        )


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_output_columns(self, sample_ohlcv: pd.DataFrame):
        """preprocess 출력에 필수 컬럼 존재."""
        config = VolRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_regime",
            "regime_strength",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame):
        """워밍업 기간 이후 NaN 없음."""
        config = VolRegimeConfig()
        result = preprocess(sample_ohlcv, config)
        warmup = config.warmup_periods()

        # 주요 지표 컬럼에서 warmup 이후 NaN 확인
        check_cols = ["returns", "realized_vol", "atr", "drawdown"]
        for col in check_cols:
            col_data = result[col].iloc[warmup:]
            assert not col_data.isna().any(), f"NaN found in {col} after warmup"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame):
        """원본 DataFrame이 수정되지 않음."""
        config = VolRegimeConfig()
        original_cols = list(sample_ohlcv.columns)
        original_values = sample_ohlcv["close"].values.copy()

        _ = preprocess(sample_ohlcv, config)

        assert list(sample_ohlcv.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv["close"].values, original_values)

    def test_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        config = VolRegimeConfig()
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_regime_is_percentile(self, sample_ohlcv: pd.DataFrame):
        """vol_regime 컬럼이 0~1 범위의 percentile rank."""
        config = VolRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        vol_regime = result["vol_regime"].dropna()
        assert len(vol_regime) > 0
        assert vol_regime.min() >= 0.0
        assert vol_regime.max() <= 1.0

    def test_drawdown_is_non_positive(self, sample_ohlcv: pd.DataFrame):
        """drawdown은 항상 0 이하."""
        config = VolRegimeConfig()
        result = preprocess(sample_ohlcv, config)

        drawdown = result["drawdown"].dropna()
        assert (drawdown <= 0).all()
