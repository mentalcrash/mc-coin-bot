"""Tests for src.market.indicators.

각 지표의 기본 동작, edge case, 기존 preprocessor 대비 parity를 검증합니다.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.market.indicators import (
    adx,
    atr,
    bb_position,
    bollinger_bands,
    cci,
    chaikin_money_flow,
    donchian_channel,
    drawdown,
    efficiency_ratio,
    ema,
    ema_cross,
    fractal_dimension,
    garman_klass_volatility,
    hurst_exponent,
    kama,
    keltner_channels,
    log_returns,
    macd,
    mean_reversion_score,
    momentum,
    obv,
    parkinson_volatility,
    price_acceleration,
    realized_volatility,
    roc,
    rolling_return,
    rolling_zscore,
    rsi,
    rsi_divergence,
    simple_returns,
    sma,
    sma_cross,
    squeeze_detect,
    stochastic,
    trend_strength,
    vol_percentile_rank,
    vol_regime,
    volatility_of_volatility,
    volatility_scalar,
    volume_macd,
    volume_weighted_returns,
    williams_r,
    yang_zhang_volatility,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """100-bar OHLCV 테스트 데이터."""
    rng = np.random.default_rng(42)
    n = 100
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def close_series(ohlcv_df: pd.DataFrame) -> pd.Series:
    result: pd.Series = ohlcv_df["close"]  # type: ignore[assignment]
    return result


@pytest.fixture
def high_series(ohlcv_df: pd.DataFrame) -> pd.Series:
    result: pd.Series = ohlcv_df["high"]  # type: ignore[assignment]
    return result


@pytest.fixture
def low_series(ohlcv_df: pd.DataFrame) -> pd.Series:
    result: pd.Series = ohlcv_df["low"]  # type: ignore[assignment]
    return result


@pytest.fixture
def volume_series(ohlcv_df: pd.DataFrame) -> pd.Series:
    result: pd.Series = ohlcv_df["volume"]  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# log_returns / simple_returns
# ---------------------------------------------------------------------------


class TestReturns:
    def test_log_returns_basic(self, close_series: pd.Series) -> None:
        result = log_returns(close_series)
        assert len(result) == len(close_series)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(np.log(close_series.iloc[1] / close_series.iloc[0]))

    def test_simple_returns_basic(self, close_series: pd.Series) -> None:
        result = simple_returns(close_series)
        assert len(result) == len(close_series)
        assert pd.isna(result.iloc[0])
        expected = (close_series.iloc[1] - close_series.iloc[0]) / close_series.iloc[0]
        assert result.iloc[1] == pytest.approx(expected)

    def test_log_returns_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            log_returns(pd.Series([], dtype=float))

    def test_simple_returns_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            simple_returns(pd.Series([], dtype=float))

    def test_log_returns_name(self, close_series: pd.Series) -> None:
        assert log_returns(close_series).name == "returns"

    def test_single_value(self) -> None:
        s = pd.Series([100.0])
        result = log_returns(s)
        assert len(result) == 1
        assert pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# realized_volatility / volatility_scalar
# ---------------------------------------------------------------------------


class TestVolatility:
    def test_realized_volatility_basic(self, close_series: pd.Series) -> None:
        rets = log_returns(close_series)
        result = realized_volatility(rets, window=30)
        assert len(result) == len(close_series)
        # First 30 values should be NaN (min_periods=window)
        assert pd.isna(result.iloc[29])
        assert not pd.isna(result.iloc[30])

    def test_realized_volatility_min_periods(self, close_series: pd.Series) -> None:
        rets = log_returns(close_series)
        result = realized_volatility(rets, window=30, min_periods=10)
        # Should have values earlier with min_periods=10
        assert not pd.isna(result.iloc[10])

    def test_realized_volatility_annualization(self, close_series: pd.Series) -> None:
        rets = log_returns(close_series)
        vol_365 = realized_volatility(rets, window=30, annualization_factor=365.0)
        vol_252 = realized_volatility(rets, window=30, annualization_factor=252.0)
        # 365-day annualized vol should be higher
        valid = ~pd.isna(vol_365) & ~pd.isna(vol_252)
        assert (vol_365[valid] > vol_252[valid]).all()

    def test_volatility_scalar_basic(self) -> None:
        vol = pd.Series([0.2, 0.4, 0.6, 0.8])
        result = volatility_scalar(vol, vol_target=0.4)
        expected = pd.Series([2.0, 1.0, 0.4 / 0.6, 0.5])
        assert_series_equal(result, expected)

    def test_volatility_scalar_min_clamp(self) -> None:
        vol = pd.Series([0.01, 0.02, 0.05, 0.1])
        result = volatility_scalar(vol, vol_target=0.4, min_volatility=0.05)
        # First two should be clamped to 0.05
        assert result.iloc[0] == pytest.approx(0.4 / 0.05)
        assert result.iloc[1] == pytest.approx(0.4 / 0.05)


# ---------------------------------------------------------------------------
# ATR / ADX
# ---------------------------------------------------------------------------


class TestATR:
    def test_atr_basic(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        result = atr(high_series, low_series, close_series, period=14)
        assert len(result) == len(close_series)
        assert result.name == "atr"
        # ATR must be positive where computed
        valid = result.dropna()
        assert (valid > 0).all()

    def test_atr_nan_warmup(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        result = atr(high_series, low_series, close_series, period=14)
        # First 13 values should be NaN (min_periods=14)
        assert pd.isna(result.iloc[12])


class TestADX:
    def test_adx_range(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        result = adx(high_series, low_series, close_series, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_adx_name(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        assert adx(high_series, low_series, close_series).name == "adx"


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_drawdown_basic(self) -> None:
        close = pd.Series([100, 110, 105, 120, 90])
        result = drawdown(close)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(0.0)  # new high
        assert result.iloc[2] == pytest.approx((105 - 110) / 110)
        assert result.iloc[3] == pytest.approx(0.0)  # new high
        assert result.iloc[4] == pytest.approx((90 - 120) / 120)

    def test_drawdown_always_leq_zero(self, close_series: pd.Series) -> None:
        result = drawdown(close_series)
        assert (result <= 0).all()

    def test_drawdown_name(self, close_series: pd.Series) -> None:
        assert drawdown(close_series).name == "drawdown"


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_range(self, close_series: pd.Series) -> None:
        result = rsi(close_series, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_overbought(self) -> None:
        """monotonically rising → RSI near 100."""
        close = pd.Series(np.arange(1, 51, dtype=float))
        result = rsi(close, period=14)
        assert result.iloc[-1] > 90

    def test_rsi_oversold(self) -> None:
        """monotonically falling → RSI near 0."""
        close = pd.Series(np.arange(50, 0, -1, dtype=float))
        result = rsi(close, period=14)
        assert result.iloc[-1] < 10

    def test_rsi_name(self, close_series: pd.Series) -> None:
        assert rsi(close_series).name == "rsi"


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_bollinger_basic(self, close_series: pd.Series) -> None:
        upper, middle, lower = bollinger_bands(close_series, period=20, std_dev=2.0)
        valid_idx = ~pd.isna(upper)
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_bollinger_names(self, close_series: pd.Series) -> None:
        upper, middle, lower = bollinger_bands(close_series)
        assert upper.name == "bb_upper"
        assert middle.name == "bb_middle"
        assert lower.name == "bb_lower"


# ---------------------------------------------------------------------------
# SMA / EMA
# ---------------------------------------------------------------------------


class TestMovingAverages:
    def test_sma_basic(self, close_series: pd.Series) -> None:
        result = sma(close_series, period=20)
        assert pd.isna(result.iloc[18])
        assert not pd.isna(result.iloc[19])

    def test_ema_basic(self, close_series: pd.Series) -> None:
        result = ema(close_series, span=20)
        # EMA starts computing from first value
        assert not pd.isna(result.iloc[0])

    def test_sma_value(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, period=3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_macd_basic(self, close_series: pd.Series) -> None:
        line, signal_line, histogram = macd(close_series)
        assert len(line) == len(close_series)
        assert len(signal_line) == len(close_series)
        assert len(histogram) == len(close_series)

    def test_macd_histogram_equals_diff(self, close_series: pd.Series) -> None:
        line, signal_line, histogram = macd(close_series)
        diff: pd.Series = line - signal_line  # type: ignore[assignment]
        assert_series_equal(histogram, diff, check_names=False)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


class TestStochastic:
    def test_stochastic_range(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        k, _d = stochastic(high_series, low_series, close_series)
        valid_k = k.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_stochastic_d_is_smoothed_k(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        k, d = stochastic(high_series, low_series, close_series, d_period=3)
        expected_d = k.rolling(3).mean()
        valid = ~pd.isna(d) & ~pd.isna(expected_d)
        assert_series_equal(d[valid], expected_d[valid], check_names=False)


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------


class TestDonchianChannel:
    def test_donchian_basic(self, high_series: pd.Series, low_series: pd.Series) -> None:
        upper, _middle, lower = donchian_channel(high_series, low_series, period=20)
        valid = ~pd.isna(upper)
        assert (upper[valid] >= lower[valid]).all()

    def test_donchian_names(self, high_series: pd.Series, low_series: pd.Series) -> None:
        upper, middle, lower = donchian_channel(high_series, low_series, period=20)
        assert upper.name == "dc_upper"
        assert middle.name == "dc_middle"
        assert lower.name == "dc_lower"


# ---------------------------------------------------------------------------
# Keltner Channels
# ---------------------------------------------------------------------------


class TestKeltnerChannels:
    def test_keltner_basic(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        upper, _middle, lower = keltner_channels(high_series, low_series, close_series)
        valid = ~pd.isna(upper) & ~pd.isna(lower)
        assert (upper[valid] >= lower[valid]).all()

    def test_keltner_names(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        upper, middle, lower = keltner_channels(high_series, low_series, close_series)
        assert upper.name == "kc_upper"
        assert middle.name == "kc_middle"
        assert lower.name == "kc_lower"


# ---------------------------------------------------------------------------
# CCI / Williams %R / ROC / OBV
# ---------------------------------------------------------------------------


class TestCCI:
    def test_cci_basic(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        result = cci(high_series, low_series, close_series, period=20)
        assert len(result) == len(close_series)
        valid = result.dropna()
        assert len(valid) > 0


class TestWilliamsR:
    def test_range(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        result = williams_r(high_series, low_series, close_series, period=14)
        valid = result.dropna()
        assert (valid >= -100).all()
        assert (valid <= 0).all()


class TestROC:
    def test_roc_basic(self, close_series: pd.Series) -> None:
        result = roc(close_series, period=10)
        assert len(result) == len(close_series)
        assert pd.isna(result.iloc[0])

    def test_roc_value(self) -> None:
        s = pd.Series([100.0, 110.0, 120.0])
        result = roc(s, period=1)
        assert result.iloc[1] == pytest.approx(0.1)
        assert result.iloc[2] == pytest.approx(120 / 110 - 1)


class TestOBV:
    def test_obv_basic(self, close_series: pd.Series, volume_series: pd.Series) -> None:
        result = obv(close_series, volume_series)
        assert len(result) == len(close_series)


# ---------------------------------------------------------------------------
# Volume-Weighted Returns
# ---------------------------------------------------------------------------


class TestVolumeWeightedReturns:
    def test_basic(self, close_series: pd.Series, volume_series: pd.Series) -> None:
        rets = log_returns(close_series)
        result = volume_weighted_returns(rets, volume_series, window=30)
        assert len(result) == len(close_series)
        # First 30 values NaN (min_periods=window)
        assert pd.isna(result.iloc[28])
        assert not pd.isna(result.iloc[30])

    def test_min_periods(self, close_series: pd.Series, volume_series: pd.Series) -> None:
        rets = log_returns(close_series)
        result = volume_weighted_returns(rets, volume_series, window=30, min_periods=10)
        assert not pd.isna(result.iloc[10])


# ---------------------------------------------------------------------------
# Parity tests — 기존 preprocessor 함수와 수치 동일성 검증
# ---------------------------------------------------------------------------


class TestParityWithTSMOM:
    """src.market.indicators vs src.strategy.tsmom.preprocessor 수치 동일성."""

    def test_log_returns_parity(self, close_series: pd.Series) -> None:
        from src.strategy.tsmom.preprocessor import calculate_returns

        expected = calculate_returns(close_series, use_log=True)
        actual = log_returns(close_series)
        assert_series_equal(actual, expected, check_names=False)

    def test_simple_returns_parity(self, close_series: pd.Series) -> None:
        from src.strategy.tsmom.preprocessor import calculate_returns

        expected = calculate_returns(close_series, use_log=False)
        actual = simple_returns(close_series)
        assert_series_equal(actual, expected, check_names=False)

    def test_realized_volatility_parity(self, close_series: pd.Series) -> None:
        from src.strategy.tsmom.preprocessor import (
            calculate_realized_volatility,
            calculate_returns,
        )

        rets = calculate_returns(close_series, use_log=True)
        expected = calculate_realized_volatility(rets, window=30)
        actual = realized_volatility(rets, window=30)
        assert_series_equal(actual, expected, check_names=False)

    def test_volatility_scalar_parity(self) -> None:
        from src.strategy.tsmom.preprocessor import calculate_volatility_scalar

        vol = pd.Series([0.1, 0.2, 0.3, 0.5, 0.8])
        expected = calculate_volatility_scalar(vol, vol_target=0.4)
        actual = volatility_scalar(vol, vol_target=0.4)
        assert_series_equal(actual, expected, check_names=False)

    def test_atr_parity(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        from src.strategy.tsmom.preprocessor import calculate_atr

        expected = calculate_atr(high_series, low_series, close_series, period=14)
        actual = atr(high_series, low_series, close_series, period=14)
        assert_series_equal(actual, expected, check_names=False)

    def test_adx_parity(
        self, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series
    ) -> None:
        from src.strategy.tsmom.preprocessor import calculate_adx

        expected = calculate_adx(high_series, low_series, close_series, period=14)
        actual = adx(high_series, low_series, close_series, period=14)
        assert_series_equal(actual, expected, check_names=False)

    def test_drawdown_parity(self, close_series: pd.Series) -> None:
        from src.strategy.tsmom.preprocessor import calculate_drawdown

        expected = calculate_drawdown(close_series)
        actual = drawdown(close_series)
        assert_series_equal(actual, expected, check_names=False)

    def test_volume_weighted_returns_parity(
        self, close_series: pd.Series, volume_series: pd.Series
    ) -> None:
        from src.strategy.tsmom.preprocessor import (
            calculate_returns,
            calculate_volume_weighted_returns,
        )

        rets = calculate_returns(close_series, use_log=True)
        expected = calculate_volume_weighted_returns(rets, volume_series, window=30)
        actual = volume_weighted_returns(rets, volume_series, window=30)
        assert_series_equal(actual, expected, check_names=False)


class TestParityWithBBRSI:
    """BB-RSI preprocessor가 indicators를 사용하므로 preprocess 결과 검증."""

    def test_preprocess_produces_rsi(self) -> None:
        """bb_rsi.preprocess가 indicators.rsi와 동일한 RSI를 생성하는지 확인."""
        from src.strategy.bb_rsi.config import BBRSIConfig
        from src.strategy.bb_rsi.preprocessor import preprocess

        rng = np.random.default_rng(42)
        n = 100
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame(
            {
                "open": close + rng.normal(0, 0.5, n),
                "high": close + rng.uniform(0.5, 2, n),
                "low": close - rng.uniform(0.5, 2, n),
                "close": close,
                "volume": rng.uniform(1000, 5000, n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        config = BBRSIConfig()
        result = preprocess(df, config)
        assert "rsi" in result.columns
        expected = rsi(df["close"].astype(float), config.rsi_period)
        valid = ~pd.isna(result["rsi"]) & ~pd.isna(expected)
        assert_series_equal(
            result["rsi"][valid].reset_index(drop=True),
            expected[valid].reset_index(drop=True),
            check_names=False,
        )


class TestNewIndicators:
    """Phase 1: 15개 신규 추출 지표 테스트."""

    # --- rolling_return ---
    def test_rolling_return_log(self, close_series: pd.Series) -> None:
        result = rolling_return(close_series, period=5, use_log=True)
        expected = np.log(close_series / close_series.shift(5))
        valid = ~pd.isna(result) & ~pd.isna(expected)
        assert_series_equal(result[valid], expected[valid], check_names=False)

    def test_rolling_return_simple(self, close_series: pd.Series) -> None:
        result = rolling_return(close_series, period=5, use_log=False)
        expected = close_series.pct_change(5)
        valid = ~pd.isna(result) & ~pd.isna(expected)
        assert_series_equal(result[valid], expected[valid], check_names=False)

    def test_rolling_return_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            rolling_return(pd.Series([], dtype=float), period=5)

    # --- parkinson_volatility ---
    def test_parkinson_volatility_basic(
        self, high_series: pd.Series, low_series: pd.Series
    ) -> None:
        result = parkinson_volatility(high_series, low_series)
        assert len(result) == len(high_series)
        assert result.name == "parkinson_vol"
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_parkinson_volatility_known_value(self) -> None:
        high = pd.Series([110.0, 120.0])
        low = pd.Series([100.0, 100.0])
        result = parkinson_volatility(high, low)
        expected_0 = np.sqrt(1.0 / (4 * np.log(2)) * np.log(110 / 100) ** 2)
        assert result.iloc[0] == pytest.approx(expected_0)

    # --- garman_klass_volatility ---
    def test_garman_klass_basic(self, ohlcv_df: pd.DataFrame) -> None:
        o: pd.Series = ohlcv_df["open"]  # type: ignore[assignment]
        h: pd.Series = ohlcv_df["high"]  # type: ignore[assignment]
        l_: pd.Series = ohlcv_df["low"]  # type: ignore[assignment]
        c: pd.Series = ohlcv_df["close"]  # type: ignore[assignment]
        result = garman_klass_volatility(o, h, l_, c)
        assert len(result) == len(ohlcv_df)
        assert result.name == "gk_var"

    # --- vol_regime ---
    def test_vol_regime_range(self, close_series: pd.Series) -> None:
        rets = log_returns(close_series)
        result = vol_regime(rets, vol_lookback=20, vol_rank_lookback=50)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    # --- efficiency_ratio ---
    def test_efficiency_ratio_basic(self, close_series: pd.Series) -> None:
        result = efficiency_ratio(close_series, period=10)
        assert len(result) == len(close_series)
        assert result.name == "efficiency_ratio"
        # ER is between 0 and 1
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.01).all()  # small tolerance

    def test_efficiency_ratio_trending(self) -> None:
        """강한 추세 → ER ~ 1."""
        close = pd.Series(np.arange(1.0, 51.0))
        result = efficiency_ratio(close, period=10)
        assert result.iloc[-1] == pytest.approx(1.0, abs=0.01)

    # --- kama ---
    def test_kama_basic(self, close_series: pd.Series) -> None:
        result = kama(close_series, er_lookback=10, fast_period=2, slow_period=30)
        assert len(result) == len(close_series)
        assert result.name == "kama"
        # KAMA should be close to close on average
        valid = ~pd.isna(result)
        assert abs(result[valid].mean() - close_series[valid].mean()) < 5

    # --- momentum ---
    def test_momentum_basic(self, close_series: pd.Series) -> None:
        result = momentum(close_series, period=5)
        expected = close_series.diff(5)
        valid = ~pd.isna(result) & ~pd.isna(expected)
        assert_series_equal(result[valid], expected[valid], check_names=False)

    def test_momentum_name(self, close_series: pd.Series) -> None:
        assert momentum(close_series, 5).name == "momentum"

    # --- chaikin_money_flow ---
    def test_chaikin_mf_basic(self, ohlcv_df: pd.DataFrame) -> None:
        h: pd.Series = ohlcv_df["high"]  # type: ignore[assignment]
        l_: pd.Series = ohlcv_df["low"]  # type: ignore[assignment]
        c: pd.Series = ohlcv_df["close"]  # type: ignore[assignment]
        v: pd.Series = ohlcv_df["volume"]  # type: ignore[assignment]
        result = chaikin_money_flow(h, l_, c, v, period=20)
        assert len(result) == len(ohlcv_df)
        valid = result.dropna()
        assert (valid >= -1.01).all()
        assert (valid <= 1.01).all()

    # --- bb_position ---
    def test_bb_position_range(self, close_series: pd.Series) -> None:
        result = bb_position(close_series, period=20)
        # Most values should be between 0 and 1 (not strict: can be outside bands)
        valid = result.dropna()
        assert len(valid) > 0

    # --- sma_cross ---
    def test_sma_cross_basic(self, close_series: pd.Series) -> None:
        result = sma_cross(close_series, fast=5, slow=20)
        assert len(result) == len(close_series)

    # --- ema_cross ---
    def test_ema_cross_basic(self, close_series: pd.Series) -> None:
        result = ema_cross(close_series, fast=5, slow=20)
        assert len(result) == len(close_series)

    # --- volume_macd ---
    def test_volume_macd_basic(self, volume_series: pd.Series) -> None:
        result = volume_macd(volume_series, fast=12, slow=26)
        assert len(result) == len(volume_series)

    # --- squeeze_detect ---
    def test_squeeze_detect_basic(self) -> None:
        bb_upper = pd.Series([10.0, 10.0, 12.0])
        bb_lower = pd.Series([8.0, 9.0, 6.0])
        kc_upper = pd.Series([11.0, 11.0, 11.0])
        kc_lower = pd.Series([7.0, 7.0, 7.0])
        result = squeeze_detect(bb_upper, bb_lower, kc_upper, kc_lower)
        assert result.iloc[0] is np.True_  # BB inside KC
        assert result.iloc[1] is np.True_  # BB inside KC
        assert result.iloc[2] is np.False_  # BB outside KC

    # --- rolling_zscore ---
    def test_rolling_zscore_basic(self) -> None:
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
        result = rolling_zscore(s, window=20)
        valid = result.dropna()
        # Z-scores should have approximately mean 0
        assert abs(valid.mean()) < 0.5

    def test_rolling_zscore_known_value(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_zscore(s, window=3)
        # At index 2: mean=2, std=1 → zscore=(3-2)/1=1
        assert result.iloc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 3: Modern Quant Indicators
# ---------------------------------------------------------------------------


class TestVolatilityAdvanced:
    """변동성 고급 지표 테스트."""

    def test_volatility_of_volatility_basic(self) -> None:
        """VoV 기본 계산."""
        np.random.seed(42)
        vol = pd.Series(np.abs(np.random.randn(100) * 0.02) + 0.01)
        result = volatility_of_volatility(vol, window=20)

        assert len(result) == 100
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_volatility_of_volatility_constant(self) -> None:
        """상수 변동성 → VoV = 0."""
        vol = pd.Series([0.05] * 50)
        result = volatility_of_volatility(vol, window=20)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-15)

    def test_yang_zhang_basic(self, ohlcv_df: pd.DataFrame) -> None:
        """Yang-Zhang volatility 기본 계산."""
        result = yang_zhang_volatility(
            ohlcv_df["open"], ohlcv_df["high"],
            ohlcv_df["low"], ohlcv_df["close"],
            window=20,
        )
        assert len(result) == len(ohlcv_df)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_yang_zhang_vs_realized(self, ohlcv_df: pd.DataFrame) -> None:
        """YZ vol은 range 정보를 활용하므로 close-to-close와 다름."""
        yz = yang_zhang_volatility(
            ohlcv_df["open"], ohlcv_df["high"],
            ohlcv_df["low"], ohlcv_df["close"],
            window=20,
        )
        cc_returns = log_returns(ohlcv_df["close"])
        cc_vol = cc_returns.rolling(20).std()
        # 둘 다 유효한 곳에서 비교
        valid_idx = yz.dropna().index.intersection(cc_vol.dropna().index)
        assert len(valid_idx) > 0
        # 완전히 같지는 않아야 함 (다른 추정치이므로)
        assert not np.allclose(yz.loc[valid_idx].values, cc_vol.loc[valid_idx].values)

    def test_vol_percentile_rank_range(self) -> None:
        """vol_percentile_rank 범위 0~1 확인."""
        np.random.seed(42)
        vol = pd.Series(np.abs(np.random.randn(200) * 0.02) + 0.01)
        result = vol_percentile_rank(vol, window=60)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0


class TestMicrostructure:
    """미시구조 지표 테스트."""

    def test_hurst_trending(self) -> None:
        """강한 추세 → H > 0.5."""
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        close = pd.Series(np.arange(200, dtype=float) + 100, index=idx)
        result = hurst_exponent(close, window=50)
        valid = result.dropna()
        assert len(valid) > 0
        # 완벽한 추세이므로 H > 0.5
        assert valid.median() > 0.5

    def test_hurst_range(self) -> None:
        """Hurst exponent 범위 확인."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        close = pd.Series(100 + np.cumsum(np.random.randn(200)), index=idx)
        result = hurst_exponent(close, window=50)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_fractal_dimension_basic(self) -> None:
        """프랙탈 차원 기본 계산."""
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        close = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5), index=idx)
        result = fractal_dimension(close, period=20)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 1.0
        assert valid.max() <= 2.0

    def test_price_acceleration_basic(self) -> None:
        """가격 가속도 기본 계산."""
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(100)))
        result = price_acceleration(close, fast=5, slow=20)
        assert len(result) == 100
        valid = result.dropna()
        assert len(valid) > 0

    def test_price_acceleration_known_value(self) -> None:
        """알려진 값: fast_roc - slow_roc."""
        close = pd.Series([100.0, 105.0, 110.0, 108.0, 115.0, 120.0])
        result = price_acceleration(close, fast=2, slow=4)
        # index 4: fast_roc = (115-110)/110, slow_roc = (115-100)/100
        fast_roc = (115 - 110) / 110
        slow_roc = (115 - 100) / 100
        expected = fast_roc - slow_roc
        np.testing.assert_almost_equal(result.iloc[4], expected)


class TestCrossIndicators:
    """크로스 지표 테스트."""

    def test_rsi_divergence_basic(self) -> None:
        """RSI divergence 기본 계산."""
        np.random.seed(42)
        n = 100
        close = pd.Series(100 + np.cumsum(np.random.randn(n)))
        rsi_vals = rsi(close, period=14)
        result = rsi_divergence(close, rsi_vals, window=20)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_trend_strength_basic(self) -> None:
        """trend_strength 분류 확인."""
        adx_vals = pd.Series([10.0, 30.0, 25.0, 50.0, 15.0])
        result = trend_strength(adx_vals, threshold=25.0)
        expected = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(result.values, expected.values)

    def test_trend_strength_custom_threshold(self) -> None:
        """커스텀 threshold."""
        adx_vals = pd.Series([10.0, 20.0, 30.0])
        result = trend_strength(adx_vals, threshold=15.0)
        expected = pd.Series([0.0, 1.0, 1.0])
        np.testing.assert_array_equal(result.values, expected.values)

    def test_mean_reversion_score_overextended(self) -> None:
        """과매수 상태 → 음의 MR score."""
        close = pd.Series([100.0] * 40 + [120.0] * 10)
        result = mean_reversion_score(close, window=30)
        # 마지막 값 (120)은 평균(100)보다 훨씬 높으므로 z-score 양수 → MR score 음수
        last_valid = result.dropna().iloc[-1]
        assert last_valid < 0

    def test_mean_reversion_score_oversold(self) -> None:
        """과매도 상태 → 양의 MR score."""
        close = pd.Series([100.0] * 40 + [80.0] * 10)
        result = mean_reversion_score(close, window=30)
        # 마지막 값 (80)은 평균(100)보다 낮으므로 z-score 음수 → MR score 양수
        last_valid = result.dropna().iloc[-1]
        assert last_valid > 0

    def test_mean_reversion_score_clip(self) -> None:
        """std_mult로 클리핑 확인."""
        close = pd.Series([100.0] * 40 + [200.0] * 10)
        result = mean_reversion_score(close, window=30, std_mult=2.0)
        valid = result.dropna()
        assert valid.min() >= -2.0
        assert valid.max() <= 2.0
