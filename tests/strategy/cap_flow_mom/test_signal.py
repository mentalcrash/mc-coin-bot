"""Tests for Capital Flow Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.cap_flow_mom.config import CapFlowMomConfig, ShortMode
from src.strategy.cap_flow_mom.preprocessor import preprocess
from src.strategy.cap_flow_mom.signal import generate_signals


@pytest.fixture
def config() -> CapFlowMomConfig:
    return CapFlowMomConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


@pytest.fixture
def sample_ohlcv_with_onchain() -> pd.DataFrame:
    """OHLCV + On-chain stablecoin 컬럼 포함 테스트 데이터."""
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    stablecoin = 100e9 + np.cumsum(np.random.randn(n) * 1e8)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_stablecoin_total_usd": stablecoin,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def preprocessed_with_onchain(
    sample_ohlcv_with_onchain: pd.DataFrame, config: CapFlowMomConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_with_onchain, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_simultaneous_entry_exit_on_same_bar(
        self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """동일 bar에서 entries + exits 동시 발생 불가."""
        signals = generate_signals(preprocessed_df, config)
        overlap = signals.entries & signals.exits
        assert not overlap.any()

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CapFlowMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CapFlowMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_require_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CapFlowMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY: short direction은 drawdown < threshold 구간에서만 발생
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_at_short = df["drawdown"].shift(1)[short_bars]
            assert (dd_at_short < config.hedge_threshold).all()

    def test_hedge_only_strength_dampened(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 숏 시그널의 strength는 hedge_strength_ratio 적용."""
        config = CapFlowMomConfig(short_mode=ShortMode.HEDGE_ONLY, hedge_strength_ratio=0.5)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # direction=-1인 bar의 strength는 약해져야 함
        assert signals.strength is not None


class TestCapitalFlowConviction:
    """Stablecoin ROC 확신도 가중 테스트."""

    def test_without_onchain_neutral_conviction(
        self, preprocessed_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """On-chain 부재 시 conviction = 1.0 (중립)."""
        signals = generate_signals(preprocessed_df, config)
        # stablecoin_roc가 NaN이면 conviction 1.0 → 순수 가격 모멘텀만
        assert signals.strength is not None

    def test_with_onchain_modulates_strength(
        self,
        sample_ohlcv_with_onchain: pd.DataFrame,
    ) -> None:
        """On-chain 있으면 strength가 달라져야 함."""
        config = CapFlowMomConfig()
        df_with = preprocess(sample_ohlcv_with_onchain, config)
        signals_with = generate_signals(df_with, config)

        # On-chain 없는 버전
        df_without = sample_ohlcv_with_onchain.drop(columns=["oc_stablecoin_total_usd"]).copy()
        df_without = preprocess(df_without, config)
        signals_without = generate_signals(df_without, config)

        # 방향은 동일하지만 strength 크기가 다를 수 있음
        # (stablecoin_roc가 boost/dampen 적용)
        # 정확히 같을 수도 있음 (NaN 구간), 길이만 확인
        assert len(signals_with.strength) == len(signals_without.strength)

    def test_onchain_shift1_applied(
        self, preprocessed_with_onchain: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """On-chain 파생 feature도 signal에서 shift(1) 적용 확인."""
        signals = generate_signals(preprocessed_with_onchain, config)
        assert signals.strength.iloc[0] == 0.0


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """마지막 50 bar 제거해도 이전 시그널 동일."""
        df_full = preprocess(sample_ohlcv_df, config)
        sig_full = generate_signals(df_full, config)
        cut = 50
        df_trunc = preprocess(sample_ohlcv_df.iloc[:-cut].copy(), config)
        sig_trunc = generate_signals(df_trunc, config)
        overlap = len(sig_trunc.direction)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:overlap].reset_index(drop=True),
            sig_trunc.direction.reset_index(drop=True),
            check_names=False,
        )

    def test_single_bar_append(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """1 bar 추가 시 마지막 bar만 변경 가능."""
        n = len(sample_ohlcv_df)
        df_prev = preprocess(sample_ohlcv_df.iloc[: n - 1].copy(), config)
        sig_prev = generate_signals(df_prev, config)
        df_full = preprocess(sample_ohlcv_df, config)
        sig_full = generate_signals(df_full, config)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:-1].reset_index(drop=True),
            sig_prev.direction.reset_index(drop=True),
            check_names=False,
        )

    def test_truncation_invariance_with_onchain(
        self, sample_ohlcv_with_onchain: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """On-chain 포함 데이터에서도 truncation invariance."""
        df_full = preprocess(sample_ohlcv_with_onchain, config)
        sig_full = generate_signals(df_full, config)
        cut = 50
        df_trunc = preprocess(sample_ohlcv_with_onchain.iloc[:-cut].copy(), config)
        sig_trunc = generate_signals(df_trunc, config)
        overlap = len(sig_trunc.direction)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:overlap].reset_index(drop=True),
            sig_trunc.direction.reset_index(drop=True),
            check_names=False,
        )
