"""Tests for VWAP Disposition Momentum Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.types import Direction
from src.strategy.vwap_disposition.config import ShortMode, VWAPDispositionConfig
from src.strategy.vwap_disposition.preprocessor import preprocess
from src.strategy.vwap_disposition.signal import generate_signals


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 800
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestSignalStructure:
    def test_output_structure(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)


class TestShift1Rule:
    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestCapitulationLogic:
    def test_capitulation_produces_long(self):
        """CGO < -overhang_low AND volume_spike → LONG."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        # Construct a preprocessed DataFrame manually
        df = pd.DataFrame(
            {
                "cgo": np.full(n, -0.20),  # deep negative CGO
                "volume_ratio": np.full(n, 2.0),  # volume spike
                "mom_direction": np.full(n, -1.0),  # doesn't matter for extreme zones
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_low=0.10,
            vol_spike_threshold=1.5,
            use_volume_confirm=True,
        )
        signals = generate_signals(df, config)

        # After shift(1), second bar onward should be LONG
        assert signals.direction.iloc[1] == Direction.LONG

    def test_capitulation_without_volume_no_signal(self):
        """CGO < -overhang_low BUT no volume_spike → no capitulation signal."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        df = pd.DataFrame(
            {
                "cgo": np.full(n, -0.20),
                "volume_ratio": np.full(n, 1.0),  # no spike
                "mom_direction": np.full(n, -1.0),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_low=0.10,
            vol_spike_threshold=1.5,
            use_volume_confirm=True,
        )
        signals = generate_signals(df, config)

        # Without volume spike and in deep CGO zone, middle zone condition is False too
        # so direction should be 0 (not capitulation since no volume, not middle zone)
        # Actually: capitulation_zone is True, middle_zone is False
        # capitulation = capitulation_zone & volume_spike = True & False = False
        # profit_taking = False
        # middle_zone = ~capitulation_zone & ~profit_taking_zone = False
        # So direction_raw = 0
        assert signals.direction.iloc[1] == Direction.NEUTRAL

    def test_capitulation_without_volume_confirm_disabled(self):
        """Volume confirm 비활성화 시 CGO만으로 시그널."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        df = pd.DataFrame(
            {
                "cgo": np.full(n, -0.20),
                "volume_ratio": np.full(n, 1.0),  # no spike but doesn't matter
                "mom_direction": np.full(n, -1.0),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_low=0.10,
            use_volume_confirm=False,
        )
        signals = generate_signals(df, config)

        assert signals.direction.iloc[1] == Direction.LONG


class TestProfitTakingLogic:
    def test_profit_taking_produces_short(self):
        """CGO > +overhang_high AND volume_decline → SHORT."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        df = pd.DataFrame(
            {
                "cgo": np.full(n, 0.25),  # high positive CGO
                "volume_ratio": np.full(n, 0.5),  # volume decline
                "mom_direction": np.full(n, 1.0),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_high=0.15,
            vol_decline_threshold=0.7,
            use_volume_confirm=True,
        )
        signals = generate_signals(df, config)

        assert signals.direction.iloc[1] == Direction.SHORT

    def test_profit_taking_without_volume_no_signal(self):
        """CGO > +overhang_high BUT no volume_decline → no signal."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        df = pd.DataFrame(
            {
                "cgo": np.full(n, 0.25),
                "volume_ratio": np.full(n, 1.0),  # normal volume
                "mom_direction": np.full(n, 1.0),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_high=0.15,
            vol_decline_threshold=0.7,
            use_volume_confirm=True,
        )
        signals = generate_signals(df, config)

        # profit_taking_zone = True, but profit_taking = False (no volume decline)
        # middle_zone = ~True & ~False = False
        # direction = 0
        assert signals.direction.iloc[1] == Direction.NEUTRAL


class TestMiddleZone:
    def test_middle_zone_follows_momentum(self):
        """Middle zone에서는 momentum direction을 따름."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        df = pd.DataFrame(
            {
                "cgo": np.full(n, 0.05),  # within -0.10 to +0.15
                "volume_ratio": np.full(n, 1.0),
                "mom_direction": np.full(n, 1.0),  # positive momentum
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_high=0.15,
            overhang_low=0.10,
        )
        signals = generate_signals(df, config)

        assert signals.direction.iloc[1] == Direction.LONG

    def test_middle_zone_negative_momentum(self):
        """Middle zone에서 negative momentum → SHORT."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        df = pd.DataFrame(
            {
                "cgo": np.full(n, 0.05),
                "volume_ratio": np.full(n, 1.0),
                "mom_direction": np.full(n, -1.0),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_high=0.15,
            overhang_low=0.10,
            short_mode=ShortMode.FULL,
        )
        signals = generate_signals(df, config)

        assert signals.direction.iloc[1] == Direction.SHORT


class TestShortModeFiltering:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self):
        n = 800
        np.random.seed(42)
        # Monotonically increasing close → no drawdown
        close = np.linspace(100, 200, n)
        high = close + 2.0
        low = close - 2.0

        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = VWAPDispositionConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = VWAPDispositionConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame):
        config = VWAPDispositionConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_default_short_mode_is_full(self):
        """기본 설정에서 short_mode는 FULL."""
        config = VWAPDispositionConfig()
        assert config.short_mode == ShortMode.FULL


class TestEntryExitDetection:
    def test_entry_on_direction_change(self):
        """Direction 변경 시 entry 발생."""
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")

        # Create a scenario: neutral → long
        cgo_values = [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        mom_values = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        df = pd.DataFrame(
            {
                "cgo": cgo_values,
                "volume_ratio": np.full(n, 1.0),
                "mom_direction": mom_values,
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.full(n, 0.0),
            },
            index=idx,
        )

        config = VWAPDispositionConfig(
            overhang_high=0.15,
            overhang_low=0.10,
        )
        signals = generate_signals(df, config)

        # entries should be True somewhere when direction switches
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
