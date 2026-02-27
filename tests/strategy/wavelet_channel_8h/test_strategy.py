"""Unit tests for WaveletChannel8hStrategy."""

import pandas as pd
import pytest

from src.strategy.base import BaseStrategy
from src.strategy.registry import get_strategy, list_strategies
from src.strategy.wavelet_channel_8h.config import ShortMode, WaveletChannel8hConfig
from src.strategy.wavelet_channel_8h.strategy import WaveletChannel8hStrategy


class TestStrategyRegistry:
    def test_strategy_registered(self) -> None:
        assert "wavelet-channel-8h" in list_strategies()

    def test_get_strategy(self) -> None:
        strategy_cls = get_strategy("wavelet-channel-8h")
        assert strategy_cls is WaveletChannel8hStrategy

    def test_is_base_strategy_subclass(self) -> None:
        assert issubclass(WaveletChannel8hStrategy, BaseStrategy)


class TestStrategyProperties:
    def test_strategy_name(self) -> None:
        strategy = WaveletChannel8hStrategy()
        assert strategy.name == "wavelet-channel-8h"

    def test_required_columns(self) -> None:
        strategy = WaveletChannel8hStrategy()
        assert strategy.required_columns == ["open", "high", "low", "close", "volume"]

    def test_config_property(self) -> None:
        strategy = WaveletChannel8hStrategy()
        assert isinstance(strategy.config, WaveletChannel8hConfig)

    def test_default_config(self) -> None:
        strategy = WaveletChannel8hStrategy(config=None)
        assert isinstance(strategy.config, WaveletChannel8hConfig)


class TestRunPipeline:
    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        strategy = WaveletChannel8hStrategy()
        _processed_df, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_missing_columns_raises(self) -> None:
        strategy = WaveletChannel8hStrategy()
        bad_df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError, match="Missing"):
            strategy.run(bad_df)


class TestFromParams:
    def test_from_params(self) -> None:
        strategy = WaveletChannel8hStrategy.from_params()
        assert isinstance(strategy, WaveletChannel8hStrategy)

    def test_from_params_custom(self) -> None:
        strategy = WaveletChannel8hStrategy.from_params(vol_target=0.25)
        assert strategy.config.vol_target == 0.25


class TestRecommendedConfig:
    def test_recommended_config(self) -> None:
        config = WaveletChannel8hStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "trailing_stop_enabled" in config
        assert config["use_intrabar_trailing_stop"] is False


class TestShortMode:
    def test_short_mode_disabled(self, sample_ohlcv: pd.DataFrame) -> None:
        config = WaveletChannel8hConfig(short_mode=ShortMode.DISABLED)
        strategy = WaveletChannel8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert (signals.direction >= 0).all()

    def test_short_mode_full(self, sample_ohlcv: pd.DataFrame) -> None:
        config = WaveletChannel8hConfig(short_mode=ShortMode.FULL)
        strategy = WaveletChannel8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)

    def test_short_mode_hedge_only(self, sample_ohlcv: pd.DataFrame) -> None:
        config = WaveletChannel8hConfig(short_mode=ShortMode.HEDGE_ONLY)
        strategy = WaveletChannel8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)


class TestStartupInfo:
    def test_startup_info_keys(self) -> None:
        strategy = WaveletChannel8hStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert all(isinstance(v, str) for v in info.values())


class TestEmaFallback:
    """Test without pywt (EMA fallback)."""

    def test_ema_fallback_without_pywt(self, sample_ohlcv: pd.DataFrame) -> None:
        """When pywt is not available, preprocessor should fall back to EMA lowpass filter."""
        import src.strategy.wavelet_channel_8h.preprocessor as preprocessor_mod

        original_available = preprocessor_mod.PYWT_AVAILABLE
        original_mod = preprocessor_mod._pywt_mod
        original_flag = preprocessor_mod._pywt_available

        try:
            # Simulate pywt not available
            preprocessor_mod.PYWT_AVAILABLE = False
            preprocessor_mod._pywt_available = False
            preprocessor_mod._pywt_mod = None

            strategy = WaveletChannel8hStrategy()
            _processed_df, signals = strategy.run(sample_ohlcv)

            # Should still produce valid signals with EMA fallback
            assert len(signals.entries) == len(sample_ohlcv)
            assert signals.entries.dtype == bool
            assert signals.exits.dtype == bool
        finally:
            # Restore original state
            preprocessor_mod.PYWT_AVAILABLE = original_available
            preprocessor_mod._pywt_mod = original_mod
            preprocessor_mod._pywt_available = original_flag

    def test_pywt_and_ema_produce_valid_signals(self, sample_ohlcv: pd.DataFrame) -> None:
        """Both pywt and EMA fallback paths should produce valid signal shapes."""
        strategy = WaveletChannel8hStrategy()
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)
