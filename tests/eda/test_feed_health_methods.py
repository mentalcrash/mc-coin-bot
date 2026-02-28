"""5개 Live 피드 클래스의 get_health_status() / update_cache_metrics() 존재 검증."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLiveOnchainFeedHealth:
    def test_has_health_and_cache_methods(self) -> None:
        from src.eda.onchain_feed import LiveOnchainFeed

        feed = LiveOnchainFeed(symbols=["BTC/USDT"])
        assert hasattr(feed, "get_health_status")
        assert hasattr(feed, "update_cache_metrics")

    def test_health_status_empty(self) -> None:
        from src.eda.onchain_feed import LiveOnchainFeed

        feed = LiveOnchainFeed(symbols=["BTC/USDT"])
        status = feed.get_health_status()
        assert status["symbols_cached"] == 0
        assert status["total_columns"] == 0

    def test_health_status_with_cache(self) -> None:
        from src.eda.onchain_feed import LiveOnchainFeed

        feed = LiveOnchainFeed(symbols=["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"oc_fear_greed": 50.0, "oc_tvl_usd": 100.0}}
        status = feed.get_health_status()
        assert status["symbols_cached"] == 1
        assert status["total_columns"] == 2

    def test_update_cache_metrics_no_error(self) -> None:
        from src.eda.onchain_feed import LiveOnchainFeed

        feed = LiveOnchainFeed(symbols=["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"oc_fear_greed": 50.0}}
        # Should not raise
        feed.update_cache_metrics()


class TestLiveDerivativesFeedHealth:
    def test_has_health_and_cache_methods(self) -> None:
        from src.eda.derivatives_feed import LiveDerivativesFeed

        client = MagicMock()
        feed = LiveDerivativesFeed(symbols=["BTC/USDT"], futures_client=client)
        assert hasattr(feed, "get_health_status")
        assert hasattr(feed, "update_cache_metrics")

    def test_health_status_empty(self) -> None:
        from src.eda.derivatives_feed import LiveDerivativesFeed

        client = MagicMock()
        feed = LiveDerivativesFeed(symbols=["BTC/USDT"], futures_client=client)
        status = feed.get_health_status()
        assert status["symbols_cached"] == 0
        assert status["total_columns"] == 0

    def test_health_status_with_cache(self) -> None:
        from src.eda.derivatives_feed import LiveDerivativesFeed

        client = MagicMock()
        feed = LiveDerivativesFeed(symbols=["BTC/USDT"], futures_client=client)
        feed._cache = {"BTC/USDT": {"funding_rate": 0.01, "open_interest": 1000}}
        status = feed.get_health_status()
        assert status["symbols_cached"] == 1
        assert status["total_columns"] == 2


class TestLiveMacroFeedHealth:
    def test_has_health_and_cache_methods(self) -> None:
        from src.eda.macro_feed import LiveMacroFeed

        feed = LiveMacroFeed()
        assert hasattr(feed, "get_health_status")
        assert hasattr(feed, "update_cache_metrics")

    def test_health_status_empty(self) -> None:
        from src.eda.macro_feed import LiveMacroFeed

        feed = LiveMacroFeed()
        status = feed.get_health_status()
        assert status["symbols_cached"] == 0
        assert status["total_columns"] == 0

    def test_health_status_with_cache(self) -> None:
        from src.eda.macro_feed import LiveMacroFeed

        feed = LiveMacroFeed()
        feed._cache = {"macro_dxy": 104.5, "macro_vix": 15.0}
        status = feed.get_health_status()
        assert status["symbols_cached"] == 1
        assert status["total_columns"] == 2


class TestLiveOptionsFeedHealth:
    def test_has_health_and_cache_methods(self) -> None:
        from src.eda.options_feed import LiveOptionsFeed

        feed = LiveOptionsFeed()
        assert hasattr(feed, "get_health_status")
        assert hasattr(feed, "update_cache_metrics")

    def test_health_status_with_cache(self) -> None:
        from src.eda.options_feed import LiveOptionsFeed

        feed = LiveOptionsFeed()
        feed._cache = {"opt_btc_dvol": 55.0}
        status = feed.get_health_status()
        assert status["symbols_cached"] == 1
        assert status["total_columns"] == 1


class TestLiveDerivExtFeedHealth:
    def test_has_health_and_cache_methods(self) -> None:
        from src.eda.deriv_ext_feed import LiveDerivExtFeed

        feed = LiveDerivExtFeed(symbols=["BTC/USDT"])
        assert hasattr(feed, "get_health_status")
        assert hasattr(feed, "update_cache_metrics")

    def test_health_status_with_cache(self) -> None:
        from src.eda.deriv_ext_feed import LiveDerivExtFeed

        feed = LiveDerivExtFeed(symbols=["BTC/USDT"])
        feed._cache = {"BTC/USDT": {"dext_agg_oi_close": 100.0, "dext_hl_funding": 0.01}}
        status = feed.get_health_status()
        assert status["symbols_cached"] == 1
        assert status["total_columns"] == 2

    @pytest.mark.parametrize("feed_class_path", [
        "src.eda.onchain_feed.LiveOnchainFeed",
        "src.eda.derivatives_feed.LiveDerivativesFeed",
        "src.eda.macro_feed.LiveMacroFeed",
        "src.eda.options_feed.LiveOptionsFeed",
        "src.eda.deriv_ext_feed.LiveDerivExtFeed",
    ])
    def test_update_cache_metrics_callable(self, feed_class_path: str) -> None:
        """모든 Live 피드 클래스가 update_cache_metrics() 호출 가능."""
        import importlib

        module_path, class_name = feed_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        assert callable(getattr(cls, "update_cache_metrics", None))
