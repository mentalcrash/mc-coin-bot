"""Tests for Hyperliquid compound dedup in DerivExtSilverProcessor."""

from datetime import UTC, datetime

import pandas as pd

from src.data.deriv_ext.storage import DerivExtBronzeStorage, DerivExtSilverProcessor


class TestCompoundDedup:
    """Hyperliquid compound dedup 테스트."""

    def test_hl_asset_contexts_dedup(self, tmp_path: object) -> None:
        """hl_asset_contexts: [date, coin] dedup."""
        from src.config.settings import IngestionSettings

        settings = IngestionSettings(
            deriv_ext_bronze_dir=tmp_path / "bronze",  # type: ignore[operator]
            deriv_ext_silver_dir=tmp_path / "silver",  # type: ignore[operator]
        )

        bronze = DerivExtBronzeStorage(settings)
        silver = DerivExtSilverProcessor(settings, bronze_storage=bronze)

        # 동일 시간, 다른 coin → 모두 유지
        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=UTC)
        df = pd.DataFrame(
            {
                "date": [now, now],
                "coin": ["BTC", "ETH"],
                "mark_price": [95000.0, 3200.0],
                "open_interest": [500000000.0, 100000000.0],
                "funding": [0.0001, -0.0005],
                "day_ntl_vlm": [5000000000.0, 2000000000.0],
                "source": ["hyperliquid", "hyperliquid"],
            }
        )

        bronze.save(df, "hyperliquid", "hl_asset_contexts")
        silver_path = silver.process("hyperliquid", "hl_asset_contexts")

        result = pd.read_parquet(silver_path)
        # Both BTC and ETH should be kept (same date, different coin)
        assert len(result) == 2

    def test_hl_predicted_fundings_dedup(self, tmp_path: object) -> None:
        """hl_predicted_fundings: [date, coin, venue] dedup."""
        from src.config.settings import IngestionSettings

        settings = IngestionSettings(
            deriv_ext_bronze_dir=tmp_path / "bronze",  # type: ignore[operator]
            deriv_ext_silver_dir=tmp_path / "silver",  # type: ignore[operator]
        )

        bronze = DerivExtBronzeStorage(settings)
        silver = DerivExtSilverProcessor(settings, bronze_storage=bronze)

        now = datetime(2026, 2, 18, 12, 0, 0, tzinfo=UTC)
        df = pd.DataFrame(
            {
                "date": [now, now, now],
                "coin": ["BTC", "BTC", "ETH"],
                "venue": ["Binance", "Bybit", "Binance"],
                "predicted_funding": [0.0002, 0.00015, -0.0001],
                "source": ["hyperliquid", "hyperliquid", "hyperliquid"],
            }
        )

        bronze.save(df, "hyperliquid", "hl_predicted_fundings")
        silver_path = silver.process("hyperliquid", "hl_predicted_fundings")

        result = pd.read_parquet(silver_path)
        # All 3 rows should be kept (different coin/venue combinations)
        assert len(result) == 3

    def test_coinalyze_still_date_only_dedup(self, tmp_path: object) -> None:
        """기존 Coinalyze는 date-only dedup 유지."""
        from src.config.settings import IngestionSettings

        settings = IngestionSettings(
            deriv_ext_bronze_dir=tmp_path / "bronze",  # type: ignore[operator]
            deriv_ext_silver_dir=tmp_path / "silver",  # type: ignore[operator]
        )

        bronze = DerivExtBronzeStorage(settings)
        silver = DerivExtSilverProcessor(settings, bronze_storage=bronze)

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-02-17", "2026-02-17", "2026-02-18"], utc=True),
                "symbol": ["BTCUSDT.6", "BTCUSDT.6", "BTCUSDT.6"],
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
                "source": ["coinalyze", "coinalyze", "coinalyze"],
            }
        )

        bronze.save(df, "coinalyze", "btc_agg_oi")
        silver_path = silver.process("coinalyze", "btc_agg_oi")

        result = pd.read_parquet(silver_path)
        # Duplicate on 2026-02-17 should be deduped → 2 rows
        assert len(result) == 2
