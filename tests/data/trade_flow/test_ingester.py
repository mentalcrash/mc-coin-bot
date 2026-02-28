"""Tests for AggTradesIngester — mock 기반 단위 테스트."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.trade_flow.ingester import AggTradesIngester

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: str) -> IngestionSettings:
    """테스트용 IngestionSettings."""
    from pathlib import Path

    tmp = Path(str(tmp_path))
    return IngestionSettings(
        trade_flow_silver_dir=tmp / "silver" / "trade_flow",
    )


@pytest.fixture
def ingester(settings: IngestionSettings) -> AggTradesIngester:
    return AggTradesIngester(settings)


def _make_csv_bytes(n_rows: int = 100, seed: int = 42) -> bytes:
    """테스트용 aggTrades CSV 생성 → ZIP bytes 반환."""
    rng = np.random.default_rng(seed)

    # 2024-01-01 00:00 ~ 2024-01-01 12:00 범위 (1 bar)
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC in ms
    timestamps = base_ts + rng.integers(0, 12 * 3600 * 1000, size=n_rows)
    timestamps.sort()

    df = pd.DataFrame(
        {
            "agg_trade_id": range(1, n_rows + 1),
            "price": rng.uniform(40000, 45000, size=n_rows),
            "quantity": rng.exponential(0.5, size=n_rows),
            "first_trade_id": range(1, n_rows + 1),
            "last_trade_id": range(1, n_rows + 1),
            "transact_time": timestamps,
            "is_buyer_maker": rng.choice([True, False], size=n_rows),
        }
    )

    # CSV to ZIP in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=True)
    csv_bytes = csv_buffer.getvalue().encode()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BTCUSDT-aggTrades-2024-01.csv", csv_bytes)

    return zip_buffer.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAggTradesIngester:
    def test_build_monthly_url(self, ingester: AggTradesIngester) -> None:
        """URL 패턴 검증."""
        url = ingester._build_monthly_url("BTCUSDT", 2024, 1)
        assert url == (
            "https://data.binance.vision/data/futures/um/monthly/aggTrades"
            "/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip"
        )

    def test_build_monthly_url_two_digit_month(self, ingester: AggTradesIngester) -> None:
        """월 두 자리 패딩 확인."""
        url = ingester._build_monthly_url("ETHUSDT", 2025, 12)
        assert "2025-12.zip" in url

    def test_parse_zip(self, ingester: AggTradesIngester) -> None:
        """ZIP → DataFrame 파싱 검증."""
        zip_data = _make_csv_bytes(n_rows=50)
        df = ingester._parse_zip(zip_data)

        assert len(df) == 50
        assert "quantity" in df.columns
        assert "is_buyer_maker" in df.columns
        assert df.index.name == "timestamp"
        assert df.index.is_monotonic_increasing

    def test_compute_bars(self, ingester: AggTradesIngester) -> None:
        """aggTrades → 12H bar 피처 계산."""
        zip_data = _make_csv_bytes(n_rows=100)
        trades_df = ingester._parse_zip(zip_data)
        bars = ingester._compute_bars(trades_df)

        assert not bars.empty
        assert "tflow_cvd" in bars.columns
        assert "tflow_buy_ratio" in bars.columns
        assert "tflow_intensity" in bars.columns
        assert "tflow_large_ratio" in bars.columns
        assert "tflow_abs_order_imbalance" in bars.columns

    @pytest.mark.asyncio
    async def test_ingest_saves_silver(self, ingester: AggTradesIngester) -> None:
        """ingest() → Silver Parquet 파일 생성 확인."""
        zip_data = _make_csv_bytes(n_rows=200)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.read = AsyncMock(return_value=zip_data)
        mock_resp.text = AsyncMock(return_value="")

        # checksum 404 → skip verification
        mock_checksum_resp = AsyncMock()
        mock_checksum_resp.status = 404

        mock_session = AsyncMock()

        async def mock_get(url: str) -> AsyncMock:
            cm = AsyncMock()
            if url.endswith(".CHECKSUM"):
                cm.__aenter__ = AsyncMock(return_value=mock_checksum_resp)
            else:
                cm.__aenter__ = AsyncMock(return_value=mock_resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        mock_session.get = lambda url: mock_get(url).__await__().__next__()

        # aiohttp.ClientSession mock — 더 간단한 방식
        with patch("src.data.trade_flow.ingester.aiohttp.ClientSession") as mock_session_cls:
            session_cm = AsyncMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = session_cm

            # _process_month, _process_month_daily를 직접 mock
            bar_df = ingester._compute_bars(ingester._parse_zip(zip_data))
            with (
                patch.object(
                    ingester,
                    "_process_month",
                    new_callable=AsyncMock,
                ) as mock_process,
                patch.object(
                    ingester,
                    "_process_month_daily",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                # 1월만 데이터 반환, 나머지 None
                side_effects = [bar_df] + [None] * 11
                mock_process.side_effect = side_effects

                path = await ingester.ingest("BTC/USDT", 2024, verify_checksum=False)

        assert path.exists()
        assert path.suffix == ".parquet"

        # Silver 파일 읽기 검증
        result = pd.read_parquet(path)
        assert "tflow_cvd" in result.columns
        assert "tflow_vpin" in result.columns

    def test_parse_zip_empty(self, ingester: AggTradesIngester) -> None:
        """빈 ZIP → 빈 DataFrame."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("empty.txt", "")

        df = ingester._parse_zip(zip_buffer.getvalue())
        assert df.empty
