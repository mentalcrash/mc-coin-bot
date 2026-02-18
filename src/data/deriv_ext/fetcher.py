"""Coinalyze Extended Derivatives data fetcher — 8 datasets.

Aggregated OI (BTC/ETH), Funding Rate (BTC/ETH),
Liquidations (BTC/ETH), CVD (BTC/ETH)

Rules Applied:
    - #12 Data Engineering: Vectorized DataFrame 변환
    - #23 Exception Handling: Domain-driven hierarchy
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.data.deriv_ext.client import AsyncCoinalyzeClient, AsyncHyperliquidClient

# Coinalyze symbol mapping: asset → symbol (Binance primary, suffix .6)
ASSET_SYMBOLS: dict[str, str] = {
    "BTC": "BTCUSDT.6",
    "ETH": "ETHUSDT.6",
}

# Dataset name → asset mapping
COINALYZE_DATASETS: dict[str, str] = {
    "btc_agg_oi": "BTC",
    "eth_agg_oi": "ETH",
    "btc_agg_funding": "BTC",
    "eth_agg_funding": "ETH",
    "btc_liquidations": "BTC",
    "eth_liquidations": "ETH",
    "btc_cvd": "BTC",
    "eth_cvd": "ETH",
}

# Hyperliquid datasets
HYPERLIQUID_DATASETS: dict[str, str] = {
    "hl_asset_contexts": "metaAndAssetCtxs",
    "hl_predicted_fundings": "predictedFundings",
}
HL_TARGET_COINS: set[str] = {"BTC", "ETH"}


def _ts_to_dt(ts: int | float) -> datetime:
    """Unix seconds → UTC datetime."""
    return datetime.fromtimestamp(ts, tz=UTC)


class CoinalyzeFetcher:
    """Coinalyze API 데이터 fetcher.

    Example:
        >>> async with AsyncCoinalyzeClient("coinalyze", api_key="...") as client:
        ...     fetcher = CoinalyzeFetcher(client)
        ...     df = await fetcher.fetch_agg_oi("BTC")
    """

    def __init__(self, client: AsyncCoinalyzeClient) -> None:
        self._client = client

    async def fetch_agg_oi(
        self,
        asset: str = "BTC",
        start: str = "2020-01-01",
        end: str = "2099-12-31",
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Aggregated Open Interest 히스토리.

        Args:
            asset: "BTC" or "ETH"
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)
            interval: 해상도 ("daily")

        Returns:
            DataFrame[date, symbol, open, high, low, close, source]
        """
        symbol = ASSET_SYMBOLS[asset.upper()]
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        now_ts = int(datetime.now(UTC).timestamp())
        end_ts = min(end_ts, now_ts)

        params = {
            "symbols": symbol,
            "interval": interval,
            "from": start_ts,
            "to": end_ts,
        }

        response = await self._client.get("open-interest-history", params=params)
        data = response.json()

        rows = self._parse_ohlc_response(data, symbol)
        if not rows:
            logger.warning("No aggregated OI data for {}", asset)
            return pd.DataFrame(
                columns=pd.Index(["date", "symbol", "open", "high", "low", "close", "source"])
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        logger.info("Fetched Coinalyze agg OI {}: {} bars", asset, len(df))
        return df

    async def fetch_agg_funding(
        self,
        asset: str = "BTC",
        start: str = "2020-01-01",
        end: str = "2099-12-31",
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Aggregated Funding Rate 히스토리.

        Args:
            asset: "BTC" or "ETH"
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)
            interval: 해상도 ("daily")

        Returns:
            DataFrame[date, symbol, open, high, low, close, source]
        """
        symbol = ASSET_SYMBOLS[asset.upper()]
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        now_ts = int(datetime.now(UTC).timestamp())
        end_ts = min(end_ts, now_ts)

        params = {
            "symbols": symbol,
            "interval": interval,
            "from": start_ts,
            "to": end_ts,
        }

        response = await self._client.get("funding-rate-history", params=params)
        data = response.json()

        rows = self._parse_ohlc_response(data, symbol)
        if not rows:
            logger.warning("No aggregated funding data for {}", asset)
            return pd.DataFrame(
                columns=pd.Index(["date", "symbol", "open", "high", "low", "close", "source"])
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        logger.info("Fetched Coinalyze agg funding {}: {} bars", asset, len(df))
        return df

    async def fetch_liquidations(
        self,
        asset: str = "BTC",
        start: str = "2020-01-01",
        end: str = "2099-12-31",
        interval: str = "1hour",
    ) -> pd.DataFrame:
        """Liquidation 히스토리 (long/short volume).

        Args:
            asset: "BTC" or "ETH"
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)
            interval: 해상도 ("1hour")

        Returns:
            DataFrame[date, symbol, long_volume, short_volume, source]
        """
        symbol = ASSET_SYMBOLS[asset.upper()]
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        now_ts = int(datetime.now(UTC).timestamp())
        end_ts = min(end_ts, now_ts)

        params = {
            "symbols": symbol,
            "interval": interval,
            "from": start_ts,
            "to": end_ts,
        }

        response = await self._client.get("liquidation-history", params=params)
        data = response.json()

        rows = self._parse_liquidation_response(data, symbol)
        if not rows:
            logger.warning("No liquidation data for {}", asset)
            return pd.DataFrame(
                columns=pd.Index(["date", "symbol", "long_volume", "short_volume", "source"])
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        logger.info("Fetched Coinalyze liquidations {}: {} bars", asset, len(df))
        return df

    async def fetch_cvd(
        self,
        asset: str = "BTC",
        start: str = "2020-01-01",
        end: str = "2099-12-31",
        interval: str = "daily",
    ) -> pd.DataFrame:
        """CVD (Cumulative Volume Delta) 히스토리.

        Args:
            asset: "BTC" or "ETH"
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)
            interval: 해상도 ("daily")

        Returns:
            DataFrame[date, symbol, open, high, low, close, volume, buy_volume, source]
        """
        symbol = ASSET_SYMBOLS[asset.upper()]
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp())
        now_ts = int(datetime.now(UTC).timestamp())
        end_ts = min(end_ts, now_ts)

        params = {
            "symbols": symbol,
            "interval": interval,
            "from": start_ts,
            "to": end_ts,
        }

        response = await self._client.get("ohlcv-history", params=params)
        data = response.json()

        rows = self._parse_cvd_response(data, symbol)
        if not rows:
            logger.warning("No CVD data for {}", asset)
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "date",
                        "symbol",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "buy_volume",
                        "source",
                    ]
                )
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        logger.info("Fetched Coinalyze CVD {}: {} bars", asset, len(df))
        return df

    def _parse_ohlc_response(
        self, data: list[dict[str, object]], symbol: str
    ) -> list[dict[str, object]]:
        """Parse Coinalyze OHLC response → row dicts.

        Response format: [{"symbol": "...", "history": [{"t": ..., "o": ..., "h": ..., "l": ..., "c": ...}, ...]}]
        """
        if not data:
            return []

        first = data[0]
        history = first.get("history", [])
        if not isinstance(history, list):
            return []

        return [
            {
                "date": _ts_to_dt(int(entry["t"])),  # type: ignore[arg-type]
                "symbol": symbol,
                "open": Decimal(str(entry["o"])),
                "high": Decimal(str(entry["h"])),
                "low": Decimal(str(entry["l"])),
                "close": Decimal(str(entry["c"])),
                "source": "coinalyze",
            }
            for entry in history
        ]

    def _parse_liquidation_response(
        self, data: list[dict[str, object]], symbol: str
    ) -> list[dict[str, object]]:
        """Parse Coinalyze liquidation response → row dicts.

        Response format: [{"symbol": "...", "history": [{"t": ..., "l": ..., "s": ...}, ...]}]
        where l = long_volume, s = short_volume
        """
        if not data:
            return []

        first = data[0]
        history = first.get("history", [])
        if not isinstance(history, list):
            return []

        return [
            {
                "date": _ts_to_dt(int(entry["t"])),  # type: ignore[arg-type]
                "symbol": symbol,
                "long_volume": Decimal(str(entry["l"])),
                "short_volume": Decimal(str(entry["s"])),
                "source": "coinalyze",
            }
            for entry in history
        ]

    def _parse_cvd_response(
        self, data: list[dict[str, object]], symbol: str
    ) -> list[dict[str, object]]:
        """Parse Coinalyze OHLCV (CVD) response → row dicts.

        Response format: [{"symbol": "...", "history": [{"t": ..., "o": ..., "h": ..., "l": ..., "c": ..., "v": ..., "bv": ...}, ...]}]
        """
        if not data:
            return []

        first = data[0]
        history = first.get("history", [])
        if not isinstance(history, list):
            return []

        return [
            {
                "date": _ts_to_dt(int(entry["t"])),  # type: ignore[arg-type]
                "symbol": symbol,
                "open": Decimal(str(entry["o"])),
                "high": Decimal(str(entry["h"])),
                "low": Decimal(str(entry["l"])),
                "close": Decimal(str(entry["c"])),
                "volume": Decimal(str(entry["v"])),
                "buy_volume": Decimal(str(entry["bv"])),
                "source": "coinalyze",
            }
            for entry in history
        ]


class HyperliquidFetcher:
    """Hyperliquid API 데이터 fetcher (POST-only, snapshot).

    Example:
        >>> async with AsyncHyperliquidClient() as client:
        ...     fetcher = HyperliquidFetcher(client)
        ...     df = await fetcher.fetch_asset_contexts()
    """

    def __init__(self, client: AsyncHyperliquidClient) -> None:
        self._client = client

    async def fetch_asset_contexts(self) -> pd.DataFrame:
        """Hyperliquid metaAndAssetCtxs 스냅샷.

        Returns:
            DataFrame[date, coin, mark_price, open_interest, funding, premium, day_ntl_vlm, source]
        """
        response = await self._client.post("info", json={"type": "metaAndAssetCtxs"})
        data = response.json()

        # Response: [meta, [ctx1, ctx2, ...]]
        if not isinstance(data, list) or len(data) < 2:  # noqa: PLR2004
            logger.warning("Unexpected Hyperliquid metaAndAssetCtxs response format")
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "date",
                        "coin",
                        "mark_price",
                        "open_interest",
                        "funding",
                        "premium",
                        "day_ntl_vlm",
                        "source",
                    ]
                )
            )

        meta = data[0]
        asset_ctxs = data[1]

        # meta contains universe with coin names
        universe = meta.get("universe", [])

        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []

        for i, ctx in enumerate(asset_ctxs):
            if i >= len(universe):
                break
            coin = universe[i].get("name", "")
            if coin not in HL_TARGET_COINS:
                continue

            rows.append(
                {
                    "date": now,
                    "coin": coin,
                    "mark_price": Decimal(str(ctx.get("markPx", 0))),
                    "open_interest": Decimal(str(ctx.get("openInterest", 0))),
                    "funding": Decimal(str(ctx.get("funding", 0))),
                    "premium": Decimal(str(ctx.get("premium", 0)))
                    if ctx.get("premium") is not None
                    else None,
                    "day_ntl_vlm": Decimal(str(ctx.get("dayNtlVlm", 0))),
                    "source": "hyperliquid",
                }
            )

        if not rows:
            logger.warning("No BTC/ETH data in Hyperliquid metaAndAssetCtxs")
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "date",
                        "coin",
                        "mark_price",
                        "open_interest",
                        "funding",
                        "premium",
                        "day_ntl_vlm",
                        "source",
                    ]
                )
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        logger.info("Fetched Hyperliquid asset contexts: {} rows", len(df))
        return df

    async def fetch_predicted_fundings(self) -> pd.DataFrame:
        """Hyperliquid predictedFundings 스냅샷 (cross-venue 비교).

        Returns:
            DataFrame[date, coin, venue, predicted_funding, source]
        """
        response = await self._client.post("info", json={"type": "predictedFundings"})
        data = response.json()

        if not isinstance(data, list):
            logger.warning("Unexpected Hyperliquid predictedFundings response format")
            return pd.DataFrame(
                columns=pd.Index(["date", "coin", "venue", "predicted_funding", "source"])
            )

        now = datetime.now(UTC)
        rows: list[dict[str, object]] = []

        for entry in data:
            coin = entry.get("coin", "")
            if coin not in HL_TARGET_COINS:
                continue

            venues = entry.get("venues", [])
            for venue_data in venues:
                if not isinstance(venue_data, list) or len(venue_data) < 2:  # noqa: PLR2004
                    continue
                venue_name = str(venue_data[0])
                predicted_rate = venue_data[1]
                rows.append(
                    {
                        "date": now,
                        "coin": coin,
                        "venue": venue_name,
                        "predicted_funding": Decimal(str(predicted_rate)),
                        "source": "hyperliquid",
                    }
                )

        if not rows:
            logger.warning("No BTC/ETH data in Hyperliquid predictedFundings")
            return pd.DataFrame(
                columns=pd.Index(["date", "coin", "venue", "predicted_funding", "source"])
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        logger.info("Fetched Hyperliquid predicted fundings: {} rows", len(df))
        return df


async def route_fetch(
    fetcher: CoinalyzeFetcher | HyperliquidFetcher,
    source: str,
    name: str,
) -> pd.DataFrame:
    """(source, name) 쌍을 적절한 fetcher 메서드로 라우팅.

    Args:
        fetcher: CoinalyzeFetcher 인스턴스
        source: 데이터 소스 ("coinalyze")
        name: 데이터 이름 (e.g., "btc_agg_oi")

    Returns:
        Fetched DataFrame

    Raises:
        ValueError: 알 수 없는 source 또는 name
    """
    if source == "coinalyze":
        if not isinstance(fetcher, CoinalyzeFetcher):
            msg = "Coinalyze source requires CoinalyzeFetcher"
            raise TypeError(msg)
        asset = COINALYZE_DATASETS.get(name)
        if asset is None:
            msg = f"Unknown deriv_ext dataset: {name}. Valid: {', '.join(COINALYZE_DATASETS)}"
            raise ValueError(msg)

        if name.endswith("_agg_oi"):
            return await fetcher.fetch_agg_oi(asset)
        if name.endswith("_agg_funding"):
            return await fetcher.fetch_agg_funding(asset)
        if name.endswith("_liquidations"):
            return await fetcher.fetch_liquidations(asset)
        if name.endswith("_cvd"):
            return await fetcher.fetch_cvd(asset)

        msg = f"No handler for deriv_ext dataset: {name}"
        raise ValueError(msg)

    if source == "hyperliquid":
        if not isinstance(fetcher, HyperliquidFetcher):
            msg = "Hyperliquid source requires HyperliquidFetcher"
            raise TypeError(msg)
        if name == "hl_asset_contexts":
            return await fetcher.fetch_asset_contexts()
        if name == "hl_predicted_fundings":
            return await fetcher.fetch_predicted_fundings()
        msg = f"Unknown Hyperliquid dataset: {name}. Valid: {', '.join(HYPERLIQUID_DATASETS)}"
        raise ValueError(msg)

    msg = f"Unknown deriv_ext source: {source}. Valid: coinalyze, hyperliquid"
    raise ValueError(msg)
