"""Macro data fetcher — FRED API + yfinance.

FRED: 7 시리즈 (DXY, Gold, DGS10, DGS2, T10Y2Y, VIX, M2)
yfinance: 6 ETF (SPY, QQQ, GLD, TLT, UUP, HYG)

Rules Applied:
    - #12 Data Engineering: Vectorized DataFrame 변환
    - #23 Exception Handling: Domain-driven hierarchy
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.data.macro.client import YFinanceClient

if TYPE_CHECKING:
    from src.data.macro.client import AsyncCoinGeckoClient, AsyncMacroClient

# FRED API base URL
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED 시리즈 매핑: name → series_id
FRED_SERIES: dict[str, str] = {
    "dxy": "DTWEXBGS",
    "gold": "GOLDAMGBD228NLBM",
    "dgs10": "DGS10",
    "dgs2": "DGS2",
    "t10y2y": "T10Y2Y",
    "vix": "VIXCLS",
    "m2": "M2SL",
}

# yfinance 티커 매핑: name → ticker
YFINANCE_TICKERS: dict[str, str] = {
    "spy": "SPY",
    "qqq": "QQQ",
    "gld": "GLD",
    "tlt": "TLT",
    "uup": "UUP",
    "hyg": "HYG",
}

# CoinGecko 데이터셋 매핑: name → endpoint
COINGECKO_DATASETS: dict[str, str] = {
    "global_metrics": "global",
    "defi_global": "global/decentralized_finance_defi",
}


class MacroFetcher:
    """FRED API + yfinance 데이터 fetcher.

    Example:
        >>> async with AsyncMacroClient("fred") as client:
        ...     fetcher = MacroFetcher(client, api_key="your_key")
        ...     df = await fetcher.fetch_fred_series("dxy")
    """

    def __init__(
        self,
        fred_client: AsyncMacroClient,
        api_key: str,
        coingecko_client: AsyncCoinGeckoClient | None = None,
    ) -> None:
        self._fred_client = fred_client
        self._api_key = api_key
        self._yf_client = YFinanceClient()
        self._cg_client = coingecko_client

    async def fetch_fred_series(
        self,
        name: str,
        start: str = "2010-01-01",
        end: str = "2099-12-31",
    ) -> pd.DataFrame:
        """FRED 시리즈 데이터 가져오기.

        Args:
            name: 시리즈 이름 (e.g., "dxy", "vix")
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)

        Returns:
            DataFrame[date, value, series_id, source]

        Raises:
            ValueError: 알 수 없는 시리즈 이름
        """
        series_id = FRED_SERIES.get(name)
        if series_id is None:
            msg = f"Unknown FRED series: {name}. Valid: {', '.join(FRED_SERIES)}"
            raise ValueError(msg)

        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }

        response = await self._fred_client.get(FRED_API_URL, params=params)
        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            logger.warning("No FRED observations for {}/{}", name, series_id)
            return pd.DataFrame(columns=pd.Index(["date", "value", "series_id", "source"]))

        rows: list[dict[str, str | Decimal | None]] = []
        for obs in observations:
            raw_val = obs.get("value", ".")
            value: Decimal | None = None if raw_val == "." else Decimal(str(raw_val))
            rows.append(
                {
                    "date": obs["date"],
                    "value": value,
                    "series_id": series_id,
                    "source": "fred",
                }
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info(
            "Fetched FRED {}/{}: {} observations",
            name,
            series_id,
            len(df),
        )
        return df

    async def fetch_yfinance_ticker(
        self,
        name: str,
        start: str = "2010-01-01",
        end: str = "2099-12-31",
    ) -> pd.DataFrame:
        """yfinance 티커 OHLCV 데이터 가져오기.

        Args:
            name: 티커 이름 (e.g., "spy", "qqq")
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)

        Returns:
            DataFrame[date, open, high, low, close, volume, ticker, source]

        Raises:
            ValueError: 알 수 없는 티커 이름
        """
        ticker = YFINANCE_TICKERS.get(name)
        if ticker is None:
            msg = f"Unknown yfinance ticker: {name}. Valid: {', '.join(YFINANCE_TICKERS)}"
            raise ValueError(msg)

        raw_df = await self._yf_client.fetch_ticker(ticker, start, end)

        if raw_df.empty:
            logger.warning("No yfinance data for {}/{}", name, ticker)
            return pd.DataFrame(
                columns=pd.Index(
                    ["date", "open", "high", "low", "close", "volume", "ticker", "source"]
                )
            )

        # yfinance returns DatetimeIndex with MultiIndex columns for single ticker
        # Flatten if needed
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)

        df = pd.DataFrame(
            {
                "date": raw_df.index,
                "open": raw_df["Open"].to_numpy(),
                "high": raw_df["High"].to_numpy(),
                "low": raw_df["Low"].to_numpy(),
                "close": raw_df["Close"].to_numpy(),
                "volume": raw_df["Volume"].to_numpy(),
                "ticker": ticker,
                "source": "yfinance",
            }
        )

        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info(
            "Fetched yfinance {}/{}: {} bars",
            name,
            ticker,
            len(df),
        )
        return df

    async def fetch_coingecko_global(self) -> pd.DataFrame:
        """CoinGecko /global 스냅샷 가져오기.

        Returns:
            DataFrame[date, btc_dominance, eth_dominance, total_market_cap_usd,
                       total_volume_usd, active_cryptocurrencies, source]

        Raises:
            RuntimeError: CoinGecko client not configured.
        """
        if self._cg_client is None:
            msg = "CoinGecko client not configured"
            raise RuntimeError(msg)

        response = await self._cg_client.get("global")
        data = response.json().get("data", {})

        if not data:
            logger.warning("No CoinGecko global data returned")
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "date",
                        "btc_dominance",
                        "eth_dominance",
                        "total_market_cap_usd",
                        "total_volume_usd",
                        "active_cryptocurrencies",
                        "source",
                    ]
                )
            )

        from datetime import UTC, datetime

        now = datetime.now(UTC)
        total_mcap = data.get("total_market_cap", {})
        total_vol = data.get("total_volume", {})

        row = {
            "date": now,
            "btc_dominance": Decimal(str(data.get("market_cap_percentage", {}).get("btc", 0))),
            "eth_dominance": Decimal(str(data.get("market_cap_percentage", {}).get("eth", 0))),
            "total_market_cap_usd": Decimal(str(total_mcap.get("usd", 0))),
            "total_volume_usd": Decimal(str(total_vol.get("usd", 0))),
            "active_cryptocurrencies": int(data.get("active_cryptocurrencies", 0)),
            "source": "coingecko",
        }

        df = pd.DataFrame([row])
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info("Fetched CoinGecko global snapshot: 1 row")
        return df

    async def fetch_coingecko_defi(self) -> pd.DataFrame:
        """CoinGecko /global/decentralized_finance_defi 스냅샷 가져오기.

        Returns:
            DataFrame[date, defi_market_cap, defi_to_eth_ratio, defi_dominance, source]

        Raises:
            RuntimeError: CoinGecko client not configured.
        """
        if self._cg_client is None:
            msg = "CoinGecko client not configured"
            raise RuntimeError(msg)

        response = await self._cg_client.get("global/decentralized_finance_defi")
        data = response.json().get("data", {})

        if not data:
            logger.warning("No CoinGecko DeFi data returned")
            return pd.DataFrame(
                columns=pd.Index(
                    ["date", "defi_market_cap", "defi_to_eth_ratio", "defi_dominance", "source"]
                )
            )

        from datetime import UTC, datetime

        now = datetime.now(UTC)

        row = {
            "date": now,
            "defi_market_cap": Decimal(str(data.get("defi_market_cap", 0))),
            "defi_to_eth_ratio": Decimal(str(data.get("defi_to_eth_ratio", 0))),
            "defi_dominance": Decimal(str(data.get("defi_dominance", 0))),
            "source": "coingecko",
        }

        df = pd.DataFrame([row])
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info("Fetched CoinGecko DeFi snapshot: 1 row")
        return df


async def route_fetch(
    fetcher: MacroFetcher,
    source: str,
    name: str,
) -> pd.DataFrame:
    """(source, name) 쌍을 적절한 fetcher 메서드로 라우팅.

    Args:
        fetcher: MacroFetcher 인스턴스
        source: 데이터 소스 ("fred" or "yfinance")
        name: 데이터 이름 (e.g., "dxy", "spy")

    Returns:
        Fetched DataFrame

    Raises:
        ValueError: 알 수 없는 source
    """
    if source == "fred":
        return await fetcher.fetch_fred_series(name)
    if source == "yfinance":
        return await fetcher.fetch_yfinance_ticker(name)
    if source == "coingecko":
        if name == "global_metrics":
            return await fetcher.fetch_coingecko_global()
        if name == "defi_global":
            return await fetcher.fetch_coingecko_defi()
        msg = f"Unknown CoinGecko dataset: {name}. Valid: {', '.join(COINGECKO_DATASETS)}"
        raise ValueError(msg)
    msg = f"Unknown macro source: {source}. Valid: fred, yfinance, coingecko"
    raise ValueError(msg)
