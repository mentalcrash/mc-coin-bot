"""On-chain data fetchers (DeFiLlama stablecoins/TVL/DEX, Coin Metrics).

Fetches:
- DeFiLlama stablecoin supply: total, chain, individual
- DeFiLlama TVL: total, per-chain historical
- DeFiLlama DEX volume: daily aggregate
- Coin Metrics Community API: MVRV, RealCap, NVTAdj90 등

Rules Applied:
    - #23 Exception Handling: Domain-driven hierarchy
    - Decimal for financial values
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.data.onchain.client import AsyncOnchainClient

# DeFiLlama API base URLs
DEFILLAMA_BASE_URL = "https://stablecoins.llama.fi"
DEFILLAMA_API_URL = "https://api.llama.fi"

# Coin Metrics Community API
COINMETRICS_BASE_URL = "https://community-api.coinmetrics.io/v4"
CM_METRICS = [
    "MVRV",
    "RealCap",
    "AdrActCnt",
    "TxTfrValAdjUSD",
    "TxTfrValMeanUSD",
    "TxTfrValMedUSD",
    "TxCnt",
    "NVTAdj90",
    "VtyRet30d",
]
CM_ASSETS = ["btc", "eth"]
CM_PAGE_SIZE = 10000

# Top 5 chains for stablecoin analysis
DEFI_CHAINS = ["Ethereum", "Tron", "BSC", "Arbitrum", "Solana"]

# Key stablecoin IDs on DeFiLlama
STABLECOIN_IDS: dict[str, int] = {
    "USDT": 1,
    "USDC": 2,
}


class OnchainFetcher:
    """DeFiLlama stablecoin data fetcher.

    Fetches and parses stablecoin supply data from DeFiLlama API.
    Client is injected for testability.

    Example:
        >>> async with AsyncOnchainClient("defillama") as client:
        ...     fetcher = OnchainFetcher(client)
        ...     df = await fetcher.fetch_stablecoin_total()
    """

    def __init__(self, client: AsyncOnchainClient) -> None:
        """Initialize fetcher with HTTP client.

        Args:
            client: Initialized AsyncOnchainClient instance.
        """
        self._client = client

    async def fetch_stablecoin_total(self) -> pd.DataFrame:
        """Fetch total stablecoin supply across all chains.

        API: GET /stablecoincharts/all

        Returns:
            DataFrame with columns [date, total_circulating_usd, source]
        """
        url = f"{DEFILLAMA_BASE_URL}/stablecoincharts/all"
        logger.info(f"Fetching total stablecoin supply from {url}")

        response = await self._client.get(url)
        data = response.json()

        if not data:
            logger.warning("Empty response from /stablecoincharts/all")
            return pd.DataFrame(columns=pd.Index(["date", "total_circulating_usd", "source"]))

        rows: list[dict[str, object]] = []
        for entry in data:
            date_val = entry.get("date")
            total_circ = entry.get("totalCirculating", {})

            # totalCirculating can be dict with peggedUSD or direct value
            if isinstance(total_circ, dict):
                pegged_usd = total_circ.get("peggedUSD", 0)
            else:
                pegged_usd = total_circ or 0

            rows.append(
                {
                    "date": pd.Timestamp.fromtimestamp(int(date_val), tz="UTC"),
                    "total_circulating_usd": Decimal(str(pegged_usd)),
                    "source": "defillama",
                }
            )

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} total stablecoin supply records")
        return df

    async def fetch_stablecoin_by_chain(self, chain: str) -> pd.DataFrame:
        """Fetch stablecoin supply for a specific chain.

        API: GET /stablecoincharts/{chain}

        Args:
            chain: Chain name (e.g., "Ethereum", "Tron")

        Returns:
            DataFrame with columns [date, chain, total_circulating_usd, total_minted_usd]
        """
        url = f"{DEFILLAMA_BASE_URL}/stablecoincharts/{chain}"
        logger.info(f"Fetching stablecoin supply for chain={chain}")

        response = await self._client.get(url)
        data = response.json()

        if not data:
            logger.warning(f"Empty response from /stablecoincharts/{chain}")
            return pd.DataFrame(
                columns=pd.Index(["date", "chain", "total_circulating_usd", "total_minted_usd"])
            )

        rows: list[dict[str, object]] = []
        for entry in data:
            date_val = entry.get("date")
            total_circ = entry.get("totalCirculating", {})
            total_minted = entry.get("totalMinted", {})

            circ_usd = (
                total_circ.get("peggedUSD", 0) if isinstance(total_circ, dict) else total_circ or 0
            )
            minted_usd = (
                total_minted.get("peggedUSD") if isinstance(total_minted, dict) else total_minted
            )

            rows.append(
                {
                    "date": pd.Timestamp.fromtimestamp(int(date_val), tz="UTC"),
                    "chain": chain,
                    "total_circulating_usd": Decimal(str(circ_usd)),
                    "total_minted_usd": Decimal(str(minted_usd))
                    if minted_usd is not None
                    else None,
                }
            )

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} stablecoin records for {chain}")
        return df

    async def fetch_stablecoin_individual(self, sc_id: int, name: str) -> pd.DataFrame:
        """Fetch individual stablecoin circulating supply.

        API: GET /stablecoin/{id}
        Sums chainBalances across all chains for each date.

        Args:
            sc_id: DeFiLlama stablecoin ID (e.g., 1 for USDT)
            name: Stablecoin name (e.g., "USDT")

        Returns:
            DataFrame with columns [date, stablecoin_id, name, circulating_usd]
        """
        url = f"{DEFILLAMA_BASE_URL}/stablecoin/{sc_id}"
        logger.info(f"Fetching individual stablecoin {name} (id={sc_id})")

        response = await self._client.get(url)
        data = response.json()

        chain_balances = data.get("chainBalances", {})
        if not chain_balances:
            logger.warning(f"Empty chainBalances for {name} (id={sc_id})")
            return pd.DataFrame(
                columns=pd.Index(["date", "stablecoin_id", "name", "circulating_usd"])
            )

        # Aggregate across all chains per date
        date_totals: dict[int, Decimal] = {}
        for chain_data in chain_balances.values():
            tokens = chain_data.get("tokens", [])
            for entry in tokens:
                date_val = int(entry.get("date", 0))
                circulating = entry.get("circulating", {})
                pegged_usd = circulating.get("peggedUSD", 0) if isinstance(circulating, dict) else 0
                date_totals[date_val] = date_totals.get(date_val, Decimal(0)) + Decimal(
                    str(pegged_usd)
                )

        rows: list[dict[str, object]] = []
        for date_ts, total_usd in sorted(date_totals.items()):
            rows.append(
                {
                    "date": pd.Timestamp.fromtimestamp(date_ts, tz="UTC"),
                    "stablecoin_id": sc_id,
                    "name": name,
                    "circulating_usd": total_usd,
                }
            )

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} records for {name} (cross-chain aggregated)")
        return df

    async def fetch_coinmetrics(
        self,
        asset: str,
        metrics: list[str] | None = None,
        start: str = "2020-01-01",
        end: str = "",
    ) -> pd.DataFrame:
        """Coin Metrics Community API에서 일별 on-chain 메트릭 수집.

        API: GET /timeseries/asset-metrics

        Args:
            asset: 자산 심볼 (예: "btc", "eth")
            metrics: 수집할 메트릭 리스트 (None → CM_METRICS 전체)
            start: 시작 날짜 (ISO 8601, 기본 2020-01-01)
            end: 종료 날짜 (ISO 8601, 빈 문자열이면 생략)

        Returns:
            DataFrame [time, asset, MVRV, RealCap, ...] — flat 구조
        """
        if metrics is None:
            metrics = CM_METRICS

        columns = ["time", "asset", *metrics]

        url = f"{COINMETRICS_BASE_URL}/timeseries/asset-metrics"
        params: dict[str, str | int] = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": start,
            "page_size": CM_PAGE_SIZE,
        }
        if end:
            params["end_time"] = end

        logger.info(f"Fetching Coin Metrics for asset={asset}, metrics={len(metrics)}")

        all_rows: list[dict[str, object]] = []
        next_page_url: str | None = None

        while True:
            if next_page_url is None:
                response = await self._client.get(url, params=params)
            else:
                response = await self._client.get(next_page_url)

            data = response.json()
            entries = data.get("data", [])

            for entry in entries:
                row: dict[str, object] = {
                    "time": pd.Timestamp(entry["time"]),
                    "asset": entry.get("asset", asset),
                }
                for m in metrics:
                    raw_val = entry.get(m)
                    if raw_val is None or raw_val == "":
                        row[m] = None
                    else:
                        row[m] = Decimal(str(raw_val))
                all_rows.append(row)

            next_page_url = data.get("next_page_url")
            if not next_page_url:
                break

        if not all_rows:
            logger.warning(f"Empty response from Coin Metrics for {asset}")
            return pd.DataFrame(columns=pd.Index(columns))

        df = pd.DataFrame(all_rows, columns=pd.Index(columns))
        logger.info(f"Fetched {len(df)} Coin Metrics records for {asset}")
        return df

    async def fetch_tvl(self, chain: str = "") -> pd.DataFrame:
        """Fetch historical TVL (DeFiLlama).

        API: GET {DEFILLAMA_API_URL}/v2/historicalChainTvl/{chain}
        chain="" → 전체 체인 합산 TVL.

        Args:
            chain: 체인 이름 (빈 문자열이면 전체 합산)

        Returns:
            DataFrame with columns [date, chain, tvl_usd]
        """
        suffix = f"/{chain}" if chain else ""
        url = f"{DEFILLAMA_API_URL}/v2/historicalChainTvl{suffix}"
        chain_label = chain if chain else "all"
        logger.info(f"Fetching TVL for chain={chain_label} from {url}")

        response = await self._client.get(url)
        data = response.json()

        if not data:
            logger.warning(f"Empty TVL response for chain={chain_label}")
            return pd.DataFrame(columns=pd.Index(["date", "chain", "tvl_usd"]))

        rows: list[dict[str, object]] = []
        for entry in data:
            date_val = entry.get("date")
            tvl = entry.get("tvl", 0)
            rows.append(
                {
                    "date": pd.Timestamp.fromtimestamp(int(date_val), tz="UTC"),
                    "chain": chain_label,
                    "tvl_usd": Decimal(str(tvl)),
                }
            )

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} TVL records for chain={chain_label}")
        return df

    async def fetch_dex_volume(self) -> pd.DataFrame:
        """Fetch daily aggregate DEX volume (DeFiLlama).

        API: GET {DEFILLAMA_API_URL}/overview/dexs

        Returns:
            DataFrame with columns [date, volume_usd, source]
        """
        url = f"{DEFILLAMA_API_URL}/overview/dexs"
        logger.info(f"Fetching DEX volume from {url}")

        response = await self._client.get(url)
        data = response.json()

        chart = data.get("totalDataChart", []) if isinstance(data, dict) else []
        if not chart:
            logger.warning("Empty totalDataChart in DEX volume response")
            return pd.DataFrame(columns=pd.Index(["date", "volume_usd", "source"]))

        rows: list[dict[str, object]] = []
        for entry in chart:
            ts, volume = entry[0], entry[1]
            rows.append(
                {
                    "date": pd.Timestamp.fromtimestamp(int(ts), tz="UTC"),
                    "volume_usd": Decimal(str(volume)),
                    "source": "defillama",
                }
            )

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df)} DEX volume records")
        return df
