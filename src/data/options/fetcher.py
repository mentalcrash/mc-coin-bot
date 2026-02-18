"""Deribit Options data fetcher — 5 datasets.

DVOL (BTC/ETH), Put/Call Ratio, Historical Vol, Term Structure, Max Pain

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
    from src.data.options.client import AsyncOptionsClient

# Deribit 데이터셋 매핑: name → currency
DERIBIT_DATASETS: dict[str, str] = {
    "btc_dvol": "BTC",
    "eth_dvol": "ETH",
    "btc_pc_ratio": "BTC",
    "btc_hist_vol": "BTC",
    "btc_term_structure": "BTC",
    "btc_max_pain": "BTC",
}

# Milliseconds per day
_MS_PER_DAY = 86_400_000

# Minimum entries required for historical vol parsing
_MIN_HIST_VOL_ENTRIES = 2

# Minimum futures contracts needed for term structure
_MIN_FUTURES_FOR_TERM = 2

# Expected parts count in option instrument name (e.g., BTC-26JAN24-40000-C)
_OPTION_NAME_PARTS = 4


_MAX_PAIN_EMPTY_COLUMNS = pd.Index(
    [
        "date",
        "currency",
        "expiry",
        "max_pain_strike",
        "total_oi",
        "source",
    ]
)


def _parse_option_instruments(instruments: list[dict[str, object]]) -> list[dict[str, object]]:
    """Option instrument book summary → parsed options list."""
    options: list[dict[str, object]] = []
    for inst in instruments:
        name = inst.get("instrument_name", "")
        parts = str(name).split("-")
        if len(parts) < _OPTION_NAME_PARTS:
            continue
        options.append(
            {
                "expiry": parts[1],
                "strike": Decimal(parts[2]),
                "option_type": parts[3],
                "oi": Decimal(str(inst.get("open_interest", 0))),
            }
        )
    return options


def _compute_max_pain_strike(
    exp_options: list[dict[str, object]],
) -> tuple[object, Decimal]:
    """Max Pain 계산: total loss 최소 행사가 반환.

    Returns:
        (max_pain_strike, total_oi)
    """
    strikes = sorted({o["strike"] for o in exp_options})  # type: ignore[type-var]

    call_oi_map: dict[Decimal, Decimal] = {}
    put_oi_map: dict[Decimal, Decimal] = {}
    for o in exp_options:
        strike = o["strike"]
        oi = o["oi"]
        if o["option_type"] == "C":
            call_oi_map[strike] = call_oi_map.get(strike, Decimal(0)) + oi  # type: ignore[arg-type]
        else:
            put_oi_map[strike] = put_oi_map.get(strike, Decimal(0)) + oi  # type: ignore[arg-type]

    max_pain_strike = strikes[0]
    min_loss = Decimal("Infinity")
    total_oi: Decimal = sum(o["oi"] for o in exp_options)  # type: ignore[assignment,misc]

    for test_strike in strikes:
        loss = Decimal(0)
        for strike in strikes:
            c_oi = call_oi_map.get(strike, Decimal(0))  # type: ignore[arg-type]
            if test_strike > strike:  # type: ignore[operator]
                loss += (test_strike - strike) * c_oi  # type: ignore[operator]
            p_oi = put_oi_map.get(strike, Decimal(0))  # type: ignore[arg-type]
            if strike > test_strike:  # type: ignore[operator]
                loss += (strike - test_strike) * p_oi  # type: ignore[operator]

        if loss < min_loss:
            min_loss = loss
            max_pain_strike = test_strike

    return max_pain_strike, total_oi


class OptionsFetcher:
    """Deribit Public API 데이터 fetcher.

    Example:
        >>> async with AsyncOptionsClient() as client:
        ...     fetcher = OptionsFetcher(client)
        ...     df = await fetcher.fetch_dvol("BTC")
    """

    def __init__(self, client: AsyncOptionsClient) -> None:
        self._client = client

    async def fetch_dvol(
        self,
        currency: str = "BTC",
        start: str = "2021-03-01",
        end: str = "2099-12-31",
    ) -> pd.DataFrame:
        """DVOL 히스토리 (get_volatility_index_data).

        Args:
            currency: "BTC" or "ETH"
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)

        Returns:
            DataFrame[date, currency, open, high, low, close, volume, source]
        """
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)

        # Clamp end to now
        now_ts = int(datetime.now(UTC).timestamp() * 1000)
        end_ts = min(end_ts, now_ts)

        all_rows: list[dict[str, object]] = []

        # Paginate in 720-day chunks
        chunk_ms = 720 * _MS_PER_DAY
        cursor = start_ts

        daily_resolution = 86400  # seconds per day

        while cursor < end_ts:
            chunk_end = min(cursor + chunk_ms, end_ts)
            params = {
                "currency": currency.upper(),
                "start_timestamp": cursor,
                "end_timestamp": chunk_end,
                "resolution": daily_resolution,
            }

            response = await self._client.get("get_volatility_index_data", params=params)
            data = response.json()
            result = data.get("result", {})

            # Response: {"result": {"data": [[ts, open, high, low, close], ...], "continuation": ...}}
            bars = result.get("data", [])
            if not bars:
                cursor = chunk_end
                continue

            for bar in bars:
                if not isinstance(bar, list) or len(bar) < 5:  # noqa: PLR2004
                    continue
                all_rows.append(
                    {
                        "date": datetime.fromtimestamp(bar[0] / 1000, tz=UTC),
                        "currency": currency.upper(),
                        "open": Decimal(str(bar[1])),
                        "high": Decimal(str(bar[2])),
                        "low": Decimal(str(bar[3])),
                        "close": Decimal(str(bar[4])),
                        "volume": None,
                        "source": "deribit",
                    }
                )

            cursor = chunk_end

        if not all_rows:
            logger.warning("No DVOL data for {}", currency)
            return pd.DataFrame(
                columns=pd.Index(
                    ["date", "currency", "open", "high", "low", "close", "volume", "source"]
                )
            )

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info("Fetched Deribit DVOL {}: {} bars", currency, len(df))
        return df

    async def fetch_pc_ratio(self, currency: str = "BTC") -> pd.DataFrame:
        """Put/Call OI Ratio 스냅샷.

        히스토리 불가 → 일일 스냅샷 적재 방식.

        Args:
            currency: "BTC" or "ETH"

        Returns:
            DataFrame[date, currency, put_oi, call_oi, pc_ratio, source]
        """
        params = {"currency": currency.upper(), "kind": "option"}
        response = await self._client.get("get_book_summary_by_currency", params=params)
        data = response.json()
        instruments = data.get("result", [])

        if not instruments:
            logger.warning("No book summary for {}", currency)
            return pd.DataFrame(
                columns=pd.Index(["date", "currency", "put_oi", "call_oi", "pc_ratio", "source"])
            )

        put_oi = Decimal(0)
        call_oi = Decimal(0)

        for inst in instruments:
            oi = Decimal(str(inst.get("open_interest", 0)))
            name = inst.get("instrument_name", "")
            if name.endswith("-P"):
                put_oi += oi
            elif name.endswith("-C"):
                call_oi += oi

        pc_ratio = put_oi / call_oi if call_oi > 0 else Decimal(0)
        now = datetime.now(UTC)

        df = pd.DataFrame(
            [
                {
                    "date": now,
                    "currency": currency.upper(),
                    "put_oi": put_oi,
                    "call_oi": call_oi,
                    "pc_ratio": pc_ratio,
                    "source": "deribit",
                }
            ]
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info(
            "Fetched Deribit P/C ratio {}: put={}, call={}, ratio={:.3f}",
            currency,
            put_oi,
            call_oi,
            pc_ratio,
        )
        return df

    async def fetch_hist_vol(self, currency: str = "BTC") -> pd.DataFrame:
        """Realized Volatility (7/30/60/90/120/180/365D).

        Args:
            currency: "BTC" or "ETH"

        Returns:
            DataFrame[date, currency, vol_7d, ..., vol_365d, source]
        """
        params = {"currency": currency.upper()}
        response = await self._client.get("get_historical_volatility", params=params)
        data = response.json()
        result = data.get("result", [])

        if not result:
            logger.warning("No historical volatility data for {}", currency)
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "date",
                        "currency",
                        "vol_7d",
                        "vol_30d",
                        "vol_60d",
                        "vol_90d",
                        "vol_120d",
                        "vol_180d",
                        "vol_365d",
                        "source",
                    ]
                )
            )

        # result is [[timestamp_ms, 7d, 30d, 60d, 90d, 120d, 180d, 365d], ...]
        vol_keys = ["vol_7d", "vol_30d", "vol_60d", "vol_90d", "vol_120d", "vol_180d", "vol_365d"]
        rows: list[dict[str, object]] = []

        for entry in result:
            if not isinstance(entry, list) or len(entry) < _MIN_HIST_VOL_ENTRIES:
                continue
            row: dict[str, object] = {
                "date": datetime.fromtimestamp(entry[0] / 1000, tz=UTC),
                "currency": currency.upper(),
                "source": "deribit",
            }
            for i, key in enumerate(vol_keys):
                row[key] = Decimal(str(entry[i + 1])) if i + 1 < len(entry) else None
            rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "date",
                        "currency",
                        "vol_7d",
                        "vol_30d",
                        "vol_60d",
                        "vol_90d",
                        "vol_120d",
                        "vol_180d",
                        "vol_365d",
                        "source",
                    ]
                )
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info("Fetched Deribit hist vol {}: {} observations", currency, len(df))
        return df

    async def fetch_term_structure(self, currency: str = "BTC") -> pd.DataFrame:
        """Futures Term Structure (contango/backwardation).

        1. get_instruments → 활성 futures 목록
        2. 만기 가장 가까운 2개 선택
        3. ticker → 각각 mark_price 조회
        4. get_index_price → 현물 index_price
        5. basis_pct 계산

        Args:
            currency: "BTC" or "ETH"

        Returns:
            DataFrame[date, currency, near_expiry, far_expiry,
                      near_basis_pct, far_basis_pct, slope, source]
        """
        # Step 1: Active futures
        params = {"currency": currency.upper(), "kind": "future", "expired": "false"}
        response = await self._client.get("get_instruments", params=params)
        data = response.json()
        instruments = data.get("result", [])

        # Filter out perpetual (no expiry), sort by expiration
        futures = [
            inst
            for inst in instruments
            if inst.get("settlement_period") != "perpetual"
            and inst.get("expiration_timestamp", 0) > 0
        ]
        futures.sort(key=lambda x: x["expiration_timestamp"])

        if len(futures) < _MIN_FUTURES_FOR_TERM:
            logger.warning("Not enough futures for term structure ({})", currency)
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "date",
                        "currency",
                        "near_expiry",
                        "far_expiry",
                        "near_basis_pct",
                        "far_basis_pct",
                        "slope",
                        "source",
                    ]
                )
            )

        near = futures[0]
        far = futures[1]

        # Step 2: Get ticker for near/far
        near_resp = await self._client.get(
            "ticker", params={"instrument_name": near["instrument_name"]}
        )
        far_resp = await self._client.get(
            "ticker", params={"instrument_name": far["instrument_name"]}
        )

        near_price = Decimal(str(near_resp.json()["result"]["mark_price"]))
        far_price = Decimal(str(far_resp.json()["result"]["mark_price"]))

        # Step 3: Index price
        index_resp = await self._client.get(
            "get_index_price", params={"index_name": f"{currency.lower()}_usd"}
        )
        index_price = Decimal(str(index_resp.json()["result"]["index_price"]))

        # Step 4: Calculate basis
        near_basis = (near_price - index_price) / index_price * 100
        far_basis = (far_price - index_price) / index_price * 100
        slope = far_basis - near_basis
        now = datetime.now(UTC)

        df = pd.DataFrame(
            [
                {
                    "date": now,
                    "currency": currency.upper(),
                    "near_expiry": near["instrument_name"],
                    "far_expiry": far["instrument_name"],
                    "near_basis_pct": near_basis,
                    "far_basis_pct": far_basis,
                    "slope": slope,
                    "source": "deribit",
                }
            ]
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info(
            "Fetched Deribit term structure {}: near={} ({:.2f}%), far={} ({:.2f}%), slope={:.2f}",
            currency,
            near["instrument_name"],
            near_basis,
            far["instrument_name"],
            far_basis,
            slope,
        )
        return df

    async def fetch_max_pain(self, currency: str = "BTC") -> pd.DataFrame:
        """Options Max Pain by nearest expiry.

        1. get_book_summary_by_currency → option 전체 OI
        2. 가장 가까운 만기의 strike별 OI 집계
        3. Max Pain = total loss 최소 행사가 계산

        Args:
            currency: "BTC" or "ETH"

        Returns:
            DataFrame[date, currency, expiry, max_pain_strike, total_oi, source]
        """
        params = {"currency": currency.upper(), "kind": "option"}
        response = await self._client.get("get_book_summary_by_currency", params=params)
        data = response.json()
        instruments = data.get("result", [])

        if not instruments:
            logger.warning("No option data for max pain ({})", currency)
            return pd.DataFrame(columns=_MAX_PAIN_EMPTY_COLUMNS)

        options = _parse_option_instruments(instruments)
        if not options:
            return pd.DataFrame(columns=_MAX_PAIN_EMPTY_COLUMNS)

        # Find nearest expiry
        expiries: list[str] = sorted(str(o["expiry"]) for o in options)
        nearest_expiry = next(iter(dict.fromkeys(expiries)))

        # Filter to nearest expiry and compute max pain
        exp_options = [o for o in options if o["expiry"] == nearest_expiry]
        max_pain_strike, total_oi = _compute_max_pain_strike(exp_options)

        now = datetime.now(UTC)
        df = pd.DataFrame(
            [
                {
                    "date": now,
                    "currency": currency.upper(),
                    "expiry": str(nearest_expiry),
                    "max_pain_strike": max_pain_strike,
                    "total_oi": total_oi,
                    "source": "deribit",
                }
            ]
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)

        logger.info(
            "Fetched Deribit max pain {}: expiry={}, strike={}, total_oi={}",
            currency,
            nearest_expiry,
            max_pain_strike,
            total_oi,
        )
        return df


async def route_fetch(
    fetcher: OptionsFetcher,
    source: str,
    name: str,
) -> pd.DataFrame:
    """(source, name) 쌍을 적절한 fetcher 메서드로 라우팅.

    Args:
        fetcher: OptionsFetcher 인스턴스
        source: 데이터 소스 ("deribit")
        name: 데이터 이름 (e.g., "btc_dvol", "btc_pc_ratio")

    Returns:
        Fetched DataFrame

    Raises:
        ValueError: 알 수 없는 source 또는 name
    """
    if source != "deribit":
        msg = f"Unknown options source: {source}. Valid: deribit"
        raise ValueError(msg)

    currency = DERIBIT_DATASETS.get(name)
    if currency is None:
        msg = f"Unknown options dataset: {name}. Valid: {', '.join(DERIBIT_DATASETS)}"
        raise ValueError(msg)

    if name.endswith("_dvol"):
        return await fetcher.fetch_dvol(currency)
    if name == "btc_pc_ratio":
        return await fetcher.fetch_pc_ratio(currency)
    if name == "btc_hist_vol":
        return await fetcher.fetch_hist_vol(currency)
    if name == "btc_term_structure":
        return await fetcher.fetch_term_structure(currency)
    if name == "btc_max_pain":
        return await fetcher.fetch_max_pain(currency)

    msg = f"No handler for options dataset: {name}"
    raise ValueError(msg)
