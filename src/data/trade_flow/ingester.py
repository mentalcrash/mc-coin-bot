"""AggTrades Ingester — data.binance.vision에서 역사적 aggTrades 수집.

Bronze 생략: raw aggTrades (TB 단위)를 저장하지 않고
인메모리 처리 → 12H bar-level 피처만 Silver에 저장.

URL 패턴 (monthly):
    https://data.binance.vision/data/futures/um/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY}-{MM}.zip
URL 패턴 (daily fallback):
    https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY}-{MM}-{DD}.zip

월별 아카이브가 2-3개월 lag이 있으므로, 최근 월은 daily archive에서 보완한다.

CSV 컬럼 (헤더 없음):
    agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
"""

import calendar
import contextlib
import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.data.trade_flow.features import compute_bar_features, compute_vpin

# CSV 컬럼 정의 (헤더 없는 파일)
_CSV_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]

_CSV_DTYPES = {
    "agg_trade_id": np.int64,
    "price": np.float64,
    "quantity": np.float64,
    "first_trade_id": np.int64,
    "last_trade_id": np.int64,
    "transact_time": np.int64,
    "is_buyer_maker": bool,
}

# Binance Vision base URLs
_BASE_URL_MONTHLY = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
_BASE_URL_DAILY = "https://data.binance.vision/data/futures/um/daily/aggTrades"

# 12H resample 규칙
_BAR_FREQ = "12h"
_BAR_HOURS = 12.0

# HTTP timeout
_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=600)  # 10분


class AggTradesIngester:
    """data.binance.vision에서 aggTrades ZIP 다운로드 → 12H 피처 Silver 생성."""

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self._settings = settings or get_settings()

    async def ingest(
        self,
        symbol: str,
        year: int,
        *,
        verify_checksum: bool = True,
        daily_fallback: bool = True,
    ) -> Path:
        """1년치 aggTrades 다운로드 + 12H 집계 + Silver 저장.

        월별 아카이브를 우선 시도하고, 없는 월은 daily archive에서 보완한다.

        Args:
            symbol: 거래 심볼 (e.g., "BTC/USDT")
            year: 연도
            verify_checksum: CHECKSUM 파일 검증 여부
            daily_fallback: 월별 아카이브 미존재 시 daily archive 시도

        Returns:
            저장된 Silver Parquet 경로

        Raises:
            ValueError: 다운로드 가능한 데이터가 없을 경우
        """
        binance_symbol = symbol.replace("/", "")  # BTC/USDT → BTCUSDT
        logger.info(f"Ingesting aggTrades: {symbol} {year}")

        # 1. 월별 데이터 수집 + 12H bar 피처 계산
        all_bar_features: list[pd.DataFrame] = []
        missing_months: list[int] = []

        async with aiohttp.ClientSession(timeout=_DOWNLOAD_TIMEOUT) as session:
            for month in range(1, 13):
                bar_df = await self._process_month(
                    session, binance_symbol, year, month, verify_checksum=verify_checksum
                )
                if bar_df is not None:
                    all_bar_features.append(bar_df)
                else:
                    missing_months.append(month)

            # Daily fallback: 월별 미존재 월에 대해 일별 아카이브 시도
            if daily_fallback and missing_months:
                for month in missing_months:
                    bar_df = await self._process_month_daily(
                        session, binance_symbol, year, month, verify_checksum=verify_checksum
                    )
                    if bar_df is not None:
                        all_bar_features.append(bar_df)

        if not all_bar_features:
            msg = f"No aggTrades data found for {symbol} {year}"
            raise ValueError(msg)

        # 2. 연도 단위 병합
        combined = pd.concat(all_bar_features).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # 3. VPIN 후처리 (연도 전체에 대해 rolling)
        vpin = compute_vpin(combined)  # type: ignore[arg-type]
        combined["tflow_vpin"] = vpin

        # 4. Silver 저장
        silver_path = self._settings.get_trade_flow_silver_path(symbol, year)
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(silver_path, compression="zstd")

        logger.info(
            "Trade flow Silver saved: {} ({} bars, {} ~ {})",
            silver_path,
            len(combined),
            combined.index[0],
            combined.index[-1],
        )
        return silver_path

    async def _process_month(
        self,
        session: aiohttp.ClientSession,
        binance_symbol: str,
        year: int,
        month: int,
        *,
        verify_checksum: bool = True,
    ) -> pd.DataFrame | None:
        """단일 월 aggTrades 다운로드 → 12H bar 피처 계산.

        Returns:
            12H bar 피처 DataFrame. 데이터 없으면 None.
        """
        url = self._build_monthly_url(binance_symbol, year, month)
        logger.debug(f"Downloading: {url}")

        try:
            zip_data = await self._download_zip(session, url)
        except (aiohttp.ClientError, ValueError):
            logger.debug(f"No data for {binance_symbol} {year}-{month:02d}")
            return None

        # Checksum 검증
        if verify_checksum:
            checksum_url = f"{url}.CHECKSUM"
            try:
                await self._verify_checksum(session, checksum_url, zip_data)
            except Exception:
                logger.warning(
                    f"Checksum verification failed for {binance_symbol} {year}-{month:02d}, proceeding"
                )

        # ZIP → CSV 파싱
        trades_df = self._parse_zip(zip_data)
        if trades_df.empty:
            return None

        # 12H bar 분할 + 피처 계산
        return self._compute_bars(trades_df)

    def _build_monthly_url(self, binance_symbol: str, year: int, month: int) -> str:
        """월별 aggTrades ZIP URL 생성."""
        filename = f"{binance_symbol}-aggTrades-{year}-{month:02d}.zip"
        return f"{_BASE_URL_MONTHLY}/{binance_symbol}/{filename}"

    def _build_daily_url(self, binance_symbol: str, year: int, month: int, day: int) -> str:
        """일별 aggTrades ZIP URL 생성."""
        filename = f"{binance_symbol}-aggTrades-{year}-{month:02d}-{day:02d}.zip"
        return f"{_BASE_URL_DAILY}/{binance_symbol}/{filename}"

    async def _process_month_daily(
        self,
        session: aiohttp.ClientSession,
        binance_symbol: str,
        year: int,
        month: int,
        *,
        verify_checksum: bool = True,
    ) -> pd.DataFrame | None:
        """월별 아카이브 미존재 시 daily archive로 보완.

        해당 월의 모든 날에 대해 daily ZIP을 다운로드하고 병합한다.

        Returns:
            12H bar 피처 DataFrame. 데이터 없으면 None.
        """
        # 미래 날짜 제외
        now = datetime.now(tz=UTC)
        days_in_month = calendar.monthrange(year, month)[1]
        all_trades: list[pd.DataFrame] = []

        for day in range(1, days_in_month + 1):
            # 미래 날짜 스킵
            if datetime(year, month, day, tzinfo=UTC) > now:
                break

            url = self._build_daily_url(binance_symbol, year, month, day)
            try:
                zip_data = await self._download_zip(session, url)
            except (aiohttp.ClientError, ValueError):
                continue

            if verify_checksum:
                with contextlib.suppress(Exception):
                    await self._verify_checksum(session, f"{url}.CHECKSUM", zip_data)

            trades_df = self._parse_zip(zip_data)
            if not trades_df.empty:
                all_trades.append(trades_df)

        if not all_trades:
            logger.debug(f"No daily data for {binance_symbol} {year}-{month:02d}")
            return None

        combined_trades = pd.concat(all_trades).sort_index()
        logger.info(
            "Daily fallback: {} {}-{:02d} ({} trades from {} days)",
            binance_symbol,
            year,
            month,
            len(combined_trades),
            len(all_trades),
        )
        return self._compute_bars(combined_trades)

    async def _download_zip(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> bytes:
        """ZIP 파일 다운로드.

        Raises:
            ValueError: HTTP 404 등 에러
        """
        async with session.get(url) as resp:
            if resp.status != 200:  # noqa: PLR2004
                msg = f"HTTP {resp.status} for {url}"
                raise ValueError(msg)
            return await resp.read()

    async def _verify_checksum(
        self,
        session: aiohttp.ClientSession,
        checksum_url: str,
        zip_data: bytes,
    ) -> None:
        """CHECKSUM 파일로 SHA256 검증.

        Raises:
            ValueError: 체크섬 불일치
        """
        async with session.get(checksum_url) as resp:
            if resp.status != 200:  # noqa: PLR2004
                msg = f"Checksum file not found: {checksum_url}"
                raise ValueError(msg)
            checksum_text = await resp.text()

        # 형식: "<sha256>  <filename>"
        expected_hash = checksum_text.strip().split()[0].lower()
        actual_hash = hashlib.sha256(zip_data).hexdigest().lower()

        if actual_hash != expected_hash:
            msg = f"Checksum mismatch: expected {expected_hash}, got {actual_hash}"
            raise ValueError(msg)

    def _parse_zip(self, zip_data: bytes) -> pd.DataFrame:
        """ZIP → CSV 파싱.

        Returns:
            파싱된 aggTrades DataFrame.
        """
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return pd.DataFrame()

            with zf.open(csv_names[0]) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    names=_CSV_COLUMNS,
                    dtype=_CSV_DTYPES,  # type: ignore[arg-type]
                    header=0,  # 첫 행이 헤더일 수 있음 — 스킵
                )

        # timestamp 변환
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        return df

    def _compute_bars(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """aggTrades → 12H bar 피처 계산.

        Args:
            trades_df: timestamp-indexed aggTrades DataFrame

        Returns:
            12H bar-level 피처 DataFrame
        """
        # 12H 기준으로 그룹화
        groups = trades_df.resample(_BAR_FREQ)

        rows: list[dict[str, float]] = []
        indices: list[pd.Timestamp] = []

        for bar_start, group_df in groups:
            if group_df.empty:
                continue
            features = compute_bar_features(
                group_df,
                bar_hours=_BAR_HOURS,
            )
            rows.append(features)
            indices.append(bar_start)  # type: ignore[arg-type]

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows, index=pd.DatetimeIndex(indices, name="timestamp"))
        return result
