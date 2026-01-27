"""Data fetcher with retry logic and progress tracking.

This module provides async data fetching capabilities with built-in
retry logic using tenacity. It fetches OHLCV data from exchanges and
validates through Pydantic schemas before storage.

Features:
    - Exponential backoff retry for transient errors
    - Progress tracking with tqdm
    - Batched fetching for large date ranges
    - Structured logging at every step

Rules Applied:
    - #14 CCXT Trading: Async, retry for NetworkError/RateLimitExceeded
    - #23 Exception Handling: Retry vs Fail-fast
    - #15 Logging: Structured logging with context
"""

import asyncio
from datetime import UTC, datetime

from loguru import logger
from tenacity import (
    RetryCallState,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import NetworkError, RateLimitError
from src.exchange.binance_client import BinanceClient
from src.models.ohlcv import OHLCVBatch, OHLCVCandle

# 1분 = 60,000 ms
MS_PER_MINUTE = 60_000
# 1년 ≈ 525,600분 (365일)
MINUTES_PER_YEAR = 525_600


def _get_year_timestamps(year: int) -> tuple[int, int]:
    """해당 연도의 시작/끝 타임스탬프 반환 (Unix ms).

    Args:
        year: 연도 (예: 2025)

    Returns:
        (시작 타임스탬프, 종료 타임스탬프) Unix ms
    """
    start_dt = datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC)
    end_dt = datetime(year, 12, 31, 23, 59, 0, tzinfo=UTC)
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    return start_ts, end_ts


class DataFetcher:
    """비동기 데이터 페처 with 재시도 로직.

    Binance에서 OHLCV 데이터를 배치로 수집합니다.
    네트워크 오류나 레이트 리밋 발생 시 지수 백오프로 재시도합니다.

    Attributes:
        settings: 설정 객체
        client: BinanceClient 인스턴스 (선택)

    Example:
        >>> fetcher = DataFetcher()
        >>> batch = await fetcher.fetch_year("BTC/USDT", 2025)
        >>> print(f"Fetched {batch.candle_count} candles")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        client: BinanceClient | None = None,
    ) -> None:
        """DataFetcher 초기화.

        Args:
            settings: 설정 객체 (None이면 기본 설정 사용)
            client: BinanceClient 인스턴스 (None이면 새로 생성)
        """
        self.settings = settings or get_settings()
        self._client = client
        self._owned_client = client is None  # 클라이언트를 직접 생성했는지

    def _create_retry_decorator(self):
        """tenacity 재시도 데코레이터 생성."""
        return retry(
            retry=retry_if_exception_type((NetworkError, RateLimitError)),
            wait=wait_exponential(
                multiplier=1,
                min=2,
                max=self.settings.backoff_max,
            ),
            stop=stop_after_attempt(self.settings.max_retries),
            before_sleep=self._log_retry,
            reraise=True,
        )

    def _log_retry(self, retry_state: RetryCallState) -> None:
        """재시도 전 로깅 콜백."""
        outcome = retry_state.outcome
        exception = outcome.exception() if outcome else None
        attempt = retry_state.attempt_number
        next_wait = retry_state.next_action.sleep if retry_state.next_action else 0
        logger.warning(
            f"Retry attempt {attempt}/{self.settings.max_retries}",
            extra={
                "attempt": attempt,
                "max_retries": self.settings.max_retries,
                "error": str(exception) if exception else "unknown",
                "next_wait": next_wait,
            },
        )

    async def _fetch_batch_with_retry(
        self,
        client: BinanceClient,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
    ) -> OHLCVBatch:
        """재시도 로직이 적용된 배치 페칭.

        Args:
            client: BinanceClient 인스턴스
            symbol: 거래 심볼
            timeframe: 타임프레임
            since: 시작 타임스탬프 (Unix ms)
            limit: 최대 캔들 수

        Returns:
            OHLCVBatch 객체

        Raises:
            NetworkError: 최대 재시도 후에도 실패 시
            RateLimitError: 최대 재시도 후에도 실패 시
        """
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        async def _fetch():
            return await client.fetch_ohlcv(symbol, timeframe, since, limit)

        return await _fetch()

    async def _fetch_with_client(  # noqa: PLR0913
        self,
        client: BinanceClient,
        symbol: str,
        start_ts: int,
        end_ts: int,
        timeframe: str,
        pbar: tqdm,  # type: ignore[reportUnknownParameterType]
    ) -> tuple[list[OHLCVCandle], int]:
        """Client를 사용해서 데이터 수집 (내부 헬퍼).

        Args:
            client: BinanceClient 인스턴스
            symbol: 거래 심볼
            start_ts: 시작 타임스탬프
            end_ts: 종료 타임스탬프
            timeframe: 타임프레임
            pbar: 진행률 바

        Returns:
            (캔들 리스트, 배치 수)
        """
        batch_size = self.settings.batch_size
        timeframe_ms = MS_PER_MINUTE

        # 심볼 유효성 검사
        if not client.is_symbol_valid(symbol):
            msg = f"Invalid symbol: {symbol}"
            raise ValueError(msg)

        all_candles: list[OHLCVCandle] = []
        current_ts = start_ts
        batch_count = 0

        while current_ts < end_ts:
            try:
                batch = await self._fetch_batch_with_retry(
                    client,
                    symbol,
                    timeframe,
                    current_ts,
                    batch_size,
                )

                if batch.is_empty:
                    logger.warning(
                        f"Empty batch received at {current_ts}",
                        extra={"timestamp": current_ts},
                    )
                    # 빈 배치면 다음 구간으로 이동
                    current_ts += batch_size * timeframe_ms
                else:
                    all_candles.extend(batch.candles)
                    # 마지막 캔들 다음 시점으로 이동
                    last_ts = int(batch.candles[-1].timestamp.timestamp() * 1000)
                    current_ts = last_ts + timeframe_ms

                    logger.debug(
                        f"Batch {batch_count + 1} completed",
                        extra={
                            "batch": batch_count + 1,
                            "candles_in_batch": batch.candle_count,
                            "total_candles": len(all_candles),
                        },
                    )

                batch_count += 1
                pbar.update(1)

                # API 레이트 리밋 준수를 위한 대기
                base_delay = 0.2
                if batch_count % 100 == 0:
                    await asyncio.sleep(2.0)  # 100배치마다 2초 휴식
                else:
                    await asyncio.sleep(base_delay)

            except RetryError as e:
                logger.error(
                    f"Max retries exceeded at timestamp {current_ts}",
                    extra={"timestamp": current_ts, "error": str(e)},
                )
                raise

        return all_candles, batch_count

    async def fetch_year(
        self,
        symbol: str,
        year: int,
        timeframe: str = "1m",
        show_progress: bool = True,
    ) -> OHLCVBatch:
        """1년치 OHLCV 데이터 수집.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            year: 수집 연도
            timeframe: 타임프레임 (기본: "1m")
            show_progress: 진행률 표시 여부

        Returns:
            1년치 데이터가 담긴 OHLCVBatch

        Raises:
            ValueError: 유효하지 않은 심볼
            NetworkError: 네트워크 오류
            RateLimitError: 레이트 리밋 초과
        """
        start_ts, end_ts = _get_year_timestamps(year)
        batch_size = self.settings.batch_size
        timeframe_ms = MS_PER_MINUTE  # 1분봉 기준

        # 예상 배치 수 계산
        total_minutes = (end_ts - start_ts) // timeframe_ms
        total_batches = (total_minutes // batch_size) + 1

        logger.info(
            f"Starting data fetch for {symbol} year {year}",
            extra={
                "symbol": symbol,
                "year": year,
                "timeframe": timeframe,
                "expected_candles": total_minutes,
                "total_batches": total_batches,
            },
        )

        # 진행률 바 설정
        pbar = tqdm(
            total=total_batches,
            desc=f"Fetching {symbol} {year}",
            unit="batch",
            disable=not show_progress,
        )

        try:
            # Client 재사용: 전달받았으면 그것 사용, 아니면 새로 생성
            if self._client:
                all_candles, batch_count = await self._fetch_with_client(
                    self._client,
                    symbol,
                    start_ts,
                    end_ts,
                    timeframe,
                    pbar,
                )
            else:
                # Backward compatibility: client가 없으면 새로 생성
                async with BinanceClient(self.settings) as client:
                    all_candles, batch_count = await self._fetch_with_client(
                        client,
                        symbol,
                        start_ts,
                        end_ts,
                        timeframe,
                        pbar,
                    )
        finally:
            pbar.close()

        # 결과 배치 생성
        result = OHLCVBatch(
            symbol=symbol,
            timeframe=timeframe,
            exchange="binance",
            candles=tuple(all_candles),
            fetched_at=datetime.now(UTC),
        )

        logger.info(
            f"Data fetch completed for {symbol} year {year}",
            extra={
                "symbol": symbol,
                "year": year,
                "total_candles": result.candle_count,
                "batches_processed": batch_count,
            },
        )

        return result

    async def fetch_range(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        timeframe: str = "1m",
        show_progress: bool = True,
    ) -> OHLCVBatch:
        """특정 기간의 OHLCV 데이터 수집.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            start_ts: 시작 타임스탬프 (Unix ms)
            end_ts: 종료 타임스탬프 (Unix ms)
            timeframe: 타임프레임 (기본: "1m")
            show_progress: 진행률 표시 여부

        Returns:
            기간 데이터가 담긴 OHLCVBatch
        """
        batch_size = self.settings.batch_size
        timeframe_ms = MS_PER_MINUTE

        total_minutes = (end_ts - start_ts) // timeframe_ms
        total_batches = (total_minutes // batch_size) + 1

        logger.info(
            f"Starting range fetch for {symbol}",
            extra={
                "symbol": symbol,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "expected_candles": total_minutes,
            },
        )

        all_candles: list[OHLCVCandle] = []
        current_ts = start_ts

        pbar = tqdm(
            total=total_batches,
            desc=f"Fetching {symbol}",
            unit="batch",
            disable=not show_progress,
        )

        async with BinanceClient(self.settings) as client:
            if not client.is_symbol_valid(symbol):
                msg = f"Invalid symbol: {symbol}"
                raise ValueError(msg)

            while current_ts < end_ts:
                batch = await self._fetch_batch_with_retry(
                    client,
                    symbol,
                    timeframe,
                    current_ts,
                    batch_size,
                )

                if not batch.is_empty:
                    all_candles.extend(batch.candles)
                    last_ts = int(batch.candles[-1].timestamp.timestamp() * 1000)
                    current_ts = last_ts + timeframe_ms
                else:
                    current_ts += batch_size * timeframe_ms

                pbar.update(1)
                await asyncio.sleep(0.1)

        pbar.close()

        return OHLCVBatch(
            symbol=symbol,
            timeframe=timeframe,
            exchange="binance",
            candles=tuple(all_candles),
            fetched_at=datetime.now(UTC),
        )
