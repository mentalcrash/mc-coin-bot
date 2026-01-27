"""Data collector for Binance candle data."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.binance_client import BinanceClient
from src.data.schemas import RawBinanceKline
from src.data.storage import (
    fill_missing_candles,
    get_parquet_path,
    save_klines_to_parquet,
)


def get_year_timestamps(year: int) -> tuple[int, int]:
    """연도의 시작/종료 타임스탬프(밀리초) 반환.

    Args:
        year: 연도 (예: 2024)

    Returns:
        (시작 타임스탬프, 종료 타임스탬프) 튜플
    """
    start = datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC)
    end = datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    return start_ms, end_ms


async def collect_symbol_year(
    client: BinanceClient,
    symbol: str,
    year: int,
    data_dir: Path | None = None,
    delay_ms: int = 25,
    fill_gaps: bool = True,
) -> Path:
    """특정 심볼의 1년치 캔들 데이터 수집.

    Args:
        client: BinanceClient 인스턴스
        symbol: 심볼 (예: 'BTCUSDT')
        year: 연도 (예: 2024)
        data_dir: 데이터 저장 디렉토리
        delay_ms: 요청 간 딜레이 (밀리초)
        fill_gaps: 빈 데이터 채우기 여부

    Returns:
        저장된 Parquet 파일 경로
    """
    logger.info(f"Collecting {symbol} {year}...")

    start_ms, end_ms = get_year_timestamps(year)

    # 현재 시간보다 미래인 경우 현재 시간까지만 수집
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    if end_ms > now_ms:
        end_ms = now_ms
        logger.info(f"Adjusting end time to current time for {year}")

    # 페이지네이션으로 데이터 수집
    klines = await client.fetch_klines_paginated(
        symbol=symbol,
        start_time=start_ms,
        end_time=end_ms,
        interval="1m",
        delay_ms=delay_ms,
    )

    if not klines:
        logger.warning(f"No data collected for {symbol} {year}")
        return get_parquet_path(symbol, year, data_dir)

    # 빈 데이터 채우기
    if fill_gaps:
        df = pd.DataFrame([k.to_dict() for k in klines])
        df = fill_missing_candles(df, year)

        # DataFrame을 다시 RawBinanceKline 리스트로 변환
        klines = [
            RawBinanceKline(
                open_time=row["open_time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                close_time=row["close_time"],
                quote_volume=row["quote_volume"],
                trades=int(row["trades"]),
                taker_buy_volume=row["taker_buy_volume"],
                taker_buy_quote_volume=row["taker_buy_quote_volume"],
            )
            for _, row in df.iterrows()
        ]

    # Parquet 저장
    return save_klines_to_parquet(klines, symbol, year, data_dir)


async def collect_symbols(
    symbols: list[str],
    years: list[int],
    data_dir: Path | None = None,
    delay_ms: int = 25,
    fill_gaps: bool = True,
    skip_existing: bool = True,
) -> dict[str, list[Path]]:
    """여러 심볼의 캔들 데이터 수집.

    Args:
        symbols: 심볼 리스트
        years: 연도 리스트
        data_dir: 데이터 저장 디렉토리
        delay_ms: 요청 간 딜레이 (밀리초)
        fill_gaps: 빈 데이터 채우기 여부
        skip_existing: 이미 존재하는 파일 건너뛰기

    Returns:
        심볼별 저장된 파일 경로 딕셔너리
    """
    results: dict[str, list[Path]] = {}

    total_tasks = len(symbols) * len(years)
    completed = 0

    async with BinanceClient(rate_limit=True) as client:
        for symbol in symbols:
            results[symbol] = []

            for year in years:
                # 이미 존재하는 파일 건너뛰기
                if skip_existing:
                    existing_path = get_parquet_path(symbol, year, data_dir)
                    if existing_path.exists():
                        logger.info(f"Skipping {symbol} {year} (already exists)")
                        results[symbol].append(existing_path)
                        completed += 1
                        continue

                try:
                    file_path = await collect_symbol_year(
                        client=client,
                        symbol=symbol,
                        year=year,
                        data_dir=data_dir,
                        delay_ms=delay_ms,
                        fill_gaps=fill_gaps,
                    )
                    results[symbol].append(file_path)

                except Exception as e:
                    logger.error(f"Error collecting {symbol} {year}: {e}")

                completed += 1
                logger.info(
                    f"Progress: {completed}/{total_tasks} ({completed / total_tasks * 100:.1f}%)"
                )

    return results


async def collect_top_symbols(
    top_n: int = 100,
    years: list[int] | None = None,
    data_dir: Path | None = None,
    delay_ms: int = 25,
    fill_gaps: bool = True,
    skip_existing: bool = True,
) -> dict[str, list[Path]]:
    """거래대금 기준 상위 N개 심볼의 캔들 데이터 수집.

    Args:
        top_n: 상위 N개 심볼 수
        years: 연도 리스트 (기본값: [2023, 2024, 2025])
        data_dir: 데이터 저장 디렉토리
        delay_ms: 요청 간 딜레이 (밀리초)
        fill_gaps: 빈 데이터 채우기 여부
        skip_existing: 이미 존재하는 파일 건너뛰기

    Returns:
        심볼별 저장된 파일 경로 딕셔너리
    """
    if years is None:
        years = [2023, 2024, 2025]

    logger.info(f"Starting collection of top {top_n} symbols for years {years}")

    # 상위 심볼 목록 조회
    async with BinanceClient(rate_limit=True) as client:
        symbols = await client.get_top_usdt_tickers(top_n)

    logger.info(f"Collecting data for {len(symbols)} symbols")

    # 데이터 수집
    results = await collect_symbols(
        symbols=symbols,
        years=years,
        data_dir=data_dir,
        delay_ms=delay_ms,
        fill_gaps=fill_gaps,
        skip_existing=skip_existing,
    )

    # 수집 완료 요약
    total_files = sum(len(files) for files in results.values())
    logger.info(f"Collection completed: {total_files} files saved")

    return results


def run_collection(
    symbols: list[str] | None = None,
    top_n: int | None = None,
    years: list[int] | None = None,
    data_dir: Path | None = None,
    delay_ms: int = 25,
    fill_gaps: bool = True,
    skip_existing: bool = True,
) -> dict[str, list[Path]]:
    """동기 방식으로 데이터 수집 실행.

    Args:
        symbols: 심볼 리스트 (None이면 top_n 사용)
        top_n: 상위 N개 심볼 수 (symbols가 None일 때 사용)
        years: 연도 리스트
        data_dir: 데이터 저장 디렉토리
        delay_ms: 요청 간 딜레이 (밀리초)
        fill_gaps: 빈 데이터 채우기 여부
        skip_existing: 이미 존재하는 파일 건너뛰기

    Returns:
        심볼별 저장된 파일 경로 딕셔너리
    """
    if years is None:
        years = [2023, 2024, 2025]

    if symbols:
        return asyncio.run(
            collect_symbols(
                symbols=symbols,
                years=years,
                data_dir=data_dir,
                delay_ms=delay_ms,
                fill_gaps=fill_gaps,
                skip_existing=skip_existing,
            )
        )

    return asyncio.run(
        collect_top_symbols(
            top_n=top_n or 100,
            years=years,
            data_dir=data_dir,
            delay_ms=delay_ms,
            fill_gaps=fill_gaps,
            skip_existing=skip_existing,
        )
    )
