#!/usr/bin/env python3
"""MATIC → POL 데이터 스티칭 스크립트.

Binance는 2024-09-10에 MATIC/USDT → POL/USDT 심볼을 전환했습니다.
이 스크립트는 두 심볼의 데이터를 수집하고 결합하여 POL/USDT로 통합합니다.

1단계: MATIC/USDT OHLCV 2020-2024 수집 → Bronze → Silver
2단계: POL/USDT OHLCV 2024-2026 수집 → Bronze → Silver
3단계: 2024년 파일 결합 (MATIC ~Sep 9 + POL Sep 13~)
4단계: 결과를 POL_USDT 디렉토리에 저장
5단계: Funding Rate도 동일 패턴으로 결합

3일 갭(Sep 10-12)은 Silver forward-fill로 자연 처리됩니다.

Usage:
    uv run python scripts/stitch_matic_pol.py
    uv run python scripts/stitch_matic_pol.py --skip-existing
    uv run python scripts/stitch_matic_pol.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Project root를 path에 추가
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.settings import get_settings
from src.core.logger import setup_logger
from src.data.bronze import BronzeStorage
from src.data.derivatives_fetcher import DerivativesFetcher
from src.data.derivatives_storage import (
    DerivativesBronzeStorage,
    DerivativesSilverProcessor,
)
from src.data.fetcher import DataFetcher
from src.data.silver import SilverProcessor
from src.exchange.binance_client import BinanceClient
from src.exchange.binance_futures_client import BinanceFuturesClient

# 전환일: 2024-09-10 00:00 UTC (이전=MATIC, 이후=POL)
_CUTOVER_TS = pd.Timestamp("2024-09-10", tz="UTC")

OLD_SYMBOL = "MATIC/USDT"
NEW_SYMBOL = "POL/USDT"

# MATIC: 2020-2024, POL: 2024-2026
MATIC_YEARS = list(range(2020, 2025))  # 2020, 2021, 2022, 2023, 2024
POL_YEARS = list(range(2024, 2027))  # 2024, 2025, 2026


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


async def _fetch_ohlcv(symbol: str, years: list[int], skip_existing: bool) -> None:
    """OHLCV Bronze 수집."""
    settings = get_settings()
    settings.ensure_directories()
    bronze = BronzeStorage(settings)

    async with BinanceClient(settings) as client:
        fetcher = DataFetcher(settings, client=client)
        for year in years:
            if skip_existing and bronze.exists(symbol, year):
                logger.info("OHLCV skip (exists): {} {}", symbol, year)
                continue
            logger.info("Fetching OHLCV: {} {}", symbol, year)
            batch = await fetcher.fetch_year(symbol, year, show_progress=False)
            if batch.is_empty:
                logger.warning("Empty OHLCV: {} {}", symbol, year)
                continue
            bronze.save(batch, year)
            logger.info("Saved OHLCV Bronze: {} {} ({} candles)", symbol, year, batch.candle_count)
            await asyncio.sleep(0.5)


async def _fetch_fr(symbol: str, years: list[int], skip_existing: bool) -> None:
    """Funding Rate Bronze 수집 (fr_only)."""
    settings = get_settings()
    settings.ensure_directories()
    deriv_bronze = DerivativesBronzeStorage(settings)

    async with BinanceFuturesClient(settings) as client:
        fetcher = DerivativesFetcher(client=client, settings=settings)
        for year in years:
            if skip_existing and deriv_bronze.exists(symbol, year):
                logger.info("FR skip (exists): {} {}", symbol, year)
                continue
            logger.info("Fetching FR: {} {}", symbol, year)
            batch = await fetcher.fetch_year(symbol, year, fr_only=True)
            if batch.is_empty:
                logger.warning("Empty FR: {} {}", symbol, year)
                continue
            deriv_bronze.save(batch, year)
            logger.info("Saved FR Bronze: {} {} ({} records)", symbol, year, len(batch.funding_rates))
            await asyncio.sleep(0.5)


def _stitch_ohlcv_2024() -> None:
    """2024년 OHLCV: MATIC(~Sep 9) + POL(Sep 13~) 결합 → POL_USDT/2024.parquet."""
    settings = get_settings()
    bronze = BronzeStorage(settings)

    matic_path = settings.get_bronze_path(OLD_SYMBOL, 2024)
    pol_path = settings.get_bronze_path(NEW_SYMBOL, 2024)

    if not matic_path.exists():
        logger.warning("MATIC 2024 Bronze not found: {}", matic_path)
        return
    if not pol_path.exists():
        logger.warning("POL 2024 Bronze not found: {}", pol_path)
        return

    df_matic = bronze.load(OLD_SYMBOL, 2024)
    df_pol = bronze.load(NEW_SYMBOL, 2024)

    # MATIC: 전환일 이전만, POL: 전환일 이후만
    df_matic_pre = df_matic[df_matic.index < _CUTOVER_TS]
    df_pol_post = df_pol[df_pol.index >= _CUTOVER_TS]

    logger.info(
        "Stitching 2024 OHLCV: MATIC {} rows + POL {} rows",
        len(df_matic_pre),
        len(df_pol_post),
    )

    combined = pd.concat([df_matic_pre, df_pol_post]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # POL_USDT 디렉토리에 저장
    out_path = settings.get_bronze_path(NEW_SYMBOL, 2024)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, compression="zstd")
    logger.info("Stitched 2024 OHLCV saved: {} ({} rows)", out_path, len(combined))


def _stitch_fr_2024() -> None:
    """2024년 FR: MATIC(~Sep 9) + POL(Sep 13~) 결합."""
    settings = get_settings()
    deriv_bronze = DerivativesBronzeStorage(settings)

    matic_path = settings.get_bronze_deriv_path(OLD_SYMBOL, 2024)
    pol_path = settings.get_bronze_deriv_path(NEW_SYMBOL, 2024)

    if not matic_path.exists():
        logger.warning("MATIC 2024 FR Bronze not found: {}", matic_path)
        return
    if not pol_path.exists():
        logger.warning("POL 2024 FR Bronze not found: {}", pol_path)
        return

    df_matic = deriv_bronze.load(OLD_SYMBOL, 2024)
    df_pol = deriv_bronze.load(NEW_SYMBOL, 2024)

    df_matic_pre = df_matic[df_matic.index < _CUTOVER_TS]
    df_pol_post = df_pol[df_pol.index >= _CUTOVER_TS]

    logger.info(
        "Stitching 2024 FR: MATIC {} rows + POL {} rows",
        len(df_matic_pre),
        len(df_pol_post),
    )

    combined = pd.concat([df_matic_pre, df_pol_post]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    out_path = settings.get_bronze_deriv_path(NEW_SYMBOL, 2024)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, compression="zstd")
    logger.info("Stitched 2024 FR saved: {} ({} rows)", out_path, len(combined))


def _copy_matic_as_pol(years: list[int]) -> None:
    """전환 이전 연도(2020-2023)는 MATIC Bronze를 POL 디렉토리로 복사."""
    settings = get_settings()
    for year in years:
        if year >= 2024:
            continue

        # OHLCV
        src = settings.get_bronze_path(OLD_SYMBOL, year)
        dst = settings.get_bronze_path(NEW_SYMBOL, year)
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logger.info("Copied OHLCV Bronze: {} → {}", src, dst)

        # FR
        src_fr = settings.get_bronze_deriv_path(OLD_SYMBOL, year)
        dst_fr = settings.get_bronze_deriv_path(NEW_SYMBOL, year)
        if src_fr.exists() and not dst_fr.exists():
            dst_fr.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_fr, dst_fr)
            logger.info("Copied FR Bronze: {} → {}", src_fr, dst_fr)


def _process_silver() -> None:
    """POL_USDT Bronze → Silver 변환 (전 연도)."""
    settings = get_settings()
    silver = SilverProcessor(settings)
    deriv_silver = DerivativesSilverProcessor(settings)

    all_years = list(range(2020, 2027))

    for year in all_years:
        # OHLCV Silver
        bronze_path = settings.get_bronze_path(NEW_SYMBOL, year)
        if bronze_path.exists():
            try:
                path = silver.process(NEW_SYMBOL, year, validate=True)
                logger.info("Silver OHLCV: {}", path)
            except Exception as e:
                logger.error("Silver OHLCV failed {} {}: {}", NEW_SYMBOL, year, e)

        # FR Silver
        bronze_fr_path = settings.get_bronze_deriv_path(NEW_SYMBOL, year)
        if bronze_fr_path.exists():
            try:
                path = deriv_silver.process(NEW_SYMBOL, year, validate=True)
                logger.info("Silver FR: {}", path)
            except Exception as e:
                logger.error("Silver FR failed {} {}: {}", NEW_SYMBOL, year, e)


async def _run(skip_existing: bool) -> None:
    """전체 스티칭 파이프라인 실행."""
    # 1. MATIC OHLCV 수집 (2020-2024)
    logger.info("=== Step 1/6: MATIC OHLCV 2020-2024 ===")
    await _fetch_ohlcv(OLD_SYMBOL, MATIC_YEARS, skip_existing)

    # 2. POL OHLCV 수집 (2024-2026)
    logger.info("=== Step 2/6: POL OHLCV 2024-2026 ===")
    await _fetch_ohlcv(NEW_SYMBOL, POL_YEARS, skip_existing)

    # 3. MATIC FR 수집 (2020-2024)
    logger.info("=== Step 3/6: MATIC FR 2020-2024 ===")
    await _fetch_fr(OLD_SYMBOL, MATIC_YEARS, skip_existing)

    # 4. POL FR 수집 (2024-2026)
    logger.info("=== Step 4/6: POL FR 2024-2026 ===")
    await _fetch_fr(NEW_SYMBOL, POL_YEARS, skip_existing)

    # 5. 2020-2023: MATIC → POL 복사 + 2024 스티칭
    logger.info("=== Step 5/6: Stitch & Copy ===")
    _copy_matic_as_pol(MATIC_YEARS)
    _stitch_ohlcv_2024()
    _stitch_fr_2024()

    # 6. Silver 처리
    logger.info("=== Step 6/6: Silver Processing ===")
    _process_silver()

    logger.info("=== Stitch Complete ===")


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="MATIC/POL data stitching")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if Bronze exists")
    parser.add_argument("--dry-run", action="store_true", help="Show plan only")
    args = parser.parse_args()

    setup_logger(console_level="INFO")

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("MATIC OHLCV years: {}", MATIC_YEARS)
        logger.info("POL OHLCV years: {}", POL_YEARS)
        logger.info("Cutover date: {}", _CUTOVER_TS)
        logger.info("Steps: fetch MATIC → fetch POL → copy/stitch → Silver")
        return

    asyncio.run(_run(skip_existing=args.skip_existing))


if __name__ == "__main__":
    main()
