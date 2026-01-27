"""Parquet storage utilities for candle data."""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from src.data.schemas import CandleRecord, RawBinanceKline

# 기본 데이터 저장 경로
DEFAULT_DATA_DIR = Path("data/binance/candles")


def get_parquet_path(symbol: str, year: int, data_dir: Path | None = None) -> Path:
    """Parquet 파일 경로 생성.

    Args:
        symbol: 심볼 (예: 'BTCUSDT')
        year: 연도 (예: 2024)
        data_dir: 데이터 디렉토리 (기본값: data/binance/candles)

    Returns:
        Parquet 파일 경로
    """
    base_dir = data_dir or DEFAULT_DATA_DIR
    return base_dir / symbol / f"{symbol}_{year}.parquet"


def save_klines_to_parquet(
    klines: list[RawBinanceKline],
    symbol: str,
    year: int,
    data_dir: Path | None = None,
) -> Path:
    """캔들 데이터를 Parquet 파일로 저장.

    Args:
        klines: RawBinanceKline 리스트
        symbol: 심볼 (예: 'BTCUSDT')
        year: 연도 (예: 2024)
        data_dir: 데이터 디렉토리

    Returns:
        저장된 파일 경로
    """
    if not klines:
        logger.warning(f"No klines to save for {symbol} {year}")
        return get_parquet_path(symbol, year, data_dir)

    # DataFrame 생성
    df = pd.DataFrame([k.to_dict() for k in klines])

    # 파일 경로 생성
    file_path = get_parquet_path(symbol, year, data_dir)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # PyArrow 스키마 정의 (원본 스키마 유지)
    schema = pa.schema(
        [
            ("open_time", pa.int64()),
            ("open", pa.string()),
            ("high", pa.string()),
            ("low", pa.string()),
            ("close", pa.string()),
            ("volume", pa.string()),
            ("close_time", pa.int64()),
            ("quote_volume", pa.string()),
            ("trades", pa.int64()),
            ("taker_buy_volume", pa.string()),
            ("taker_buy_quote_volume", pa.string()),
        ]
    )

    # Parquet로 저장
    table = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(table, file_path, compression="snappy")

    logger.info(f"Saved {len(klines)} candles to {file_path}")
    return file_path


def load_klines_from_parquet(
    symbol: str,
    year: int,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Parquet 파일에서 캔들 데이터 로드.

    Args:
        symbol: 심볼 (예: 'BTCUSDT')
        year: 연도 (예: 2024)
        data_dir: 데이터 디렉토리

    Returns:
        캔들 데이터 DataFrame
    """
    file_path = get_parquet_path(symbol, year, data_dir)

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} candles from {file_path}")
    return df


def load_candles_as_records(
    symbol: str,
    year: int,
    data_dir: Path | None = None,
) -> list[CandleRecord]:
    """Parquet 파일에서 캔들 데이터를 CandleRecord 리스트로 로드.

    Args:
        symbol: 심볼 (예: 'BTCUSDT')
        year: 연도 (예: 2024)
        data_dir: 데이터 디렉토리

    Returns:
        CandleRecord 리스트
    """
    df = load_klines_from_parquet(symbol, year, data_dir)

    if df.empty:
        return []

    records = []
    for _, row in df.iterrows():
        raw = RawBinanceKline(
            open_time=row["open_time"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            close_time=row["close_time"],
            quote_volume=row["quote_volume"],
            trades=row["trades"],
            taker_buy_volume=row["taker_buy_volume"],
            taker_buy_quote_volume=row["taker_buy_quote_volume"],
        )
        records.append(CandleRecord.from_raw_kline(raw))

    return records


def fill_missing_candles(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """빈 캔들 데이터를 이전 값으로 채우기.

    1분 간격으로 연속된 데이터를 보장합니다.
    빈 캔들은 이전 캔들의 close 값으로 OHLC를 채웁니다.

    Args:
        df: 원본 캔들 DataFrame
        year: 연도

    Returns:
        빈 데이터가 채워진 DataFrame
    """
    if df.empty:
        return df

    # 타임스탬프를 datetime으로 변환
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("datetime")

    # 해당 연도의 시작/종료 시간
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year}-12-31 23:59", tz="UTC")

    # 실제 데이터 범위로 제한 (연도 전체가 아닌 데이터가 있는 범위만)
    data_start = df.index.min()
    data_end = df.index.max()

    # 데이터가 있는 범위만 처리
    actual_start = max(start, data_start)
    actual_end = min(end, data_end)

    # 1분 간격으로 모든 타임스탬프 생성
    expected_timestamps = pd.date_range(actual_start, actual_end, freq="1min")

    # 중복 인덱스 제거 (같은 시간에 여러 캔들이 있는 경우)
    df = df[~df.index.duplicated(keep="first")]

    # reindex로 빈 타임스탬프 추가
    df = df.reindex(expected_timestamps)

    # 빈 데이터 수 확인
    missing_count = df["close"].isna().sum()
    if missing_count > 0:
        logger.info(f"Filling {missing_count} missing candles for {year}")

    # close 값을 먼저 forward fill
    df["close"] = df["close"].ffill()

    # open, high, low는 close 값으로 채움 (거래 없음 = 가격 변동 없음)
    for col in ["open", "high", "low"]:
        df[col] = df[col].fillna(df["close"])

    # volume과 trades는 0으로 채움
    df["volume"] = df["volume"].fillna("0")
    df["quote_volume"] = df["quote_volume"].fillna("0")
    df["taker_buy_volume"] = df["taker_buy_volume"].fillna("0")
    df["taker_buy_quote_volume"] = df["taker_buy_quote_volume"].fillna("0")
    df["trades"] = df["trades"].fillna(0).astype(int)

    # open_time과 close_time 재계산
    df["open_time"] = df.index.astype("int64") // 10**6  # datetime to ms
    df["close_time"] = df["open_time"] + 60000 - 1  # 1분봉 종료 시간

    # 인덱스를 일반 컬럼으로 변환하여 반환
    return df.reset_index(drop=True)


def validate_candle_data(
    symbol: str,
    year: int,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """저장된 캔들 데이터의 무결성 검증.

    Args:
        symbol: 심볼 (예: 'BTCUSDT')
        year: 연도 (예: 2024)
        data_dir: 데이터 디렉토리

    Returns:
        검증 결과 딕셔너리
    """
    file_path = get_parquet_path(symbol, year, data_dir)

    result = {
        "symbol": symbol,
        "year": year,
        "file_exists": file_path.exists(),
        "total_candles": 0,
        "expected_candles": 0,
        "missing_candles": 0,
        "duplicate_candles": 0,
        "data_range": None,
        "file_size_mb": 0,
    }

    if not file_path.exists():
        return result

    df = pd.read_parquet(file_path)
    result["total_candles"] = len(df)
    result["file_size_mb"] = round(file_path.stat().st_size / (1024 * 1024), 2)

    if df.empty:
        return result

    # 데이터 범위 확인
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    data_start = df["datetime"].min()
    data_end = df["datetime"].max()
    result["data_range"] = f"{data_start} ~ {data_end}"

    # 예상 캔들 수 계산
    expected = int((data_end - data_start).total_seconds() / 60) + 1
    result["expected_candles"] = expected
    result["missing_candles"] = expected - len(df)

    # 중복 확인
    result["duplicate_candles"] = df["open_time"].duplicated().sum()

    return result


def list_available_data(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """저장된 데이터 목록 조회.

    Args:
        data_dir: 데이터 디렉토리

    Returns:
        저장된 데이터 정보 리스트
    """
    base_dir = data_dir or DEFAULT_DATA_DIR

    if not base_dir.exists():
        return []

    data_list = []
    for symbol_dir in sorted(base_dir.iterdir()):
        if symbol_dir.is_dir():
            for parquet_file in sorted(symbol_dir.glob("*.parquet")):
                # 파일명에서 연도 추출
                parts = parquet_file.stem.split("_")
                if len(parts) >= 2:
                    year = int(parts[-1])
                    symbol = "_".join(parts[:-1])
                    data_list.append(
                        {
                            "symbol": symbol,
                            "year": year,
                            "path": str(parquet_file),
                            "size_mb": round(parquet_file.stat().st_size / (1024 * 1024), 2),
                        }
                    )

    return data_list
