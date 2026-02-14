"""Bronze/Silver storage for derivatives data (Funding Rate, OI, LS Ratio, Taker Ratio).

기존 bronze.py / silver.py 패턴 미러링.
경로: data/{bronze,silver}/{SYMBOL}/{YEAR}_deriv.parquet

Rules Applied:
    - #12 Data Engineering: Parquet, zstd compression
    - Medallion Architecture: Bronze = raw, Silver = cleaned
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import DataIntegrityError, StorageError

if TYPE_CHECKING:
    from pathlib import Path

    from src.models.derivatives import DerivativesBatch

# Bronze DataFrame 컬럼 스키마
DERIV_COLUMNS = [
    "funding_rate",
    "mark_price",
    "open_interest",
    "oi_value",
    "ls_ratio",
    "long_pct",
    "short_pct",
    "taker_ratio",
    "taker_buy_vol",
    "taker_sell_vol",
]


def batch_to_dataframe(batch: DerivativesBatch) -> pd.DataFrame:
    """DerivativesBatch를 단일 DataFrame으로 변환.

    서로 다른 granularity의 데이터를 공통 DatetimeIndex(union)로 정렬합니다.

    Args:
        batch: DerivativesBatch 객체

    Returns:
        DatetimeIndex 기반 DataFrame (DERIV_COLUMNS)
    """
    frames: list[pd.DataFrame] = []

    # Funding Rate (8h 간격)
    if batch.funding_rates:
        records = [
            {
                "timestamp": r.timestamp,
                "funding_rate": float(r.funding_rate),
                "mark_price": float(r.mark_price),
            }
            for r in batch.funding_rates
        ]
        df_fr = pd.DataFrame(records).set_index("timestamp")
        df_fr.index = pd.to_datetime(df_fr.index, utc=True)
        frames.append(df_fr)

    # Open Interest (1h 간격)
    if batch.open_interest:
        records = [
            {
                "timestamp": r.timestamp,
                "open_interest": float(r.sum_open_interest),
                "oi_value": float(r.sum_open_interest_value),
            }
            for r in batch.open_interest
        ]
        df_oi = pd.DataFrame(records).set_index("timestamp")
        df_oi.index = pd.to_datetime(df_oi.index, utc=True)
        frames.append(df_oi)

    # Long/Short Ratio (1h 간격)
    if batch.long_short_ratios:
        records = [
            {
                "timestamp": r.timestamp,
                "ls_ratio": float(r.long_short_ratio),
                "long_pct": float(r.long_account),
                "short_pct": float(r.short_account),
            }
            for r in batch.long_short_ratios
        ]
        df_ls = pd.DataFrame(records).set_index("timestamp")
        df_ls.index = pd.to_datetime(df_ls.index, utc=True)
        frames.append(df_ls)

    # Taker Ratio (1h 간격)
    if batch.taker_ratios:
        records = [
            {
                "timestamp": r.timestamp,
                "taker_ratio": float(r.buy_sell_ratio),
                "taker_buy_vol": float(r.buy_vol),
                "taker_sell_vol": float(r.sell_vol),
            }
            for r in batch.taker_ratios
        ]
        df_tk = pd.DataFrame(records).set_index("timestamp")
        df_tk.index = pd.to_datetime(df_tk.index, utc=True)
        frames.append(df_tk)

    if not frames:
        return pd.DataFrame(columns=pd.Index(DERIV_COLUMNS))

    # union join으로 모든 시점 보존
    combined = frames[0]
    for df in frames[1:]:
        combined = combined.join(df, how="outer")

    combined = combined.sort_index()

    # 누락 컬럼 보충
    for col in DERIV_COLUMNS:
        if col not in combined.columns:
            combined[col] = float("nan")

    result: pd.DataFrame = combined.loc[:, DERIV_COLUMNS]
    return result


class DerivativesBronzeStorage:
    """Bronze 계층 저장소 — 파생상품 데이터.

    변형 없이 원본 그대로 저장합니다.

    Example:
        >>> storage = DerivativesBronzeStorage()
        >>> path = storage.save(batch, 2024)
    """

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def save(self, batch: DerivativesBatch, year: int) -> Path:
        """Bronze 데이터 저장.

        Args:
            batch: DerivativesBatch 객체
            year: 저장 연도

        Returns:
            저장된 파일 경로

        Raises:
            ValueError: 비어있는 배치
            StorageError: 저장 실패
        """
        if batch.is_empty:
            msg = "Cannot save empty derivatives batch"
            raise ValueError(msg)

        path = self.settings.get_bronze_deriv_path(batch.symbol, year)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = batch_to_dataframe(batch)
            df.to_parquet(path, compression="zstd", index=True)
            logger.info(
                f"Bronze derivatives saved: {path}",
                extra={
                    "path": str(path),
                    "symbol": batch.symbol,
                    "year": year,
                    "rows": len(df),
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to save Bronze derivatives to {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return path

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """Bronze 데이터 로드.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            pandas DataFrame

        Raises:
            StorageError: 파일 없음 또는 로드 실패
        """
        path = self.settings.get_bronze_deriv_path(symbol, year)
        if not path.exists():
            raise StorageError(
                f"Bronze derivatives file not found: {path}",
                context={"path": str(path), "symbol": symbol, "year": year},
            )
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            logger.debug(
                f"Bronze derivatives loaded: {path}",
                extra={"path": str(path), "rows": len(df)},
            )
        except Exception as e:
            raise StorageError(
                f"Failed to load Bronze derivatives from {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return df

    def exists(self, symbol: str, year: int) -> bool:
        """Bronze 파일 존재 여부."""
        return self.settings.get_bronze_deriv_path(symbol, year).exists()

    def append(self, batch: DerivativesBatch, year: int) -> Path:
        """기존 Bronze 데이터에 추가 (중복 제거).

        Args:
            batch: DerivativesBatch 객체
            year: 저장 연도

        Returns:
            저장된 파일 경로
        """
        if batch.is_empty:
            msg = "Cannot append empty derivatives batch"
            raise ValueError(msg)

        path = self.settings.get_bronze_deriv_path(batch.symbol, year)
        if not path.exists():
            return self.save(batch, year)

        try:
            existing_df = self.load(batch.symbol, year)
            new_df = batch_to_dataframe(batch)
            combined = pd.concat([existing_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined.to_parquet(path, compression="zstd", index=True)
            logger.info(
                f"Bronze derivatives appended: {path}",
                extra={
                    "path": str(path),
                    "existing_rows": len(existing_df),
                    "new_rows": len(new_df),
                    "total_rows": len(combined),
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to append Bronze derivatives to {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return path

    def get_info(self, symbol: str, year: int) -> dict[str, Any] | None:
        """Bronze 파일 정보."""
        path = self.settings.get_bronze_deriv_path(symbol, year)
        if not path.exists():
            return None
        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        }


@dataclass(frozen=True)
class DerivGapReport:
    """파생상품 데이터 갭 리포트.

    Attributes:
        total_rows: 전체 행 수
        nan_counts: 컬럼별 NaN 수
        filled_counts: 컬럼별 채워진 NaN 수
        first_timestamp: 첫 행 시각
        last_timestamp: 마지막 행 시각
    """

    total_rows: int
    nan_counts: dict[str, int]
    filled_counts: dict[str, int]
    first_timestamp: datetime | None
    last_timestamp: datetime | None


class DerivativesSilverProcessor:
    """Silver 계층 프로세서 — 파생상품 데이터.

    Bronze 데이터를 forward-fill로 결측치를 채우고 저장합니다.
    OHLCV와 달리 완전 인덱스 생성 없이 존재하는 데이터의 NaN만 채웁니다.

    Example:
        >>> processor = DerivativesSilverProcessor()
        >>> path = processor.process("BTC/USDT", 2024)
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        bronze_storage: DerivativesBronzeStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bronze_storage = bronze_storage or DerivativesBronzeStorage(self.settings)

    def analyze_gaps(self, symbol: str, year: int) -> DerivGapReport:
        """Bronze 데이터 갭 분석.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            DerivGapReport 객체
        """
        df = self.bronze_storage.load(symbol, year)
        nan_counts: dict[str, int] = {}
        for col in DERIV_COLUMNS:
            if col in df.columns:
                nan_counts[col] = int(df[col].isna().sum())

        first_ts: datetime | None = None
        last_ts: datetime | None = None
        if not df.empty:
            ts_min: pd.Timestamp = df.index.min()  # type: ignore[assignment]
            ts_max: pd.Timestamp = df.index.max()  # type: ignore[assignment]
            first_ts = ts_min.to_pydatetime()
            last_ts = ts_max.to_pydatetime()
            if first_ts.tzinfo is None:
                first_ts = first_ts.replace(tzinfo=UTC)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=UTC)

        return DerivGapReport(
            total_rows=len(df),
            nan_counts=nan_counts,
            filled_counts={},  # process() 후 설정
            first_timestamp=first_ts,
            last_timestamp=last_ts,
        )

    def process(self, symbol: str, year: int, *, validate: bool = True) -> Path:
        """Bronze → Silver 변환 (forward-fill 적용).

        Args:
            symbol: 거래 심볼
            year: 연도
            validate: 저장 전 검증 여부

        Returns:
            저장된 Silver 파일 경로
        """
        logger.info(
            f"Processing derivatives Bronze → Silver for {symbol} {year}",
            extra={"symbol": symbol, "year": year},
        )

        df = self.bronze_storage.load(symbol, year)

        # NaN forward-fill
        nan_before: dict[str, int] = {}
        for col in DERIV_COLUMNS:
            if col in df.columns:
                nan_before[col] = int(df[col].isna().sum())

        df_filled = df.ffill()

        nan_after: dict[str, int] = {}
        for col in DERIV_COLUMNS:
            if col in df_filled.columns:
                nan_after[col] = int(df_filled[col].isna().sum())

        filled_counts = {
            col: nan_before.get(col, 0) - nan_after.get(col, 0) for col in DERIV_COLUMNS
        }

        if validate:
            self._validate_data(df_filled, symbol, year)

        silver_path = self.settings.get_silver_deriv_path(symbol, year)
        silver_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df_filled.to_parquet(silver_path, compression="zstd", index=True)
            logger.info(
                f"Silver derivatives saved: {silver_path}",
                extra={
                    "path": str(silver_path),
                    "symbol": symbol,
                    "year": year,
                    "rows": len(df_filled),
                    "filled": sum(filled_counts.values()),
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to save Silver derivatives to {silver_path}",
                context={"path": str(silver_path), "error": str(e)},
            ) from e
        else:
            return silver_path

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """Silver 데이터 로드.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            pandas DataFrame
        """
        path = self.settings.get_silver_deriv_path(symbol, year)
        if not path.exists():
            raise StorageError(
                f"Silver derivatives file not found: {path}",
                context={"path": str(path), "symbol": symbol, "year": year},
            )
        try:
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            logger.debug(
                f"Silver derivatives loaded: {path}",
                extra={"path": str(path), "rows": len(df)},
            )
        except Exception as e:
            raise StorageError(
                f"Failed to load Silver derivatives from {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return df

    def exists(self, symbol: str, year: int) -> bool:
        """Silver 파일 존재 여부."""
        return self.settings.get_silver_deriv_path(symbol, year).exists()

    def _validate_data(self, df: pd.DataFrame, symbol: str, year: int) -> None:
        """데이터 유효성 검증.

        forward-fill 후 첫 행 NaN은 허용 (backward-fill 미적용).
        음수 값은 경고만 발행.
        """
        errors: list[str] = []

        if df.empty:
            errors.append("DataFrame is empty")

        # 가격/수량 컬럼 음수 검사 (경고만)
        positive_cols = [
            "mark_price",
            "open_interest",
            "oi_value",
            "taker_buy_vol",
            "taker_sell_vol",
        ]
        for col in positive_cols:
            if col in df.columns:
                negative: int = (df[col].dropna() < 0).sum()  # type: ignore[assignment]
                if negative > 0:
                    logger.warning(
                        f"Found {negative} negative {col} values",
                        extra={"symbol": symbol, "year": year, "column": col},
                    )

        if errors:
            raise DataIntegrityError(
                f"Derivatives data validation failed for {symbol} {year}",
                context={"symbol": symbol, "year": year, "errors": errors},
            )
