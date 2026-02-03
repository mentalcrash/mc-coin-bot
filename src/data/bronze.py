"""Bronze layer storage for raw OHLCV data.

This module implements the Bronze layer of the Medallion Architecture.
Data is stored exactly as received from the API without any transformation.

Features:
    - No data transformation (preserves original values)
    - Parquet format with zstd compression
    - Metadata preservation (fetched_at, exchange, symbol)
    - Path pattern: data/bronze/{SYMBOL}/{year}.parquet

Rules Applied:
    - #12 Data Engineering: Parquet, zstd compression
    - Medallion Architecture: Bronze = raw data
"""

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import StorageError
from src.models.ohlcv import OHLCVBatch


class BronzeStorage:
    """Bronze 계층 저장소 (원본 데이터 보존).

    API에서 수집한 데이터를 변형 없이 Parquet로 저장합니다.
    데이터 복구나 재처리가 필요할 때 Bronze 데이터를 사용합니다.

    Attributes:
        settings: 설정 객체

    Example:
        >>> storage = BronzeStorage()
        >>> path = storage.save(batch)
        >>> print(f"Saved to {path}")
        Saved to data/bronze/BTC_USDT/2025.parquet
    """

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        """BronzeStorage 초기화.

        Args:
            settings: 설정 객체 (None이면 기본 설정 사용)
        """
        self.settings = settings or get_settings()

    def _ensure_directory(self, path: Path) -> None:
        """디렉토리 생성 (존재하지 않으면).

        Args:
            path: 파일 경로 (부모 디렉토리 생성)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

    def _batch_to_dataframe(self, batch: OHLCVBatch) -> pd.DataFrame:
        """OHLCVBatch를 DataFrame으로 변환 (변형 없음).

        Args:
            batch: OHLCVBatch 객체

        Returns:
            pandas DataFrame
        """
        # Pydantic model_dump()로 직렬화
        records = [candle.model_dump() for candle in batch.candles]
        df = pd.DataFrame(records)

        # timestamp를 인덱스로 설정 (UTC timezone 유지)
        if not df.empty:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()

        return df

    def save(self, batch: OHLCVBatch, year: int | None = None) -> Path:
        """Bronze 데이터 저장 (변형 없음).

        Args:
            batch: OHLCVBatch 객체
            year: 저장 연도 (None이면 첫 캔들 기준)

        Returns:
            저장된 파일 경로

        Raises:
            StorageError: 저장 실패 시
            ValueError: 비어있는 배치
        """
        if batch.is_empty:
            msg = "Cannot save empty batch"
            raise ValueError(msg)

        # 연도 결정
        if year is None:
            year = batch.candles[0].timestamp.year

        # 경로 생성
        path = self.settings.get_bronze_path(batch.symbol, year)
        self._ensure_directory(path)

        try:
            # DataFrame 변환 (변형 없음)
            df = self._batch_to_dataframe(batch)

            # Parquet 저장 (zstd 압축)
            df.to_parquet(
                path,
                compression="zstd",
                index=True,
            )

            logger.info(
                f"Bronze data saved: {path}",
                extra={
                    "path": str(path),
                    "symbol": batch.symbol,
                    "year": year,
                    "candles": len(df),
                    "size_bytes": path.stat().st_size,
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to save Bronze data to {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return path

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """Bronze 데이터 로드.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            year: 연도

        Returns:
            pandas DataFrame

        Raises:
            StorageError: 파일이 없거나 로드 실패 시
        """
        path = self.settings.get_bronze_path(symbol, year)

        if not path.exists():
            raise StorageError(
                f"Bronze file not found: {path}",
                context={"path": str(path), "symbol": symbol, "year": year},
            )

        try:
            df = pd.read_parquet(path)

            # 인덱스가 datetime이 아니면 변환
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)

            logger.debug(
                f"Bronze data loaded: {path}",
                extra={
                    "path": str(path),
                    "rows": len(df),
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to load Bronze data from {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return df

    def exists(self, symbol: str, year: int) -> bool:
        """Bronze 파일 존재 여부 확인.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            파일이 존재하면 True
        """
        path = self.settings.get_bronze_path(symbol, year)
        return path.exists()

    def get_info(self, symbol: str, year: int) -> dict[str, Any] | None:
        """Bronze 파일 정보 반환.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            파일 정보 딕셔너리 또는 None
        """
        path = self.settings.get_bronze_path(symbol, year)

        if not path.exists():
            return None

        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
        }

    def append(self, batch: OHLCVBatch, year: int | None = None) -> Path:
        """기존 Bronze 데이터에 추가 (중복 제거).

        기존 파일이 있으면 새 데이터를 추가하고,
        없으면 새로 생성합니다.

        Args:
            batch: OHLCVBatch 객체
            year: 저장 연도

        Returns:
            저장된 파일 경로
        """
        if batch.is_empty:
            msg = "Cannot append empty batch"
            raise ValueError(msg)

        if year is None:
            year = batch.candles[0].timestamp.year

        path = self.settings.get_bronze_path(batch.symbol, year)

        # 기존 파일이 없으면 새로 저장
        if not path.exists():
            return self.save(batch, year)

        try:
            # 기존 데이터 로드
            existing_df = self.load(batch.symbol, year)

            # 새 데이터 변환
            new_df = self._batch_to_dataframe(batch)

            # 병합 및 중복 제거
            combined_df = cast("pd.DataFrame", pd.concat([existing_df, new_df]))
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            combined_df = combined_df.sort_index()

            # 저장
            combined_df.to_parquet(path, compression="zstd", index=True)

            logger.info(
                f"Bronze data appended: {path}",
                extra={
                    "path": str(path),
                    "existing_rows": len(existing_df),
                    "new_rows": len(new_df),
                    "total_rows": len(combined_df),
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to append Bronze data to {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return path
