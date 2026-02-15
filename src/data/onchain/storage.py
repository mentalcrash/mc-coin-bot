"""Bronze/Silver storage for on-chain data (DeFiLlama stablecoins, etc.).

Bronze: append-only parquet (zstd compression)
Silver: dedup → sort → UTC enforce → save

Path convention: data/{bronze,silver}/onchain/{source}/{name}.parquet

Rules Applied:
    - #12 Data Engineering: Parquet, zstd compression
    - Medallion Architecture: Bronze = raw, Silver = cleaned
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import DataIntegrityError, StorageError

if TYPE_CHECKING:
    from pathlib import Path


class OnchainBronzeStorage:
    """Bronze 계층 저장소 — on-chain 데이터.

    변형 없이 원본 그대로 저장합니다.
    OHLCV와 달리 daily granularity, source 기반 경로 사용.

    Example:
        >>> storage = OnchainBronzeStorage()
        >>> path = storage.save(df, "defillama", "stablecoin_total")
    """

    def __init__(self, settings: IngestionSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def save(self, df: pd.DataFrame, source: str, name: str) -> Path:
        """Bronze 데이터 저장.

        Args:
            df: 저장할 DataFrame
            source: 데이터 소스 (예: "defillama")
            name: 데이터 이름 (예: "stablecoin_total")

        Returns:
            저장된 파일 경로

        Raises:
            ValueError: 빈 DataFrame
            StorageError: 저장 실패
        """
        if df.empty:
            msg = f"Cannot save empty DataFrame for {source}/{name}"
            raise ValueError(msg)

        path = self.settings.get_onchain_bronze_path(source, name)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(path, compression="zstd", index=True)
            logger.info(
                f"Bronze onchain saved: {path}",
                extra={"path": str(path), "source": source, "name": name, "rows": len(df)},
            )
        except Exception as e:
            raise StorageError(
                f"Failed to save Bronze onchain to {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return path

    def append(self, df: pd.DataFrame, source: str, name: str, dedup_col: str = "date") -> Path:
        """기존 Bronze 데이터에 추가 (중복 제거).

        Args:
            df: 추가할 DataFrame
            source: 데이터 소스
            name: 데이터 이름
            dedup_col: 중복 제거 기준 컬럼

        Returns:
            저장된 파일 경로
        """
        if df.empty:
            msg = f"Cannot append empty DataFrame for {source}/{name}"
            raise ValueError(msg)

        path = self.settings.get_onchain_bronze_path(source, name)
        if not path.exists():
            return self.save(df, source, name)

        try:
            existing_df = self.load(source, name)
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[dedup_col], keep="last")
            combined = combined.sort_values(dedup_col).reset_index(drop=True)
            combined.to_parquet(path, compression="zstd", index=True)
            logger.info(
                f"Bronze onchain appended: {path}",
                extra={
                    "path": str(path),
                    "existing_rows": len(existing_df),
                    "new_rows": len(df),
                    "total_rows": len(combined),
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to append Bronze onchain to {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return path

    def load(self, source: str, name: str) -> pd.DataFrame:
        """Bronze 데이터 로드.

        Args:
            source: 데이터 소스
            name: 데이터 이름

        Returns:
            pandas DataFrame

        Raises:
            StorageError: 파일 없음 또는 로드 실패
        """
        path = self.settings.get_onchain_bronze_path(source, name)
        if not path.exists():
            raise StorageError(
                f"Bronze onchain file not found: {path}",
                context={"path": str(path), "source": source, "name": name},
            )
        try:
            df = pd.read_parquet(path)
            logger.debug(
                f"Bronze onchain loaded: {path}",
                extra={"path": str(path), "rows": len(df)},
            )
        except Exception as e:
            raise StorageError(
                f"Failed to load Bronze onchain from {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return df

    def exists(self, source: str, name: str) -> bool:
        """Bronze 파일 존재 여부."""
        return self.settings.get_onchain_bronze_path(source, name).exists()

    def get_info(self, source: str, name: str) -> dict[str, Any] | None:
        """Bronze 파일 정보."""
        path = self.settings.get_onchain_bronze_path(source, name)
        if not path.exists():
            return None
        stat = path.stat()
        return {
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        }


class OnchainSilverProcessor:
    """Silver 계층 프로세서 — on-chain 데이터.

    Bronze 데이터를 정제합니다: 중복 제거 → 정렬 → UTC enforce → 저장.

    Example:
        >>> processor = OnchainSilverProcessor()
        >>> path = processor.process("defillama", "stablecoin_total")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        bronze_storage: OnchainBronzeStorage | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bronze_storage = bronze_storage or OnchainBronzeStorage(self.settings)

    def process(
        self, source: str, name: str, *, date_col: str = "date", sort_col: str = "date"
    ) -> Path:
        """Bronze → Silver 변환.

        Args:
            source: 데이터 소스
            name: 데이터 이름
            date_col: datetime 컬럼 이름
            sort_col: 정렬 기준 컬럼

        Returns:
            저장된 Silver 파일 경로

        Raises:
            StorageError: Bronze 파일 없음 또는 저장 실패
        """
        logger.info(
            f"Processing onchain Bronze → Silver for {source}/{name}",
            extra={"source": source, "name": name},
        )

        df = self.bronze_storage.load(source, name)

        # Dedup
        rows_before = len(df)
        df = df.drop_duplicates(subset=[date_col], keep="last")
        rows_deduped = rows_before - len(df)

        # Sort
        df = df.sort_values(sort_col).reset_index(drop=True)

        # UTC enforce
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], utc=True)

        # Validate
        self._validate_data(df, source, name)

        silver_path = self.settings.get_onchain_silver_path(source, name)
        silver_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df.to_parquet(silver_path, compression="zstd", index=True)
            logger.info(
                f"Silver onchain saved: {silver_path}",
                extra={
                    "path": str(silver_path),
                    "source": source,
                    "name": name,
                    "rows": len(df),
                    "deduped": rows_deduped,
                },
            )
        except Exception as e:
            raise StorageError(
                f"Failed to save Silver onchain to {silver_path}",
                context={"path": str(silver_path), "error": str(e)},
            ) from e
        else:
            return silver_path

    def load(self, source: str, name: str) -> pd.DataFrame:
        """Silver 데이터 로드.

        Args:
            source: 데이터 소스
            name: 데이터 이름

        Returns:
            pandas DataFrame

        Raises:
            StorageError: 파일 없음 또는 로드 실패
        """
        path = self.settings.get_onchain_silver_path(source, name)
        if not path.exists():
            raise StorageError(
                f"Silver onchain file not found: {path}",
                context={"path": str(path), "source": source, "name": name},
            )
        try:
            df = pd.read_parquet(path)
            logger.debug(
                f"Silver onchain loaded: {path}",
                extra={"path": str(path), "rows": len(df)},
            )
        except Exception as e:
            raise StorageError(
                f"Failed to load Silver onchain from {path}",
                context={"path": str(path), "error": str(e)},
            ) from e
        else:
            return df

    def exists(self, source: str, name: str) -> bool:
        """Silver 파일 존재 여부."""
        return self.settings.get_onchain_silver_path(source, name).exists()

    def _validate_data(self, df: pd.DataFrame, source: str, name: str) -> None:
        """데이터 유효성 검증."""
        errors: list[str] = []

        if df.empty:
            errors.append("DataFrame is empty")

        if errors:
            raise DataIntegrityError(
                f"Onchain data validation failed for {source}/{name}",
                context={"source": source, "name": name, "errors": errors},
            )
