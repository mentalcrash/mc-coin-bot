"""Silver layer processor for gap-filled OHLCV data.

This module implements the Silver layer of the Medallion Architecture.
Bronze data is loaded and processed with gap-filling (forward fill)
to create continuous time series data suitable for analysis.

Features:
    - Gap detection and reporting
    - Forward fill (ffill) for missing data
    - Complete DatetimeIndex generation
    - Data validation before storage

Rules Applied:
    - #12 Data Engineering: Parquet, vectorized operations
    - Medallion Architecture: Silver = cleaned/validated data
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import DataIntegrityError, StorageError
from src.data.bronze import BronzeStorage


@dataclass(frozen=True)
class GapReport:
    """갭 분석 리포트.

    Attributes:
        expected_count: 예상 캔들 수
        actual_count: 실제 캔들 수
        gap_count: 결측 캔들 수
        gap_percentage: 결측 비율 (%)
        first_timestamp: 첫 캔들 시각
        last_timestamp: 마지막 캔들 시각
    """

    expected_count: int
    actual_count: int
    gap_count: int
    gap_percentage: float
    first_timestamp: datetime | None
    last_timestamp: datetime | None

    def __str__(self) -> str:
        """리포트 문자열 표현."""
        return (
            f"GapReport(expected={self.expected_count:,}, "
            f"actual={self.actual_count:,}, "
            f"gaps={self.gap_count:,}, "
            f"gap_pct={self.gap_percentage:.2f}%)"
        )


class SilverProcessor:
    """Silver 계층 프로세서 (갭 필링).

    Bronze 데이터를 로드하여 결측치를 채우고 연속적인 시계열을 생성합니다.
    분석이나 백테스팅에 적합한 형태로 데이터를 정제합니다.

    Attributes:
        settings: 설정 객체
        bronze_storage: Bronze 저장소 인스턴스

    Example:
        >>> processor = SilverProcessor()
        >>> report = processor.analyze_gaps("BTC/USDT", 2025)
        >>> print(report)
        GapReport(expected=525600, actual=520000, gaps=5600, gap_pct=1.07%)
        >>> path = processor.process("BTC/USDT", 2025)
        >>> print(f"Saved to {path}")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        bronze_storage: BronzeStorage | None = None,
    ) -> None:
        """SilverProcessor 초기화.

        Args:
            settings: 설정 객체
            bronze_storage: Bronze 저장소 (None이면 새로 생성)
        """
        self.settings = settings or get_settings()
        self.bronze_storage = bronze_storage or BronzeStorage(self.settings)

    def _get_year_index(self, year: int) -> pd.DatetimeIndex:
        """해당 연도의 완전한 1분 단위 DatetimeIndex 생성.

        Args:
            year: 연도

        Returns:
            1분 간격의 DatetimeIndex (UTC)
        """
        start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{year}-12-31 23:59:00", tz="UTC")
        return pd.date_range(start, end, freq="1min")

    def analyze_gaps(self, symbol: str, year: int) -> GapReport:
        """Bronze 데이터의 갭 분석.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            GapReport 객체

        Raises:
            StorageError: Bronze 파일이 없을 경우
        """
        # Bronze 데이터 로드
        df = self.bronze_storage.load(symbol, year)

        # 예상 인덱스
        expected_index = self._get_year_index(year)
        expected_count = len(expected_index)
        actual_count = len(df)
        gap_count = expected_count - actual_count

        # 시간 범위
        first_timestamp: datetime | None = None
        last_timestamp: datetime | None = None

        if not df.empty:
            first_ts: pd.Timestamp = df.index.min()  # type: ignore[assignment]
            last_ts: pd.Timestamp = df.index.max()  # type: ignore[assignment]
            # pandas Timestamp를 datetime으로 변환
            first_timestamp = first_ts.to_pydatetime()
            if first_timestamp.tzinfo is None:
                first_timestamp = first_timestamp.replace(tzinfo=UTC)
            last_timestamp = last_ts.to_pydatetime()
            if last_timestamp.tzinfo is None:
                last_timestamp = last_timestamp.replace(tzinfo=UTC)

        report = GapReport(
            expected_count=expected_count,
            actual_count=actual_count,
            gap_count=gap_count,
            gap_percentage=(gap_count / expected_count * 100) if expected_count > 0 else 0,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
        )

        logger.info(
            f"Gap analysis for {symbol} {year}: {report}",
            extra={
                "symbol": symbol,
                "year": year,
                "expected": expected_count,
                "actual": actual_count,
                "gaps": gap_count,
                "gap_pct": report.gap_percentage,
            },
        )

        return report

    def _fill_gaps(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """결측치 채우기 (Forward Fill).

        Args:
            df: 원본 DataFrame
            year: 연도 (완전한 인덱스 생성용)

        Returns:
            갭이 채워진 DataFrame
        """
        # 1. 완전한 인덱스 생성
        complete_index = self._get_year_index(year)

        # 2. Reindex (결측 행 생성)
        df_reindexed = df.reindex(complete_index)

        # 3. Forward Fill (이전 값으로 채우기)
        df_filled = df_reindexed.ffill()

        # 4. Backward Fill (첫 행이 NaN인 경우)
        df_filled = df_filled.bfill()

        return df_filled

    def _validate_data(self, df: pd.DataFrame, symbol: str, year: int) -> None:
        """데이터 유효성 검증.

        Args:
            df: 검증할 DataFrame
            symbol: 심볼 (로깅용)
            year: 연도 (로깅용)

        Raises:
            DataIntegrityError: 검증 실패 시
        """
        errors: list[str] = []

        # 1. NaN 검사
        nan_count: int = df.isna().sum().sum()  # type: ignore[assignment]
        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN values after gap filling")

        # 2. 가격 검사 (0 또는 음수)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                invalid: int = (df[col] <= 0).sum()  # type: ignore[assignment]
                if invalid > 0:
                    logger.warning(
                        f"Found {invalid} zero/negative {col} values",
                        extra={"symbol": symbol, "year": year, "column": col, "count": invalid},
                    )

        # 3. 거래량 검사 (음수)
        if "volume" in df.columns:
            negative_vol: int = (df["volume"] < 0).sum()  # type: ignore[assignment]
            if negative_vol > 0:
                errors.append(f"Found {negative_vol} negative volume values")

        # 4. 인덱스 연속성 검사
        expected_count = len(self._get_year_index(year))
        actual_count = len(df)
        if actual_count != expected_count:
            errors.append(
                f"Row count mismatch: expected {expected_count}, got {actual_count}"
            )

        if errors:
            raise DataIntegrityError(
                f"Data validation failed for {symbol} {year}",
                context={"symbol": symbol, "year": year, "errors": errors},
            )

        logger.debug(
            f"Data validation passed for {symbol} {year}",
            extra={"symbol": symbol, "year": year, "rows": len(df)},
        )

    def process(self, symbol: str, year: int, validate: bool = True) -> Path:
        """Bronze → Silver 변환 (갭 필링 적용).

        Args:
            symbol: 거래 심볼
            year: 연도
            validate: 저장 전 검증 여부

        Returns:
            저장된 Silver 파일 경로

        Raises:
            StorageError: Bronze 파일이 없거나 저장 실패 시
            DataIntegrityError: 검증 실패 시
        """
        logger.info(
            f"Processing Bronze → Silver for {symbol} {year}",
            extra={"symbol": symbol, "year": year},
        )

        # 1. 갭 분석
        gap_report = self.analyze_gaps(symbol, year)

        # 2. Bronze 데이터 로드
        df = self.bronze_storage.load(symbol, year)

        # 3. 갭 필링
        df_filled = self._fill_gaps(df, year)

        # 4. 검증
        if validate:
            self._validate_data(df_filled, symbol, year)

        # 5. Silver 저장
        silver_path = self.settings.get_silver_path(symbol, year)
        silver_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df_filled.to_parquet(silver_path, compression="zstd", index=True)

            logger.info(
                f"Silver data saved: {silver_path}",
                extra={
                    "path": str(silver_path),
                    "symbol": symbol,
                    "year": year,
                    "rows": len(df_filled),
                    "gaps_filled": gap_report.gap_count,
                    "size_bytes": silver_path.stat().st_size,
                },
            )

            return silver_path

        except Exception as e:
            raise StorageError(
                f"Failed to save Silver data to {silver_path}",
                context={"path": str(silver_path), "error": str(e)},
            ) from e

    def load(self, symbol: str, year: int) -> pd.DataFrame:
        """Silver 데이터 로드.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            pandas DataFrame

        Raises:
            StorageError: 파일이 없거나 로드 실패 시
        """
        path = self.settings.get_silver_path(symbol, year)

        if not path.exists():
            raise StorageError(
                f"Silver file not found: {path}",
                context={"path": str(path), "symbol": symbol, "year": year},
            )

        try:
            df = pd.read_parquet(path)

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)

            logger.debug(
                f"Silver data loaded: {path}",
                extra={"path": str(path), "rows": len(df)},
            )

            return df

        except Exception as e:
            raise StorageError(
                f"Failed to load Silver data from {path}",
                context={"path": str(path), "error": str(e)},
            ) from e

    def exists(self, symbol: str, year: int) -> bool:
        """Silver 파일 존재 여부 확인.

        Args:
            symbol: 거래 심볼
            year: 연도

        Returns:
            파일이 존재하면 True
        """
        path = self.settings.get_silver_path(symbol, year)
        return path.exists()
