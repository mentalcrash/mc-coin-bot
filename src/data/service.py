"""Market Data Service for data access abstraction.

이 모듈은 데이터 접근을 추상화하는 서비스 레이어를 제공합니다.
Repository Pattern을 적용하여 CLI, EDA, 백테스트 등에서 동일한 인터페이스로
데이터에 접근할 수 있습니다.

Features:
    - 연도별 Silver 데이터 로드 및 병합
    - 타임프레임별 리샘플링 (1m → 1h, 1D 등)
    - 기간 필터링
    - MarketDataSet으로 래핑하여 반환

Rules Applied:
    - #10 Python Standards: Modern typing, async-ready design
    - #12 Data Engineering: Parquet, vectorized resampling
    - #15 Logging Standards: Structured logging
"""

from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from src.config.settings import IngestionSettings, get_settings
from src.core.exceptions import DataNotFoundError
from src.data.market_data import MarketDataRequest, MarketDataSet
from src.data.silver import SilverProcessor


class MarketDataService:
    """시장 데이터 서비스 (Repository Pattern).

    Silver 데이터를 로드하고, 요청된 타임프레임으로 리샘플링하여
    MarketDataSet으로 래핑하여 반환합니다.

    Attributes:
        settings: 설정 객체
        silver_processor: Silver 데이터 프로세서

    Example:
        >>> service = MarketDataService()
        >>> request = MarketDataRequest(
        ...     symbol="BTC/USDT",
        ...     timeframe="1D",
        ...     start=datetime(2024, 1, 1, tzinfo=UTC),
        ...     end=datetime(2025, 12, 31, tzinfo=UTC),
        ... )
        >>> data = service.get(request)
        >>> print(f"Loaded {data.periods} daily candles")
    """

    def __init__(
        self,
        settings: IngestionSettings | None = None,
        silver_processor: SilverProcessor | None = None,
    ) -> None:
        """MarketDataService 초기화.

        Args:
            settings: 설정 객체 (None이면 기본값)
            silver_processor: Silver 프로세서 (None이면 새로 생성)
        """
        self.settings = settings or get_settings()
        self.silver_processor = silver_processor or SilverProcessor(self.settings)

    def get(self, request: MarketDataRequest) -> MarketDataSet:
        """데이터 요청 처리.

        1. 요청된 기간에 해당하는 연도별 Silver 데이터 로드
        2. 데이터 병합 및 기간 필터링
        3. 타임프레임에 맞게 리샘플링
        4. MarketDataSet으로 래핑

        Args:
            request: 데이터 요청 객체

        Returns:
            MarketDataSet 객체

        Raises:
            DataNotFoundError: 요청된 기간에 데이터가 없을 경우
        """
        logger.debug("=" * 60)
        logger.debug("MarketDataService.get() 시작")
        logger.debug(f"  Request: symbol={request.symbol}, timeframe={request.timeframe}")
        logger.debug(f"  Period: {request.start.date()} ~ {request.end.date()}")

        # 1. 연도 범위 계산
        years = list(range(request.start.year, request.end.year + 1))
        logger.debug(f"[1/6] 연도 범위 계산: {years}")

        # 2. 연도별 Silver 데이터 로드 및 병합
        logger.debug("[2/6] Silver 데이터 로드 시작...")
        dfs: list[pd.DataFrame] = []
        total_rows = 0
        for year in years:
            try:
                df = self.silver_processor.load(request.symbol, year)
                dfs.append(df)
                total_rows += len(df)
                logger.debug(f"  - {year}: {len(df):,} rows (누적: {total_rows:,})")
            except Exception:
                logger.warning(f"  - {year}: 데이터 없음 (스킵)")

        if not dfs:
            raise DataNotFoundError(
                f"No data found for {request.symbol}",
                context={
                    "symbol": request.symbol,
                    "years": years,
                    "timeframe": request.timeframe,
                },
            )

        logger.debug(f"  총 로드: {len(dfs)}개 연도, {total_rows:,} rows")

        # 3. 병합 및 정렬
        logger.debug("[3/6] 데이터 병합 및 정렬...")
        combined = pd.concat(dfs).sort_index()
        before_dedup = len(combined)
        combined = combined[~combined.index.duplicated(keep="first")]
        after_dedup = len(combined)
        if before_dedup != after_dedup:
            logger.debug(
                f"  - 중복 제거: {before_dedup:,} → {after_dedup:,} ({before_dedup - after_dedup:,} 제거)"
            )

        # 4. 기간 필터링
        logger.debug("[4/6] 기간 필터링...")
        # DatetimeIndex를 tz-aware로 변환 (필요시)
        if isinstance(combined.index, pd.DatetimeIndex) and combined.index.tz is None:
            combined.index = combined.index.tz_localize("UTC")
            logger.debug("  - 인덱스 UTC 변환 완료")

        # start/end를 Timestamp로 변환 (이미 tz-aware일 수 있음)
        start_ts = pd.Timestamp(request.start)
        end_ts = pd.Timestamp(request.end)
        # tz-naive면 UTC로 설정
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")
        filtered = combined.loc[start_ts:end_ts]
        logger.debug(
            f"  - 필터링 결과: {len(filtered):,} rows ({start_ts.date()} ~ {end_ts.date()})"
        )

        if filtered.empty:
            raise DataNotFoundError(
                f"No data in requested period for {request.symbol}",
                context={
                    "symbol": request.symbol,
                    "start": str(request.start),
                    "end": str(request.end),
                },
            )

        # 5. 리샘플링 (1m이 아닌 경우)
        if request.timeframe != "1m":
            logger.debug(f"[5/6] 리샘플링: 1m → {request.timeframe}...")
            resampled = self._resample(filtered, request.timeframe)
        else:
            logger.debug("[5/6] 리샘플링: 불필요 (이미 1m)")
            resampled = filtered

        # 6. 실제 데이터 범위 추출
        logger.debug("[6/6] MarketDataSet 생성...")
        actual_start: datetime = resampled.index[0].to_pydatetime()  # type: ignore[union-attr]
        actual_end: datetime = resampled.index[-1].to_pydatetime()  # type: ignore[union-attr]

        # UTC 보장
        if actual_start.tzinfo is None:
            actual_start = actual_start.replace(tzinfo=UTC)
        if actual_end.tzinfo is None:
            actual_end = actual_end.replace(tzinfo=UTC)

        # 데이터 품질 검증 로그
        logger.debug("  데이터 품질 검증:")
        logger.debug(f"    - 기간: {(actual_end - actual_start).days} days")
        logger.debug(f"    - Candles: {len(resampled):,}")
        logger.debug(
            f"    - Price range: ${resampled['close'].min():,.2f} ~ ${resampled['close'].max():,.2f}"
        )
        logger.debug(f"    - Volume total: {resampled['volume'].sum():,.0f}")

        logger.info(
            f"MarketDataService.get() 완료: {request.symbol} {request.timeframe} [{len(resampled):,} candles]"
        )
        logger.debug("=" * 60)

        return MarketDataSet(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start=actual_start,
            end=actual_end,
            ohlcv=resampled,
        )

    def _resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """OHLCV 데이터 리샘플링.

        1분봉 데이터를 지정된 타임프레임으로 리샘플링합니다.

        Args:
            df: 1분봉 DataFrame
            timeframe: 목표 타임프레임 (예: "1h", "1D")

        Returns:
            리샘플링된 DataFrame
        """
        # 타임프레임을 pandas resample 규칙으로 변환
        resample_rule_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1D": "1D",
            "1d": "1D",
            "1W": "1W",
            "1w": "1W",
        }
        rule = resample_rule_map.get(timeframe, timeframe)

        # OHLCV 리샘플링 규칙
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        resampled: pd.DataFrame = df.resample(rule).agg(agg_rules).dropna()  # type: ignore[assignment]

        # Decimal 타입을 float64로 변환 (Parquet에서 로드된 경우)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in resampled.columns:
                resampled[col] = pd.to_numeric(resampled[col], errors="coerce")

        logger.debug(f"Resampled {len(df):,} → {len(resampled):,} candles (1m → {timeframe})")

        return resampled

    def available_years(self, symbol: str) -> list[int]:
        """사용 가능한 연도 목록 조회.

        Args:
            symbol: 거래 심볼

        Returns:
            Silver 데이터가 있는 연도 목록
        """
        return [
            year
            for year in range(2017, datetime.now(UTC).year + 1)
            if self.silver_processor.exists(symbol, year)
        ]
