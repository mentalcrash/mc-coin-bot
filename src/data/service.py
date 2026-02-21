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
from src.data.market_data import MarketDataRequest, MarketDataSet, MultiSymbolData
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

    def get(
        self,
        request: MarketDataRequest,
    ) -> MarketDataSet:
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

        # 5.5. Data enrichment (always-on, graceful degradation)
        resampled = self._maybe_enrich_derivatives(resampled, request)
        resampled = self._maybe_enrich_onchain(resampled, request)
        resampled = self._maybe_enrich_macro(resampled, request)
        resampled = self._maybe_enrich_options(resampled, request)
        resampled = self._maybe_enrich_deriv_ext(resampled, request)

        # 5.6. Enrichment NaN 비율 검사
        self._check_enrichment_nan_ratios(
            resampled, ohlcv_cols={"open", "high", "low", "close", "volume"}
        )

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

    def _maybe_enrich_derivatives(
        self,
        df: pd.DataFrame,
        request: MarketDataRequest,
    ) -> pd.DataFrame:
        """Derivatives enrichment (always-on, graceful degradation).

        Args:
            df: 리샘플링된 OHLCV DataFrame
            request: 데이터 요청

        Returns:
            원본 또는 enriched DataFrame
        """
        try:
            from src.data.derivatives_service import DerivativesDataService

            deriv_service = DerivativesDataService(self.settings)
            result = deriv_service.enrich(df, request.symbol, request.start, request.end)
        except Exception:
            logger.debug("Derivatives data not available — skipping enrichment")
            return df
        else:
            logger.debug("Derivatives enrichment applied")
            return result

    def _maybe_enrich_onchain(
        self,
        df: pd.DataFrame,
        request: MarketDataRequest,
    ) -> pd.DataFrame:
        """On-chain enrichment (always-on, graceful degradation).

        OnchainDataService.precompute()를 사용하여 catalog 기반
        oc_* 컬럼을 OHLCV DataFrame에 merge_asof로 주입한다.
        Publication lag가 자동 적용되어 lookahead bias가 방지된다.
        """
        try:
            from src.data.onchain.service import OnchainDataService

            onchain_service = OnchainDataService(settings=self.settings)
            enriched = onchain_service.precompute(
                symbol=request.symbol,
                ohlcv_index=df.index,
            )
        except Exception:
            logger.debug("On-chain data not available — skipping enrichment")
            return df
        else:
            if enriched.columns.empty:
                logger.debug("No on-chain data available — skipping enrichment")
                return df
            # Drop columns already present in df to avoid overlap errors
            overlap = set(enriched.columns) & set(df.columns)
            if overlap:
                enriched = enriched.drop(columns=list(overlap))
            if enriched.columns.empty:
                return df
            result = df.join(enriched, how="left")
            oc_cols = [c for c in enriched.columns if c.startswith("oc_")]
            logger.debug(
                f"On-chain enrichment applied: {len(oc_cols)} columns ({', '.join(oc_cols[:5])}...)"
            )
            return result

    def _maybe_enrich_macro(
        self,
        df: pd.DataFrame,
        request: MarketDataRequest,
    ) -> pd.DataFrame:
        """Macro enrichment (always-on, graceful degradation).

        MacroDataService.precompute()를 사용하여 macro_* 컬럼 주입.
        GLOBAL scope — 모든 자산 동일.
        """
        try:
            from src.data.macro.service import MacroDataService

            macro_service = MacroDataService(settings=self.settings)
            enriched = macro_service.precompute(ohlcv_index=df.index)
        except Exception:
            logger.debug("Macro data not available — skipping enrichment")
            return df
        else:
            if enriched.columns.empty:
                logger.debug("No macro data available — skipping enrichment")
                return df
            # Drop columns already present in df to avoid overlap errors
            overlap = set(enriched.columns) & set(df.columns)
            if overlap:
                enriched = enriched.drop(columns=list(overlap))
            if enriched.columns.empty:
                return df
            result = df.join(enriched, how="left")
            macro_cols = [c for c in enriched.columns if c.startswith("macro_")]
            logger.debug(
                f"Macro enrichment applied: {len(macro_cols)} columns ({', '.join(macro_cols[:5])}...)"
            )
            return result

    def _maybe_enrich_options(
        self,
        df: pd.DataFrame,
        request: MarketDataRequest,
    ) -> pd.DataFrame:
        """Options enrichment (always-on, graceful degradation).

        OptionsDataService.precompute()를 사용하여 opt_* 컬럼 주입.
        GLOBAL scope — 모든 자산 동일.
        """
        try:
            from src.data.options.service import OptionsDataService

            options_service = OptionsDataService(settings=self.settings)
            enriched = options_service.precompute(ohlcv_index=df.index)
        except Exception:
            logger.debug("Options data not available — skipping enrichment")
            return df
        else:
            if enriched.columns.empty:
                logger.debug("No options data available — skipping enrichment")
                return df
            # Drop columns already present in df to avoid overlap errors
            overlap = set(enriched.columns) & set(df.columns)
            if overlap:
                enriched = enriched.drop(columns=list(overlap))
            if enriched.columns.empty:
                return df
            result = df.join(enriched, how="left")
            opt_cols = [c for c in enriched.columns if c.startswith("opt_")]
            logger.debug(
                f"Options enrichment applied: {len(opt_cols)} columns ({', '.join(opt_cols[:5])}...)"
            )
            return result

    def _maybe_enrich_deriv_ext(
        self,
        df: pd.DataFrame,
        request: MarketDataRequest,
    ) -> pd.DataFrame:
        """Extended derivatives enrichment (always-on, graceful degradation).

        DerivExtDataService.precompute()를 사용하여 dext_* 컬럼 주입.
        PER-ASSET scope — symbol에서 asset 추출.
        """
        try:
            from src.data.deriv_ext.service import DerivExtDataService

            asset = request.symbol.split("/")[0]
            deriv_ext_service = DerivExtDataService(settings=self.settings)
            enriched = deriv_ext_service.precompute(ohlcv_index=df.index, asset=asset)
        except Exception:
            logger.debug("Deriv-ext data not available — skipping enrichment")
            return df
        else:
            if enriched.columns.empty:
                logger.debug("No deriv-ext data available — skipping enrichment")
                return df
            # Drop columns already present in df to avoid overlap errors
            overlap = set(enriched.columns) & set(df.columns)
            if overlap:
                enriched = enriched.drop(columns=list(overlap))
            if enriched.columns.empty:
                return df
            result = df.join(enriched, how="left")
            dext_cols = [c for c in enriched.columns if c.startswith("dext_")]
            logger.debug(
                f"Deriv-ext enrichment applied: {len(dext_cols)} columns ({', '.join(dext_cols[:5])}...)"
            )
            return result

    def _check_enrichment_nan_ratios(
        self,
        df: pd.DataFrame,
        ohlcv_cols: set[str],
    ) -> None:
        """Enrichment 컬럼의 NaN 비율을 검사하여 경고를 로깅합니다.

        enriched 컬럼(oc_*, macro_*, opt_*, dext_*, funding_rate 등)만 대상으로
        NaN 비율이 높은 컬럼에 대해 경고를 출력합니다.

        Args:
            df: enrichment이 적용된 DataFrame
            ohlcv_cols: OHLCV 기본 컬럼 (검사 제외)
        """
        enrichment_prefixes = ("oc_", "macro_", "opt_", "dext_", "funding_rate", "open_interest")
        enriched_cols = [
            c
            for c in df.columns
            if c not in ohlcv_cols and any(c.startswith(p) for p in enrichment_prefixes)
        ]
        if not enriched_cols:
            return

        n_rows = len(df)
        if n_rows == 0:
            return

        warn_threshold = 0.3
        drop_threshold = 0.8

        high_nan_cols: list[tuple[str, float]] = []
        drop_candidate_cols: list[tuple[str, float]] = []

        for col in enriched_cols:
            nan_ratio = float(df[col].isna().sum()) / n_rows
            if nan_ratio > drop_threshold:
                drop_candidate_cols.append((col, nan_ratio))
            elif nan_ratio > warn_threshold:
                high_nan_cols.append((col, nan_ratio))

        if high_nan_cols:
            col_details = ", ".join(f"{c}={r:.0%}" for c, r in high_nan_cols)
            logger.warning(
                "Enrichment NaN > {:.0%}: {}",
                warn_threshold,
                col_details,
            )

        if drop_candidate_cols:
            col_details = ", ".join(f"{c}={r:.0%}" for c, r in drop_candidate_cols)
            logger.warning(
                "Enrichment NaN > {:.0%} (consider dropping): {}",
                drop_threshold,
                col_details,
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

    def get_multi(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> MultiSymbolData:
        """여러 심볼의 Silver 데이터를 일괄 로드.

        각 심볼에 대해 get()을 호출하고 MultiSymbolData로 래핑합니다.

        Args:
            symbols: 심볼 목록 (예: ["BTC/USDT", "ETH/USDT"])
            timeframe: 타임프레임 (예: "1D")
            start: 시작 시각
            end: 종료 시각

        Returns:
            MultiSymbolData 객체

        Raises:
            DataNotFoundError: 어떤 심볼의 데이터도 찾을 수 없을 경우
        """
        logger.info(f"Loading multi-symbol data: {len(symbols)} symbols, {timeframe}")

        ohlcv: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            request = MarketDataRequest(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
            )
            data = self.get(request)
            ohlcv[symbol] = data.ohlcv

        # 실제 데이터의 공통 시작/종료 시각 결정
        all_starts = [df.index[0].to_pydatetime() for df in ohlcv.values()]  # type: ignore[union-attr]
        all_ends = [df.index[-1].to_pydatetime() for df in ohlcv.values()]  # type: ignore[union-attr]
        actual_start = max(all_starts)
        actual_end = min(all_ends)

        # UTC 보장
        if actual_start.tzinfo is None:
            actual_start = actual_start.replace(tzinfo=UTC)
        if actual_end.tzinfo is None:
            actual_end = actual_end.replace(tzinfo=UTC)

        logger.info(
            f"Multi-symbol data loaded: {len(symbols)} symbols, {actual_start.date()} ~ {actual_end.date()}"
        )

        return MultiSymbolData(
            symbols=symbols,
            timeframe=timeframe,
            start=actual_start,
            end=actual_end,
            ohlcv=ohlcv,
        )

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
