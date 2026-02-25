# Taker Buy Base Volume 인프라 추가 기획서

**상태**: Phase 1 완료 / Phase 2~6 미착수
**작성일**: 2026-02-23
**교훈 참조**: Lesson #15 (BVC 근사 검증 실패)

## 1. 배경 및 동기

### 문제

- ccxt `fetch_ohlcv()`는 Binance klines API의 12개 컬럼 중 6개(OHLCV)만 반환
- BVC(Bulk Volume Classification) 근사는 실제 taker flow와 괴리 (교훈 #15)
- Derivatives taker ratio API는 **30일 제한**으로 백테스트 불가

### 해결

Binance klines API 동일 호출에서 **column 9** (`taker_buy_base_volume`)를 캡처:

- 전 에셋, 전 기간, 1m 해상도의 실제 taker flow 데이터
- 추가 API 비용 없음 (동일 endpoint, 동일 rate limit)
- CVD-Price Divergence 등 고가치 전략의 핵심 입력

### Binance klines 컬럼 맵

| Index | Field | 현재 사용 |
|-------|-------|-----------|
| 0 | open_time | timestamp |
| 1 | open | open |
| 2 | high | high |
| 3 | low | low |
| 4 | close | close |
| 5 | volume | volume |
| 6 | close_time | - |
| 7 | quote_asset_volume | - |
| 8 | number_of_trades | - |
| **9** | **taker_buy_base_volume** | **Phase 2에서 추가** |
| 10 | taker_buy_quote_volume | - |
| 11 | ignore | - |

## 2. 의존성 그래프

```text
Phase 1: 데이터 모델 + 지표 (독립)  ← ✅ 완료
    ↓
Phase 2: API 캡처 (Phase 1 의존)
    ↓
Phase 3: Storage 파이프라인 (Phase 1+2 의존)
    ↓
Phase 4: EDA 통합 (Phase 1+3 의존)
    ↓
Phase 5: 데이터 재수집 (Phase 2+3 의존)
    ↓
Phase 6: 전략 통합 (Phase 5 이후, 별도 P2P3 세션)
```

## 3. Phase 상세

### Phase 1: 데이터 모델 + 지표 ✅ 완료

**산출물:**

| 파일 | 변경 | 상태 |
|------|------|------|
| `catalogs/datasets.yaml` | `binance_taker_buy_base_volume` 엔트리 추가 | ✅ |
| `catalogs/indicators.yaml` | `taker_cvd`, `taker_buy_ratio` 2개 추가 + `cvd_price_divergence` notes 수정 | ✅ |
| `src/market/indicators/microstructure.py` | `taker_cvd()`, `taker_buy_ratio()` 함수 추가 | ✅ |
| `src/market/indicators/__init__.py` | import + `__all__` 등록 | ✅ |
| `tests/market/indicators/test_microstructure.py` | 9개 테스트 추가 (24 passed) | ✅ |

**지표 사양:**

```python
def taker_cvd(taker_buy_base_volume, volume) -> pd.Series:
    """CVD = cumsum(2*taker_buy - volume)"""

def taker_buy_ratio(taker_buy_base_volume, volume, window=14) -> pd.Series:
    """Rolling taker_buy / volume (0~1)"""
```

### Phase 2: API 캡처

**목표**: Binance klines raw 응답에서 column 9 캡처.

#### 2-1. OHLCVCandle 모델 확장

**파일**: `src/models/ohlcv.py` (라인 ~50)

```python
# 현재
class OHLCVCandle(BaseModel):
    timestamp: datetime
    open: Decimal = Field(..., gt=0)
    high: Decimal = Field(..., gt=0)
    low: Decimal = Field(..., gt=0)
    close: Decimal = Field(..., gt=0)
    volume: Decimal = Field(..., ge=0)

# 변경 후
class OHLCVCandle(BaseModel):
    timestamp: datetime
    open: Decimal = Field(..., gt=0)
    high: Decimal = Field(..., gt=0)
    low: Decimal = Field(..., gt=0)
    close: Decimal = Field(..., gt=0)
    volume: Decimal = Field(..., ge=0)
    taker_buy_base_volume: Decimal | None = Field(default=None, ge=0)
```

**역호환성**: `None` 기본값 → 기존 코드 영향 없음.

#### 2-2. BinanceClient.fetch_ohlcv_extended()

**파일**: `src/exchange/binance_client.py` (라인 ~247)

```python
# 기존 fetch_ohlcv() 유지 (backward compatible)
# 새 메서드 추가:
async def fetch_ohlcv_extended(
    self, symbol: str, timeframe: str = "1m",
    since: int | None = None, limit: int | None = None,
) -> OHLCVBatch:
    """publicGetKlines() 직접 호출 → 12 컬럼 raw response."""
```

**구현 방안:**

- ccxt `publicGetKlines()` 사용 (raw REST endpoint)
- 응답에서 `c[9]`를 `taker_buy_base_volume`으로 매핑
- 기존 `fetch_ohlcv()`는 변경 없이 유지

#### 2-3. DataFetcher 전환

**파일**: `src/data/fetcher.py` (라인 ~149)

- `_fetch_batch_with_retry()` 내부에서 `fetch_ohlcv_extended()` 호출
- 기존 파라미터 동일 (symbol, timeframe, since, limit)

**테스트:**

- `test_binance_client_extended.py`: raw 응답 파싱 + OHLCVCandle 매핑
- `test_fetcher_extended.py`: DataFetcher → extended 호출 확인

### Phase 3: Storage 파이프라인

#### 3-1. Bronze 레이어

**파일**: `src/data/bronze.py`

- 변경 불필요: `OHLCVCandle.model_dump()` → parquet 자동 전파
- `taker_buy_base_volume` 컬럼이 자동으로 포함됨

#### 3-2. Silver 레이어 검증

**파일**: `src/data/silver.py` (라인 ~209)

```python
# _validate_data()에 추가:
if "taker_buy_base_volume" in df.columns:
    negative = (df["taker_buy_base_volume"] < 0).sum()
    if negative > 0:
        errors.append(f"Found {negative} negative taker_buy_base_volume values")
```

#### 3-3. Resample 규칙

**파일**: `src/data/service.py` (라인 ~469)

```python
agg_rules = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}
# 조건부 추가:
if "taker_buy_base_volume" in df.columns:
    agg_rules["taker_buy_base_volume"] = "sum"
```

#### 3-4. Enrichment NaN 체크

**파일**: `src/data/service.py` (라인 ~399)

```python
# ohlcv_cols 확장:
ohlcv_cols = {"open", "high", "low", "close", "volume"}
if "taker_buy_base_volume" in resampled.columns:
    ohlcv_cols.add("taker_buy_base_volume")
```

**테스트:**

- `test_silver_taker_validation.py`: 음수 검증
- `test_service_resample_taker.py`: sum 집계 확인

### Phase 4: EDA 통합

#### 4-1. BarEvent 확장

**파일**: `src/core/events.py` (라인 ~87)

```python
class BarEvent(BaseModel):
    # ... 기존 필드 ...
    volume: float
    taker_buy_base_volume: float | None = None  # 추가
    bar_timestamp: datetime
```

#### 4-2. PartialCandle 확장

**파일**: `src/eda/candle_aggregator.py` (라인 ~60)

```python
@dataclass
class PartialCandle:
    # ... 기존 필드 ...
    volume: float
    taker_buy_base_volume: float | None = None  # 추가
    # ...

# 집계 로직 (라인 ~148):
partial.volume += bar.volume
if bar.taker_buy_base_volume is not None:
    if partial.taker_buy_base_volume is None:
        partial.taker_buy_base_volume = bar.taker_buy_base_volume
    else:
        partial.taker_buy_base_volume += bar.taker_buy_base_volume
```

#### 4-3. HistoricalDataFeed

**파일**: `src/eda/data_feed.py` (라인 ~171)

```python
# parquet 로드 시 optional 컬럼 처리:
taker_buy = (
    float(row["taker_buy_base_volume"])
    if "taker_buy_base_volume" in df.columns
    and not pd.isna(row.get("taker_buy_base_volume"))
    else None
)

bar_1m = BarEvent(
    # ... 기존 필드 ...
    taker_buy_base_volume=taker_buy,
)
```

#### 4-4. StrategyEngine 버퍼

**파일**: `src/eda/strategy_engine.py` (라인 ~120)

```python
buf = {
    "open": bar.open,
    "high": bar.high,
    "low": bar.low,
    "close": bar.close,
    "volume": bar.volume,
}
if bar.taker_buy_base_volume is not None:
    buf["taker_buy_base_volume"] = bar.taker_buy_base_volume
self._buffers[symbol].append(buf)
```

#### 4-5. LiveDataFeed

- MVP에서는 `None` (ccxt `watch_ohlcv()`도 6 컬럼만 반환)
- 라이브에서는 DerivativesFeed의 taker ratio로 대체

**테스트:**

- `test_bar_event_taker.py`: BarEvent 직렬화/역직렬화
- `test_candle_aggregator_taker.py`: PartialCandle sum 집계
- `test_data_feed_taker.py`: parquet → BarEvent 전달 (유/무 컬럼)
- `test_strategy_engine_taker.py`: 버퍼 포함 확인

### Phase 5: 데이터 재수집

**대상 에셋** (우선순위):

1. BTCUSDT
2. ETHUSDT
3. BNBUSDT
4. SOLUSDT
5. DOGEUSDT

**실행:**

```bash
uv run mcbot ingest pipeline BTCUSDT
uv run mcbot ingest pipeline ETHUSDT
# ...
```

**검증 체크리스트:**

- [ ] Silver parquet에 `taker_buy_base_volume` 컬럼 존재
- [ ] 값 범위: 0 이상, volume 이하
- [ ] NaN 비율 < 1%
- [ ] 1D resample 시 sum 정합성 (1440 bars 합산)

### Phase 6: 전략 통합 (미래)

- CVD-Price-Divergence 전략 구현 (별도 P2P3 세션)
- `taker_cvd()` + `cvd_price_divergence()` 조합
- `taker_buy_ratio()` 필터 (> 0.5 buyer dominant 구간)

## 4. 리스크 평가

| 리스크 | 영향 | 완화 |
|--------|------|------|
| Spot vs Futures klines 혼동 | 다른 taker_buy 값 | 현재 OHLCV는 Spot 기반 → `publicGetKlines` 사용 |
| 기존 parquet에 컬럼 없음 | 로드 실패 | `OHLCVCandle.taker_buy_base_volume = None` (Optional) |
| Gap-fill 시 CVD 왜곡 | forward-fill된 volume → CVD 왜곡 | gap-fill된 구간 CVD는 신뢰하지 않음 (전략에서 필터) |
| Rate limit | IP 밴 | 기존 endpoint 동일 (1200 req/min), 추가 호출 없음 |
| LiveDataFeed 미지원 | 실시간 CVD 불가 | MVP: historical만, live는 DerivativesFeed taker ratio 대체 |
| publicGetKlines 응답 형식 변경 | 파싱 실패 | 방어적 파싱 + 길이 검증 (`len(row) >= 10`) |

## 5. 역호환성 보장

1. **OHLCVCandle**: `taker_buy_base_volume: Decimal | None = Field(default=None)` → 기존 코드 영향 없음
2. **BarEvent**: `taker_buy_base_volume: float | None = None` → 기존 이벤트 핸들러 영향 없음
3. **Resample**: 조건부 agg_rules 추가 (`if col in df.columns`) → 기존 파이프라인 안전
4. **StrategyEngine**: 조건부 버퍼 추가 → 기존 전략 DataFrame에 영향 없음
5. **기존 Parquet**: 컬럼 없으면 `None`으로 처리 → 재수집 전에도 정상 동작
