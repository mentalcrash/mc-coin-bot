---
paths:
  - "src/data/**"
---

# Data Layer Rules (Medallion Architecture)

## Layer Overview

| Layer | Location | Purpose | Policy |
|-------|----------|---------|--------|
| **Bronze** | `data/bronze/` | Raw Binance OHLCV | Append-only |
| **Silver** | `data/silver/` | Validated & gap-filled | Idempotent |
| **Gold** | Memory | Strategy features | On-the-fly |

## Bronze Layer

- **원시 데이터 저장:** 변환 없이 Binance API 데이터 그대로
- **파티셔닝:** `data/bronze/{SYMBOL}/{YEAR}.parquet`
- **쓰기 정책:** Append-only (덮어쓰기 금지)

```python
# Example path
data/bronze/BTC_USDT/2024.parquet
data/bronze/BTC_USDT/2025.parquet
```

## Silver Layer

- **정제 데이터:** 검증, 갭 채우기, 중복 제거 완료
- **갭 처리:** Forward-fill로 채우기
- **타임스탬프:** DatetimeIndex, UTC 타임존 필수

```python
# ❌ Bad (no timezone)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ✅ Good (UTC explicit)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.set_index('timestamp')
```

## Gold Layer

- **전략별 피처:** 기술적 지표, 파생 변수
- **생성 시점:** 백테스트/실거래 시 on-the-fly 계산
- **저장 정책:** 디스크 저장 없음 (메모리만)

## Parquet Storage

- **포맷:** Parquet (컬럼형 압축)
- **백엔드:** PyArrow 권장

```python
# Read with PyArrow backend
df = pd.read_parquet(path, dtype_backend="pyarrow")
```

## Data Quality Checks

Silver 레이어 처리 시 검증:

1. 시간 갭 탐지 (1분 기준)
2. 중복 타임스탬프 제거
3. 가격 이상치 검증 (급등락 체크)
4. 타임스탬프 정렬 확인

## Derivatives Data Layer

OHLCV 외 파생 데이터 (Funding Rate, OI, LS Ratio, Taker Ratio):

| Layer | Path | Policy |
|-------|------|--------|
| **Bronze** | `data/{bronze}/{SYMBOL}/{YEAR}_deriv.parquet` | Append-only, raw |
| **Silver** | `data/{silver}/{SYMBOL}/{YEAR}_deriv.parquet` | 1H resample + forward-fill |

- **데이터 소스**: Binance Futures API
- **Silver 처리**: 1시간 리샘플, forward-fill, 중복 제거
- **스토리지**: `src/data/derivatives_storage.py` (Bronze/Silver 저장/로드)
- **서비스**: `src/data/derivatives_service.py` (파이프라인 오케스트레이션)

## Pandas Best Practices

```python
# ✅ Immutable operations
df = df.fillna(0)
df = df.drop(columns=['col'])

# ❌ Mutable operations (PD002)
df.fillna(0, inplace=True)
df.drop(columns=['col'], inplace=True)
```
