---
paths:
  - "src/data/**"
  - "src/catalog/**"
---

# Data Layer Rules (Medallion Architecture)

## Layer Overview

| Layer | Location | Purpose | Policy |
|-------|----------|---------|--------|
| **Bronze** | `data/bronze/` | Raw Binance OHLCV | Append-only |
| **Silver** | `data/silver/` | Validated & gap-filled | Idempotent |
| **Gold** | Memory | Strategy features | On-the-fly |

## Bronze Layer

- 변환 없이 Binance API 데이터 그대로 저장
- 파티셔닝: `data/bronze/{SYMBOL}/{YEAR}.parquet`
- Append-only (덮어쓰기 금지)

## Silver Layer

- 검증, 갭 채우기 (forward-fill), 중복 제거
- **타임스탬프**: DatetimeIndex, UTC 필수

```python
# UTC explicit
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.set_index("timestamp")
```

## Derivatives Data Layer

OHLCV 외 파생 데이터 (Funding Rate, OI, LS Ratio, Taker Ratio):

| Layer | Path | Policy |
|-------|------|--------|
| **Bronze** | `data/{bronze}/{SYMBOL}/{YEAR}_deriv.parquet` | Append-only |
| **Silver** | `data/{silver}/{SYMBOL}/{YEAR}_deriv.parquet` | 1H resample + forward-fill |

## Data Catalog

YAML 기반 데이터셋 메타데이터 (`catalogs/datasets.yaml`):

| Component | Location |
|-----------|----------|
| **YAML** | `catalogs/datasets.yaml` (14 sources, 75 datasets SSOT) |
| **Models** | `src/catalog/models.py` |
| **Store** | `src/catalog/store.py` (GateCriteriaStore 패턴) |

### Store API

```python
from src.catalog.store import DataCatalogStore

store = DataCatalogStore()
store.load("btc_metrics")
store.get_by_type(DataType.ONCHAIN)
store.get_by_group("stablecoin")
store.get_batch_definitions("stablecoin")
store.build_precompute_map(["BTC/USDT"])
```

## Additional Data Modules

On-chain과 동일 패턴. 각 모듈: `client.py`, `models.py`, `fetcher.py`, `storage.py`, `service.py`.

| Module | Path | Scope |
|--------|------|-------|
| **Macro** | `src/data/macro/` | GLOBAL (DXY, VIX 등) |
| **Options** | `src/data/options/` | GLOBAL (DVOL 등) |
| **DerivExt** | `src/data/deriv_ext/` | PER-ASSET |
