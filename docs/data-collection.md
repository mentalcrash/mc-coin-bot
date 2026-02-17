# 데이터 수집 가이드

> 모든 데이터는 **Medallion Architecture** (Bronze → Silver → Gold)로 관리됩니다.

---

## 아키텍처 개요

```
                    ┌─ Binance Spot API ──────── OHLCV 1m
External APIs ──────├─ Binance Futures API ───── Derivatives (Funding/OI/LS/Taker)
                    └─ On-chain APIs (6개 소스) ─ Stablecoin/TVL/MVRV/Sentiment/...
                              │
                              ▼
                    ┌─────────────────────────┐
                    │  Bronze (Append-Only)    │  ← 변환 없이 원본 그대로
                    │  data/bronze/            │
                    └──────────┬──────────────┘
                              │  validate + dedup + gap-fill
                              ▼
                    ┌─────────────────────────┐
                    │  Silver (Cleaned)        │  ← 백테스트/전략에서 사용
                    │  data/silver/            │
                    └──────────┬──────────────┘
                              │  on-the-fly 계산 (FeatureStore, Indicator Library)
                              ▼
                    ┌─────────────────────────┐
                    │  Gold (Features)         │  ← 메모리 전용, 디스크 저장 없음
                    └─────────────────────────┘
```

---

## 데이터 소스 종합표

### OHLCV (가격 데이터)

| 소스 | 데이터 | 해상도 | 히스토리 | 저장 경로 |
|------|--------|--------|---------|----------|
| Binance Spot | OHLCV (Open/High/Low/Close/Volume) | **1분** | 전체 (2017~) | `data/{layer}/{SYMBOL}/{YEAR}.parquet` |

- 모든 타임프레임은 1분봉에서 실시간 집계 (`CandleAggregator`)
- Silver: 갭 탐지 + forward-fill + 중복 제거 + 가격 이상치 검증

### Derivatives (파생상품 데이터)

| 소스 | 데이터 | 해상도 | 히스토리 | 저장 경로 |
|------|--------|--------|---------|----------|
| Binance Futures | Funding Rate | 8H | 전체 | `data/{layer}/{SYMBOL}/{YEAR}_deriv.parquet` |
| Binance Futures | Open Interest | 1H | 30일 제한 | 〃 |
| Binance Futures | Long/Short Ratio | 1H | 30일 제한 | 〃 |
| Binance Futures | Taker Buy/Sell Ratio | 1H | 30일 제한 | 〃 |

- Silver: 1시간 리샘플 + forward-fill + 중복 제거
- OI/LS/Taker는 Binance API 30일 제한 → **일일 cron 수집 필요**

### On-chain (온체인 데이터)

총 **6개 소스**, **22개 데이터셋**을 수집합니다.

| 소스 | 데이터 | 데이터셋 수 | 해상도 | 히스토리 |
|------|--------|:-----------:|--------|---------|
| **DeFiLlama** | Stablecoin Supply (전체/체인/개별) | 7 | Daily | 2020~ |
| **DeFiLlama** | TVL (전체/체인별) | 6 | Daily | 2020~ |
| **DeFiLlama** | DEX Volume | 1 | Daily | 2020~ |
| **Coin Metrics** | 12개 Community 메트릭 (BTC/ETH) | 2 | Daily | 2009~ |
| **Alternative.me** | Fear & Greed Index | 1 | Daily | 2018~ |
| **Blockchain.com** | Hash Rate, Miners Revenue, Tx Fees (BTC) — ⚠️ 아래 참고 | 3 | Daily | 2009~ |
| **Etherscan** | ETH Supply, Staking, Burned Fees | 1 | Snapshot | Near-RT |
| **mempool.space** | BTC Hashrate, Difficulty | 1 | Real-time | Rolling |

**저장 경로**: `data/{layer}/onchain/{source}/{name}.parquet`

<details>
<summary>데이터셋 전체 목록 (22개)</summary>

| # | Source | Name | 설명 |
|---|--------|------|------|
| 1 | defillama | stablecoin_total | 전체 stablecoin mcap |
| 2 | defillama | stablecoin_chain_Ethereum | Ethereum 체인 stablecoin |
| 3 | defillama | stablecoin_chain_Tron | Tron 체인 stablecoin |
| 4 | defillama | stablecoin_chain_BSC | BSC 체인 stablecoin |
| 5 | defillama | stablecoin_chain_Arbitrum | Arbitrum 체인 stablecoin |
| 6 | defillama | stablecoin_chain_Solana | Solana 체인 stablecoin |
| 7 | defillama | stablecoin_usdt | USDT 개별 공급량 |
| 8 | defillama | stablecoin_usdc | USDC 개별 공급량 |
| 9 | defillama | tvl_total | 전체 체인 합산 TVL |
| 10 | defillama | tvl_chain_Ethereum | Ethereum TVL |
| 11 | defillama | tvl_chain_Tron | Tron TVL |
| 12 | defillama | tvl_chain_BSC | BSC TVL |
| 13 | defillama | tvl_chain_Arbitrum | Arbitrum TVL |
| 14 | defillama | tvl_chain_Solana | Solana TVL |
| 15 | defillama | dex_volume | 전체 DEX 일일 볼륨 |
| 16 | coinmetrics | btc_metrics | BTC 메트릭 (12 Community metrics) |
| 17 | coinmetrics | eth_metrics | ETH 메트릭 (12 Community metrics) |
| 18 | alternative_me | fear_greed | Fear & Greed Index |
| 19 | blockchain_com | bc_hash-rate | BTC 해시레이트 |
| 20 | blockchain_com | bc_miners-revenue | BTC 마이너 수익 |
| 21 | blockchain_com | bc_transaction-fees-usd | BTC 트랜잭션 수수료 |
| 22 | etherscan | eth_supply | ETH 공급량/스테이킹/소각 |
| 23 | mempool_space | mining | BTC 해시레이트/난이도 |

</details>

> **Coin Metrics Community API 메트릭 마이그레이션 완료 (2026-02-18)**
>
> 2025-10-28 Community tier 변경으로 기존 9개 메트릭 중 7개가 Pro 전용 전환.
> 12개 Community 무료 메트릭으로 교체 완료 (`src/data/onchain/fetcher.py`).
>
> | 기존 (deprecated) | 대체 (Community) | `oc_*` 컬럼명 |
> |-------------------|-----------------|--------------|
> | `MVRV` | `CapMVRVCur` | `oc_mvrv` |
> | `RealCap` | `CapMrktCurUSD` | `oc_mktcap_usd` |
> | `TxTfrValAdjUSD` | `FlowInExUSD` | `oc_flow_in_ex_usd` |
> | `TxTfrValMeanUSD` | `FlowOutExUSD` | `oc_flow_out_ex_usd` |
> | `TxTfrValMedUSD` | `TxTfrCnt` | `oc_txtfrcnt` |
> | `VtyRet30d` | `ROI30d` | `oc_roi_30d` |
> | `NVTAdj90` | (제거 — Community 대체 없음) | — |
> | — | `HashRate` (신규) | `oc_cm_hashrate` |
> | — | `SplyCur` (신규) | `oc_supply` |
> | — | `IssTotUSD` (신규) | `oc_issuance_usd` |
> | — | `BlkCnt` (신규) | `oc_blkcnt` |
> | `AdrActCnt` (유지) | `AdrActCnt` | `oc_adractcnt` |
> | `TxCnt` (유지) | `TxCnt` | `oc_txcnt` |

> **⚠️ Blockchain.com Charts API 장애 (2026-02-18 확인)**
>
> `api.blockchain.info/charts/*` 엔드포인트가 **502/503 Bad Gateway**를 반환하고 있습니다.
> 3개 차트(hash-rate, miners-revenue, transaction-fees-usd) 모두 동일 증상입니다.
>
> - 공식 상태 페이지([status.blockchain.com](https://status.blockchain.com))에는 "정상" 표시 — Charts API가 별도 모니터링 대상이 아닌 것으로 추정
> - 커뮤니티/Reddit에 관련 보고 없음 (이 API 사용자가 극소수)
> - **대체 소스**: mempool.space에서 BTC 해시레이트/난이도를 이미 수집 중이므로, `bc_hash-rate`는 `mempool_space/mining`으로 대체 가능
> - **현재 코드 이슈**: `AsyncOnchainClient`가 5xx 에러에 대해 retry 없이 즉시 `NetworkError` raise — 5xx retry 로직 추가 필요
>
> **조치 사항**:
> 1. ✅ `src/data/onchain/client.py`에 5xx 서버 에러 retry 로직 추가 완료 (exponential backoff, 최대 3회)
> 2. Blockchain.com API 복구 시까지 `--type blockchain` 배치 수집은 실패 예상
> 3. 장기 미복구 시 해당 소스 비활성화 또는 대체 소스 전환 검토

---

## 저장 구조

```
data/
├── bronze/                              # Raw (변환 없음)
│   ├── BTC_USDT/
│   │   ├── 2024.parquet                 # OHLCV 1m
│   │   ├── 2024_deriv.parquet           # Derivatives
│   │   └── 2025.parquet
│   ├── ETH_USDT/
│   │   └── ...
│   └── onchain/                         # On-chain 데이터
│       ├── defillama/
│       │   ├── stablecoin_total.parquet
│       │   ├── stablecoin_chain_Ethereum.parquet
│       │   ├── tvl_total.parquet
│       │   └── dex_volume.parquet
│       ├── coinmetrics/
│       │   ├── btc_metrics.parquet
│       │   └── eth_metrics.parquet
│       ├── alternative_me/
│       │   └── fear_greed.parquet
│       ├── blockchain_com/
│       │   ├── bc_hash-rate.parquet
│       │   ├── bc_miners-revenue.parquet
│       │   └── bc_transaction-fees-usd.parquet
│       ├── etherscan/
│       │   └── eth_supply.parquet
│       └── mempool_space/
│           └── mining.parquet
│
└── silver/                              # Cleaned (백테스트/전략용)
    ├── BTC_USDT/                        # OHLCV: gap-filled + validated
    │   ├── 2024.parquet
    │   └── 2024_deriv.parquet           # Derivatives: 1H resample + ffill
    └── onchain/                         # On-chain: dedup + sort + UTC
        └── (bronze과 동일 구조)
```

---

## CLI 명령어

### OHLCV 수집

```bash
# Bronze → Silver 전체 파이프라인
uv run mcbot ingest pipeline BTC/USDT --year 2024 --year 2025

# 개별 레이어
uv run mcbot ingest bronze BTC/USDT --year 2024 --year 2025
uv run mcbot ingest silver BTC/USDT --year 2024 --year 2025

# 벌크 다운로드 (상위 N개 심볼)
uv run mcbot ingest bulk-download --top 100 --year 2024 --year 2025
uv run mcbot ingest bulk-download --dry-run   # 대상 미리보기

# 검증 & 정보
uv run mcbot ingest validate data/silver/BTC_USDT_1m_2025.parquet
uv run mcbot ingest info
```

### Derivatives 수집

```bash
# 단일 심볼 파이프라인
uv run mcbot ingest derivatives pipeline BTC/USDT --year 2024 --year 2025

# 개별 레이어
uv run mcbot ingest derivatives bronze BTC/USDT --year 2024
uv run mcbot ingest derivatives silver BTC/USDT --year 2024

# 배치 (기본: 8개 Tier-1/2 자산)
uv run mcbot ingest derivatives batch
uv run mcbot ingest derivatives batch -s BTC/USDT,ETH/USDT -y 2024 -y 2025
uv run mcbot ingest derivatives batch --dry-run

# 데이터 정보
uv run mcbot ingest derivatives info BTC/USDT --year 2024 --year 2025
```

### On-chain 수집

```bash
# 단일 데이터셋 파이프라인
uv run mcbot ingest onchain pipeline defillama stablecoin_total
uv run mcbot ingest onchain pipeline coinmetrics btc_metrics

# 배치 (카테고리별)
uv run mcbot ingest onchain batch --type all            # 전체 22개 데이터셋
uv run mcbot ingest onchain batch --type stablecoin     # DeFiLlama Stablecoin (7)
uv run mcbot ingest onchain batch --type tvl            # DeFiLlama TVL (6)
uv run mcbot ingest onchain batch --type dex            # DEX Volume (1)
uv run mcbot ingest onchain batch --type coinmetrics    # Coin Metrics (2)
uv run mcbot ingest onchain batch --type sentiment      # Fear & Greed (1)
uv run mcbot ingest onchain batch --type blockchain     # Blockchain.com (3)
uv run mcbot ingest onchain batch --type etherscan      # ETH Supply (1)
uv run mcbot ingest onchain batch --type mempool        # BTC Mining (1)
uv run mcbot ingest onchain batch --dry-run             # 대상 미리보기

# 데이터 정보
uv run mcbot ingest onchain info                        # 전체 인벤토리
uv run mcbot ingest onchain info --type coinmetrics     # 카테고리별
```

---

## Rate Limiting

외부 API별 요청 제한을 자동으로 준수합니다.

| 소스 | API 제한 | 적용값 | 비고 |
|------|---------|--------|------|
| Binance Spot | 1200 req/min | CCXT 내장 | IP 밴 주의 |
| Binance Futures | 1200 req/min | CCXT 내장 | IP 밴 주의 |
| DeFiLlama | ~30 req/min (비공식) | **25 req/min** | 보수적 적용 |
| Coin Metrics | 100 req/min | **90 req/min** | 10 req/6s 기준 |
| Blockchain.com | ~6 req/min | **5 req/min** | 가장 엄격 |
| Alternative.me | 제한 없음 | **10 req/min** | 보수적 |
| Etherscan | 5 req/sec (무료) | **4 req/min** | 일일 1회 스냅샷 |
| mempool.space | ~10 req/min | **8 req/min** | 보수적 |

- On-chain 클라이언트: `AsyncOnchainClient` — 소스별 `RateLimiter` + exponential backoff retry (최대 3회)
- Binance: CCXT Pro 내장 rate limiter 사용

---

## Publication Lag (Look-Ahead Bias 방지)

On-chain 데이터는 일 단위로 발행되며, 실제 접근 가능 시점에 지연(lag)이 있습니다.
`OnchainDataService.enrich()`가 자동으로 lag를 적용하여 look-ahead bias를 방지합니다.

| 소스 | Lag | 설명 |
|------|:---:|------|
| DeFiLlama | T+1 | ~06:00 UTC 확정 |
| Coin Metrics | T+1 | ~00:00-04:00 UTC 가용 |
| Alternative.me | T+1 | 안전하게 T+1 적용 |
| Blockchain.com | T+1 | ~12:00 UTC 확정 |
| Etherscan | T+0 | 스냅샷 (near real-time) |
| mempool.space | T+0 | 실시간 |

---

## 데이터 품질

### Bronze → Silver 처리 규칙

| 데이터 타입 | 중복 제거 | 갭 처리 | 정렬 | 타임존 |
|------------|:---------:|:------:|:----:|:------:|
| OHLCV | 타임스탬프 기준 | Forward-fill | 시간순 | UTC 필수 |
| Derivatives | 타임스탬프 기준 | 1H resample + ffill | 시간순 | UTC 필수 |
| On-chain | date/timestamp 기준 | 없음 (daily) | 시간순 | UTC 필수 |

### OHLCV 추가 검증
- 시간 갭 탐지 (1분 기준)
- 가격 이상치 검증 (급등락 체크)
- 타임스탬프 연속성 확인

### On-chain 추가 검증
- Decimal 타입 강제 (금융 값)
- Pydantic V2 frozen model로 불변성 보장
- Source별 date 컬럼 자동 매핑 (`date` or `timestamp`)

---

## 코드 구조

```
src/data/
├── fetcher.py                     # OHLCV DataFetcher (Binance Spot)
├── bronze.py                      # OHLCV BronzeStorage
├── silver.py                      # OHLCV SilverProcessor
├── service.py                     # OHLCV MarketDataService
│
├── derivatives_fetcher.py         # Derivatives DerivativesFetcher (Binance Futures)
├── derivatives_storage.py         # Derivatives Bronze/Silver
├── derivatives_service.py         # Derivatives DataService
│
└── onchain/                       # On-chain 데이터 모듈
    ├── client.py                  # AsyncOnchainClient (httpx + rate limit)
    ├── models.py                  # Pydantic V2 레코드 모델
    ├── fetcher.py                 # OnchainFetcher (6개 소스 fetch 메서드)
    ├── storage.py                 # OnchainBronzeStorage + SilverProcessor
    └── service.py                 # OnchainDataService (batch SSOT + enrich + precompute)

src/cli/
├── ingest.py                      # 메인 ingest CLI (OHLCV)
├── ingest_derivatives.py          # `ingest derivatives` 서브커맨드
└── ingest_onchain.py              # `ingest onchain` 서브커맨드
```

---

## 전략에서의 활용

### On-chain 데이터 → OHLCV 병합

`OnchainDataService.enrich()`로 on-chain 데이터를 OHLCV에 시간 기준 병합합니다.

```python
from src.data.onchain.service import OnchainDataService

service = OnchainDataService()

# Silver 데이터 로드
df = service.load("defillama", "stablecoin_total")

# OHLCV에 on-chain 병합 (publication lag 자동 적용)
enriched = service.enrich(ohlcv_df, "coinmetrics", "btc_metrics")
```

### EDA 백테스트 사전계산

`precompute()`로 심볼별 on-chain 피처를 `oc_*` prefix로 사전 병합합니다.

```python
# EDA 엔진에서 자동 호출
oc_df = service.precompute("BTC/USDT", ohlcv_index)
# → oc_mvrv, oc_realcap, oc_stablecoin_total_circulating_usd, ...
```

---

## 환경 변수

```bash
# .env 파일에 설정 (선택)
# BRONZE_DIR=data/bronze          # Bronze 저장 경로 (기본값)
# SILVER_DIR=data/silver          # Silver 저장 경로 (기본값)
# ETHERSCAN_API_KEY=              # Etherscan 무료 API key (on-chain etherscan 타입에 필요)
```

---

## 향후 계획

1. **Cron 자동 수집**: Derivatives(OI/LS/Taker) 30일 제한 → 일일 수집 자동화
2. **데이터 품질 알림**: 수집 실패/이상 감지 시 Discord 알림
3. **FeatureStore 통합**: Silver on-chain → FeatureStore 등록 → 전략 직접 사용
4. **On-chain Alpha 전략**: Stablecoin flow, MVRV regime filter 등 전략 후보

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [`docs/design/paid-data-sources-research.md`](design/paid-data-sources-research.md) | 유료 데이터 소스 분석 |
| [`.claude/rules/data.md`](../.claude/rules/data.md) | 데이터 레이어 코딩 규칙 (Medallion, Pandas) |
