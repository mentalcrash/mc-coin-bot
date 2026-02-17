# On-Chain 데이터 수집 구현 계획

> 작성일: 2026-02-15
> 근거 문서: [onchain-data-collection-plan.md](onchain-data-collection-plan.md)
> 범위: Tier 1 무료 데이터 소스 (DeFiLlama, Coin Metrics, Blockchain.com, Alternative.me, Etherscan, mempool.space)

---

## 0. 설계 원칙

| 원칙 | 적용 |
|------|------|
| 기존 Medallion 패턴 준수 | Bronze(append-only) → Silver(검증/gap-fill) |
| 기존 코드 패턴 재사용 | `DerivativesFetcher` / `DerivativesBronzeStorage` 패턴 기반 |
| 점진적 구현 | Phase별 독립 배포 가능, 각 Phase 끝에 테스트 통과 |
| 최소 의존성 | `httpx[http2]` 1개 추가 (async HTTP client) |

---

## 1. 아키텍처 개요

### 1.1 모듈 구조

```
src/data/onchain/
├── __init__.py                     # Public exports
├── fetcher.py                      # OnchainFetcher (소스별 fetch 메서드)
├── storage.py                      # OnchainBronzeStorage + OnchainSilverProcessor
├── models.py                       # Pydantic V2 레코드 모델
└── client.py                       # AsyncOnchainClient (httpx wrapper + rate limit)
```

### 1.2 저장 구조

```
data/bronze/onchain/
├── defillama/
│   ├── stablecoin_total_{YEAR}.parquet
│   ├── stablecoin_chain_{CHAIN}_{YEAR}.parquet
│   ├── stablecoin_individual_{ID}_{YEAR}.parquet
│   ├── tvl_total_{YEAR}.parquet
│   ├── tvl_chain_{CHAIN}_{YEAR}.parquet
│   └── dex_volume_{YEAR}.parquet
├── coinmetrics/
│   └── {ASSET}_metrics_{YEAR}.parquet       # BTC, ETH
├── blockchain_com/
│   └── {CHART_NAME}_{YEAR}.parquet          # hash-rate, miners-revenue, ...
├── sentiment/
│   └── fear_greed_{YEAR}.parquet
├── etherscan/
│   └── eth_supply_{YEAR}.parquet
└── mempool_space/
    └── mining_{YEAR}.parquet
```

Silver는 on-chain 데이터가 Daily resolution이므로 gap-fill보다 **검증 + 중복제거 + 타입정제**에 집중:

```
data/silver/onchain/
├── defillama/
│   └── (동일 구조, validated)
├── coinmetrics/
│   └── (동일 구조, validated)
└── ...
```

### 1.3 데이터 흐름

```
[Cron / CLI] → OnchainFetcher → AsyncOnchainClient → HTTP API
                    │
                    ▼
              OnchainBronzeStorage.append()   ← append-only parquet
                    │
                    ▼
              OnchainSilverProcessor.process() ← validate + dedup + type coerce
                    │
                    ▼
              data/silver/onchain/...parquet   ← 전략/백테스트에서 사용
```

---

## 2. 신규 파일 상세

### 2.1 `client.py` — Async HTTP Client

`httpx.AsyncClient` 래퍼. 소스별 rate limit을 관리한다.

```python
"""Async HTTP client with per-source rate limiting."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, requests_per_minute: float) -> None:
        self._interval = 60.0 / requests_per_minute
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = asyncio.get_event_loop().time()


# 소스별 rate limit 설정
SOURCE_RATE_LIMITS: dict[str, float] = {
    "defillama": 25.0,       # ~30 req/min 공식 미공개 → 여유 25
    "coinmetrics": 90.0,     # 10 req/6s = 100/min → 여유 90
    "blockchain_com": 5.0,   # 6 req/min → 여유 5
    "alternative_me": 10.0,  # 제한 없음 → 보수적 10
    "etherscan": 4.0,        # 5 req/sec 이지만 일일 1회 → 보수적
    "mempool_space": 8.0,    # ~10 req/min → 여유 8
}


class AsyncOnchainClient:
    """httpx.AsyncClient wrapper with rate limiting + retry."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout
        self._limiters: dict[str, RateLimiter] = {
            source: RateLimiter(rpm)
            for source, rpm in SOURCE_RATE_LIMITS.items()
        }

    async def __aenter__(self) -> AsyncOnchainClient:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            http2=True,
            follow_redirects=True,
            headers={"User-Agent": "mc-coin-bot/1.0"},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()

    async def get(
        self,
        url: str,
        *,
        source: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> httpx.Response:
        """GET with rate limiting + exponential backoff retry."""
        assert self._client is not None, "Use async with"
        limiter = self._limiters.get(source)

        for attempt in range(1, max_retries + 1):
            if limiter:
                await limiter.acquire()
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                if attempt == max_retries:
                    raise
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Retry {}/{} for {} ({}), wait {}s",
                    attempt, max_retries, url, e, wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")  # pragma: no cover
```

**설계 결정:**
- `httpx[http2]` 선택 이유: async + HTTP/2 + 타입 안전 + 기존 코드에 aiohttp가 있지만 on-chain 전용 client 분리가 관심사 분리에 적합
- 소스별 `RateLimiter`: 각 API의 rate limit이 다르므로 개별 관리
- Retry: tenacity 대신 내장 구현 (간단한 GET-only 패턴이므로)

### 2.2 `models.py` — Pydantic V2 레코드

```python
"""On-chain data record models (frozen, validated)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── DeFiLlama ──

class StablecoinSupplyRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    date: datetime
    total_circulating_usd: Decimal = Field(..., ge=0)
    source: str = "defillama"

    @field_validator("date", mode="before")
    @classmethod
    def parse_unix(cls, v: int | float | datetime) -> datetime:
        if isinstance(v, int | float):
            return datetime.fromtimestamp(v, tz=UTC)
        return v


class StablecoinChainRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    date: datetime
    chain: str
    total_circulating_usd: Decimal = Field(..., ge=0)


class StablecoinIndividualRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    date: datetime
    stablecoin_id: int
    name: str
    circulating_usd: Decimal = Field(..., ge=0)


class TvlRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    date: datetime
    chain: str              # "all" for total
    tvl_usd: Decimal = Field(..., ge=0)


class DexVolumeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    date: datetime
    volume_usd: Decimal = Field(..., ge=0)


# ── Coin Metrics ──

class CoinMetricsRecord(BaseModel):
    """Daily metric row — 필드가 동적이므로 asset + time + metrics dict."""
    model_config = ConfigDict(frozen=True)
    asset: str
    time: datetime
    metrics: dict[str, Decimal | None]  # {MVRV: 2.3, RealCap: 500B, ...}


# ── Blockchain.com ──

class BlockchainChartRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: datetime
    chart_name: str
    value: Decimal


# ── Sentiment ──

class FearGreedRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: datetime
    value: int = Field(..., ge=0, le=100)
    classification: str


# ── Etherscan ──

class EthSupplyRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: datetime
    eth_supply: Decimal
    eth2_staking: Decimal
    burnt_fees: Decimal


# ── mempool.space ──

class MempoolMiningRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    timestamp: datetime
    hashrate: Decimal          # TH/s
    difficulty: Decimal


# ── Batch Containers ──

class OnchainBatch(BaseModel):
    """소스 + 데이터 타입별 배치."""
    model_config = ConfigDict(frozen=True)
    source: str                # "defillama", "coinmetrics", ...
    data_type: str             # "stablecoin_total", "mvrv", ...
    records: tuple[BaseModel, ...]  # 위 레코드 타입 중 하나

    @property
    def count(self) -> int:
        return len(self.records)
```

### 2.3 `fetcher.py` — OnchainFetcher

소스별 fetch 메서드를 제공한다. `DerivativesFetcher` 패턴과 동일하게 client를 주입받는다.

```python
"""On-chain data fetcher — async, per-source methods."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.data.onchain.client import AsyncOnchainClient


# Coin Metrics Community API 메트릭 목록
CM_METRICS = [
    "MVRV", "RealCap", "AdrActCnt",
    "TxTfrValAdjUSD", "TxTfrValMeanUSD", "TxTfrValMedUSD",
    "TxCnt", "NVTAdj90", "VtyRet30d",
]

# DeFiLlama 모니터링 체인
DEFI_CHAINS = ["Ethereum", "Tron", "BSC", "Arbitrum", "Solana"]

# Blockchain.com chart 종류
BC_CHARTS = ["hash-rate", "miners-revenue", "transaction-fees-usd"]

# Stablecoin ID (DeFiLlama)
STABLECOIN_IDS = {"USDT": 1, "USDC": 2}


class OnchainFetcher:
    """Tier 1 무료 on-chain 데이터 수집기."""

    def __init__(self, client: AsyncOnchainClient) -> None:
        self._client = client

    # ── DeFiLlama: Stablecoin ──

    async def fetch_stablecoin_total(self) -> pd.DataFrame: ...
    async def fetch_stablecoin_by_chain(self, chain: str) -> pd.DataFrame: ...
    async def fetch_stablecoin_individual(self, sc_id: int) -> pd.DataFrame: ...

    # ── DeFiLlama: TVL + DEX ──

    async def fetch_tvl(self, chain: str = "") -> pd.DataFrame: ...
    async def fetch_dex_volume(self) -> pd.DataFrame: ...

    # ── Coin Metrics Community ──

    async def fetch_coinmetrics(
        self, asset: str, start: str, end: str,
    ) -> pd.DataFrame: ...

    # ── Blockchain.com ──

    async def fetch_blockchain_chart(
        self, chart_name: str, timespan: str = "5years",
    ) -> pd.DataFrame: ...

    # ── Fear & Greed ──

    async def fetch_fear_greed(self) -> pd.DataFrame: ...

    # ── Etherscan ──

    async def fetch_eth_supply(self, api_key: str) -> pd.DataFrame: ...

    # ── mempool.space ──

    async def fetch_mempool_mining(self) -> pd.DataFrame: ...
```

### 2.4 `storage.py` — Bronze/Silver

```python
"""On-chain Bronze (append-only) + Silver (validate + dedup) storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


class OnchainBronzeStorage:
    """Append-only parquet storage for on-chain raw data."""

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir / "onchain"

    def _path(self, source: str, name: str) -> Path:
        return self._base / source / f"{name}.parquet"

    def save(self, df: pd.DataFrame, source: str, name: str) -> Path:
        """Write or append to Bronze parquet."""
        path = self._path(source, name)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=["date"] if "date" in df.columns else None,
                keep="last",
            )

        df.to_parquet(path, compression="zstd", index=False)
        logger.info("Bronze saved: {} ({} rows)", path, len(df))
        return path

    def load(self, source: str, name: str) -> pd.DataFrame | None:
        path = self._path(source, name)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, source: str, name: str) -> bool:
        return self._path(source, name).exists()


class OnchainSilverProcessor:
    """Validate + dedup + type coerce for Silver layer."""

    def __init__(self, bronze_dir: Path, silver_dir: Path) -> None:
        self._bronze = OnchainBronzeStorage(bronze_dir)
        self._silver_base = silver_dir / "onchain"

    def process(self, source: str, name: str) -> Path:
        """Bronze → Silver: validate, dedup, coerce types."""
        df = self._bronze.load(source, name)
        if df is None:
            msg = f"Bronze not found: {source}/{name}"
            raise FileNotFoundError(msg)

        # 1. Dedup
        date_col = "date" if "date" in df.columns else "timestamp"
        df = df.drop_duplicates(subset=[date_col], keep="last")

        # 2. Sort by time
        df = df.sort_values(date_col).reset_index(drop=True)

        # 3. UTC enforce
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], utc=True)

        # 4. Save
        out = self._silver_base / source / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, compression="zstd", index=False)
        logger.info("Silver processed: {} ({} rows)", out, len(df))
        return out
```

---

## 3. CLI 확장

`src/cli/ingest.py`에 `onchain` 서브커맨드 추가:

```bash
# 전체 on-chain 데이터 수집 (Phase 1~6 전체)
uv run mcbot ingest onchain all

# 소스별 개별 수집
uv run mcbot ingest onchain defillama        # Stablecoin + TVL + DEX
uv run mcbot ingest onchain coinmetrics      # MVRV + RealCap + Network metrics
uv run mcbot ingest onchain blockchain-com   # Hash rate + Miner revenue
uv run mcbot ingest onchain fear-greed       # Fear & Greed Index
uv run mcbot ingest onchain etherscan        # ETH supply (--api-key 필요)
uv run mcbot ingest onchain mempool          # BTC mempool mining data

# 옵션
# --start 2020-01-01     시작일 (기본: 소스별 최대 히스토리)
# --end   2026-02-15     종료일 (기본: 오늘)
# --silver               Silver 처리까지 수행 (기본: Bronze만)
# -v / --verbose
```

---

## 4. Settings 확장

```python
# src/config/settings.py 에 추가
onchain_bronze_dir: Path = Field(default=Path("data/bronze/onchain"))
onchain_silver_dir: Path = Field(default=Path("data/silver/onchain"))
etherscan_api_key: SecretStr = Field(default=SecretStr(""))
```

`.env.example` 에 추가:
```env
# On-Chain Data (선택)
# ETHERSCAN_API_KEY=     # Etherscan 무료 API key (Phase 7에서 필요)
```

---

## 5. 의존성 추가

```toml
# pyproject.toml [project.optional-dependencies]
onchain = [
    "httpx[http2]>=0.27",              # Async HTTP client
    "coinmetrics-api-client>=2024.9",  # Coin Metrics Community API
]
```

설치: `uv add --optional onchain "httpx[http2]>=0.27" "coinmetrics-api-client>=2024.9"`

> **참고**: `coinmetrics-api-client`는 Phase 2에서 사용. Phase 1은 httpx만으로 충분.

---

## 6. 구현 Phase

### Phase 1: 인프라 + DeFiLlama Stablecoins (Alpha 최상)

**목표**: 핵심 인프라 구축 + 가장 높은 alpha 데이터 수집

| 작업 | 파일 | 설명 |
|------|------|------|
| 1-1 | `pyproject.toml` | `httpx[http2]` 의존성 추가 |
| 1-2 | `src/config/settings.py` | `onchain_bronze_dir`, `onchain_silver_dir` 설정 추가 |
| 1-3 | `src/data/onchain/client.py` | `AsyncOnchainClient` + `RateLimiter` 구현 |
| 1-4 | `src/data/onchain/models.py` | `StablecoinSupplyRecord`, `StablecoinChainRecord`, `StablecoinIndividualRecord` |
| 1-5 | `src/data/onchain/storage.py` | `OnchainBronzeStorage` + `OnchainSilverProcessor` |
| 1-6 | `src/data/onchain/fetcher.py` | `fetch_stablecoin_total()`, `fetch_stablecoin_by_chain()`, `fetch_stablecoin_individual()` |
| 1-7 | `src/data/onchain/__init__.py` | Public exports |
| 1-8 | `tests/unit/data/test_onchain_client.py` | RateLimiter + Client 테스트 (httpx mock) |
| 1-9 | `tests/unit/data/test_onchain_storage.py` | Bronze append/load + Silver process 테스트 |
| 1-10 | `tests/unit/data/test_onchain_fetcher_defillama.py` | Stablecoin fetcher 테스트 (response mock) |

**수집 데이터**:
- `stablecoin_total_{YEAR}` — 전체 stablecoin mcap 히스토리 (2020~)
- `stablecoin_chain_{CHAIN}_{YEAR}` — Ethereum/Tron/BSC/Arbitrum/Solana별 (2020~)
- `stablecoin_individual_{ID}_{YEAR}` — USDT(1), USDC(2) 개별 (2020~)

**검증 기준**: `ruff check` + `pyright` + `pytest` 0 error

---

### Phase 2: Coin Metrics (MVRV + RealCap)

**목표**: BTC/ETH 핵심 on-chain 밸류에이션 지표

| 작업 | 파일 | 설명 |
|------|------|------|
| 2-1 | `pyproject.toml` | `coinmetrics-api-client` 의존성 추가 |
| 2-2 | `src/data/onchain/models.py` | `CoinMetricsRecord` 추가 |
| 2-3 | `src/data/onchain/fetcher.py` | `fetch_coinmetrics()` 구현 — rate limit 10req/6s 준수 |
| 2-4 | `tests/unit/data/test_onchain_fetcher_coinmetrics.py` | CM API mock 테스트 |

**수집 메트릭**: `MVRV`, `RealCap`, `AdrActCnt`, `TxTfrValAdjUSD`, `TxTfrValMeanUSD`, `TxTfrValMedUSD`, `TxCnt`, `NVTAdj90`, `VtyRet30d`
**자산**: BTC, ETH

---

### Phase 3: DeFiLlama TVL + DEX Volume

**목표**: DEX 활동 및 DeFi 자금 흐름 포착

| 작업 | 파일 | 설명 |
|------|------|------|
| 3-1 | `src/data/onchain/models.py` | `TvlRecord`, `DexVolumeRecord` 추가 |
| 3-2 | `src/data/onchain/fetcher.py` | `fetch_tvl()`, `fetch_dex_volume()` 구현 |
| 3-3 | `tests/unit/data/test_onchain_fetcher_tvl_dex.py` | TVL/DEX mock 테스트 |

**수집 데이터**:
- `tvl_total_{YEAR}` — 전체 체인 합산 TVL
- `tvl_chain_{CHAIN}_{YEAR}` — 주요 5개 체인별 TVL
- `dex_volume_{YEAR}` — 전체 DEX 볼륨

---

### Phase 4: CLI + Orchestration

**목표**: CLI 커맨드로 수집 실행 + 전체 파이프라인 오케스트레이션

| 작업 | 파일 | 설명 |
|------|------|------|
| 4-1 | `src/cli/ingest.py` | `onchain` 서브커맨드 그룹 추가 |
| 4-2 | `src/data/onchain/service.py` | `OnchainDataService` — collect_all(), collect_source() 오케스트레이션 |
| 4-3 | `tests/unit/cli/test_ingest_onchain.py` | CLI 통합 테스트 |

**CLI 명령**:
```bash
uv run mcbot ingest onchain all
uv run mcbot ingest onchain defillama
uv run mcbot ingest onchain coinmetrics
# ...
```

---

### Phase 5: Fear & Greed Index

**목표**: 시장 심리 보조 지표

| 작업 | 파일 | 설명 |
|------|------|------|
| 5-1 | `src/data/onchain/models.py` | `FearGreedRecord` 추가 |
| 5-2 | `src/data/onchain/fetcher.py` | `fetch_fear_greed()` 구현 |
| 5-3 | `tests/unit/data/test_onchain_fetcher_sentiment.py` | F&G mock 테스트 |

---

### Phase 6: Blockchain.com Charts (BTC 보조)

**목표**: BTC 네트워크 건강도 보조 지표

| 작업 | 파일 | 설명 |
|------|------|------|
| 6-1 | `src/data/onchain/models.py` | `BlockchainChartRecord` 추가 |
| 6-2 | `src/data/onchain/fetcher.py` | `fetch_blockchain_chart()` 구현 |
| 6-3 | `tests/unit/data/test_onchain_fetcher_blockchain.py` | BC chart mock 테스트 |

**수집 차트**: `hash-rate`, `miners-revenue`, `transaction-fees-usd`

---

### Phase 7: Etherscan (ETH 보조)

**목표**: ETH supply/staking/burn 모니터링

| 작업 | 파일 | 설명 |
|------|------|------|
| 7-1 | `src/config/settings.py` | `etherscan_api_key` 설정 추가 |
| 7-2 | `src/data/onchain/models.py` | `EthSupplyRecord` 추가 |
| 7-3 | `src/data/onchain/fetcher.py` | `fetch_eth_supply()` 구현 |
| 7-4 | `tests/unit/data/test_onchain_fetcher_etherscan.py` | Etherscan mock 테스트 |

> **Note**: API key 발급 필요 (무료). `.env`에 `ETHERSCAN_API_KEY` 설정.

---

### Phase 8: mempool.space (BTC 라이브 보조)

**목표**: BTC 네트워크 혼잡도 모니터링 (라이브용)

| 작업 | 파일 | 설명 |
|------|------|------|
| 8-1 | `src/data/onchain/models.py` | `MempoolMiningRecord` 추가 |
| 8-2 | `src/data/onchain/fetcher.py` | `fetch_mempool_mining()` 구현 |
| 8-3 | `tests/unit/data/test_onchain_fetcher_mempool.py` | Mempool mock 테스트 |

---

## 7. 테스트 전략

### 단위 테스트 (모든 Phase)

```python
# 패턴: httpx response mock으로 외부 API 의존성 제거
@pytest.fixture
def mock_response():
    """DeFiLlama stablecoin response fixture."""
    return [
        {"date": 1609459200, "totalCirculating": {"peggedUSD": 28_000_000_000}},
        {"date": 1609545600, "totalCirculating": {"peggedUSD": 28_500_000_000}},
    ]

async def test_fetch_stablecoin_total(mock_response, respx_mock):
    respx_mock.get("https://stablecoins.llama.fi/stablecoincharts/all").respond(
        json=mock_response
    )
    async with AsyncOnchainClient() as client:
        fetcher = OnchainFetcher(client)
        df = await fetcher.fetch_stablecoin_total()
    assert len(df) == 2
    assert "total_circulating_usd" in df.columns
```

### 통합 테스트 (Phase 4)

```python
# CLI → Service → Fetcher → Storage 전체 파이프라인
# Bronze → Silver 변환 검증
# 중복 append 시 dedup 동작 검증
```

### 테스트 의존성

```toml
# pyproject.toml [project.optional-dependencies.dev] 에 추가
"respx>=0.21",  # httpx mock library
```

---

## 8. Phase별 예상 파일 수

| Phase | 신규 파일 | 수정 파일 | 테스트 파일 |
|:-----:|:---------:|:---------:|:-----------:|
| 1 | 5 (`client`, `models`, `storage`, `fetcher`, `__init__`) | 2 (`pyproject.toml`, `settings.py`) | 3 |
| 2 | 0 | 2 (`models`, `fetcher`) | 1 |
| 3 | 0 | 2 (`models`, `fetcher`) | 1 |
| 4 | 1 (`service.py`) | 2 (`ingest.py`, `data/__init__.py`) | 1 |
| 5 | 0 | 2 (`models`, `fetcher`) | 1 |
| 6 | 0 | 2 (`models`, `fetcher`) | 1 |
| 7 | 0 | 3 (`models`, `fetcher`, `settings`) | 1 |
| 8 | 0 | 2 (`models`, `fetcher`) | 1 |
| **합계** | **6** | — | **10** |

---

## 9. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|--------|------|------|
| DeFiLlama rate limit 공식 미공개 | 429 응답 가능 | 보수적 25 req/min + exponential backoff |
| Coin Metrics Community 무료 tier 변경 | 데이터 접근 불가 | API 응답 캐싱 + 대안 endpoint 준비 |
| API 응답 스키마 변경 | 파싱 실패 | Pydantic validation + 명확한 에러 로깅 |
| 대량 히스토리 다운로드 시 메모리 | OOM | 연도별 파티셔닝 + 배치 처리 |
| Etherscan API key 노출 | 보안 | SecretStr + .env (gitignore) |

---

## 10. 향후 확장 (Out of Scope)

구현 후 다음 단계로 고려:

1. **FeatureStore 통합**: Silver on-chain 데이터를 `FeatureStore`에 등록하여 전략에서 직접 사용
2. **Cron 자동화**: 일일 수집 스케줄러 (systemd timer 또는 Python scheduler)
3. **데이터 품질 모니터링**: 수집 실패 알림 (Discord)
4. **On-chain Alpha 전략 구현**: Stablecoin flow, MVRV regime filter 등 전략 후보 발굴
5. **Binance 30일 데이터 축적 cron**: 기존 `DerivativesFetcher` 확장 (운영상 가장 시급하지만 별도 작업)
