# On-Chain 데이터 수집 계획

> 작성일: 2026-02-14 | 갱신일: 2026-02-15
> 목적: 유료 API 없이 수집 가능한 온체인 데이터 소스 정리 + Alpha 근거 + 수집 파이프라인 설계

---

## 1. 무료 데이터 소스

### Tier 1: 완전 무료, 높은 가치

#### A. DeFiLlama API (Stablecoins + TVL + DEX)

- **URL**: `https://api.llama.fi` (TVL), `https://stablecoins.llama.fi` (Stablecoins)
- **API Key**: 불필요
- **Rate Limit**: ~30 req/min (공식 미공개)
- **히스토리**: 2020년~ (5년+)
- **License**: Open, 출처 표기 권장

| Endpoint | 설명 | 백테스트 | 라이브 |
|----------|------|:--------:|:------:|
| `GET /stablecoins/stablecoincharts/all` | 전체 stablecoin 히스토리컬 mcap | O | O |
| `GET /stablecoins/stablecoincharts/{chain}` | 체인별 stablecoin 히스토리컬 | O | O |
| `GET /stablecoins/stablecoin/{id}` | 개별 stablecoin 상세 (USDT=1, USDC=2) | O | O |
| `GET /stablecoins/stablecoins` | 전체 stablecoin 목록 + 현재 유통량 | - | O |
| `GET /v2/historicalChainTvl` | 전체 체인 히스토리컬 TVL | O | O |
| `GET /v2/historicalChainTvl/{chain}` | 특정 체인 히스토리컬 TVL | O | O |
| `GET /overview/dexs` | 전체 DEX 볼륨 집계 (dailyChart 포함) | O | O |
| `GET /overview/dexs/{chain}` | 체인별 DEX 볼륨 | O | O |
| `GET /overview/fees` | 프로토콜 수수료 개요 | O | O |

**Pro (유료) 전용**: Yields, Bridges, Derivatives, ETF Data, Token Unlocks

```python
# 전체 stablecoin 히스토리컬 mcap
resp = httpx.get("https://stablecoins.llama.fi/stablecoincharts/all")
# [{"date": 1609459200, "totalCirculating": {"peggedUSD": 28000000000}, ...}, ...]

# 개별 stablecoin (USDT = id 1, USDC = id 2)
resp = httpx.get("https://stablecoins.llama.fi/stablecoin/1")

# Ethereum TVL 히스토리
resp = httpx.get("https://api.llama.fi/v2/historicalChainTvl/Ethereum")
# [{"date": 1609459200, "tvl": 21500000000}, ...]

# 전체 DEX 볼륨 히스토리
resp = httpx.get("https://api.llama.fi/overview/dexs")
daily_volumes = resp.json()["totalDataChart"]  # [[timestamp, volume], ...]
```

---

#### B. Coin Metrics Community API

- **URL**: `https://community-api.coinmetrics.io/v4`
- **API Key**: 불필요
- **Rate Limit**: 10 req / 6초 sliding window (~1.6 RPS)
- **해상도**: Daily
- **히스토리**: BTC genesis (2009)~, ETH genesis~
- **License**: Creative Commons (비상업용)
- **Python Client**: `coinmetrics-api-client` (PyPI)
- **지원 자산**: BTC, ETH, LTC, BCH, XRP, DOGE 등 주요 L1 (BTC 기준 147개 metric)

| Metric ID | 설명 | Alpha 등급 |
|-----------|------|:----------:|
| `MVRV` | Market Value / Realized Value | **최상** |
| `RealCap` | Realized Capitalization | **상** |
| `AdrActCnt` | Active addresses 수 | 중 |
| `TxTfrValAdjUSD` | Adjusted transfer value (노이즈 제거) | 중 |
| `TxTfrValMeanUSD` | 평균 전송 금액 | 중 (whale proxy) |
| `TxTfrValMedUSD` | 중간값 전송 금액 | 중 (whale proxy) |
| `TxCnt` | Transaction count | 보조 |
| `FeeMean` / `FeeTotal` | 수수료 데이터 | 보조 |
| `VtyRet30d` / `VtyRet60d` | 가격 변동성 | 보조 |
| `CurSply` | 현재 공급량 | 보조 |
| `PriceUSD` | 자산 가격 | 보조 |
| `NVTAdj90` | NVT Ratio (90일 이동평균) | **약화 중** |
| `DiffMean` | Mining difficulty | 약 |

```python
from coinmetrics.api_client import CoinMetricsClient

client = CoinMetricsClient()  # API key 불필요
df = client.get_asset_metrics(
    assets="btc",
    metrics=["MVRV", "RealCap", "AdrActCnt", "TxTfrValAdjUSD", "TxTfrValMeanUSD", "TxTfrValMedUSD"],
    start_time="2020-01-01",
    end_time="2026-02-15",
    frequency="1d",
).to_dataframe()
```

---

#### C. Binance Derivatives (기존 수집 확장)

이미 `src/data/derivatives_fetcher.py`에서 Funding Rate, OI, L/S Ratio, Taker Ratio 수집 중.

| 데이터 | 상태 | 해상도 | 히스토리 |
|--------|------|:------:|:--------:|
| Funding Rate | **수집 중** | 8H | Full (listing 이후) |
| Open Interest | **수집 중** | 1H | 최근 30일 → 축적 필수 |
| Long/Short Ratio | **수집 중** | 1H | 최근 30일 → 축적 필수 |
| Taker Buy/Sell Ratio | **수집 중** | 1H | 최근 30일 → 축적 필수 |

**추가 수집 대상** (아직 미구현):

| Endpoint | 설명 | 히스토리 |
|----------|------|:--------:|
| `GET /futures/data/topLongShortPositionRatio` | Top trader L/S ratio (positions) | 30일 |
| `GET /futures/data/topLongShortAccountRatio` | Top trader L/S ratio (accounts) | 30일 |

> **운영 과제**: 30일 히스토리 제한 → 일일 cron job으로 축적 필수 (가장 시급)

---

#### D. Alternative.me Fear & Greed Index

- **URL**: `https://api.alternative.me/fng/`
- **API Key**: 불필요
- **Rate Limit**: 제한 없음 (합리적 사용)
- **해상도**: Daily
- **히스토리**: 2018년 2월~ (8년)

```python
resp = httpx.get("https://api.alternative.me/fng/?limit=0&format=json")
data = resp.json()["data"]
# [{"value": "73", "value_classification": "Greed", "timestamp": "1707868800"}, ...]
```

---

#### E. Blockchain.com Charts API (Bitcoin 전용)

- **URL**: `https://api.blockchain.info/charts/{chart-name}`
- **API Key**: 불필요
- **Rate Limit**: 6 req/min (10초당 1 request)
- **해상도**: Daily
- **히스토리**: BTC genesis (2009)~

| Chart Name | 설명 | Alpha |
|------------|------|:-----:|
| `hash-rate` | Network hash rate (TH/s) | 보조 |
| `miners-revenue` | 채굴자 수익 | 보조 |
| `transaction-fees-usd` | 총 transaction fees (USD) | 보조 |
| `n-transactions` | 일일 confirmed transaction 수 | 보조 |
| `mempool-size` | Mempool 크기 (bytes) | 보조 |
| `difficulty` | Mining difficulty | 약 |

**Parameters**: `timespan` (e.g. `5years`), `format` (`json`/`csv`), `sampled` (`true`/`false`)

```python
url = "https://api.blockchain.info/charts/hash-rate"
params = {"timespan": "5years", "format": "json", "sampled": "false"}
resp = httpx.get(url, params=params)
data = resp.json()  # {"values": [{"x": unix_ts, "y": value}, ...]}
```

---

#### F. mempool.space API (Bitcoin 전용)

- **URL**: `https://mempool.space/api`
- **API Key**: 불필요
- **Rate Limit**: ~10 req/min (공식 미공개)

| Endpoint | 백테스트 | 라이브 | 용도 |
|----------|:--------:|:------:|------|
| `GET /v1/fees/recommended` | X (현재만) | O | BTC 네트워크 혼잡도 |
| `GET /v1/fees/mempool-blocks` | X (현재만) | O | 블록 예측 |
| `GET /v1/mining/hashrate/{interval}` | O (3y까지) | O | Hashrate 추이 |
| `GET /v1/mining/difficulty-adjustments` | O | O | Difficulty 이력 |

> **판정**: 라이브 트레이딩 거래 비용 모니터링에만 활용. Alpha 소스로는 약함.

---

### Tier 2: Freemium (제한적 무료)

| Source | 무료 범위 | 한계 | API Key | 판정 |
|--------|-----------|------|---------|------|
| **Etherscan** | Gas oracle, ETH supply+staking+burned | 5 req/sec, 100K/day | 무료 발급 필요 | ETH 특화 보조 지표로 활용 가능 |
| **Glassnode** | 웹 대시보드 일부 | API는 Professional 이상 유료 add-on | 가입 필요 | **수집 비권장** — Coin Metrics가 대안 |
| **CoinGecko** | 가격/시총/볼륨 (18K+ 코인) | 30 req/min, 10K/월 | Demo key 무료 | 가격 데이터는 Binance로 충분, DeFi token 시총만 보조 |
| **Santiment** | 2년 히스토리 (최근 30일 제외) | 라이브 불가 | 가입 필요 | **수집 비권장** — 30일 갭으로 라이브 불가 |

#### Etherscan API (ETH 전용)

```python
# ETH Supply (including staking + burned)
resp = httpx.get(
    "https://api.etherscan.io/api",
    params={
        "module": "stats",
        "action": "ethsupply2",
        "apikey": "YOUR_FREE_API_KEY",
    },
)
# {"EthSupply": "120123456...", "Eth2Staking": "...", "BurntFees": "..."}
```

### 수집 비권장 (무료 tier 실용성 부족)

| Source | 이유 |
|--------|------|
| **CoinGlass API** | 유료 전용 (웹 UI만 무료) |
| **CryptoQuant API** | 유료 (대시보드만 무료) |
| **Santiment** | 최근 30일 제외 → 라이브 불가 |
| **Whale Alert** | 7일 trial 후 유료 ($20+/mo) |
| **Arkham Intelligence** | 웹 대시보드만 무료, API 유료 |
| **Glassnode API** | 2025-26 기준 API는 Professional 이상 optional add-on |

---

## 2. Alpha 근거 + 전략 아이디어

### 최우선: Stablecoin Flows (Alpha 등급: 최상)

**가장 깨끗한 on-chain 자금 유입 시그널.**

- Tether mint = 시장 진입 자금. 2025년 stablecoin 총 supply $300B 근접
- Stablecoin Velocity (거래량/시총) — 2026년 가장 깨끗한 on-chain 활동 시그널
- USDT vs USDC 비율 변화 — 시장 선호도/규제 환경 반영
- 체인별 Flow — Ethereum→Tron = 개인 송금, Ethereum→L2 = DeFi 활동

**학술 근거:**
- BIS Working Paper No 1270: Stablecoin 자금 흐름이 안전자산 가격에 영향
- NY Fed Research: Stablecoin과 crypto shock의 관계 분석

**전략 아이디어:**

| Signal | 구현 | 기대 효과 |
|--------|------|----------|
| `Δ(stablecoin_supply) / 7d` | 7일 supply 변화율 | 자금 유입 모멘텀 |
| `stablecoin_velocity = volume / mcap` | DeFiLlama volume / supply | 시장 활동 레벨 |
| `usdt_dominance = usdt / total` | 비율 추이 | 시장 구조 변화 |
| `chain_flow = Δ(chain_stablecoin) / dt` | 체인별 변화율 비교 | 자금 이동 방향 |

---

### 최우선: MVRV + Realized Cap (Alpha 등급: 상)

- MVRV > 3.7 → 과열, MVRV < 1 → 저평가. 사이클 Top/Bottom 판별에 학계/실무 모두 검증
- Realized Cap = on-chain 비용 기반 시가총액
- 2025-26년에도 유효성 유지 (ETF 시대에도 BTC UTXO 기반 계산은 영향 적음)

**전략 아이디어:**

| Signal | 구현 | 기대 효과 |
|--------|------|----------|
| `mvrv_zscore` | (MV - RealCap) / std(MV - RealCap) | 사이클 위치 정량화 |
| `mvrv_regime` | >3.7 과열 / <1 저평가 / 중립 | Regime filter |
| `realcap_momentum` | Δ(RealCap) / 30d | On-chain 비용 기반 모멘텀 |

---

### 최우선: DEX Volume / CEX-DEX Ratio (Alpha 등급: 상)

- 2025년 11월 기준 DEX/CEX spot ratio ~20%, perps ratio ~11.7%
- DEX 비중 급등 = retail 활동 증가 시그널
- 구조적 시장 변화의 leading indicator

**전략 아이디어:**

| Signal | 구현 | 기대 효과 |
|--------|------|----------|
| `dex_volume_momentum` | Δ(dex_volume) / 7d | DEX 활동 증가 감지 |
| `dex_cex_ratio_trend` | dex_volume / cex_volume 추이 | 구조적 전환 포착 |

---

### 최우선: Derivatives 3차원 (Alpha 등급: 상)

- FR + OI + Price 3차원 분석이 단순 FR 대비 우위 확인
- Funding rate > 0.05%/8h → 과도한 레버리지 축적 → liquidation cascade 선행
- SSRN 논문 검증: Binance BTC perpetual funding rate의 OOS 예측 가능성

> **주의**: FR은 trailing indicator 성격 — 모멘텀의 부산물일 수 있으므로 단독 사용보다 conditioning factor로 활용

---

### 보조: Whale Activity Proxy (Alpha 등급: 중)

직접적인 whale tracking은 무료로 불가능. Coin Metrics 간접 proxy가 유일한 대안.

```python
# Mean/Median 비율로 whale 활동 간접 추정
whale_proxy = TxTfrValMeanUSD / TxTfrValMedUSD
# 비율이 높을수록 대형 거래가 활발
```

---

### 보조: Fear & Greed Index (Alpha 등급: 중)

- Extreme Fear (<20) → 매수 기회, Extreme Greed (>80) → 경계
- Contrarian indicator로 검증됨
- 단독 alpha 약함, regime filter 보조 지표로 활용

---

### 참고용 (낮은 가치 또는 신뢰도 하락)

| Metric | 이유 |
|--------|------|
| **NVT** | ETF/기관 off-chain 거래 증가로 on-chain tx volume 기반 지표 신뢰도 저하 |
| **Difficulty** | 반응 느림, ~2주 단위 조정으로 예측 가능 |
| **Block Rewards** | Halving 스케줄로 완전히 예측 가능 |
| **Mempool Data** | 가격 방향성과 약한 상관, 히스토리컬 제한 |

---

## 3. 2025-2026 On-Chain Alpha 트렌드

### 유효한 시그널

| Signal | 등급 | 설명 | 학술/실무 근거 |
|--------|:----:|------|----------------|
| **Stablecoin Supply 변화** | 최상 | 신규 자금 유입 프록시 ($300B+) | BIS WP 1270, NY Fed Research |
| **Stablecoin Velocity** | 최상 | 거래량/시총 비율 — 가장 깨끗한 활동 시그널 | TRM Labs 2025 Report |
| **CEX vs DEX Volume Divergence** | 상 | DEX 월간 $857B — 구조적 시장 변화 | CoinGecko Research |
| **FR + OI + Price 3차원** | 상 | 단순 FR 대비 우위 | SSRN, Presto Research |
| **MVRV** | 상 | 사이클 위치 판단에 유효 | David Puell, 다수 학술 논문 |
| **Exchange Reserve 감소** | 상 | 2025년 multi-year 저점, 공급 제약 | 무료 API 없음 (참고만) |

### 구조적 변화 (주의 사항)

1. **NVT 신뢰도 하락**: Off-chain 거래 (ETF, 기관 custody) 증가로 on-chain tx volume 기반 지표 약화
2. **Exchange Flow 의미 변화**: 2022-23년 exchange inflow = 매도 압력 → 2025-26년 ETF 차익거래 등 다양한 의미
3. **기관 자금은 on-chain에 보이지 않음**: ETF 통한 BTC 매수는 기관 custody로 진행 → on-chain 미반영
4. **Signal Decay**: 널리 알려진 on-chain 시그널은 self-regulating mechanism에 의해 alpha 감소 경향

---

## 4. 종합 우선순위 Matrix

| 순위 | 데이터 | 소스 | 해상도 | 히스토리 | Alpha | 난이도 | 비용 |
|:----:|--------|------|:------:|:--------:|:-----:|:------:|:----:|
| **1** | Stablecoin Supply + Chain별 | DeFiLlama | Daily | 5년+ | **최상** | 낮음 | 무료 |
| **2** | MVRV + Realized Cap | Coin Metrics | Daily | Full | **상** | 낮음 | 무료 |
| **3** | DEX Volume (CEX/DEX ratio) | DeFiLlama | Daily | 4년+ | **상** | 낮음 | 무료 |
| **4** | FR + OI + Price 3차원 | Binance | 1H-8H | 축적 중 | **상** | 완료 | 무료 |
| **5** | L/S Ratio + Taker Ratio | Binance | 1H | 30일→축적 | **중-상** | 완료 | 무료 |
| **6** | Active Addr + Tx Value + Whale Proxy | Coin Metrics | Daily | Full | **중** | 낮음 | 무료 |
| **7** | DeFi TVL by Chain | DeFiLlama | Daily | 5년+ | **중** | 낮음 | 무료 |
| **8** | Fear & Greed Index | Alternative.me | Daily | 8년 | **중** | 최소 | 무료 |
| **9** | Hash Rate + Miner Revenue | Blockchain.com | Daily | Full | **보조** | 낮음 | 무료 |
| **10** | ETH Supply/Staking/Burned | Etherscan | Daily | Full | **보조** | 낮음 | 무료 |
| **11** | BTC Mempool Fees | mempool.space | RT | 제한 | **약** | 낮음 | 무료 |
| --- | Liquidation (실시간 축적만) | Binance WS | RT | 없음 | 상 (데이터 부족) | 중 | 무료 |
| --- | SOPR / Exchange Reserve | Glassnode | Daily | Full | 상 (접근 불가) | — | **유료** |
| --- | Whale Tracking (직접) | Whale Alert | RT | 수년 | 중 (접근 불가) | — | **유료** |

---

## 5. 수집 Pipeline 설계

### 저장 구조 (기존 Medallion 패턴 확장)

```
data/bronze/onchain/
├── defillama/
│   ├── stablecoin_supply_2026.parquet        # 전체 stablecoin mcap 히스토리
│   ├── stablecoin_by_chain_2026.parquet      # 체인별 stablecoin 분포
│   ├── stablecoin_individual_2026.parquet    # USDT/USDC 개별 추이
│   ├── tvl_chains_2026.parquet              # 체인별 TVL
│   └── dex_volume_2026.parquet              # DEX 볼륨 집계
├── coinmetrics/
│   ├── BTC_metrics_2026.parquet             # MVRV, RealCap, AdrAct, TxTfr, ...
│   └── ETH_metrics_2026.parquet
├── binance_derivatives/
│   ├── top_ls_ratio_2026.parquet            # Top trader L/S ratio (신규)
│   └── top_ls_account_2026.parquet          # Top trader L/S account (신규)
├── blockchain_com/
│   ├── hash_rate_2026.parquet
│   ├── miners_revenue_2026.parquet
│   └── transaction_fees_2026.parquet
├── sentiment/
│   └── fear_greed_2026.parquet
├── etherscan/
│   └── eth_supply_2026.parquet
└── mempool_space/
    └── fees_mining_2026.parquet
```

### 수집 주기

| 데이터 | 주기 | 이유 |
|--------|:----:|------|
| DeFiLlama (Stablecoins, TVL, DEX) | 1일 1회 | Daily resolution, API 부하 최소화 |
| Coin Metrics (MVRV, RealCap, Network) | 1일 1회 | Daily resolution, rate limit 고려 |
| Blockchain.com charts | 1일 1회 | Daily data |
| Fear & Greed Index | 1일 1회 | Daily update (00:00 UTC) |
| Etherscan (ETH supply) | 1일 1회 | Daily 변화 충분 |
| Binance L/S Ratio, Taker Ratio | 4시간마다 | **30일 히스토리 제한 → 축적 필수** |
| Binance Top Trader Ratios | 4시간마다 | **30일 히스토리 제한 → 축적 필수** |
| mempool.space fees | 4시간마다 | Fee 변동 빠름, 라이브 모니터링용 |

### 의존성

```toml
# pyproject.toml [project.optional-dependencies]
onchain = [
    "coinmetrics-api-client",  # Coin Metrics Community API
]
# httpx, pyarrow, pandas 는 기존 의존성
```

### 수집 Pipeline 골격

```python
"""on-chain data collector — 일일/4시간 cron 실행."""

from __future__ import annotations

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

BRONZE_ONCHAIN = Path("data/bronze/onchain")


class OnchainCollector:
    """무료 on-chain 데이터 수집기."""

    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self._client = http_client or httpx.Client(timeout=30.0)

    # ── DeFiLlama Stablecoins (최우선) ──

    def fetch_stablecoin_mcap(self) -> pd.DataFrame:
        """전체 stablecoin 히스토리컬 mcap."""
        resp = self._client.get("https://stablecoins.llama.fi/stablecoincharts/all")
        resp.raise_for_status()
        rows = [
            {"date": d["date"], "totalCirculating": d["totalCirculating"]["peggedUSD"]}
            for d in resp.json()
        ]
        return pd.DataFrame(rows)

    def fetch_stablecoin_by_chain(self, chain: str) -> pd.DataFrame:
        """체인별 stablecoin 히스토리컬."""
        resp = self._client.get(f"https://stablecoins.llama.fi/stablecoincharts/{chain}")
        resp.raise_for_status()
        rows = [
            {"date": d["date"], "totalCirculating": d["totalCirculating"]["peggedUSD"]}
            for d in resp.json()
        ]
        return pd.DataFrame(rows)

    def fetch_stablecoin_individual(self, stablecoin_id: int) -> pd.DataFrame:
        """개별 stablecoin 상세 (USDT=1, USDC=2)."""
        resp = self._client.get(f"https://stablecoins.llama.fi/stablecoin/{stablecoin_id}")
        resp.raise_for_status()
        data = resp.json()
        chains = data.get("chainBalances", {})
        # 전체 supply 히스토리 추출
        tokens = data.get("tokens", [])
        rows = [
            {"date": t["date"], "circulating": t["circulating"]["peggedUSD"]}
            for t in tokens
            if "circulating" in t and "peggedUSD" in t.get("circulating", {})
        ]
        return pd.DataFrame(rows)

    # ── DeFiLlama TVL + DEX ──

    def fetch_chain_tvl(self, chain: str = "") -> pd.DataFrame:
        """체인별 히스토리컬 TVL."""
        url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}".rstrip("/")
        resp = self._client.get(url)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())

    def fetch_dex_volume(self) -> pd.DataFrame:
        """전체 DEX 볼륨 히스토리."""
        resp = self._client.get("https://api.llama.fi/overview/dexs")
        resp.raise_for_status()
        chart = resp.json().get("totalDataChart", [])
        rows = [{"date": row[0], "volume": row[1]} for row in chart]
        return pd.DataFrame(rows)

    # ── Coin Metrics Community ──

    def fetch_coinmetrics(
        self, assets: list[str], metrics: list[str], start: str, end: str,
    ) -> pd.DataFrame:
        """Coin Metrics Community API — daily metrics."""
        url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        params = {
            "assets": ",".join(assets),
            "metrics": ",".join(metrics),
            "start_time": start, "end_time": end,
            "frequency": "1d", "page_size": 10000,
        }
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return pd.DataFrame(resp.json()["data"])

    # ── Blockchain.com Charts ──

    def fetch_blockchain_chart(
        self, chart_name: str, timespan: str = "5years",
    ) -> pd.DataFrame:
        """Blockchain.com chart data (BTC only)."""
        url = f"https://api.blockchain.info/charts/{chart_name}"
        params = {"timespan": timespan, "format": "json", "sampled": "false"}
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        values = resp.json()["values"]
        df = pd.DataFrame(values)
        df["timestamp"] = pd.to_datetime(df["x"], unit="s", utc=True)
        return df.rename(columns={"y": chart_name})[["timestamp", chart_name]]

    # ── Fear & Greed Index ──

    def fetch_fear_greed(self) -> pd.DataFrame:
        """Alternative.me Fear & Greed Index — 전체 히스토리."""
        resp = self._client.get("https://api.alternative.me/fng/?limit=0&format=json")
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["value"] = df["value"].astype(int)
        return df[["timestamp", "value", "value_classification"]]

    # ── Etherscan ──

    def fetch_eth_supply(self, api_key: str) -> dict[str, str]:
        """ETH supply (total + staking + burned)."""
        resp = self._client.get(
            "https://api.etherscan.io/api",
            params={"module": "stats", "action": "ethsupply2", "apikey": api_key},
        )
        resp.raise_for_status()
        return resp.json()["result"]
```

### Parquet 저장

```python
def save_parquet(df: pd.DataFrame, source: str, name: str, year: int) -> Path:
    """Bronze onchain parquet 저장."""
    out_dir = BRONZE_ONCHAIN / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{year}.parquet"
    pq.write_table(
        pa.Table.from_pandas(df),
        path,
        compression="snappy",
    )
    return path
```

---

## 6. 구현 우선순위

| Phase | 대상 | 소스 | 주기 | 난이도 | 비고 |
|:-----:|------|------|:----:|:------:|------|
| **1** | Stablecoin Supply + Chain별 + USDT/USDC | DeFiLlama | Daily | 낮음 | 최고 alpha, 가장 쉬운 구현 |
| **2** | MVRV + RealCap + AdrAct + TxTfrVal | Coin Metrics | Daily | 낮음 | 공식 Python client |
| **3** | DEX Volume + TVL by Chain | DeFiLlama | Daily | 낮음 | Phase 1과 같은 API |
| **4** | Binance 30일 데이터 축적 cron | Binance | 4H | 낮음 | 기존 패턴 확장, **운영상 가장 시급** |
| **5** | Fear & Greed Index | Alternative.me | Daily | 최소 | 5줄 구현 |
| **6** | Hash Rate + Miner Revenue | Blockchain.com | Daily | 낮음 | BTC 전용 보조 |
| **7** | ETH Supply/Staking/Burned | Etherscan | Daily | 낮음 | API key 발급 필요 |
| **8** | mempool.space (BTC fees) | mempool.space | 4H | 낮음 | 라이브 모니터링용 |

---

## 7. References

| Source | URL | 설명 |
|--------|-----|------|
| BIS Working Paper No 1270 | `https://www.bis.org/publ/work1270.pdf` | Stablecoin 자금 흐름과 안전자산 가격 |
| NY Fed Research | `https://libertystreeteconomics.newyorkfed.org/2025/04/stablecoins-and-crypto-shocks-an-update/` | Stablecoin과 crypto shock |
| SSRN: Funding Rate Predictability | `https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424` | Binance BTC FR OOS 예측 |
| Presto Research | `https://www.prestolabs.io/research/can-funding-rate-predict-price-change` | FR 가격 예측력 분석 |
| MVRV Ratio (David Puell) | `https://medium.com/@kenoshaking/bitcoin-market-value-to-realized-value-mvrv-ratio-3ebc914dbaee` | MVRV 원저자 설명 |
| TRM Labs 2025 Report | `https://www.trmlabs.com/reports-and-whitepapers/2025-crypto-adoption-and-stablecoin-usage-report` | Stablecoin 시장 규모 |
| CoinGecko DEX/CEX Ratio | `https://www.coingecko.com/research/publications/dex-to-cex-ratio` | DEX/CEX 비율 분석 |
| DeFiLlama API Docs | `https://api-docs.defillama.com/` | API 공식 문서 |
| Coin Metrics Community Docs | `https://gitbook-docs.coinmetrics.io/packages/coin-metrics-community-data` | Community API 문서 |
