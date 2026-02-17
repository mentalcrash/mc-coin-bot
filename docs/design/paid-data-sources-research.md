# 유료 데이터 소스 리서치

> 작성일: 2026-02-15
> 목적: 유료 API 도입 시 가장 높은 ROI를 제공하는 데이터 소스 정리 + Alpha 근거 + 비용 분석
> 상태: **리서치 단계** — 실제 적용 여부 미정

---

## 1. 투자 우선순위 요약

### Tier 1: 즉시 투자 권장 (합계 ~$228/mo)

| Provider | Plan | 월 비용 | 핵심 가치 | Alpha 등급 |
|----------|------|---------|-----------|:----------:|
| **CoinGlass** | Startup | $79/mo | Cross-exchange 파생 aggregation, Liquidation Heatmap, ETF Flow | **상** |
| **Laevitas** | Premium | $50/mo | Options Vol Surface/Skew — forward-looking sentiment | **상** |
| **CryptoQuant** | Professional | $99/mo | Exchange Reserve, Miner Flow, Stablecoin Supply Ratio | **상** |

### Tier 2: 전략 확장 시 투자 (합계 ~$1,000-1,200/mo)

| Provider | Plan | 월 비용 | 핵심 가치 | Alpha 등급 |
|----------|------|---------|-----------|:----------:|
| **Tardis.dev** | Solo/Pro | ~$200-400/mo | Historical LOB data, tick-level trades/liquidations | **최상** |
| **Glassnode** | Professional | ~$799/mo | Entity-adjusted SOPR, HODL Waves — 독점 metrics | **상** |

### Tier 3: Free tier 활용 ($0)

| Provider | 활용 | 비고 |
|----------|------|------|
| **Arkham Intelligence** | Whale/institution entity tracking | Free API로 정부·기관 지갑 추적 |
| **Dune Analytics** | Custom SQL on-chain research | 2,500 credits/mo 무료 |
| **Coinalyze** | Aggregated derivatives (40 req/min) | CoinGlass 경량 대안 |

---

## 2. 상세 분석: On-Chain Analytics

### A. CryptoQuant (Professional $99/mo) — **Tier 1 권장**

- **URL**: `https://cryptoquant.com`
- **API**: REST + GraphQL
- **해상도**: 분 단위 (Professional 이상)
- **커버리지**: BTC, ETH + 주요 L1, 500+ exchanges

| Metric | 설명 | Alpha 등급 | 무료 대안 유무 |
|--------|------|:----------:|:--------------:|
| **Exchange Reserve** | Exchange별 자산 보유량 추이 | **상** | 없음 |
| **Exchange Netflow** | 거래소 순유입/유출 | **상** | 없음 |
| **Miner-to-Exchange Flow** | 채굴자 매도 압력 | **중-상** | 없음 |
| **Stablecoin Supply Ratio** | BTC mcap / Stablecoin supply | **상** | DeFiLlama로 부분 대체 가능 |
| **Fund Flow Ratio** | 거래소 유입 / 전체 on-chain tx | **중** | 없음 |
| **Estimated Leverage Ratio** | OI / Exchange Reserve | **상** | 없음 |
| MVRV, NVT, SOPR | 기본 on-chain valuation | 중 | Coin Metrics Community로 MVRV 대체 가능 |

**Glassnode 대비 장점:**
- 1/8 가격 ($99 vs $799)으로 핵심 Exchange Flow 데이터 커버
- Derivatives 데이터 통합 (Estimated Leverage Ratio 등)
- 분 단위 granularity (Glassnode Professional과 동등)

**Glassnode에서만 가능한 것:**
- Entity-adjusted metrics (내부 전송 제거)
- Realized Cap HODL Waves
- Long/Short-Term Holder 분류
- 800+ metrics (CryptoQuant 대비 깊이)

**결론**: Exchange Flow + derivatives 통합이 필요하면 CryptoQuant, macro cycle 정밀 분석이 필요하면 Glassnode.

```python
# CryptoQuant API 예시 (Professional)
import httpx

headers = {"Authorization": "Bearer YOUR_API_KEY"}

# Exchange Reserve (BTC)
resp = httpx.get(
    "https://api.cryptoquant.com/v1/btc/exchange-flows/reserve",
    headers=headers,
    params={"exchange": "all_exchange", "window": "day", "limit": 365},
)

# Estimated Leverage Ratio
resp = httpx.get(
    "https://api.cryptoquant.com/v1/btc/market-data/estimated-leverage-ratio",
    headers=headers,
    params={"window": "day", "limit": 365},
)
```

---

### B. Glassnode (Professional ~$799/mo) — **Tier 2**

- **URL**: `https://glassnode.com`
- **API**: REST, well-documented
- **해상도**: 10분 (Professional), 1시간 (Advanced $29), Daily (Free)
- **커버리지**: BTC, ETH, LTC + 주요 ERC-20, 800+ metrics

| Tier | 가격 | 해상도 | Historical | API |
|------|------|:------:|:----------:|:---:|
| Free | $0 | Daily | 제한 | X |
| Advanced | ~$29/mo | 1시간 | 1개월 | X |
| Professional | ~$799/mo | 10분 | Full | O (add-on) |
| Enterprise | 협의 | 10분 | Full | O |

**독점 Metrics (다른 곳에서 불가):**

| Metric | 설명 | Alpha 등급 |
|--------|------|:----------:|
| **Entity-Adjusted SOPR** | 내부 전송 제거 → 실제 수익/손실 행동 | **최상** |
| **Realized Cap HODL Waves** | 코인 연령별 Realized Cap 분포 | **상** |
| **LTH/STH Supply** | 장기/단기 보유자 공급량 분류 | **상** |
| **Liveliness** | Coin Days Destroyed / Created 비율 | **중-상** |
| **Entity-Adjusted CDD** | 실제 경제적 의미의 Coin Days Destroyed | **중** |

**Advanced ($29/mo) 한계:**
- Historical 1개월만 제공 → **백테스트 불가능**
- API 미포함
- 결론: Advanced는 사실상 무용. Professional 아니면 Free로 충분

```python
# Glassnode API 예시 (Professional + API add-on)
import httpx

params = {
    "a": "BTC",
    "api_key": "YOUR_API_KEY",
    "s": "1609459200",  # 2021-01-01
    "u": "1739577600",  # 2025-02-15
    "i": "24h",
}

# Entity-Adjusted SOPR
resp = httpx.get(
    "https://api.glassnode.com/v1/metrics/indicators/sopr_adjusted",
    params=params,
)

# LTH/STH Supply
resp = httpx.get(
    "https://api.glassnode.com/v1/metrics/supply/sth_sum",
    params=params,
)
```

---

### C. Nansen (Pioneer $99/mo) — **DeFi/Altcoin 확장 시만**

- **URL**: `https://nansen.ai`
- **핵심**: 500M+ wallet labels, Smart Money tracking
- **커버리지**: EVM chains 중심 (Ethereum, BSC, Polygon, Arbitrum, ...)

| Tier | 가격 | Wallet Labels | API |
|------|------|:-------------:|:---:|
| Free | $0 | 제한 | X |
| Pioneer | $99/mo | 250M+ | 기본 |
| Professional | $999/mo | 500M+ | Full |
| Enterprise | $2,000+/mo | 전체 | 전용 |

**MC Coin Bot 맥락 평가:**
- BTC/ETH CEX futures 전략에는 **직접적 alpha 낮음**
- DeFi token flow, altcoin whale tracking이 핵심 use case
- Smart Money wallet 대량 이동은 시장 전환점 시그널로 가치 있으나, 빈도 낮음
- **결론: 현재 전략에는 불필요. DeFi/altcoin 확장 시 재평가**

---

### D. Santiment (SanAPI Pro ~$149/mo) — **Social+On-Chain 통합**

- **URL**: `https://santiment.net`
- **API**: GraphQL (`sanpy` Python client)
- **해상도**: Daily (on-chain), 5분 (social)
- **히스토리**: 2014년~
- **커버리지**: 2,000+ crypto assets

| Tier | 가격 | 핵심 기능 |
|------|------|-----------|
| Free | $0 | 제한적 metrics |
| Sanbase Pro | ~$49/mo | Dashboard, 기본 API |
| SanAPI Pro | ~$149/mo | Full API, on-chain + social + dev |
| Business | 협의 | Enterprise API |

**유일한 3-in-1 Provider (On-Chain + Social + Dev Activity):**

| Metric | 설명 | Alpha 등급 |
|--------|------|:----------:|
| **Social Volume** | Twitter/Reddit/Telegram 멘션 빈도 | **중** (contrarian) |
| **Weighted Sentiment** | 감성 가중 소셜 점수 | **중** |
| **Development Activity** | GitHub commit frequency per project | **보조** |
| **Token Age Consumed** | Dormant token 이동량 | **중-상** |
| **Exchange Inflow/Outflow** | 거래소 유출입 | 중 (CryptoQuant이 더 정밀) |

**판정**: Social sentiment 단독은 noisy. Extreme sentiment만 contrarian으로 활용 가치 있음. CryptoQuant과 중복되는 on-chain 부분 많음. **Social data가 필요할 때만 고려**.

```python
# Santiment API 예시
import san  # pip install sanpy

san.ApiConfig.api_key = "YOUR_API_KEY"

# Social Volume
df = san.get(
    "social_volume_total",
    slug="bitcoin",
    from_date="2024-01-01",
    to_date="2026-02-15",
    interval="1d",
)

# Token Age Consumed
df = san.get(
    "age_consumed",
    slug="bitcoin",
    from_date="2024-01-01",
    to_date="2026-02-15",
    interval="1d",
)
```

---

## 3. 상세 분석: Derivatives Data

### A. CoinGlass (Startup $79/mo) — **Tier 1 권장**

- **URL**: `https://www.coinglass.com`
- **API**: REST V4
- **Latency**: < 1분

| Tier | 월 비용 | 연 비용 | Endpoints | Rate Limit |
|------|---------|---------|:---------:|:---------:|
| Hobbyist | $29/mo | $348/yr | 70+ | 30/min |
| **Startup** | **$79/mo** | **$948/yr** | **80+** | **80/min** |
| Standard | $299/mo | $3,588/yr | 90+ | 300/min |
| Professional | $699/mo | $8,388/yr | 100+ | 1,200/min |

**핵심 데이터 (현재 Binance 단독 대비 추가 가치):**

| Data | 설명 | 현재 보유 | CoinGlass 추가 가치 |
|------|------|:---------:|:-------------------:|
| **Aggregated OI** | 전체 exchange OI 합산 | Binance만 | 12+ exchanges 합산 |
| **Aggregated Funding** | Exchange별 FR 비교 | Binance만 | Arbitrage 기회 포착 |
| **Liquidation Heatmap** | 가격 레벨별 청산 집중도 | 없음 | **Support/Resistance 식별** |
| **Liquidation Data** | Cross-exchange 실시간 청산 | 없음 | **Cascade 예측** |
| **ETF Flow** | BTC/ETH ETF 일일 순유입 | 없음 | **기관 자금 흐름** |
| **Options OI/Volume** | 옵션 시장 구조 | 없음 | Max Pain, P/C Ratio |
| **Exchange Balance** | Exchange별 자산 보유 변화 | 없음 | 유출입 추이 |

**전략 아이디어:**

| Signal | 구현 | 기대 효과 |
|--------|------|-----------|
| `agg_oi_change` | Δ(total_oi) / 4h | 전체 시장 레버리지 변화 |
| `funding_dispersion` | std(FR across exchanges) | Exchange 간 FR 괴리 → arbitrage/squeeze |
| `liquidation_cluster` | heatmap density at price level | 자기실현적 S/R 레벨 |
| `etf_flow_momentum` | Σ(etf_netflow) / 5d | 기관 자금 유입 모멘텀 |

```python
# CoinGlass API V4 예시
import httpx

headers = {"CG-API-KEY": "YOUR_API_KEY"}
base = "https://open-api-v4.coinglass.com/api"

# Aggregated Open Interest (historical)
resp = httpx.get(
    f"{base}/futures/openInterest/ohlc-history",
    headers=headers,
    params={"symbol": "BTC", "interval": "4h", "limit": 500},
)

# Liquidation data
resp = httpx.get(
    f"{base}/futures/liquidation/v2/history",
    headers=headers,
    params={"symbol": "BTC", "interval": "1h", "limit": 500},
)

# ETF Flow (BTC)
resp = httpx.get(
    f"{base}/index/bitcoin-etf-flow-total",
    headers=headers,
)

# Aggregated Funding Rate
resp = httpx.get(
    f"{base}/futures/fundingRate/ohlc-history",
    headers=headers,
    params={"symbol": "BTC", "interval": "8h", "limit": 500},
)
```

---

### B. Laevitas (Premium $50/mo) — **Tier 1 권장**

- **URL**: `https://www.laevitas.ch`
- **핵심**: Options volatility surface, skew, Greeks
- **커버리지**: Deribit, Binance, OKX, Bybit options

| Tier | 가격 | 핵심 기능 |
|------|------|-----------|
| Free | $0 | 기본 차트 |
| **Premium** | **$50/mo** | Enhanced analytics, 일부 API |
| Enterprise | $500/mo | Full API, advanced features |

**핵심 데이터 (forward-looking indicators):**

| Metric | 설명 | Alpha 등급 |
|--------|------|:----------:|
| **ATM Implied Volatility** | 현재 시장 기대 변동성 (7d/30d/90d) | **상** |
| **25-Delta Skew** | Put vs Call IV 차이 → 방향성 기대 | **최상** |
| **Volatility Surface** | Strike × Expiry IV matrix | **상** |
| **Put/Call Ratio** | 풋/콜 OI 비율 | **중-상** |
| **Max Pain** | 옵션 만기일 최대 손실 가격 | **중** |
| **Options Block Trades** | 대형 옵션 거래 흐름 | **중** |
| **Gamma Exposure (GEX)** | 딜러 감마 노출 → 변동성 예측 | **상** |

**왜 Forward-Looking인가:**
- 가격/on-chain/sentiment 데이터는 모두 **과거 기반 (backward-looking)**
- Options IV/Skew는 시장 참여자들의 **미래 기대치를 직접 반영**
- 25-delta skew 급등 → 하방 hedging 수요 증가 → 시장 불안 선행 지표
- 2024 BTC ETF 승인 이후 options 시장 유동성 급증 → 신뢰도 상승

**전략 아이디어:**

| Signal | 구현 | 기대 효과 |
|--------|------|-----------|
| `iv_term_structure` | IV(7d) / IV(30d) | Contango/Backwardation → 단기 변동성 기대 |
| `skew_regime` | 25d skew z-score | Extreme put skew → contrarian long |
| `gex_sign` | Aggregate dealer gamma | Positive GEX → low vol, Negative GEX → high vol |
| `vol_risk_premium` | IV - RV(realized) | 과대/과소 평가된 변동성 |

---

### C. Coinalyze (Free) — **Tier 3, 추가 투자 불필요**

- **URL**: `https://coinalyze.net`
- **Free**: 40 req/min, aggregated OI/Funding/Liquidation
- **Ad-Free**: $10.95/mo (광고 제거)
- **판정**: CoinGlass 도입 전 경량 대안으로 활용 가능. CoinGlass 대비 depth 부족.

---

## 4. 상세 분석: Institutional-Grade Market Data

### A. Tardis.dev (Solo ~$200-400/mo) — **Tier 2, 최고 Alpha 잠재력**

- **URL**: `https://tardis.dev`
- **핵심**: Historical tick-level order book + trades data
- **커버리지**: 30+ exchanges (Binance, Deribit, OKX, Bybit, Kraken 등)
- **히스토리**: 2019-03-30~
- **최소 주문**: $300

| Tier | 가격 | 비고 |
|------|------|------|
| Solo | ~$200-300/mo | 단일 API key, 연구용 |
| Pro | ~$300-500/mo | 상업적 사용 가능 |
| Business | ~$500+/mo | 10 API keys |
| Invoice | $6,000+ | 대량 구매 |

**핵심 데이터:**

| Data | 설명 | Alpha 등급 |
|------|------|:----------:|
| **L2 Order Book Snapshots** | 호가창 전체 depth 스냅샷 | **최상** |
| **L2 Incremental Updates** | 호가창 실시간 변화 | **최상** |
| **Tick-level Trades** | 모든 개별 체결 (aggTrade) | **상** |
| **Liquidations** | 개별 청산 이벤트 | **상** |
| **Options Chains** | Deribit 옵션 전체 데이터 | **상** |

**학술 근거 (Order Flow Alpha):**
- Anastasopoulos & Gradojevic (2025): Order flow conditioning으로 **Sharpe 1.44 → 3.19** (2.2배 향상)
- Order book imbalance, trade flow imbalance → 단기 가격 방향 예측
- LOB microstructure 기반 ML 모델이 현재 학술적으로 가장 강력한 alpha 영역

**전략 아이디어:**

| Signal | 구현 | 기대 효과 |
|--------|------|-----------|
| `order_book_imbalance` | (bid_depth - ask_depth) / total | 즉각적 수급 불균형 |
| `trade_flow_imbalance` | (buy_volume - sell_volume) / total | 공격적 매수/매도 강도 |
| `vpin` | Volume-Synchronized PIN | 정보 비대칭 기반 독성 흐름 감지 |
| `lob_gradient` | 호가 단계별 depth 기울기 | 숨겨진 유동성/벽 감지 |

**주의**: LOB data는 용량이 매우 크고 (수 TB), 처리·모델링에 ML 역량 필요. 단순 rule-based 전략에는 과잉 투자일 수 있음.

```python
# Tardis.dev Python client 예시
from tardis_dev import datasets

datasets.download(
    exchange="binance-futures",
    data_types=["book_snapshot_25", "trades", "liquidations"],
    from_date="2024-01-01",
    to_date="2026-02-15",
    symbols=["BTCUSDT"],
    api_key="YOUR_API_KEY",
)
```

---

### B. Kaiko — **개인 Quant에게 비권장**

- **가격**: 연 $9,500~$55,000 (평균 ~$28,500/yr)
- **핵심**: Normalized trade data (100+ exchanges), L1/L2 order book
- **Bloomberg Terminal 연동**
- **판정**: Tardis.dev가 동급 데이터를 1/5~1/10 가격에 제공. **Kaiko 대신 Tardis.dev 권장**.

### C. CCData (CryptoCompare) — **Free tier로 충분**

- **가격**: Free (7일 minute-level) / Commercial (미공개) / Enterprise (협의)
- **커버리지**: 5,600+ coins, 260K+ pairs
- **판정**: Broad coverage이지만 Tardis 대비 depth 부족. 기존 Binance OHLCV로 충분히 커버. **추가 투자 불필요**.

### D. Amberdata — **Options 전략 확장 시만**

- **가격**: 협의 (기관급)
- **핵심**: DeFi analytics + Derivatives volatility surface
- **판정**: Laevitas가 동급 options data를 $50/mo에 제공. **Laevitas 우선**.

---

## 5. 상세 분석: Alternative Data

### A. Arkham Intelligence (Free) — **즉시 활용 권장**

- **URL**: `https://intel.arkm.com`
- **Free API**: 20 req/sec, entity labels, address attribution
- **핵심**: Real-world entity ↔ blockchain address 매핑

| 추적 가능 Entity | 예시 |
|-------------------|------|
| 정부 지갑 | US DOJ, German BKA, Mt.Gox trustee |
| 기관 | BlackRock, Grayscale, MicroStrategy |
| 거래소 | Binance, Coinbase cold/hot wallets |
| Whale 개인 | 대형 보유자 |

**활용**: 대형 entity 이동 → event-driven signal. 유료 API 불필요.

### B. Dune Analytics (Free 2,500 credits/mo)

- **URL**: `https://dune.com`
- **핵심**: Custom SQL on 1+ PB multi-chain indexed data
- **유료**: Plus $399/mo, Premium $999/mo

**활용 예시:**
```sql
-- DEX volume by protocol (last 7 days)
SELECT project, SUM(amount_usd) as volume
FROM dex.trades
WHERE block_time > NOW() - INTERVAL '7 days'
GROUP BY project ORDER BY volume DESC;
```

**판정**: Free tier로 research 충분. 실시간 트레이딩보다는 alpha 발굴 도구.

### C. Token Terminal — **DeFi 확장 시만**

- **가격**: Pro $350/mo
- **핵심**: DeFi protocol P/S, P/E, Revenue, Fees
- **판정**: BTC/ETH futures 전략에 직접 alpha 없음. **현재 불필요**.

### D. LunarCrush — **투자 비권장**

- **가격**: Builder $240/mo
- **핵심**: Social sentiment (Galaxy Score)
- **판정**: Santiment이 on-chain+social 통합으로 더 나은 선택. Social 단독은 alpha 약함.

### E. The TIE — **기관급, 개인 비권장**

- **가격**: ~$60,000/yr (추정)
- **핵심**: Point-in-time social data, 90%+ bot filtering
- **판정**: 최고 품질 sentiment이나 비용 대비 효용 낮음. **Santiment이 합리적 대안**.

---

## 6. 2025-2026 데이터 트렌드

### 유효한 트렌드

| 트렌드 | 설명 | 관련 소스 | 학술/실무 근거 |
|--------|------|-----------|----------------|
| **Options Vol Surface** | BTC/ETH options 시장 급성장 → IV/Skew 핵심 지표화 | Laevitas, Amberdata | Deribit OI $30B+ (2025) |
| **Order Flow ML** | LOB microstructure → ML 모델 최강 alpha | Tardis.dev | Sharpe 1.44→3.19 (EFMA 2025) |
| **Cross-exchange Liquidation** | Aggregated liquidation cascade 예측 | CoinGlass | 2024-25 대형 cascade 다수 |
| **ETF Flow Tracking** | 2024 BTC ETF 승인 → 기관 자금 흐름 필수 지표화 | CoinGlass, CryptoQuant | BlackRock IBIT $50B+ AUM |
| **Stablecoin Velocity** | Supply 변화 < Velocity(거래량/시총) | DeFiLlama (무료) | TRM Labs 2025 Report |
| **AI-filtered Sentiment** | Raw social → bot/spam 제거 후 정제 | The TIE, Santiment | Signal-to-noise 3x 향상 |

### 구조적 변화 (주의 사항)

1. **ETF 시대의 On-Chain 한계**: 기관 자금은 custody를 통해 흐름 → on-chain에 미반영. ETF flow 데이터가 이를 보완
2. **Signal Decay 가속**: 널리 알려진 on-chain 지표(MVRV, SOPR)는 자기규제 메커니즘으로 alpha 감소 추세
3. **Options 시장 성숙**: 2024-25 BTC/ETH options 유동성 급증 → vol surface 신뢰도 상승. 과거에는 thin market으로 노이즈가 많았으나 현재는 실용 수준
4. **Micro-structure Alpha 부상**: 전통 금융의 order flow 연구가 crypto에 본격 적용. LOB data 기반 ML이 차세대 alpha 영역

---

## 7. MC Coin Bot 맥락 권장 로드맵

현재 상태: BTC/ETH 중심 CEX futures, 1D timeframe 앙상블 전략, 무료 데이터만 사용 중.

### Phase 1: Derivatives 강화 ($128/mo)

```
CoinGlass Startup ($79) + Laevitas Premium ($50) = $129/mo
```

- Cross-exchange OI/Funding/Liquidation aggregation
- Options IV/Skew → forward-looking regime filter
- 기존 Binance-only 파생 데이터의 blind spot 해소
- ETF Flow 데이터 추가

**예상 효과**: 파생상품 포지셔닝 분석 정밀도 향상, forward-looking signal 추가

### Phase 2: On-Chain Flow 추가 ($99/mo 추가, 합계 $228/mo)

```
+ CryptoQuant Professional ($99) = 합계 $228/mo
```

- Exchange Reserve + Netflow → 수급 분석
- Estimated Leverage Ratio → 시스템 리스크 감지
- Stablecoin Supply Ratio → 자금 유입 정밀화
- Miner Flow → BTC 매도 압력 선행 지표

**예상 효과**: On-chain flow 기반 regime filter 추가, 기존 DeFiLlama 무료 데이터 보완

### Phase 3: Order Flow ML (추가 $200-400/mo, 합계 $430-630/mo)

```
+ Tardis.dev Solo/Pro ($200-400) = 합계 $430-630/mo
```

- LOB data 기반 ML 모델 개발
- Trade flow imbalance, VPIN 등 microstructure signal
- 학술적으로 가장 높은 alpha 잠재력 (Sharpe 3.19)
- **전제**: ML 모델링 역량 + 대용량 데이터 처리 인프라

**예상 효과**: 단기 alpha 획득, 기존 daily timeframe에서 intraday 확장 가능

### Phase 4: Macro Cycle 정밀화 (필요 시, 추가 ~$800/mo)

```
+ Glassnode Professional ($799) = 합계 $1,200-1,400/mo
```

- Entity-adjusted SOPR, HODL Waves → 사이클 위치 정밀화
- CryptoQuant으로 80% 커버되므로, 나머지 20%가 필요할 때만
- **판정**: ROI 대비 Phase 1-3이 우선

---

## 8. 비용 대비 분석

| 월 예산 | 구성 | 추가 데이터 | Alpha 기대 |
|---------|------|-------------|:----------:|
| **$0** | 현재 (무료만) | DeFiLlama + Coin Metrics + Binance | 기준선 |
| **$129** | CoinGlass + Laevitas | Cross-exchange derivatives + Vol surface | **+15-25%** |
| **$228** | + CryptoQuant | Exchange flow + on-chain metrics | **+25-35%** |
| **$430-630** | + Tardis.dev | Order book microstructure | **+40-60%** |
| **$1,200+** | + Glassnode Pro | Entity-adjusted full suite | **+45-65%** |

> **주의**: Alpha 기대치는 데이터 자체의 정보량 기준 추정이며, 실제 전략 성과는 signal engineering과 모델링 품질에 크게 의존합니다.

---

## 9. References

| Source | URL | 설명 |
|--------|-----|------|
| CoinGlass API Pricing | `https://www.coinglass.com/pricing` | 유료 플랜 비교 |
| CryptoQuant Pricing | `https://cryptoquant.com/pricing` | 유료 플랜 비교 |
| Glassnode Studio | `https://studio.glassnode.com` | 플랜별 metric 목록 |
| Laevitas | `https://www.laevitas.ch` | Options analytics |
| Tardis.dev | `https://tardis.dev` | Historical LOB data |
| Nansen | `https://www.nansen.ai/plans` | Wallet labeling 플랜 |
| Santiment API | `https://academy.santiment.net/products-and-plans/sanapi-plans/` | API 플랜 |
| Arkham Intelligence | `https://intel.arkm.com/api` | Free entity tracking API |
| Dune Analytics | `https://dune.com/pricing` | SQL on-chain queries |
| Token Terminal | `https://tokenterminal.com/pricing` | DeFi fundamentals |
| Order Flow & Crypto Returns | `https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/OrderFlowpaper.pdf` | Sharpe 3.19 학술 근거 |
| BTC Price Prediction (On-Chain) | `https://www.sciencedirect.com/science/article/pii/S266682702500057X` | On-chain 예측력 연구 |
| Kaiko Buyer Guide | `https://www.vendr.com/buyer-guides/kaiko` | 기관급 가격 참고 |
| CoinGecko DEX/CEX Ratio | `https://www.coingecko.com/research/publications/dex-to-cex-ratio` | DEX 시장 비중 |
