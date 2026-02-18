# 무료 데이터 소스 확장 계획

> 작성일: 2026-02-18
> 목적: 추가 비용 없이 수집 가능한 데이터 소스 분석 + Alpha 근거 + 구현 계획
> 상태: **설계 단계** — 사용자 승인 후 구현 진행
> 관련: [`paid-data-sources-research.md`](paid-data-sources-research.md) (유료 소스 리서치)

---

## 1. 현재 상태 분석

### 수집 중인 데이터 (8개 소스, 23개 데이터셋)

| # | 소스 | 카테고리 | 데이터셋 수 | 해상도 |
|---|------|---------|:-----------:|--------|
| 1 | Binance Spot | OHLCV | 심볼별 | 1분 |
| 2 | Binance Futures | Derivatives | 4종/심볼 | 1H~8H |
| 3 | DeFiLlama | On-chain (Stablecoin/TVL/DEX) | 15 | Daily |
| 4 | Coin Metrics | On-chain (BTC/ETH metrics) | 2 | Daily |
| 5 | Alternative.me | Sentiment (Fear & Greed) | 1 | Daily |
| 6 | Blockchain.com | On-chain (BTC mining) | 3 | Daily |
| 7 | Etherscan | On-chain (ETH supply) | 1 | Snapshot |
| 8 | mempool.space | On-chain (BTC hashrate) | 1 | Real-time |

### 미수집 영역 (Gap 분석)

| 영역 | 현재 상태 | 리스크/기회 |
|------|----------|------------|
| **매크로/Cross-Asset** | 완전 부재 | DXY-BTC 역상관(-0.6~-0.8) 미반영. M2→BTC 6~12개월 선행관계 미활용 |
| **Options/IV** | 완전 부재 | Forward-looking 지표 전무. Realized vol만으로 regime 판단 |
| **멀티거래소 파생상품** | Binance 단독 | Cross-exchange OI divergence, aggregated liquidation 불가 |
| **Market Structure** | BTC dominance 없음 | 자산 rotation 시그널 미감지 |
| **Binance 추가 엔드포인트** | 일부만 사용 | Top Trader L/S ratio, liquidation stream 미수집 |

### 핵심 결론

1D 타임프레임 앙상블 전략에 **매크로 상관관계**와 **옵션 IV** 데이터가 가장 큰 부가가치를 제공.
두 영역 모두 완전 무료로 수집 가능.

---

## 2. 추천 소스 우선순위 요약

### Tier 1: 즉시 도입 (높은 Alpha, 완전 무료)

| 소스 | 데이터 | API 비용 | Alpha 등급 | 구현 난이도 |
|------|--------|---------|:----------:|:----------:|
| **FRED API** | DXY, Gold, Treasury, VIX, M2, Fed B/S | $0 (키 등록) | **최상** | 낮음 |
| **yfinance** | SPY, GLD, TLT, UUP (cross-asset ETFs) | $0 (인증 없음) | **최상** | 낮음 |
| **Deribit Public API** | DVOL, Put/Call, Options OI, Term Structure | $0 (인증 없음) | **상** | 중간 |
| **Coinalyze** | 멀티거래소 OI/Funding/Liquidation/CVD | $0 (키 등록) | **상** | 중간 |

### Tier 2: 다음 단계 도입 (보통 Alpha, 완전 무료)

| 소스 | 데이터 | API 비용 | Alpha 등급 | 구현 난이도 |
|------|--------|---------|:----------:|:----------:|
| **CoinGecko Demo** | BTC Dominance, Global MCap, DeFi stats | $0 (키 등록) | **중** | 낮음 |
| **Hyperliquid** | On-chain perp OI/Funding/Cross-venue 비교 | $0 (인증 없음) | **중** | 중간 |
| **Binance 추가 엔드포인트** | Top Trader L/S ratio (Account/Position) | $0 (기존 키) | **중** | 낮음 |
| **Dune Analytics** | Custom SQL on-chain 쿼리 | $0 (2,500 credits/mo) | **중** | 높음 |

### 평가 후 비추천 (무료 tier 부적합)

| 소스 | 비추천 이유 |
|------|-----------|
| Glassnode | 무료 tier: Daily only, API 미포함, 히스토리 제한 |
| CryptoQuant | 무료 API 없음 ($39+/mo) |
| Santiment | 무료 3개월 히스토리 → 백테스트 불가 |
| CoinMarketCap | 무료 tier에 히스토리 데이터 없음 |
| Token Terminal | 유료 전용 ($200+/mo) |
| Kaiko | Enterprise 전용 ($1,000+/mo) |

---

## 3. Tier 1 상세 분석

### 3.1 FRED API — 매크로 경제 지표

**Federal Reserve Economic Data**: 미국 연방준비제도 공식 경제 데이터 API.

| 속성 | 값 |
|------|-----|
| Base URL | `https://api.stlouisfed.org/fred/` |
| 인증 | API key (무료 등록: `fred.stlouisfed.org`) |
| Rate Limit | ~120 req/min (매우 관대) |
| 히스토리 | 수십 년 (시리즈별 상이) |
| 응답 형식 | JSON, XML |

**수집 대상 시리즈 (7개):**

| Series ID | 데이터 | 해상도 | Alpha 시그널 |
|-----------|--------|--------|-------------|
| `DTWEXBGS` | Broad Dollar Index (DXY 대용) | Daily | BTC-DXY 역상관 regime filter |
| `GOLDAMGBD228NLBM` | Gold Price (London Fix) | Daily | BTC-Gold 상대강도 = risk-on/off |
| `DGS10` | 10Y Treasury Yield | Daily | Real yield 상승 → crypto 약세 |
| `DGS2` | 2Y Treasury Yield | Daily | 단기 금리 변화 → 유동성 영향 |
| `T10Y2Y` | 10Y-2Y Spread (Yield Curve) | Daily | Inversion → recession → risk-off |
| `VIXCLS` | VIX Volatility Index | Daily | VIX > 30 = crypto risk-off regime |
| `M2SL` | M2 Money Supply | Monthly | M2 증가율 → BTC 6~12개월 선행 |

**API 호출 예시:**

```python
import httpx

params = {
    "series_id": "DTWEXBGS",
    "api_key": "YOUR_KEY",
    "file_type": "json",
    "observation_start": "2020-01-01",
    "observation_end": "2026-02-18",
}
resp = httpx.get(
    "https://api.stlouisfed.org/fred/series/observations",
    params=params,
)
# → {"observations": [{"date": "2020-01-02", "value": "121.23"}, ...]}
```

**Alpha 근거:**

- **DXY-BTC 역상관**: 2024-2025 상관계수 -0.6 ~ -0.8. Dollar 강세 시 BTC 약세 패턴 안정적
- **M2 선행관계**: M2 YoY 변화율이 BTC 가격에 6~12개월 선행 (2020-2025 backtest 검증 가능)
- **VIX regime**: VIX > 30 구간에서 BTC 평균 수익률 음수 → regime filter로 활용
- **Real yield**: (DGS10 - CPI) 상승 시 crypto 약세. 전통 자산 대비 기회비용 증가

**기대 활용 (1D 앙상블 전략):**

| Signal | 계산 | 전략 적용 |
|--------|------|----------|
| `macro_dxy_trend` | DXY 20D SMA 방향 | DXY 하락 시 롱 bias 강화 |
| `macro_vix_regime` | VIX > 30 여부 | 고변동 시 포지션 축소 |
| `macro_m2_momentum` | M2 3M/12M 변화율 | 유동성 확장 시 롱 bias |
| `macro_yield_curve` | T10Y2Y 부호 | Inversion 시 방어적 |
| `macro_real_yield` | DGS10 - CPI trend | Real yield 상승 시 축소 |

---

### 3.2 yfinance — Cross-Asset ETF 데이터

**Yahoo Finance Python wrapper**: 글로벌 주식/ETF/인덱스 무료 접근.

| 속성 | 값 |
|------|-----|
| 라이브러리 | `yfinance` (pip install) |
| 인증 | 불필요 |
| Rate Limit | 비공식 ~2,000 req/hr (보수적 사용 권장) |
| 히스토리 | 수십 년 (Daily) |
| 데이터 | OHLCV + Adj Close |

**수집 대상 티커 (6개):**

| Ticker | 데이터 | Alpha 시그널 |
|--------|--------|-------------|
| `SPY` | S&P 500 ETF | BTC-SPY rolling corr 이탈 = regime 변화 |
| `QQQ` | Nasdaq 100 ETF | Tech 상관관계 (BTC-QQQ ~0.5-0.7) |
| `GLD` | Gold ETF | BTC-Gold rotation (risk-on vs risk-off) |
| `TLT` | 20+ Year Treasury ETF | Bond momentum = risk appetite 선행 |
| `UUP` | US Dollar ETF | 실시간 DXY proxy (FRED 보완) |
| `HYG` | High Yield Bond ETF | Credit stress indicator |

**API 사용 예시:**

```python
import yfinance as yf

# 일괄 다운로드
tickers = ["SPY", "QQQ", "GLD", "TLT", "UUP", "HYG"]
df = yf.download(tickers, start="2020-01-01", end="2026-02-18", group_by="ticker")
# → MultiIndex DataFrame: (Date) × (Ticker, OHLCV)
```

**Alpha 근거:**

- **BTC-SPY rolling correlation**: 30D rolling corr 급변 시 regime 전환 시그널
- **BTC-Gold ratio momentum**: BTC/GLD 비율의 추세 반전 → 자본 rotation 포착
- **TLT momentum**: Bond rally → risk-off 전환 → crypto 약세 선행
- **HYG/LQD spread**: Credit stress 확대 → 전체 risk-off → crypto 동반 하락

**FRED와의 시너지:**

| FRED | yfinance | 시너지 |
|------|----------|--------|
| DTWEXBGS (DXY) | UUP | DXY 일일 보완 + 인트라데이 proxy |
| DGS10 (10Y yield) | TLT (역방향) | Yield ↔ Bond price 교차 검증 |
| VIXCLS (VIX) | SPY vol | 실현 변동성 vs 내재 변동성 비교 |

> **주의**: yfinance는 비공식 API이므로 Yahoo 정책 변경 시 불안정할 수 있음.
> 핵심 시리즈(DXY, VIX)는 FRED를 primary, yfinance를 secondary로 구성.

---

### 3.3 Deribit Public API — Options & Implied Volatility

**Deribit**: 글로벌 최대 crypto options 거래소. Public endpoint 완전 무료.

| 속성 | 값 |
|------|-----|
| Base URL | `https://www.deribit.com/api/v2/public/` |
| 인증 | 불필요 (public endpoints) |
| Rate Limit | Token-bucket, MarketData ~20 req/s |
| 커버리지 | BTC, ETH options + futures |
| 히스토리 | `get_tradingview_chart_data`로 전체 히스토리 |

**수집 대상 (5개 데이터셋):**

| # | 데이터셋 | Endpoint | 설명 |
|---|---------|----------|------|
| 1 | **DVOL** (30D Implied Vol) | `GET /public/ticker?instrument_name=BTC-DVOL` | Crypto VIX 등가물 |
| 2 | **Put/Call OI Ratio** | `GET /public/get_book_summary_by_currency?currency=BTC&kind=option` | 풋/콜 비율 |
| 3 | **Historical Volatility** | `GET /public/get_historical_volatility?currency=BTC` | 실현 변동성 (1~365D) |
| 4 | **Term Structure** | 여러 expiry의 futures ticker | Contango/Backwardation |
| 5 | **Options OI by Strike** | book_summary 집계 | Max Pain 계산 |

**API 호출 예시:**

```python
import httpx

# DVOL Index (30-day implied volatility)
resp = httpx.get(
    "https://www.deribit.com/api/v2/public/ticker",
    params={"instrument_name": "BTC-DVOL"},
)
dvol = resp.json()["result"]["last_price"]  # e.g., 52.3 (52.3% annualized IV)

# Put/Call Ratio 계산
resp = httpx.get(
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
    params={"currency": "BTC", "kind": "option"},
)
options = resp.json()["result"]
put_oi = sum(o["open_interest"] for o in options if "-P" in o["instrument_name"])
call_oi = sum(o["open_interest"] for o in options if "-C" in o["instrument_name"])
pc_ratio = put_oi / call_oi if call_oi > 0 else 0

# Historical Volatility
resp = httpx.get(
    "https://www.deribit.com/api/v2/public/get_historical_volatility",
    params={"currency": "BTC"},
)
# → [[timestamp, 7d_vol, 30d_vol, 60d_vol, 90d_vol, 120d_vol, 180d_vol, 365d_vol], ...]

# Historical DVOL (TradingView chart data)
resp = httpx.get(
    "https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
    params={
        "instrument_name": "BTC-DVOL",
        "resolution": "1D",
        "start_timestamp": 1609459200000,
        "end_timestamp": 1739836800000,
    },
)
```

**Alpha 근거:**

- **DVOL / Realized Vol spread**: IV > RV = 시장 불안 (put 수요 증가). Spread 축소 시 mean-reversion 기회
- **Put/Call Ratio 극값**: P/C > 1.5 → 과도한 공포 → contrarian long signal
- **Term Structure slope**: 급격한 backwardation → 단기 하방 스트레스 → 방어 모드 전환
- **Options Max Pain**: 만기 접근 시 max pain 가격대로의 수렴 경향

**기대 활용 (1D 앙상블):**

| Signal | 계산 | 전략 적용 |
|--------|------|----------|
| `dvol_regime` | DVOL z-score (60D lookback) | 고 IV 시 포지션 축소, 저 IV 시 확대 |
| `vol_risk_premium` | DVOL - RV(30D) | 양수 극값 → contrarian 기회 |
| `options_pc_ratio` | Put/Call OI ratio | > 1.5 contrarian long, < 0.5 contrarian short |
| `term_structure_slope` | Near/Far futures basis | Backwardation → 단기 방어 |

> **왜 Options 데이터가 중요한가**: 기존 데이터(OHLCV, funding, on-chain, sentiment)는 모두 **backward-looking**.
> Options IV/Skew는 시장 참여자의 **미래 기대치를 직접 반영**하는 유일한 forward-looking 데이터.

---

### 3.4 Coinalyze — 멀티거래소 파생상품 집계

| 속성 | 값 |
|------|-----|
| Base URL | `https://api.coinalyze.net/v1/` |
| 인증 | API key (무료 등록) |
| Rate Limit | 40 req/min |
| 커버리지 | Binance, OKX, Bybit, dYdX, Bitget 등 10+ exchanges |
| 히스토리 | 수년 (endpoints별 상이) |

**수집 대상 (4개 데이터셋):**

| # | 데이터셋 | Endpoint | 현재 대비 추가 가치 |
|---|---------|----------|------------------|
| 1 | **Aggregated OI** | `/v1/open-interest-history` | Binance-only → 전체 시장 OI |
| 2 | **Aggregated Funding** | `/v1/funding-rate-history` | 거래소 간 FR 편차 포착 |
| 3 | **Liquidation History** | `/v1/liquidation-history` | Cross-exchange cascade 감지 |
| 4 | **CVD** (Cumulative Volume Delta) | `/v1/ohlcv-history` | 매수/매도 강도 차이 |

**API 호출 예시:**

```python
import httpx

headers = {"api_key": "YOUR_KEY"}

# Aggregated OI (Binance Futures)
resp = httpx.get(
    "https://api.coinalyze.net/v1/open-interest-history",
    headers=headers,
    params={
        "symbols": "BTCUSDT.6",  # .6 = Binance Futures
        "interval": "1D",
        "from": 1609459200,
        "to": 1739836800,
    },
)

# Liquidation History
resp = httpx.get(
    "https://api.coinalyze.net/v1/liquidation-history",
    headers=headers,
    params={
        "symbols": "BTCUSDT.6",
        "interval": "1h",
        "from": 1609459200,
        "to": 1739836800,
    },
)
```

**Symbol suffix 규칙:**

| Exchange | Suffix | 예시 |
|----------|--------|------|
| Binance Futures | `.6` | `BTCUSDT.6` |
| OKX Perp | `.C` | `BTCUSDT.C` |
| Bybit Perp | `.A` | `BTCUSDT.A` |
| dYdX | `.8` | `BTCUSD.8` |

**Alpha 근거:**

- **Cross-exchange OI divergence**: Binance OI 상승 + 전체 OI 하락 → localized leverage (위험 신호)
- **Funding dispersion**: `std(FR across exchanges)` 급등 → squeeze 임박
- **Aggregated liquidation cascade**: 단일 exchange보다 전체 시장 liquidation이 의미 있는 S/R 레벨 식별
- **CVD-Price divergence**: 가격 상승 + CVD 하락 → hidden selling → 반전 시그널

**기대 활용:**

| Signal | 계산 | 전략 적용 |
|--------|------|----------|
| `agg_oi_change` | Δ(total_oi) / 1D | 전체 시장 레버리지 모멘텀 |
| `funding_dispersion` | std(FR across exchanges) | 극값 → squeeze 방어 |
| `liq_cluster_intensity` | Σ(liq_volume) / 24h rolling | 대규모 청산 후 반전 |
| `cvd_price_divergence` | CVD trend vs Price trend | Divergence → contrarian |

---

## 4. Tier 2 상세 분석

### 4.1 CoinGecko Demo API — 시장 구조 지표

| 속성 | 값 |
|------|-----|
| Base URL | `https://api.coingecko.com/api/v3/` |
| 인증 | API key (무료 Demo plan 등록) |
| Rate Limit | 30 req/min, **10,000 req/month** |
| 히스토리 | 최대 12년 (주요 자산) |

**수집 대상 (2개 데이터셋):**

| # | 데이터셋 | Endpoint | Alpha 시그널 |
|---|---------|----------|-------------|
| 1 | **Global Market** | `/global` | BTC dominance, total mcap, DeFi mcap |
| 2 | **DeFi Global** | `/global/decentralized_finance_defi` | DeFi TVL/MCap ratio |

```python
import httpx

headers = {"x-cg-demo-api-key": "YOUR_KEY"}

# Global metrics (BTC dominance, total market cap)
resp = httpx.get(
    "https://api.coingecko.com/api/v3/global",
    headers=headers,
)
data = resp.json()["data"]
btc_dominance = data["market_cap_percentage"]["btc"]   # e.g., 52.3
total_mcap = data["total_market_cap"]["usd"]           # e.g., 3.2T
```

**Alpha 근거:**

- **BTC Dominance trend**: 상승 = risk-off (alt → BTC rotation), 하락 = risk-on (BTC → alt rotation)
- **Total MCap velocity**: Rate of change 극값 → euphoria/capitulation 시그널

**제한사항**: 10K/month 콜 제한 → 일일 1~2회 스냅샷 수집만 가능. 히스토리 구축에 시간 필요.

---

### 4.2 Hyperliquid — On-Chain Perpetuals

| 속성 | 값 |
|------|-----|
| Base URL | `https://api.hyperliquid.xyz/info` (POST) |
| 인증 | 불필요 |
| Rate Limit | 미공식 (합리적 사용 권장) |
| 커버리지 | 100+ perp 페어 |

**수집 대상 (2개 데이터셋):**

| # | 데이터셋 | Request Body | Alpha 시그널 |
|---|---------|-------------|-------------|
| 1 | **Asset Contexts** | `{"type": "metaAndAssetCtxs"}` | 전 자산 OI/Funding/Volume 스냅샷 |
| 2 | **Cross-venue Funding** | `{"type": "predictedFundings"}` | Binance/Bybit/HL funding 비교 |

```python
import httpx

# All assets: mark price, funding, OI, volume
resp = httpx.post(
    "https://api.hyperliquid.xyz/info",
    json={"type": "metaAndAssetCtxs"},
)

# Cross-venue predicted funding
resp = httpx.post(
    "https://api.hyperliquid.xyz/info",
    json={"type": "predictedFundings"},
)
# → [{"coin": "BTC", "venues": [["Binance", 0.0001], ["Bybit", 0.00012], ["Hyperliquid", 0.00008]]}]
```

**Alpha 근거:**

- **DeFi vs CeFi funding spread**: CeFi FR > DeFi FR → CeFi 과열 (정보 우위)
- **On-chain 투명성**: Hyperliquid OI는 완전 on-chain → manipulation 불가능한 진정한 포지셔닝 데이터

---

### 4.3 Binance 추가 엔드포인트 (기존 API 확장)

현재 미수집 중인 Binance Futures 엔드포인트:

| # | Endpoint | 해상도 | Alpha 시그널 |
|---|----------|--------|-------------|
| 1 | `/fapi/v1/topLongShortAccountRatio` | 5m~1D | Top trader 포지셔닝 |
| 2 | `/fapi/v1/topLongShortPositionRatio` | 5m~1D | Top trader 크기 기반 비율 |
| 3 | `/fapi/v1/globalLongShortAccountRatio` | 5m~1D | Global (retail) 비율 |

**Alpha: Top vs Global divergence**

```
Top Trader L/S = 2.5 (상위 트레이더 롱 우세)
Global L/S = 0.8 (일반 유저 숏 우세)
→ Smart money vs Retail 괴리 = 유의미한 방향 시그널
```

> 기존 `DerivativesFetcher`에 3개 endpoint 추가만으로 구현 가능. 가장 낮은 구현 비용.

---

### 4.4 Dune Analytics (리서치 전용)

| 속성 | 값 |
|------|-----|
| Base URL | `https://api.dune.com/api/v1/` |
| 인증 | API key (무료 등록) |
| 무료 크레딧 | 2,500/month |

**활용 방안**: 정기 데이터 파이프라인보다는 **alpha 발굴 리서치** 도구로 활용.

```sql
-- 예시: DEX volume 추이 (weekly)
SELECT
    DATE_TRUNC('week', block_time) AS week,
    SUM(amount_usd) AS total_volume
FROM dex.trades
WHERE block_time > NOW() - INTERVAL '90 days'
GROUP BY 1
ORDER BY 1;
```

2,500 크레딧으로 월 10~20회 쿼리 실행 가능. 유의미한 패턴 발견 시 전용 데이터셋으로 격상.

---

## 5. 2026년 트렌드 분석

### 리테일 퀀트에서 활용 가치가 높아진 영역

| 트렌드 | 설명 | 관련 소스 | 근거 |
|--------|------|-----------|------|
| **Macro-Crypto Correlation** | BTC-macro 상관 안정화 (2023~). 매크로 무시 = 불리 | FRED, yfinance | DXY-BTC corr -0.6~-0.8 (2024-2025) |
| **Options 시장 성숙** | Deribit $30B+/mo. IV/Skew 신뢰도 상승 | Deribit | 2024 BTC ETF 이후 유동성 급증 |
| **멀티벤치 집계** | Hyperliquid/dYdX 성장 → CeFi-only 불완전 | Coinalyze, Hyperliquid | DeFi perp 시장 점유율 확대 |
| **On-chain 투명성** | Hyperliquid 완전 on-chain 오더북 → 진정한 데이터 | Hyperliquid | CeFi 대비 manipulation-free |

### 주의해야 할 구조적 변화

| 변화 | 영향 | 대응 |
|------|------|------|
| **ETF 시대**: 기관 자금 custody 통해 흐름 | On-chain에 미반영 | Macro(ETF flow) 데이터로 보완 |
| **Signal Decay**: 널리 알려진 on-chain 지표(MVRV, SOPR) alpha 감소 | 기존 metric 과신 금지 | Forward-looking(IV) + Cross-asset으로 분산 |
| **AI-driven Alpha**: 비정형 데이터(social, order flow) ML 처리 보편화 | Rule-based 한계 | 구조화된 데이터에 집중, ML은 Phase 2 |

---

## 6. 아키텍처 설계

### 6.1 모듈 구조

기존 `src/data/onchain/` 패턴을 확장. 카테고리별 서브모듈 추가.

```
src/data/
├── onchain/                    # 기존 (DeFiLlama, CoinMetrics, ...)
│
├── macro/                      # 신규: 매크로 데이터
│   ├── __init__.py
│   ├── client.py               # FREDClient + YFinanceClient
│   ├── models.py               # Pydantic V2 레코드 모델
│   ├── fetcher.py              # MacroFetcher (FRED + yfinance)
│   ├── storage.py              # Bronze/Silver 저장
│   └── service.py              # MacroDataService (enrich + precompute)
│
├── options/                    # 신규: 옵션 데이터
│   ├── __init__.py
│   ├── client.py               # DeribitClient (httpx, public endpoints)
│   ├── models.py               # DVOL, PCRatio, TermStructure 모델
│   ├── fetcher.py              # OptionsFetcher
│   ├── storage.py              # Bronze/Silver 저장
│   └── service.py              # OptionsDataService
│
└── derivatives_extended/       # 신규: 멀티거래소 파생상품
    ├── __init__.py
    ├── client.py               # CoinalyzeClient (httpx + rate limit)
    ├── models.py               # AggOI, AggFunding, Liquidation 모델
    ├── fetcher.py              # ExtDerivativesFetcher
    ├── storage.py              # Bronze/Silver 저장
    └── service.py              # ExtDerivativesService
```

### 6.2 공통 패턴 (기존 onchain 모듈과 동일)

```python
# Client: httpx + RateLimiter + retry
class FREDClient:
    """FRED API client with rate limiting."""

    def __init__(self, api_key: str, rate_limit: float = 100.0) -> None:
        self._api_key = api_key
        self._limiter = RateLimiter(rate_limit)  # 기존 onchain RateLimiter 재사용
        self._client = httpx.AsyncClient(base_url="https://api.stlouisfed.org/fred/")

    async def get_observations(
        self, series_id: str, start: str, end: str
    ) -> list[dict]:
        await self._limiter.acquire()
        resp = await self._client.get(
            "series/observations",
            params={
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start,
                "observation_end": end,
            },
        )
        resp.raise_for_status()
        return resp.json()["observations"]
```

```python
# Service: enrich() + precompute() 패턴
class MacroDataService:
    """매크로 데이터 로드 + OHLCV 병합."""

    def enrich(self, ohlcv_df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """OHLCV에 매크로 지표 병합 (publication lag 적용)."""
        ...

    def precompute(self, symbol: str, ohlcv_index: pd.DatetimeIndex) -> pd.DataFrame:
        """심볼별 매크로 피처 사전 병합 (macro_* prefix)."""
        ...
```

### 6.3 데이터 카탈로그 확장

`catalogs/datasets.yaml`에 신규 데이터셋 등록:

```yaml
# Macro - FRED
- dataset_id: fred_dxy
  source: fred
  name: dxy
  type: macro
  group: macro_rates
  description: "Broad US Dollar Index (DXY proxy)"
  frequency: daily
  history_start: "2006-01-02"
  lag_days: 1
  columns:
    - { name: date, type: date, description: "Observation date" }
    - { name: value, type: decimal, description: "Dollar index value" }
  enrichment:
    oc_col_prefix: "macro_dxy"
    merge_on: date
  strategy_hints:
    - "BTC-DXY inverse correlation regime filter"
    - "Dollar strength → crypto weakness"

- dataset_id: fred_vix
  source: fred
  name: vix
  type: macro
  group: macro_volatility
  description: "CBOE Volatility Index (VIX)"
  frequency: daily
  history_start: "1990-01-02"
  lag_days: 1
  columns:
    - { name: date, type: date }
    - { name: value, type: decimal, description: "VIX close" }
  enrichment:
    oc_col_prefix: "macro_vix"
  strategy_hints:
    - "VIX > 30 = risk-off regime for crypto"

# ... (M2, Gold, Yields, yfinance tickers, Deribit, Coinalyze도 동일 패턴)
```

### 6.4 저장 구조

```
data/
├── bronze/
│   ├── macro/                          # 신규
│   │   ├── fred/
│   │   │   ├── dxy.parquet
│   │   │   ├── gold.parquet
│   │   │   ├── dgs10.parquet
│   │   │   ├── dgs2.parquet
│   │   │   ├── t10y2y.parquet
│   │   │   ├── vix.parquet
│   │   │   └── m2.parquet
│   │   └── yfinance/
│   │       ├── spy.parquet
│   │       ├── qqq.parquet
│   │       ├── gld.parquet
│   │       ├── tlt.parquet
│   │       ├── uup.parquet
│   │       └── hyg.parquet
│   ├── options/                        # 신규
│   │   └── deribit/
│   │       ├── btc_dvol.parquet
│   │       ├── eth_dvol.parquet
│   │       ├── btc_pc_ratio.parquet
│   │       ├── btc_hist_vol.parquet
│   │       └── btc_term_structure.parquet
│   └── derivatives_ext/               # 신규
│       └── coinalyze/
│           ├── btc_agg_oi.parquet
│           ├── btc_agg_funding.parquet
│           ├── btc_liquidations.parquet
│           └── btc_cvd.parquet
│
└── silver/
    ├── macro/                          # (bronze와 동일 구조)
    ├── options/
    └── derivatives_ext/
```

### 6.5 Publication Lag

| 소스 | Lag | 설명 |
|------|:---:|------|
| FRED (Daily) | T+1 | ~16:00 ET (다음 영업일) 확정 |
| FRED (Monthly, M2) | T+14 | 약 2주 지연 발행 |
| yfinance | T+0 | 장 마감 후 즉시 (16:00 ET) |
| Deribit | T+0 | Real-time |
| Coinalyze | T+0 | Near real-time |
| CoinGecko | T+0 | Snapshot |

### 6.6 CLI 확장

```bash
# Macro 수집
uv run mcbot ingest macro pipeline fred dxy               # 단일 시리즈
uv run mcbot ingest macro pipeline yfinance spy            # 단일 티커
uv run mcbot ingest macro batch --type fred                # FRED 전체 (7)
uv run mcbot ingest macro batch --type yfinance            # yfinance 전체 (6)
uv run mcbot ingest macro batch --type all                 # 전체 (13)
uv run mcbot ingest macro info                             # 데이터 인벤토리

# Options 수집
uv run mcbot ingest options pipeline deribit btc_dvol      # 단일 데이터셋
uv run mcbot ingest options batch --type all               # 전체 (5)
uv run mcbot ingest options info

# Extended Derivatives 수집
uv run mcbot ingest deriv-ext pipeline coinalyze btc_agg_oi
uv run mcbot ingest deriv-ext batch --type all             # 전체 (4)
uv run mcbot ingest deriv-ext info
```

---

## 7. 구현 계획

### Phase 1: Macro 데이터 (FRED + yfinance) — 2~3일

| # | 작업 | 파일 | 비고 |
|---|------|------|------|
| 1 | `FREDClient` + `YFinanceClient` 구현 | `src/data/macro/client.py` | httpx + RateLimiter 재사용 |
| 2 | Pydantic V2 레코드 모델 | `src/data/macro/models.py` | frozen, Decimal |
| 3 | `MacroFetcher` (7 FRED + 6 yfinance) | `src/data/macro/fetcher.py` | |
| 4 | Bronze/Silver 저장 | `src/data/macro/storage.py` | parquet |
| 5 | `MacroDataService` (enrich + precompute) | `src/data/macro/service.py` | `macro_*` prefix |
| 6 | CLI: `ingest macro` 서브커맨드 | `src/cli/ingest_macro.py` | pipeline/batch/info |
| 7 | 데이터 카탈로그 등록 | `catalogs/datasets.yaml` | 13개 데이터셋 |
| 8 | 단위 테스트 | `tests/data/macro/` | client, fetcher, storage, service |

**환경 변수:**

```bash
# .env 추가
FRED_API_KEY=                  # https://fred.stlouisfed.org/docs/api/api_key.html
# yfinance는 API key 불필요
```

### Phase 2: Options 데이터 (Deribit) — 2~3일

| # | 작업 | 파일 | 비고 |
|---|------|------|------|
| 9 | `DeribitClient` (public endpoints) | `src/data/options/client.py` | httpx, 인증 불필요 |
| 10 | DVOL/PCRatio/TermStructure 모델 | `src/data/options/models.py` | |
| 11 | `OptionsFetcher` (5개 데이터셋) | `src/data/options/fetcher.py` | |
| 12 | Bronze/Silver 저장 | `src/data/options/storage.py` | |
| 13 | `OptionsDataService` | `src/data/options/service.py` | `opt_*` prefix |
| 14 | CLI: `ingest options` 서브커맨드 | `src/cli/ingest_options.py` | |
| 15 | 데이터 카탈로그 등록 | `catalogs/datasets.yaml` | 5개 데이터셋 |
| 16 | 단위 테스트 | `tests/data/options/` | |

### Phase 3: Extended Derivatives (Coinalyze) — 1~2일

| # | 작업 | 파일 | 비고 |
|---|------|------|------|
| 17 | `CoinalyzeClient` | `src/data/derivatives_extended/client.py` | httpx + 40 req/min |
| 18 | AggOI/Funding/Liquidation 모델 | `src/data/derivatives_extended/models.py` | |
| 19 | `ExtDerivativesFetcher` | `src/data/derivatives_extended/fetcher.py` | |
| 20 | Bronze/Silver 저장 + Service | `src/data/derivatives_extended/` | `dext_*` prefix |
| 21 | CLI: `ingest deriv-ext` 서브커맨드 | `src/cli/ingest_deriv_ext.py` | |
| 22 | 데이터 카탈로그 등록 | `catalogs/datasets.yaml` | 4개 데이터셋 |
| 23 | 단위 테스트 | `tests/data/derivatives_extended/` | |

**환경 변수:**

```bash
COINALYZE_API_KEY=             # https://coinalyze.net (무료 등록)
```

### Phase 4: Binance 추가 엔드포인트 — 0.5일

| # | 작업 | 파일 | 비고 |
|---|------|------|------|
| 24 | Top Trader L/S ratio 3개 endpoint 추가 | `src/data/derivatives_fetcher.py` | 기존 파일 확장 |
| 25 | Silver 처리 추가 | `src/data/derivatives_storage.py` | 기존 파일 확장 |
| 26 | 테스트 추가 | `tests/data/test_derivatives_fetcher.py` | |

### Phase 5: Tier 2 소스 (선택적) — 2~3일

| # | 작업 | 소스 | 비고 |
|---|------|------|------|
| 27 | CoinGecko global metrics | `src/data/macro/` 확장 | `macro_btc_dom` prefix |
| 28 | Hyperliquid perp data | `src/data/derivatives_extended/` 확장 | `dext_hl_*` prefix |
| 29 | Dune Analytics 리서치 쿼리 | `scripts/dune/` | 리서치 전용 |

### Phase 6: EDA/Backtest 통합 — 2~3일

| # | 작업 | 파일 | 비고 |
|---|------|------|------|
| 30 | StrategyEngine에 macro/options/deriv_ext enrichment 추가 | `src/eda/strategy_engine.py` | |
| 31 | BacktestEngine precompute 확장 | `src/backtest/engine.py` | |
| 32 | 전략 후보 IC 테스트 (macro signals) | `scripts/` | 신규 시그널 검증 |
| 33 | 통합 테스트 | `tests/integration/` | |

### 총 일정 추정

| Phase | 기간 | 데이터셋 수 | 누적 |
|-------|------|:---------:|:----:|
| Phase 1 (Macro) | 2~3일 | 13 | 13 |
| Phase 2 (Options) | 2~3일 | 5 | 18 |
| Phase 3 (Coinalyze) | 1~2일 | 4 | 22 |
| Phase 4 (Binance 확장) | 0.5일 | 3 | 25 |
| Phase 5 (Tier 2) | 2~3일 | 3+ | 28+ |
| Phase 6 (EDA 통합) | 2~3일 | — | — |
| **합계** | **~10~15일** | **~28개** | |

> 기존 23개 + 신규 28개 = **총 ~51개 데이터셋** (완전 무료)

---

## 8. 기대 효과

### 정량적 추정

| 지표 | 현재 (OHLCV + Deriv + On-chain) | + Macro/Options/ExtDeriv (예상) | 변화 |
|------|:------------------------------:|:-------------------------------:|:----:|
| Regime 정확도 | ~60% (RV 기반만) | ~75-80% (IV + Macro 추가) | +15~20%p |
| False signal 비율 | ~35% | ~20-25% | -10~15%p |
| MDD | 19.4% | ~15-17% | -2~4%p |
| Sharpe | 1.57 | ~1.7-1.9 | +8~20% |

> 매크로 regime filter(VIX > 30 시 축소)와 options IV regime(DVOL 극값 시 방어)의 결합이 핵심.
> 주의: 추정치는 유사 backtesting 사례 기반이며, 실제 구현 후 검증 필수.

### 정성적 효과

| 영역 | 개선 |
|------|------|
| **데이터 다각화** | Crypto-only → Crypto + TradFi + Macro 3계층 |
| **Forward-looking** | RV만 사용 → IV/Skew로 미래 기대치 반영 |
| **Cross-venue** | Binance 단독 → 10+ exchanges 집계 |
| **Regime detection** | 가격 기반 → Macro + IV + Funding 다차원 |

---

## 9. 리스크 및 주의사항

| 리스크 | 영향 | 완화 |
|--------|------|------|
| yfinance API 불안정 | 정책 변경 시 수집 중단 | FRED를 primary로, yfinance는 secondary |
| FRED 시리즈 폐지 | 특정 시리즈 중단 가능 | 대체 시리즈 매핑 테이블 유지 |
| Coinalyze 무료 tier 변경 | Rate limit 강화 또는 유료 전환 | 40 req/min 내 보수적 사용, fallback 설계 |
| Deribit 시장 구조 변화 | Options 유동성 변동 | DVOL z-score로 정규화, 절대값 의존 금지 |
| 과도한 데이터 = 과적합 | Feature 수 증가 → overfitting | IC 사전 검증 후 유의미한 시그널만 전략 투입 |
| Look-ahead bias | Macro lag 미적용 시 | Publication lag 테이블 엄격 적용 (FRED T+1, M2 T+14) |
| 데이터 품질 불일치 | 소스 간 시간대/형식 차이 | Silver 레이어에서 UTC 정규화 + 통일 schema |
| CoinGecko 월간 콜 제한 | 10K/month 소진 시 수집 불가 | 일일 1~2회만 수집, 히스토리 구축 후 증분만 |

---

## 10. 비채택 대안

### A. Glassnode Free Tier

Daily 해상도만 제공, API 미포함, 히스토리 제한.
Coin Metrics Community가 이미 유사 메트릭(MVRV, TxCnt, AdrActCnt)을 커버.
**비채택**: 무료 tier의 추가 가치 거의 없음.

### B. Santiment Free (sanpy)

3개월 히스토리 제한 → 백테스트 불가.
SAN 토큰 보유 시 full access이나 토큰 구매 = 사실상 유료.
**비채택**: 히스토리 제한이 치명적.

### C. CoinMarketCap Basic API

Latest 데이터만 제공 (히스토리 없음).
CoinGecko Demo가 동일 영역에서 더 많은 데이터 제공.
**비채택**: CoinGecko 대비 열위.

### D. Token Terminal / Kaiko / Nansen

모두 유료 전용. 무료 프로그래매틱 접근 불가.
**비채택**: 비용 발생. [`paid-data-sources-research.md`](paid-data-sources-research.md) 참조.

---

## 11. 향후 확장 경로

```
현재 (23 datasets, $0/mo)
  │
  ├─ [이 문서] Tier 1+2 무료 확장 (+28 datasets, $0/mo)
  │   → 총 ~51 datasets
  │
  ├─ [paid-data-sources-research.md] Phase 1 유료 확장
  │   → CoinGlass Startup ($79) + Laevitas Premium ($50) = $129/mo
  │   → Liquidation Heatmap, Vol Surface, ETF Flow
  │
  └─ [paid-data-sources-research.md] Phase 2+ 유료 확장
      → CryptoQuant ($99), Tardis.dev ($200-400), Glassnode ($799)
      → Exchange Flow, LOB microstructure, Entity-adjusted metrics
```

무료 소스로 기본 매크로/옵션/멀티벤치 커버리지를 확보한 후,
유료 소스는 **무료 데이터로 검증된 alpha가 있는 영역**에만 선별 투자.

---

## 12. References

| Source | URL | 설명 |
|--------|-----|------|
| FRED API Documentation | `https://fred.stlouisfed.org/docs/api/fred/` | 공식 API 문서 |
| FRED API Key 등록 | `https://fred.stlouisfed.org/docs/api/api_key.html` | 무료 키 발급 |
| yfinance (PyPI) | `https://pypi.org/project/yfinance/` | Python 라이브러리 |
| Deribit API v2 | `https://docs.deribit.com/` | Public endpoints 문서 |
| Coinalyze API | `https://api.coinalyze.net/v1/doc/` | REST API 문서 |
| CoinGecko API | `https://docs.coingecko.com/` | Demo plan 문서 |
| Hyperliquid API | `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals` | Info endpoint |
| Dune API | `https://docs.dune.com/api-reference/overview/introduction` | Query API |
| Binance Futures API | `https://developers.binance.com/docs/derivatives/usds-margined-futures/` | Top Trader endpoints |
