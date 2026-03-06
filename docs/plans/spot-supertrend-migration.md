# SuperTrend+ADX Spot 전환 계획

## 1. 전략 요약

| 항목 | 값 |
|------|-----|
| 전략 | SuperTrend + ADX (Long-Only) |
| 타임프레임 | 12H |
| 파라미터 | ATR=7, Mult=2.5, ADX(14, 25) |
| 에셋 | BTC, ETH, SOL, AVAX, XRP, FTM(Sonic) |
| 배분 | Equal Weight (1/6 = 16.7%) |
| 거래소 | Binance **Spot** |
| 청산 | 반대 신호 + Trailing Stop (3.0x ATR) |
| 포지션 | 100% (max_leverage_cap=1.0) |

### 백테스트 성과 (2020-01 ~ 2026-03)

#### 개별 에셋 (VBT 백테스트, Set B: ATR=7, Mult=2.5)

| 에셋 | Sharpe | CAGR | MDD | PF |
|------|:------:|-----:|----:|---:|
| AVAX | 1.25 | 93.5% | -64.8% | 1.47 |
| SOL | 1.22 | 82.3% | -65.2% | 1.32 |
| XRP | 1.19 | 65.4% | -48.4% | 1.89 |
| BTC | 1.15 | 36.1% | -58.5% | 1.67 |
| FTM | 1.10 | 82.9% | -70.8% | 1.46 |
| ETH | 0.92 | 32.4% | -40.6% | 1.71 |
| **평균** | **1.14** | **65.4%** | **-58.1%** | **1.59** |

#### 6에셋 포트폴리오 (EDA 백테스트, 1m→12H, Spot 비용)

| 지표 | 값 |
|------|-----|
| **Total Return** | **+1,512%** |
| **CAGR** | **59.3%** |
| **Sharpe** | **1.54** |
| **Max Drawdown** | **-30.2%** |
| Win Rate | 36.9% |
| Total Trades | 298 |
| Profit Factor | **1.77** |
| Volatility | 34.0% |
| Calmar Ratio | ~2.0 |

#### 분산 효과 (개별 평균 vs 포트폴리오)

| 지표 | 개별 평균 | 6에셋 포트폴리오 | 개선 |
|------|:---------:|:---------------:|:----:|
| Sharpe | 1.14 | **1.54** | **+35%** |
| MDD | -58.1% | **-30.2%** | **-48%** |
| CAGR | 65.4% | 59.3% | -9% |
| PF | 1.59 | **1.77** | +11% |

> CAGR 소폭 하락(-9%)을 감수하고 MDD 절반(-48%), Sharpe 35% 개선.
> 분산 투자의 교과서적 효과.

---

## 2. 현재 시스템 vs 목표 시스템

### 현재 (Futures Multi-Strategy Orchestrator)

```
LiveRunner
├── StrategyOrchestrator         ← 멀티 전략 Pod 관리
│   ├── StrategyPod × N          ← 전략별 독립 실행
│   ├── CapitalAllocator         ← 동적 자본 배분
│   ├── RiskAggregator           ← Pod 간 리스크 집계
│   ├── Netting                  ← 심볼 간 포지션 상쇄
│   └── LifecycleManager         ← 전략 승격/강등
├── StrategyEngine               ← 단일 전략 실행
├── EDAPortfolioManager          ← PM (SL/TS/Rebalance)
├── EDARiskManager               ← RM (Circuit Breaker)
├── OMS → LiveExecutor           ← Futures 주문 실행
│        └── SmartExecutor       ← Limit Order Decorator
├── ExchangeStopManager          ← Futures STOP_MARKET 안전망
├── LiveDataFeed (WebSocket)     ← 1m Spot → 12H 집계
├── DerivativesFeed              ← Funding Rate, OI
├── OnchainFeed / MacroFeed      ← 대안 데이터
├── FeatureStore                 ← 지표 계산 캐시
├── RegimeService                ← 시장 레짐 분류
├── PositionReconciler           ← 거래소-PM 정합성
├── AnalyticsEngine              ← 실시간 성과 분석
├── StateManager (SQLite)        ← 상태 영속화
└── DiscordBot                   ← 알림/리포트
```

### 목표 (Spot Single-Strategy — 간소화)

```
LiveRunner (간소화)
│
│  ┌─── 데이터 (기존 유지) ───────────────────┐
├── BinanceClient              WebSocket 1m 스트리밍
├── LiveDataFeed               1m bar 수신 + EventBus 발행
├── CandleAggregator           1m → 12H 캔들 집계
├── EventBus                   비동기 이벤트 라우팅
│  └──────────────────────────────────────────┘
│
│  ┌─── 전략 (기존 유지) ─────────────────────┐
├── StrategyEngine             SuperTrend preprocess → signal
│  └──────────────────────────────────────────┘
│
│  ┌─── 포지션/리스크 (기존 간소화) ──────────┐
├── EDAPortfolioManager        포지션 추적 + target weight
├── OMS                        주문 추상화
│  └──────────────────────────────────────────┘
│
│  ┌─── 실행 (신규) ──────────────────────────┐
├── BinanceSpotClient (신규)   Spot Market/Stop-Limit 주문
├── SpotExecutor (신규)        Spot 주문 실행
├── SpotStopManager (신규)     Stop-Limit TS 거래소 위임
│  └──────────────────────────────────────────┘
│
│  ┌─── 모니터링 (기존 유지) ─────────────────┐
├── AnalyticsEngine            실시간 PnL/Sharpe 추적
├── StateManager (SQLite)      상태 영속화
└── DiscordBot                 알림/리포트
```

**EventBus + 기존 EDA 파이프라인 유지 이유:**
- WebSocket → 1m bar가 6심볼 × 불규칙 타이밍으로 도착
- 비동기 스트리밍 + CandleAggregator 12H 집계가 EventBus 기반으로 검증됨
- Graceful shutdown, 에러 전파, Paper/Live 전환이 기존 구조에 내장

**제거하는 EventBus 구독자:**
- Orchestrator / Netting / CapitalAllocator / RiskAggregator
- RegimeService / FeatureStore
- DerivativesFeed / OnchainFeed / MacroFeed / OptionsFeed / DerivExtFeed
- SmartExecutor (Spot에서 구조 다름)
- EDARiskManager → SpotStopManager에 핵심(TS) 흡수, CircuitBreaker만 잔류

---

## 3. 자산 배분 (Asset Allocation)

### 현재: Equal Weight (1/6)

```
BTC: 16.7%  |  ETH: 16.7%  |  SOL: 16.7%
AVAX: 16.7% |  XRP: 16.7%  |  FTM: 16.7%
```

각 에셋에 동일 비중 배분. 구현이 가장 단순하고,
백테스트에서 에셋별 성과가 고르므로 (Sharpe 0.92~1.25) 합리적인 출발점.

**구현:**
```python
# SpotRunner 내부
weights = {s: 1.0 / len(symbols) for s in symbols}
# 진입 시: usdt_balance * weights[symbol] 만큼 매수
```

### 향후 고도화 옵션 (TODO)

| 방식 | 설명 | 복잡도 |
|------|------|:------:|
| **Inverse Volatility** | ATR 역수 비례 배분. 변동성 낮은 에셋에 더 많이 | 낮음 |
| **Risk Parity** | 각 에셋의 리스크 기여도 균등화 | 중간 |
| **Sharpe-Weighted** | 최근 N일 Sharpe 비례 배분 | 중간 |
| **Kelly Criterion** | 승률×PF 기반 최적 비중 | 높음 |
| **Max Drawdown Cap** | 에셋별 MDD 한도 초과 시 비중 축소 | 중간 |

**고도화 시 인터페이스:**
```python
# allocator.py (추후 구현)
class Allocator(Protocol):
    def compute_weights(self, prices: dict[str, pd.Series]) -> dict[str, float]: ...

class EqualWeight:
    def compute_weights(self, prices): return {s: 1/len(prices) for s in prices}

class InverseVolatility:
    def compute_weights(self, prices):
        atrs = {s: prices[s].pct_change().std() for s in prices}
        inv = {s: 1/v for s, v in atrs.items()}
        total = sum(inv.values())
        return {s: v/total for s, v in inv.items()}
```

> 현재는 `EqualWeight` 하드코딩. `Allocator` Protocol만 정의해두면
> 추후 교체 시 SpotRunner 코드 변경 최소화.

### 자본 변동 처리 (입출금)

YAML의 `initial_capital`은 **Paper 모드 전용**.
Live 모드에서는 시작 시 `fetch_balance()`로 거래소 실제 잔고를 조회하여 사용.

#### 방식: 12H bar마다 잔고 조회 + 신규 진입만 반영

```
12H bar 도착
  └→ fetch_balance() → 총 equity 계산
     total_equity = USDT잔고 + Σ(에셋 보유량 × 현재가)
  └→ 에셋별 목표 금액 = total_equity × weight (1/6)
  └→ 신규 진입 에셋만 새 목표 금액으로 매수
  └→ 기존 보유 에셋은 리밸런싱 하지 않음
```

#### 입금 예시

```
[현재] 총 equity $6,000 → 에셋당 $1,000
  BTC: $1,000 보유 중 (LONG)
  ETH: $1,000 보유 중 (LONG)
  SOL: 미보유 (FLAT)

     ↓  $3,000 입금

[다음 12H bar] 총 equity $9,000 → 에셋당 $1,500
  BTC: $1,000 유지 (리밸런싱 안 함)
  ETH: $1,000 유지 (리밸런싱 안 함)
  SOL: 신규 LONG 시그널 → $1,500으로 매수 ✅ (새 금액 반영)
```

#### 출금 예시

```
[현재] 총 equity $9,000 → 에셋당 $1,500
  BTC: $1,500 보유 중 (LONG)

     ↓  $3,000 출금

[다음 12H bar] 총 equity $6,000 → 에셋당 $1,000
  BTC: $1,500 유지 (리밸런싱 안 함, 자연 청산까지 대기)
  SOL: 신규 LONG 시그널 → $1,000으로 매수 (줄어든 금액 반영)
```

#### 왜 리밸런싱 안 하는가

- 12H 추세추종 전략은 포지션 교체가 자주 발생 (에셋당 연 15회)
  → 자연스럽게 새 자본이 반영됨
- 불필요한 리밸런싱 = 수수료 낭비 (Spot 0.1% × 추가 거래)
- 기존 포지션의 TS(stop price)는 진입 시 ATR 기준 → 비중 변경과 무관

---

## 4. 제거 대상 (기존 3번)

| 모듈 | 경로 | 이유 |
|------|------|------|
| **Orchestrator** | `src/orchestrator/` 전체 | 단일 전략, Pod 불필요 |
| **OrchestratedRunner** | `src/eda/orchestrated_runner.py` | Orchestrator 전용 Runner |
| **Netting** | `src/orchestrator/netting.py` | 단일 전략, 심볼 상쇄 불필요 |
| **AssetAllocator** | `src/orchestrator/asset_allocator.py` | EW 고정, 동적 배분 불필요 |
| **CapitalAllocator** | `src/orchestrator/allocator.py` | Pod 간 자본 분배 불필요 |
| **RiskAggregator** | `src/orchestrator/risk_aggregator.py` | Pod 간 리스크 집계 불필요 |
| **LifecycleManager** | `src/orchestrator/lifecycle.py` | 전략 승격/강등 불필요 |
| **Surveillance** | `src/orchestrator/surveillance.py` | 동적 에셋 탐색 불필요 |
| **VolTargeting** | `src/orchestrator/vol_targeting.py` | EW 고정 |
| **RegimeService** | `src/regime/` | ADX가 추세 필터 역할 |
| **FeatureStore** | `src/market/feature_store.py` | SuperTrend preprocessor 직접 계산 |
| **DerivativesFeed** | `src/eda/derivatives_feed.py` | Spot, Funding Rate 불필요 |
| **OnchainFeed** | `src/eda/onchain_feed.py` | 대안 데이터 미사용 |
| **MacroFeed** | `src/eda/macro_feed.py` | 대안 데이터 미사용 |
| **OptionsFeed** | `src/eda/options_feed.py` | 대안 데이터 미사용 |
| **DerivExtFeed** | `src/eda/deriv_ext_feed.py` | 대안 데이터 미사용 |
| **SmartExecutor** | `src/eda/smart_executor.py` | Spot에서 구조 다름 |
| **BinanceFuturesClient** | `src/exchange/binance_futures_client.py` | Futures 불필요 |

> **주의**: 삭제가 아닌 **비활성화/미참조**. 코드베이스는 유지하되
> LiveRunner에서 참조하지 않도록 분리.

---

## 5. 신규 개발

### 핵심 원칙

> 기존 EDA 파이프라인(EventBus + LiveDataFeed + CandleAggregator) 위에서
> **불필요한 구독자만 제거**하고, Executor + StopManager만 Spot용으로 교체.
> 기존 LiveRunner에 `spot_live()` / `spot_paper()` 팩토리 메서드 추가.

### 5.1. LiveRunner Spot 모드 추가

```python
# 기존 LiveRunner에 팩토리 메서드 추가
@classmethod
def spot_live(
    cls,
    strategy: BaseStrategy,
    symbols: list[str],
    target_timeframe: str,
    config: PortfolioManagerConfig,
    client: BinanceClient,
    spot_client: BinanceSpotClient,  # 신규
    initial_capital: float = 10000.0,
    asset_weights: dict[str, float] | None = None,
    ...
) -> LiveRunner:
    """Spot Live 모드.

    기존 live()과 동일한 구조, Executor + StopManager만 Spot용 교체.
    Orchestrator/Derivatives/Onchain 관련 초기화 스킵.
    """
    feed = LiveDataFeed(symbols, target_timeframe, client)
    executor = SpotExecutor(spot_client)  # Futures → Spot 교체
    ...
```

**기존 LiveRunner.run() 흐름 그대로 활용:**
```
WebSocket 1m → EventBus(BAR) → CandleAggregator → EventBus(BAR_12H)
→ StrategyEngine → EventBus(SIGNAL) → PM → EventBus(ORDER)
→ OMS → SpotExecutor → Binance Spot API
            └→ SpotStopManager → Stop-Limit 설정/갱신
```

### 5.2. BinanceSpotClient (신규, `src/exchange/binance_spot_client.py`)

`BinanceFuturesClient`와 동일한 패턴 (async context manager + CCXT Pro).

```python
class BinanceSpotClient:
    """Binance Spot 주문 클라이언트.

    BinanceFuturesClient와 동일한 async with 패턴.
    Spot 전용: reduceOnly/positionSide 없음.
    """

    async def create_market_order(symbol, side, amount) -> dict
    async def create_stop_limit_order(symbol, side, amount, stop_price, limit_price) -> dict
    async def cancel_order(symbol, order_id) -> None
    async def fetch_balance() -> dict[str, Decimal]  # 에셋별 잔고
    async def fetch_open_orders(symbol) -> list[dict]
```

### 5.3. SpotExecutor (신규, `src/eda/spot_executor.py`)

`ExecutorPort` 프로토콜 준수 → OMS에서 기존 LiveExecutor 자리 교체.

```python
class SpotExecutor:
    """Spot 주문 실행기. ExecutorPort 준수.

    LiveExecutor 대비 차이:
    - reduceOnly / positionSide 없음
    - Market Buy: quoteOrderQty (USDT 금액 기준)
    - Market Sell: 보유 수량 전량 매도
    """
```

### 5.4. SpotStopManager — Stop-Limit Ratchet 방식 (방식 A)

기존 `ExchangeStopManager`의 Spot 버전. EventBus BAR/FILL 이벤트 구독.

#### 왜 Stop-Limit Ratchet인가 (방식 A vs B)

| | A. Stop-Limit + Ratchet | B. 거래소 trailingDelta |
|---|---|---|
| stop 거리 | **3.0 × ATR (동적)** | 고정 % (예: 3%) |
| 변동성 적응 | ✅ ATR이 변하면 stop 거리도 변함 | ❌ 항상 같은 % |
| 봇 장애 보호 | ✅ 거래소에 주문 존재 | ✅ 거래소에 주문 존재 |
| 봇 개입 필요 | 12H마다 갱신 (하루 2번) | 없음 |
| 전략 정합성 | ✅ 백테스트와 동일 로직 | ❌ 백테스트와 다름 |

→ **방식 A 채택**. 백테스트의 ATR 기반 TS와 동일한 로직을 라이브에서 재현.

#### 동작 흐름

```
┌─────────────────────────────────────────────────────┐
│                    정상 운영                          │
│                                                     │
│  1. SIGNAL(LONG) → Market Buy 체결                   │
│     └→ 즉시 Stop-Limit Sell 설정                     │
│        stop_price = 체결가 - 3.0 × ATR               │
│        limit_price = stop_price × 0.995 (0.5% 여유)  │
│                                                     │
│  2. 12H bar 도착 (하루 2번)                           │
│     └→ 새 ATR 계산 → 새 stop = high_watermark - 3×ATR│
│        if 새 stop > 기존 stop:                       │
│           취소 → 재설정 (ratchet up ⬆️)               │
│        else:                                        │
│           유지 (절대 아래로 안 내림 🔒)                 │
│                                                     │
│  3a. SIGNAL(NEUTRAL) → 반대 신호                     │
│      └→ Stop-Limit 취소 → Market Sell (정상 청산)     │
│                                                     │
│  3b. 가격 급락 → stop_price 도달                      │
│      └→ 거래소가 자동 Limit Sell 실행 (TS 청산)        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    장애 상황                          │
│                                                     │
│  봇 다운 → Stop-Limit이 거래소에 살아있음              │
│  └→ 가격 급락 시 거래소가 자동 매도 → 자산 보호         │
│  └→ 봇 재시작 시 fetch_open_orders()로 상태 복구       │
└─────────────────────────────────────────────────────┘
```

#### Ratchet 예시 (BTC, ATR=$1,500)

```
Bar 1: 매수 $100,000
       → Stop-Limit Sell: stop=$95,500, limit=$95,023
         (100,000 - 3×1,500 = 95,500)

Bar 2: 고가 $105,000, ATR=$1,600
       → 새 stop = 105,000 - 3×1,600 = $100,200
       → 100,200 > 95,500 → 취소 + 재설정 ⬆️
       → Stop-Limit Sell: stop=$100,200, limit=$99,699

Bar 3: 고가 $103,000, ATR=$1,700
       → 새 stop = 105,000 - 3×1,700 = $99,900
         (high_watermark는 여전히 105,000)
       → 99,900 < 100,200 → 유지 🔒

Bar 4: 고가 $112,000, ATR=$1,500
       → 새 stop = 112,000 - 3×1,500 = $107,500
       → 107,500 > 100,200 → 취소 + 재설정 ⬆️
       → Stop-Limit Sell: stop=$107,500, limit=$106,963

Bar 5: 가격 $106,000으로 급락
       → stop=$107,500 발동 → limit=$106,963에 매도 시도
       → 체결 → 포지션 청산 완료
```

#### 상태 관리 (심볼별)

```python
@dataclass
class SpotStopState:
    symbol: str
    order_id: str              # 거래소 주문 ID
    stop_price: float          # 현재 설정된 stop price
    limit_price: float         # 현재 설정된 limit price
    quantity: float            # 매도 수량
    high_watermark: float      # 진입 이후 최고가
    last_atr: float            # 마지막 ATR 값
    created_at: datetime
```

#### 엣지 케이스 처리

| 상황 | 처리 |
|------|------|
| Stop-Limit 부분 체결 | 잔량 확인 → 잔량에 대해 재설정 |
| Stop 발동 + limit 미체결 (가격 관통) | 미체결 감지 → Market Sell fallback |
| 봇 재시작 | `fetch_open_orders()` → SpotStopState 복구 |
| API 실패 (stop 설정 실패) | 재시도 3회 → 실패 시 Discord CRITICAL 알림 |
| 12H bar 누락 | 마지막 stop 유지 (거래소에 살아있으므로 안전) |

### 5.5. 설정 YAML

```yaml
# config/spot_supertrend.yaml
mode: paper  # paper | live

symbols: [BTC/USDT, ETH/USDT, SOL/USDT, AVAX/USDT, XRP/USDT, FTM/USDT]
timeframe: "12h"

strategy:
  name: supertrend
  params:
    atr_period: 7
    multiplier: 2.5
    adx_period: 14
    adx_threshold: 25
    short_mode: 0  # DISABLED

portfolio:
  initial_capital: 10000.0  # Paper 모드 전용. Live는 거래소 잔고 자동 조회
  allocation: equal_weight  # 추후: inverse_vol, risk_parity
  use_trailing_stop: true
  trailing_stop_atr_multiplier: 3.0
  cost_model:  # Paper 모드 전용. Live는 거래소가 실제 수수료 차감
    maker_fee: 0.001   # Spot 0.1%
    taker_fee: 0.001
    slippage: 0.0005

discord_webhook: ${DISCORD_WEBHOOK_URL}
```

---

## 6. 기존 코드 재사용 (변경 없음)

| 모듈 | 용도 |
|------|------|
| `LiveDataFeed` | WebSocket 1m → CandleAggregator → 12H bar |
| `CandleAggregator` | 1m → 12H 캔들 집계 |
| `StrategyEngine` | SuperTrend 전략 실행 (preprocess → generate_signals) |
| `EDAPortfolioManager` | 포지션 관리 + target weight 계산 |
| `EDARiskManager` | Circuit Breaker (간소화) |
| `OMS` | Executor 추상화 계층 |
| `EventBus` | 이벤트 기반 통신 |
| `AnalyticsEngine` | 실시간 Sharpe/MDD/PnL 추적 |
| `StateManager` | SQLite 상태 영속화 |
| `PositionReconciler` | 거래소-PM 정합성 검증 |
| `DiscordBot` | 알림/일일 리포트 |
| `BinanceClient` | Spot WebSocket 데이터 스트리밍 (기존 그대로) |

---

## 7. Futures → Spot 전환 핵심 차이

| 항목 | Futures (현재) | Spot (목표) |
|------|---------------|-------------|
| **마켓** | USDT-M Linear | Spot |
| **심볼 형식** | `BTC/USDT:USDT` | `BTC/USDT` |
| **숏** | 가능 | 불가 (Long-Only 전략이므로 무관) |
| **레버리지** | 1x~125x | 없음 (1x 고정) |
| **Funding Rate** | 8h마다 발생 | 없음 (비용 절감) |
| **청산(Liquidation)** | 존재 | 없음 (안전) |
| **TS 위임** | STOP_MARKET | Stop-Limit / OCO |
| **reduceOnly** | 지원 | 미지원 (보유량 기반) |
| **positionSide** | LONG/SHORT | 없음 |
| **수수료** | Maker 0.02% / Taker 0.04% | 0.1% (BNB 보유 시 0.075%) |
| **잔고 관리** | USDT margin 단일 | 에셋별 개별 보유 |

### Spot 잔고 관리 주의사항

```
Futures: USDT 잔고 하나로 모든 심볼 거래
Spot:    BTC, ETH, SOL 등 에셋별 개별 잔고

→ EDAPortfolioManager의 equity 계산 방식 변경 필요:
  기존: cash (USDT) + unrealized PnL
  변경: USDT 잔고 + sum(각 에셋 보유량 × 현재가)
```

---

## 8. 작업 단계 (Implementation Phases)

### Phase A: 핵심 구현 (~4 파일 신규 + LiveRunner 수정)

| 파일 | 신규/수정 | 내용 |
|------|:--------:|------|
| `src/exchange/binance_spot_client.py` | 신규 | Market/Stop-Limit 주문, 잔고 조회 |
| `src/eda/spot_executor.py` | 신규 | SpotExecutor (ExecutorPort 준수) |
| `src/eda/spot_stop_manager.py` | 신규 | Stop-Limit TS 거래소 위임 |
| `src/eda/live_runner.py` | 수정 | `spot_live()` / `spot_paper()` 팩토리 추가 |
| `src/eda/portfolio_manager.py` | 수정 | Spot equity 계산 (에셋별 잔고) |

### Phase B: Paper Trading 검증

1. Paper 모드로 1주일 라이브 데이터 검증
   - 12H bar 생성 확인
   - 시그널 발생 → 로그 출력
   - TS stop price 계산 확인

### Phase C: Live 전환

1. 소액 ($100) 단일 에셋 (BTC) 실거래 테스트
2. 정상 확인 후 6개 에셋 순차 추가
3. 풀 자본 투입

---

## 9. 리스크 관리 변경 사항

### 유지
- **Trailing Stop 3.0x ATR**: 거래소 Stop-Limit으로 위임
- **Circuit Breaker**: 일일 손실 한도 초과 시 거래 중단
- **Discord 알림**: 진입/청산/에러 실시간 통보

### 제거/간소화
- **System Stop-Loss**: TS가 주력이므로 별도 SL 불필요
- **Rebalance Threshold**: All-in/All-out 전략이므로 불필요
- **Netting**: 단일 전략, 심볼 중복 없음
- **DD De-Risking**: Orchestrator 레벨 기능, 제거
- **Vol Targeting**: EW 고정, 변동성 기반 배분 불필요

### 신규
- **Spot 잔고 모니터링**: 에셋별 보유량 추적
- **Stop-Limit 상태 추적**: 거래소 미체결 주문 상태 주기적 확인
- **BNB 잔고 확인**: 수수료 할인용 BNB 최소 보유량 알림

---

## 10. 비용 비교

### 연간 비용 추정 (에셋당 ~15회 거래 기준, 6에셋 = 90회)

| 항목 | Futures | Spot | 차이 |
|------|---------|------|------|
| 수수료 (Taker) | 0.04% × 90 = 3.6% | 0.1% × 90 = 9.0% | Spot +5.4% |
| 수수료 (BNB 할인) | - | 0.075% × 90 = 6.75% | Spot +3.15% |
| Funding Rate | ~0.01% × 3/day × 365 ≈ 11% | 0% | **Spot -11%** |
| Slippage | 비슷 | 비슷 | - |
| **합계** | ~14.6% | ~6.75% (BNB) | **Spot 약 8% 절감** |

> Long-Only + 장기 홀딩(수일~수주)에서 Funding Rate 절감이 수수료 차이를 압도.
> BNB 소량 보유 시 수수료 25% 할인 → Spot이 확실히 유리.

---

## 11. 체크리스트

### 구현
- [ ] `BinanceSpotClient` 구현 (Market/Stop-Limit/잔고)
- [ ] `SpotRunner` 구현 (데이터→전략→주문 직선 흐름)
- [ ] `SpotStopManager` 구현 (Stop-Limit TS 위임 + ratchet)
- [ ] `config/spot_supertrend.yaml` 작성
- [ ] 단위 테스트 (mock exchange)

### 검증
- [ ] Paper Trading 1주 검증
- [ ] Live 소액 BTC 단일 테스트
- [ ] 6개 에셋 순차 투입

### 운영
- [ ] BNB 최소 보유량 확보 (수수료 25% 할인)
- [ ] Discord 알림 (진입/청산/TS 발동/에러)
- [ ] 봇 재시작 시 Stop-Limit 주문 복구 확인

### 향후 고도화 (TODO)
- [ ] 자산 배분 고도화 (Inverse Volatility / Risk Parity)
- [ ] 에셋 추가/제거 프로세스 (시총/유동성 기반)
- [ ] 성과 대시보드 (웹 or Grafana)
