# NautilusTrader 도입 평가 리서치

> **Research Date**: 2026-02-08
> **Scope**: NautilusTrader vs MC Coin Bot 자체 EDA 시스템 비교 평가
> **Conclusion**: 현 시점 도입 비추천 (자체 EDA 유지 권장)

---

## 목차

1. [Executive Summary](#1-executive-summary)
2. [NautilusTrader 아키텍처](#2-nautilustrader-아키텍처)
3. [주요 기능](#3-주요-기능)
4. [MC Coin Bot과의 상세 비교](#4-mc-coin-bot과의-상세-비교)
5. [도입 시 이점](#5-도입-시-이점)
6. [도입 시 단점 및 리스크](#6-도입-시-단점-및-리스크)
7. [마이그레이션 비용 분석](#7-마이그레이션-비용-분석)
8. [커뮤니티 피드백](#8-커뮤니티-피드백)
9. [결론 및 권고](#9-결론-및-권고)
10. [References](#10-references)

---

## 1. Executive Summary

### 한 줄 결론

> **현 시점에서 NautilusTrader 도입은 비용 대비 효과가 낮다.** MC Coin Bot의 자체 EDA 시스템(Phase 5-B, 397 tests)이 이미 업계 best practice를 충족하고 있으며, NautilusTrader의 핵심 강점(Rust 성능, L2/L3 시뮬레이션)은 현재 일봉/분봉 레벨 전략에서 의미 있는 차이를 만들지 않는다.

### 핵심 Trade-off

| 항목 | 자체 EDA 유지 | NautilusTrader 전환 |
|------|:---:|:---:|
| 마이그레이션 비용 | 0 | 4~8주 풀타임 |
| 커스터마이징 자유도 | 완전 자유 | Framework 제약 |
| VectorBT 연동 | 검증 완료 (Parity) | 별도 변환 필요 |
| CCXT 거래소 지원 | 전체 | Binance/Bybit 등 제한 |
| Backtest-to-Live 통일 | 별도 구현 필요 | 동일 코드 |
| 고빈도 데이터 성능 | Python 수준 | Rust core (5M rows/sec) |
| API 안정성 | 자체 관리 | Beta (breaking changes) |
| L2/L3 Orderbook 시뮬 | 미지원 | 4종 Fill Model |

---

## 2. NautilusTrader 아키텍처

### 2.1 3-Layer 구조

```
┌──────────────────────────────────────────────┐
│  Python Layer (User-facing API)              │
│  Strategy, Actor, Config, TradingNode        │
├──────────────────────────────────────────────┤
│  Cython Layer (Bridge)                       │
│  Type-safe C-level 바인딩, PyO3              │
├──────────────────────────────────────────────┤
│  Rust Layer (Foundation)                     │
│  nautilus-core, model, data, indicators,     │
│  trading, live, infrastructure, system       │
│  tokio 기반 비동기 네트워킹                    │
└──────────────────────────────────────────────┘
```

- **Rust Core**: Domain model, indicators, data engine, matching engine 모두 Rust 구현
- **Cython Bridge**: Python ↔ Rust 오버헤드 최소화
- **Python API**: 전략 개발은 순수 Python으로 가능

### 2.2 NautilusKernel

중앙 오케스트레이션 컴포넌트로, 3가지 환경에서 동일한 전략 코드 실행을 보장:

| 환경 | Data | Execution |
|------|------|-----------|
| **Backtest** | Historical | Simulated venues |
| **Sandbox** | Real-time | Simulated venues |
| **Live** | Real-time | Live venues |

- **단일 스레드 모델**: MessageBus, 전략 로직, 리스크 체크를 하나의 스레드에서 처리 → deterministic event ordering
- 백그라운드 서비스(네트워크 I/O, persistence)는 별도 스레드

### 2.3 MessageBus

- **Pub/Sub 패턴**: 이벤트 브로드캐스팅
- **Request/Response 패턴**: 응답이 필요한 작업
- Redis 또는 in-memory 백엔드 선택 가능

### 2.4 설계 원칙

- Domain-Driven Design
- Event-Driven Architecture
- Ports and Adapters 패턴
- **Crash-Only Design**: NaN, Infinity, 타입 변환 실패 시 즉시 panic
- **Component State Machine**: PRE_INITIALIZED → READY → RUNNING → STOPPED

---

## 3. 주요 기능

### 3.1 거래소 지원

| 거래소 | 상태 | 비고 |
|--------|------|------|
| **Binance** (Spot/Futures) | Stable | 가장 완성도 높은 integration |
| **Bybit** | Stable | |
| **BitMEX** | Stable | |
| **OKX** | Stable | |
| **Coinbase International** | Stable | |
| **Kraken** | Beta | |
| **dYdX v3/v4** | Stable/Building | |
| **Interactive Brokers** | Stable | |
| **Databento** | Stable | Data only |
| **Tardis** | Stable | Data only |

**주의**: CCXT 미지원. 2021년 CCXT integration이 중단되었고, 재도입 계획 없음 (모든 adapter를 Rust로 전환 중).

### 3.2 Binance Futures 상세

- `CryptoPerpetual` instrument type (USDT-M, Coin-M)
- 심볼 규약: `BTCUSDT-PERP.BINANCE`
- Hedge Mode 지원 (One-Way/Hedge)
- 주문 유형: MARKET, LIMIT, STOP_MARKET, STOP_LIMIT, TRAILING_STOP_MARKET 등
- Rate Limiting: Token bucket (Futures 2,400/min)
- 인증: HMAC, RSA, Ed25519
- Testnet 지원

### 3.3 백테스트 엔진

**두 가지 API:**
- **BacktestEngine** (Low-Level): RAM 로드, `reset()`으로 반복 최적화
- **BacktestNode** (High-Level): 배치 실행, RAM 초과 데이터 스트리밍

**Bar 실행 처리:**
- 각 Bar를 O → H → L → C 4개 가격 포인트로 분해 (각 25% 거래량)
- Adaptive mode: H/L 순서를 bar 구조에 따라 지능적 결정

**Fill Model:**

| 모델 | 설명 |
|------|------|
| `BestPriceFillModel` | 최우선 호가 체결 |
| `ThreeTierFillModel` | 3단계 체결 모델 |
| `SizeAwareFillModel` | 주문 크기 반영 |
| `VolumeSensitiveFillModel` | 거래량 민감 체결 |

### 3.4 전략 개발 패턴

```python
class MyStrategy(Strategy):
    def __init__(self, config: MyStrategyConfig) -> None:
        super().__init__(config)
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)

    def on_start(self):
        self.register_indicator_for_bars(self.bar_type, self.fast_ema)
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar):
        if self.fast_ema.value > self.slow_ema.value:
            order = self.order_factory.market(...)
            self.submit_order(order)
```

### 3.5 내장 지표

| 카테고리 | 지표 |
|---------|------|
| Moving Averages | SMA, EMA, DEMA, HMA, WMA, VWAP |
| Momentum | RSI, MACD, Aroon, CCI, Stochastics |
| Volatility | ATR, Bollinger Bands, Donchian Channels, Keltner Channels |
| Microstructure | Book Imbalance Ratio |

### 3.6 데이터 관리

- **ParquetDataCatalog**: Nautilus-specific Parquet 포맷
- 로컬 파일, S3, GCS 스토리지 지원
- 나노초 해상도 타임스탬프
- Custom Data 타입 persist 가능

### 3.7 라이브 트레이딩

- **TradingNode**: 프로세스당 단일 인스턴스 (글로벌 싱글톤)
- **Execution Reconciliation**: 시작 시 + 지속적 루프로 거래소 상태 동기화
- **State Persistence**: Redis 또는 PostgreSQL 백엔드
- **Docker 배포 지원**

---

## 4. MC Coin Bot과의 상세 비교

### 4.1 아키텍처 비교

| 영역 | MC Coin Bot (자체 EDA) | NautilusTrader |
|------|----------------------|---------------|
| **코어 언어** | Python 3.13 (asyncio) | Rust + Cython + Python |
| **이벤트 시스템** | EventBus (async Queue, flush) | MessageBus (Rust, Pub/Sub + Req/Rep) |
| **백테스트 엔진** | VBT (vectorized) + EDA (event-driven) | NautilusKernel (event-driven only) |
| **라이브 전환** | 별도 구현 필요 (Phase 8 계획) | 동일 코드 (adapter 교체만) |
| **데이터 처리** | Medallion (Bronze/Silver Parquet) | ParquetDataCatalog (Nautilus 포맷) |
| **거래소 연동** | CCXT Pro (Binance) | 네이티브 Rust adapter |
| **리스크 관리** | 자체 PM/RM/OMS 3단계 | 내장 RiskEngine |
| **비용 모델** | CostModel (maker/taker/slippage/funding) | Fill Model (4종, 확률적 파라미터) |
| **지표** | 자체 구현 (Numba JIT 3개 함수) | 20+ 내장 지표 (Rust) |
| **데이터 해상도** | 1m bar (CandleAggregator) | 나노초 tick 데이터 |
| **성능** | Python 수준 | 5M rows/sec (Rust core) |

### 4.2 기능 대응표

| MC Coin Bot 기능 | NautilusTrader 대응 | 난이도 |
|-----------------|-------------------|--------|
| BaseStrategy (4종: TSMOM, BB-RSI, Donchian, Breakout) | Strategy 클래스 재작성 | 높음 |
| EventBus (flush 기반 bar-by-bar 동기화) | MessageBus (자체 이벤트 루프) | 중간 |
| PM (vol-target, trailing stop ATR, batch order) | Strategy 내부 + RiskEngine | 높음 |
| RM (leverage/position/order validation, circuit breaker) | RiskEngine + ExecAlgorithm | 높음 |
| OMS (idempotent, executor routing) | ExecutionEngine (내장) | 낮음 |
| BacktestExecutor (open price fill, CostModel) | Fill Model (4종) | 낮음 |
| AnalyticsEngine (equity curve, PerformanceMetrics) | PortfolioAnalyzer | 중간 |
| CandleAggregator (1m→target TF) | Bar aggregation (내장) | 낮음 |
| DataFeed (flush 동기화) | DataEngine (자동 처리) | 낮음 |
| Validation (IS/OOS, WFA, CPCV, DSR, PBO) | 미제공 (자체 구현 필요) | 해당없음 |
| Parameter Sweep (VectorBT) | BacktestEngine.reset() 반복 | 높음 |

### 4.3 MC Coin Bot만의 고유 강점

1. **VBT + EDA Dual Architecture**: Vectorized sweep → Event-driven 검증, Parity 확인 완료
2. **3-Tier Validation**: IS/OOS, WFA, CPCV, DSR, PBO 통합 검증 파이프라인
3. **Batch Order Processing**: 멀티에셋 동일 equity snapshot 기반 일괄 주문
4. **PM 3-Layer Defense**: Stop-loss (intrabar) + Trailing Stop (ATR) + Vol-target rebalancing
5. **Medallion Architecture**: Bronze → Silver 데이터 품질 보장 파이프라인
6. **CCXT Pro**: 모든 거래소 연동 가능 (NautilusTrader는 제한적)

---

## 5. 도입 시 이점

### 5.1 Backtest-to-Live 코드 통일 (가장 큰 이점)

NautilusTrader의 핵심 가치 제안:

```
동일 전략 코드 → adapter 교체만으로 backtest/sandbox/live 전환
```

MC Coin Bot은 현재 VBT(backtest) + EDA(event-driven) + Live(미구현) 3중 구조이다.
NautilusTrader를 사용하면 이 격차가 해소되어 Phase 8 (Production Readiness) 구현 부담이 대폭 감소한다.

### 5.2 고성능 Rust Core

- 5M rows/sec 데이터 스트리밍
- 나노초 해상도 타임스탬프
- GC 없는 deterministic latency
- **적용 시나리오**: tick 데이터 기반 HFT, L2/L3 orderbook 전략, 대규모 멀티에셋 (수천 종목)

### 5.3 정밀한 Fill Simulation

4종 Fill Model + 확률적 파라미터로 현실적 체결 시뮬레이션:
- Queue position 반영 (`prob_fill_on_limit`)
- 주문 크기 대비 유동성 영향
- Bar를 O→H→L→C로 분해하여 intrabar 체결 시뮬레이션

### 5.4 내장 Execution Reconciliation

라이브 트레이딩 시 거래소 상태와 내부 상태를 자동 동기화:
- Startup reconciliation (cached state 복구)
- Continuous loop (in-flight 주문 모니터링)
- Rate limit 보호

### 5.5 다중 거래소 네이티브 지원

Binance, Bybit, BitMEX, OKX 등 15+ venue를 Rust 네이티브 adapter로 지원.
CCXT의 Python overhead 없이 직접 통신.

### 5.6 성숙한 주문 관리

- Emulated orders (거래소 미지원 주문 유형 로컬 에뮬레이션)
- ExecAlgorithm (TWAP, VWAP 등 실행 알고리즘)
- Batch submit/modify/cancel

---

## 6. 도입 시 단점 및 리스크

### 6.1 Beta 상태 — API 불안정 (리스크: 매우 높음)

- 현재 v1.222.0 **Beta**: "breaking changes can occur between releases"
- Stable API는 2.x에서 목표 (Rust 포팅 완료 후)
- Breaking changes는 "best-effort basis"로만 문서화
- **2주 간격 릴리스** → 빈번한 업그레이드 부담

**영향**: 마이그레이션 후에도 지속적으로 NautilusTrader 업그레이드에 따른 코드 수정 필요.

### 6.2 가파른 학습 곡선 (리스크: 높음)

공식 문서에서도 "beginners and experts alike may find the learning curve steep" 인정.

주요 pain point:
- Vectorized → Event-driven 패러다임 전환
- Rust + Cython 이중 언어 디버깅
- Order lifecycle 프로그래밍
- 복잡한 설정 (NautilusKernel, adapter config)

### 6.3 문서 부족 (리스크: 높음)

- 실전 예제, end-to-end 튜토리얼 부족
- 멀티에셋 전략, 커스텀 리스크 관리 문서 미비
- GitHub Discussion에서 "so complex and almost no docs" 불만 (2025년)
- 샘플 대부분 1~2 instrument만 다룸

### 6.4 커스터마이징 제약 (리스크: 중간-높음)

| MC Coin Bot 기능 | NautilusTrader 제약 |
|-----------------|-------------------|
| PM vol-target rebalancing | RiskEngine 커스터마이징 제한적 → Strategy 내부로 이동 필요 |
| PM batch order processing | 지원하지만 자체 로직(동일 equity snapshot)과 다를 수 있음 |
| PM trailing stop (ATR incremental) | Strategy 내부 구현 필요 |
| RM circuit breaker | RiskEngine으로 구현 가능하나 세부 동작 차이 |
| `_stopped_this_bar` guard | Framework에 해당 개념 없음 → 직접 구현 |

### 6.5 Crypto-Specific 한계 (리스크: 중간)

- **Funding rate backtest**: perpetual futures에서 funding rate 정기 차감 시뮬레이션이 명확히 문서화되지 않음
- **Liquidation engine**: 레버리지 포지션 강제 청산 시뮬레이션 정확도 불확실
- **Binance Futures margin 계산 오류**: [Issue #3420](https://github.com/nautechsystems/nautilus_trader/issues/3420) (2025년 보고)
- **CCXT 미지원**: 소규모 거래소 연동 불가

### 6.6 의존성 무게 (리스크: 중간)

| 항목 | 수치 |
|------|------|
| Wheel 크기 | 47.4 ~ 111.0 MB (플랫폼별) |
| Python 요구 | 3.12+ (3.11 지원 종료) |
| Rust MSRV | 1.92.0 |
| 추가 의존성 | Cap'n Proto, Redis (선택) |

소스 빌드 시 Rust toolchain + Cap'n Proto + Cython 컴파일 환경 필요.

### 6.7 VBT Parameter Sweep 대체 불가 (리스크: 중간)

NautilusTrader의 event-driven backtest는 VectorBT의 vectorized sweep 대비 **수십~수백 배 느림**.
Parameter sweep은 VectorBT를 유지해야 하며, 두 프레임워크 간 전략 코드 변환이 필요.

### 6.8 Validation Framework 미제공 (리스크: 높음)

MC Coin Bot의 3-Tier Validation (IS/OOS, WFA, CPCV, DSR, PBO)은 NautilusTrader에 포함되지 않음.
마이그레이션 시 이 전체를 NautilusTrader API 위에 재구현해야 함.

---

## 7. 마이그레이션 비용 분석

### 7.1 작업 항목별 난이도

| 작업 | 난이도 | 예상 기간 | 설명 |
|------|--------|----------|------|
| Strategy 전환 (4종) | 높음 | 1~2주 | BaseStrategy → NautilusTrader Strategy. on_bar/on_order_event 패턴 |
| EventBus → MessageBus | 중간 | 3~5일 | flush 기반 동기화 → NautilusTrader 이벤트 루프 |
| PM/RM 통합 | 높음 | 1~2주 | Vol-target, trailing stop, batch order, circuit breaker 재구현 |
| Data Pipeline 변환 | 높음 | 3~5일 | Medallion → ParquetDataCatalog |
| Backtest Engine 통합 | 중간 | 3~5일 | VBT + EDA → NautilusTrader BacktestEngine |
| 테스트 재작성 (397개) | 높음 | 1~2주 | NautilusTrader API 기준 |
| Validation 재구현 | 높음 | 1주 | IS/OOS, WFA, CPCV, DSR, PBO 파이프라인 |
| CLI 재구현 | 낮음 | 2~3일 | run/run-multi/run-agg 명령어 |

### 7.2 총 예상 비용

| 항목 | 예상치 |
|------|--------|
| **총 기간** | 4~8주 (풀타임) |
| **재작성 테스트** | ~400개 |
| **재작성 모듈** | ~15개 |
| **학습 곡선** | 1~2주 (NautilusTrader 패러다임 습득) |
| **리스크** | Beta API 변경으로 추가 작업 발생 가능 |

### 7.3 기회비용

4~8주의 마이그레이션 기간 동안 진행하지 못하는 작업:
- Phase 6: Dynamic Slippage, HMM Regime Detection
- Phase 7: Triple Barrier, Meta-Labeling
- Phase 8: Live Trading Adapter (자체 구현)
- 새로운 전략 연구 및 검증

---

## 8. 커뮤니티 피드백

### 8.1 긍정적 피드백

- **아키텍처 설계**: Rust core + Python API의 성능/사용성 균형
- **Backtest-to-Live**: 동일 코드 실행 가능한 설계 철학
- **활발한 개발**: 18.9k GitHub stars, 주간 커밋
- **오픈소스**: LGPL-3.0, 상업 사용 가능

### 8.2 부정적 피드백

**Hacker News (2025-07):**
- Goldman Sachs 트레이더: "규제 준수 기능이 부족하다. 실제 운용에서는 plug and play가 아니다."
- "핵심은 전략 발굴이지 인프라가 아니다. NautilusTrader는 인프라만 제공한다."
- QuantConnect/LEAN 대비 차별점 불분명

**GitHub Discussions:**
- [Discussion #1415](https://github.com/nautechsystems/nautilus_trader/discussions/1415): 5,000~8,000 ticker 멀티에셋 전략을 검토한 사용자가 **결국 채택하지 않고** 자체 엔진 구축
- "so complex and almost no docs" (2025-09)

**알려진 버그:**
- Binance Futures margin 계산 오류 ([#3420](https://github.com/nautechsystems/nautilus_trader/issues/3420))
- OTO contingency partial fill 문제 ([#2623](https://github.com/nautechsystems/nautilus_trader/issues/2623))
- Instrument 자동 로드 실패 ([#3237](https://github.com/nautechsystems/nautilus_trader/issues/3237))
- Reduce-Only 주문 잘못된 reject ([#2534](https://github.com/nautechsystems/nautilus_trader/issues/2534))

---

## 9. 결론 및 권고

### 9.1 종합 평가

| 평가 항목 | 점수 (5점 만점) | 비고 |
|----------|:---:|------|
| 아키텍처 설계 | 5 | Rust core, 3-layer, backtest/live 통일 |
| 성능 | 5 | 5M rows/sec, 나노초 해상도 |
| 기능 완성도 | 4 | 내장 지표, Fill Model, Reconciliation |
| 거래소 지원 | 3 | Binance 안정, 그 외 제한적. CCXT 미지원 |
| 문서/커뮤니티 | 2 | 부족한 문서, 소규모 실질 사용자 |
| API 안정성 | 2 | Beta, 빈번한 breaking changes |
| 학습 곡선 | 2 | 매우 가파름 |
| Crypto 특화 | 3 | 기본 지원, funding rate/liquidation 시뮬 불확실 |
| MC Coin Bot 적합성 | 2 | 마이그레이션 비용 대비 효과 낮음 |

### 9.2 시나리오별 권고

#### 시나리오 A: 현재 (일봉/분봉, 8-asset, TSMOM/BB-RSI)

> **권고: 자체 EDA 유지**

- 현재 EDA 시스템이 VBT parity 검증 완료, 397 tests
- NautilusTrader의 Rust 성능은 일봉/분봉 레벨에서 불필요
- 마이그레이션 4~8주 대비 새 기능 개발이 더 가치 있음

#### 시나리오 B: 향후 Live Trading 진입 시 (Phase 8)

> **권고: 부분 참조, 자체 구현 유지**

- NautilusTrader의 **Execution Reconciliation 패턴** 참조하여 자체 구현
- CCXT Pro 기반 LiveExecutor + Reconciliation Loop 개발
- 이유: 자체 PM/RM 로직 유지가 중요, NautilusTrader의 RiskEngine으로는 대체 어려움

#### 시나리오 C: HFT / L2 Orderbook 전략 확장 시

> **권고: NautilusTrader 도입 재검토**

- tick 데이터 기반 전략에서 Rust core 성능이 결정적
- L2/L3 orderbook 시뮬레이션이 필수
- 이 시점에서는 마이그레이션 비용이 정당화될 수 있음
- 단, 2.x Stable API 출시 이후가 적절

#### 시나리오 D: 수천 종목 멀티에셋 확장 시

> **권고: NautilusTrader 도입 재검토**

- 5M rows/sec 데이터 처리가 필요한 규모
- 현재 Python EDA의 성능 한계가 체감될 수 있음

### 9.3 자체 EDA 로드맵 우선순위

NautilusTrader에서 배울 수 있는 패턴을 자체 EDA에 점진적으로 적용:

```
Phase 6 (즉시 실행 가능)
├── 6.1 Dynamic Slippage Model (NautilusTrader의 Fill Model 패턴 참조)
├── 6.2 HMM Regime Detection
├── 6.3 Execution Reconciliation (NautilusTrader 패턴 참조)
└── 6.4 Parallel Parameter Sweep

Phase 8 (Live Trading)
├── 8.1 LiveExecutor (CCXT Pro 기반)
├── 8.2 Reconciliation Loop (NautilusTrader 패턴 참조)
│   ├── Startup reconciliation
│   ├── Continuous in-flight order monitoring
│   └── Rate limit 보호
├── 8.3 State Persistence (Redis/SQLite)
└── 8.4 Shadow Trading Mode
```

### 9.4 최종 판단

| 판단 기준 | 결론 |
|----------|------|
| **지금 전환해야 하는가?** | **No** — 비용 대비 효과 낮음 |
| **향후 전환 가능성?** | 2.x Stable + HFT 전략 시 재검토 |
| **참조할 가치가 있는가?** | **Yes** — Reconciliation, Fill Model, Adapter 패턴 |
| **병행 사용 가능한가?** | 비효율적 — VectorBT + 자체 EDA가 더 적합 |

---

## 10. References

### NautilusTrader 공식
- [Architecture](https://nautilustrader.io/docs/latest/concepts/architecture/)
- [Strategies](https://nautilustrader.io/docs/latest/concepts/strategies/)
- [Backtesting](https://nautilustrader.io/docs/latest/concepts/backtesting/)
- [Live Trading](https://nautilustrader.io/docs/latest/concepts/live/)
- [Binance Integration](https://nautilustrader.io/docs/nightly/integrations/binance/)
- [Installation](https://nautilustrader.io/docs/latest/getting_started/installation/)

### GitHub
- [Repository](https://github.com/nautechsystems/nautilus_trader) (18.9k stars)
- [Releases](https://github.com/nautechsystems/nautilus_trader/releases) (v1.222.0 Beta)
- [Discussion #1415 - Migration Experience](https://github.com/nautechsystems/nautilus_trader/discussions/1415)
- [Issue #3420 - Binance Margin Bug](https://github.com/nautechsystems/nautilus_trader/issues/3420)
- [Issue #2885 - CCXT Integration](https://github.com/nautechsystems/nautilus_trader/issues/2885)

### Community
- [Hacker News Discussion (2025-07)](https://news.ycombinator.com/item?id=44810552)
- [AutoTradeLab Framework Comparison](https://autotradelab.com/blog/backtrader-vs-nautilusttrader-vs-vectorbt-vs-zipline-reloaded)
- [DeepWiki Analysis](https://deepwiki.com/nautechsystems/nautilus_trader)

### Rust Crates
- [nautilus-core](https://crates.io/crates/nautilus-core)
- [nautilus-indicators](https://docs.rs/nautilus-indicators)
- [nautilus-data](https://crates.io/crates/nautilus-data)
