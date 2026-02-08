# VBT → EDA 전환 가이드: 설계 원칙과 2026 최신 트렌드

**날짜:** 2026-02-07
**목적:** VectorBT에서 유의미한 전략 발견 후 EDA 백테스팅으로 전환할 때의 핵심 포인트, 설계 원칙, 최신 트렌드 정리

---

## 목차

1. [왜 전환이 필요한가](#1-왜-전환이-필요한가)
2. [전환 시 핵심 위험 요소](#2-전환-시-핵심-위험-요소)
3. [MC Coin Bot 현재 상태 진단](#3-mc-coin-bot-현재-상태-진단)
4. [오류 없는 전환을 위한 설계 원칙](#4-오류-없는-전환을-위한-설계-원칙)
5. [Parity Testing 전략](#5-parity-testing-전략)
6. [2026 업계 트렌드와 Best Practices](#6-2026-업계-트렌드와-best-practices)
7. [참고 프레임워크 비교](#7-참고-프레임워크-비교)
8. [MC Coin Bot 개선 로드맵](#8-mc-coin-bot-개선-로드맵)
9. [체크리스트](#9-체크리스트)
10. [참고 자료](#10-참고-자료)

---

## 1. 왜 전환이 필요한가

### 1.1 Hybrid Approach가 2026 표준

업계 합의: **"Vectorized로 탐색, Event-Driven으로 검증"** — 이 2단계 파이프라인이 2025-2026 퀀트 트레이딩의 표준 패턴이다.

```
Stage 1: VectorBT (탐색)          Stage 2: EDA (검증)           Stage 3: Live
┌─────────────────────┐    ┌─────────────────────────┐    ┌──────────────┐
│ 수천 개 파라미터 sweep │ → │ 유망 전략 bar-by-bar 검증  │ → │ 동일 코드 배포 │
│ Numba 가속           │    │ Slippage/Fees 모델링       │    │ Paper → Live  │
│ 100-1000x 빠름       │    │ Look-ahead bias 구조적 방지 │    │ 코드 변경 제로 │
└─────────────────────┘    └─────────────────────────┘    └──────────────┘
```

### 1.2 VBT와 EDA의 근본적 차이

| 관점 | VBT (Vectorized) | EDA (Event-Driven) |
|------|-------------------|-------------------|
| **데이터 처리** | 전체 DataFrame 한번에 | Bar 단위 순차 처리 |
| **체결 모델** | Close 가격 (lookahead 가능) | **Next-Open 가격** (lookahead 차단) |
| **포지션 사이징** | 전체 시계열 벡터화 계산 | Bar별 점진적 계산 |
| **리스크 관리** | 사후 적용 (weights에 규칙 적용) | **사전 검증** (주문 전 RM 통과) |
| **코드 재사용** | 라이브 별도 구현 필요 | **동일 코드로 Live 전환** |
| **비용 모델** | 고정 비율 차감 | 이벤트별 동적 적용 |
| **Multi-Asset** | `cash_sharing=True` 단순 설정 | Batch order + equity snapshot |

### 1.3 전환의 핵심 가치

VBT에서 찾은 전략을 EDA로 전환하는 것은 단순한 "포팅"이 아니다. **다음 3가지를 동시에 달성**하는 것이다:

1. **Bias 제거**: Look-ahead bias, survivorship bias 구조적 차단
2. **현실 반영**: 실제 체결 환경 시뮬레이션 (slippage, fees, latency)
3. **Live 준비**: 전환 없이 Paper → Canary → Live로 직행

---

## 2. 전환 시 핵심 위험 요소

### 2.1 시그널 충실도 (Signal Fidelity) — 가장 중요

**문제:** VBT와 EDA에서 동일한 전략이 다른 시그널을 생성할 수 있다.

```
VBT:  df["signal"] = df["close"].pct_change(20) > 0  ← 전체 시계열 한번에 계산
EDA:  매 bar마다 buffer에 누적 → strategy.run() → 마지막 행의 시그널만 추출
```

**MC Coin Bot의 실제 사례:**
- TSMOM: VBT 95건 vs EDA 3건 (32배 차이) — signal dedup 과도
- BB-RSI: VBT 117건 vs EDA 62건 — 가장 유사 (빈번한 방향 전환)

**근본 원인:**
1. StrategyEngine의 **시그널 중복제거(dedup)** 정책
2. VBT의 **vol-target 리밸런싱**이 EDA에서 누락
3. 체결 가격 차이 (Close vs Next-Open)

### 2.2 포지션 사이징 불일치

**VBT 방식:**
```python
# 매 bar마다 변동성 기반 목표 비중 벡터화 계산
target_weights = vol_target / realized_vol  # 전체 시계열
# rebalance_threshold 초과 시 거래 발생
```

**EDA 방식 (현재 문제점):**
```python
# SignalEvent 발행 시에만 포지션 사이징
# → TSMOM처럼 direction이 오래 유지되면 리밸런싱 안 됨
# → vol-target 기반 포지션 크기 재조정 누락
```

**해결 방향:** PM이 매 bar마다 독립적으로 vol-target 리밸런싱 수행 (Phase 4.5에서 부분 해결됨)

### 2.3 Equity 계산 오류

**흔한 실수:**
```python
# 잘못된 계산
total_equity = cash + notional + unrealized_pnl  # ❌ 이중 계산!

# 올바른 계산
total_equity = cash + long_notional - short_notional  # ✅
# notional(=size*price)에 이미 unrealized PnL 포함
```

### 2.4 이벤트 순서 보장 (Event Ordering)

```
BAR → Signal → Order → Risk Check → Fill → Balance Update → 다음 BAR
      ↑                                                        ↑
      └──────────── flush() 로 체인 완료 보장 ─────────────────┘
```

**MC Coin Bot의 교훈:** `await bus.flush()` 없이는 DataFeed가 모든 bar를 한번에 큐에 넣어서, 파생 이벤트가 마지막 bar 가격으로만 체결됨.

### 2.5 Multi-Asset 복잡도

| 이슈 | 원인 | 해결 |
|------|------|------|
| 동일 equity snapshot | 자산별 순차 처리 시 먼저 체결된 자산이 equity 변경 | Batch order processing |
| Threshold scaling | 8개 자산 × 5% threshold = 너무 빈번 | `effective_threshold = threshold × asset_weight` |
| Cash competition | 자산 간 현금 경합 | `cash_sharing=True` + 일괄 주문 |

---

## 3. MC Coin Bot 현재 상태 진단

### 3.1 아키텍처 정렬도

| 영역 | 현재 상태 | 업계 Best Practice | 평가 |
|------|----------|-------------------|:----:|
| Hybrid Backtest | VBT + EDA | Vectorized screening + EDA validation | **A** |
| Event Bus | asyncio.Queue, bounded, flush() | In-process async → Redis Streams 확장 | **A** |
| PM/RM/OMS 3단계 방어 | 구현 완료 | 업계 표준 패턴 | **A** |
| Idempotent OMS | client_order_id | UUID/deterministic hash | **A** |
| Look-ahead bias 방지 | Next-Open fill | 구조적 차단 | **A** |
| Parity Testing | SimpleMomentum PASS | 실전 전략까지 확장 필요 | **B** |
| Vol-target rebalancing | PM per-bar rebalancing (Phase 4.5) | 매 bar 독립 리밸런싱 | **B** |
| CandleAggregator | 1m→target TF 집계 | Intrabar SL/TS 지원 | **A** |
| Batch Order Processing | Multi-asset 일괄 주문 | 동일 equity snapshot 보장 | **A** |

### 3.2 알려진 Gap

```
┌─────────────────────────────────────────┐
│           실전 전략 Parity 미확보          │
│                                          │
│  TSMOM:   VBT +47% vs EDA -45% (FAIL)   │
│  Breakout: VBT +16% vs EDA +632% (FAIL)  │
│  Donchian: VBT +18% vs EDA +6% (WARN)    │
│  BB-RSI:  VBT -3% vs EDA -11% (PASS)     │
│                                          │
│  → 핵심 원인: vol-target 리밸런싱 로직     │
│    차이 + signal dedup 정책               │
└─────────────────────────────────────────┘
```

### 3.3 Phase 4.5 이후 개선 현황

Phase 4.5(PM Vol-Target Rebalancing)와 Phase 5-B(Batch Order Processing) 이후:
- PM이 매 bar `_evaluate_rebalance()` 호출로 포지션 크기 재조정
- Signal dedup 제거 → 매 bar SignalEvent 발행
- `_stopped_this_bar` guard로 stop-loss 후 재진입 방지
- Multi-asset batch mode에서 동일 equity snapshot 보장

---

## 4. 오류 없는 전환을 위한 설계 원칙

### 4.1 원칙 1: "Same Code, Different Mode"

```python
# 전략 코드는 VBT와 EDA에서 동일
class TSMOMStrategy(BaseStrategy):
    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        ...  # 이 코드가 VBT에서도, EDA에서도 그대로 실행

# 차이는 "실행 환경"에만 존재
runner = EDARunner(strategy=tsmom, data=data, config=config)
result = await runner.run()  # 백테스트
# 나중에: executor만 BacktestExecutor → LiveExecutor로 교체
```

### 4.2 원칙 2: "Bar-by-Bar 이벤트 체인 완결성"

매 bar마다 이벤트 체인이 **완전히 종료**된 후 다음 bar 처리:

```python
async def replay(self) -> None:
    for bar in self._bars:
        await self._bus.publish(bar_event)
        await self._bus.flush()  # ← 이 bar의 모든 파생 이벤트 처리 완료 보장
```

**위반 시 증상:**
- 모든 주문이 마지막 bar 가격으로 체결
- 중간 bar의 stop-loss가 작동하지 않음
- Equity curve가 마지막 bar에서만 급변

### 4.3 원칙 3: "PM 주도 리밸런싱"

시그널 변화뿐 아니라, **가격/변동성 변화에 의한 포지션 드리프트**도 PM이 감지:

```python
# StrategyEngine: 매 bar SignalEvent 발행 (dedup 없음)
# PM: signal 수신 → target weight 계산 → 현재 weight과 비교
#     → rebalance_threshold 초과 시에만 OrderRequest 발행

# 핵심: PM의 should_rebalance()가 거래 빈도를 제어
# StrategyEngine은 항상 현재 상태를 보고, PM이 필터링
```

### 4.4 원칙 4: "Deterministic Replay"

```
모든 이벤트 → JSONL 기록 → 동일 시퀀스 재생 → 동일 결과
```

- EventBus의 JSONL audit log 활용
- 장애 시나리오 재현 가능
- Regression test 자동화

### 4.5 원칙 5: "Fail-Safe by Default"

```
Signal → [PM: position sizing] → [RM: pre-trade validation] → [OMS: execution]
              ↓ stop-loss              ↓ leverage check            ↓ idempotency
              ↓ trailing stop          ↓ circuit breaker           ↓ fill simulation
```

3단계 중 어느 하나라도 거부하면 주문 미실행. **보수적 기본값**:
- `system_stop_loss=0.10` (10%)
- `max_leverage_cap=3.0` (3x)
- `rebalance_threshold=0.05` (5%)

### 4.6 원칙 6: "Multi-Asset Batch Atomicity"

```python
# 단일 자산: 즉시 실행
# 멀티 자산: 시그널 수집 → 동일 equity snapshot으로 일괄 주문
if len(asset_weights) > 1:
    self._batch_mode = True
    # flush_pending_signals() 호출 시 일괄 처리
```

---

## 5. Parity Testing 전략

### 5.1 계층별 검증

```
Layer 1: Unit Test (개별 컴포넌트)
    ├── StrategyEngine: bar 수집 → 시그널 추출 → SignalEvent 발행
    ├── PM: signal → target weight → order 생성
    ├── RM: leverage check, position limit, circuit breaker
    └── OMS: idempotent execution, fill generation

Layer 2: Parity Test (VBT vs EDA)
    ├── 통제 전략 (SimpleMomentum): 수익률 ±0.5pp, 거래 수 ±5%
    ├── 실전 전략 (TSMOM/BB-RSI/...): 수익률 부호 일치, 거래 수 0.5x~2.0x
    └── Multi-Asset: 자산별 수익률 방향 일치

Layer 3: Statistical Validation
    ├── IS/OOS Split
    ├── Walk-Forward Analysis (WFA)
    ├── Combinatorial Purged Cross-Validation (CPCV)
    ├── Deflated Sharpe Ratio (DSR)
    └── Probability of Backtest Overfitting (PBO)

Layer 4: Paper Trading Parity (다음 단계)
    ├── Shadow Mode: 실시간 시그널 vs 백테스트 시그널 비교
    └── Paper Mode: 시뮬레이션 PnL vs 백테스트 PnL 비교
```

### 5.2 Parity 허용 오차 기준

| 지표 | 통제 전략 | 실전 전략 | 근거 |
|------|:--------:|:--------:|------|
| **수익률 방향** | 일치 | 일치 | 부호가 다르면 전략 자체가 다른 것 |
| **수익률 크기** | ±1pp | 0.2x~5.0x | 체결 가격/타이밍 차이 반영 |
| **거래 수** | ±5% | 0.5x~2.0x | 리밸런싱 빈도 차이 허용 |
| **Sharpe** | ±0.3 | 부호 일치 | 체결 방식에 민감 |
| **MDD** | ±5pp | 방향 유사 | Stop-loss 타이밍 차이 |

### 5.3 차이가 발생하는 구조적 원인

| 원인 | VBT | EDA | 영향도 |
|------|-----|-----|:------:|
| 체결 가격 | Close | Next-Open | 중간 |
| 포지션 사이징 시점 | 전체 벡터 | Bar별 점진 | **높음** |
| Stop-loss 체크 | Bar 종료 시 | Intrabar 가능 | 중간 |
| Equity 기준 | 고정/기간별 | 실시간 mark-to-market | 높음 |
| 거래 비용 | 비율 차감 | 이벤트별 적용 | 낮음 |

> **핵심 통찰:** VBT와 EDA의 수치가 정확히 일치하는 것은 **불가능하며 바람직하지도 않다.** EDA가 더 현실적이기 때문에, "EDA 결과가 VBT보다 보수적"인 것은 정상이다. 반대로 EDA가 VBT보다 훨씬 낙관적이면 (Breakout +632% 사례) 버그를 의심해야 한다.

---

## 6. 2026 업계 트렌드와 Best Practices

### 6.1 Backtest → Live Pipeline: 3단계 표준

```
┌──────────────┐    ┌──────────────┐    ┌────────────────────┐
│   Stage 1    │    │   Stage 2    │    │     Stage 3        │
│  Vectorized  │ →  │  Event-Driven│ →  │  Paper → Canary    │
│  Screening   │    │  Validation  │    │  → Full Live       │
│              │    │              │    │                    │
│ VectorBT     │    │ Custom EDA   │    │ 동일 코드, Executor │
│ 수천 후보 탐색│    │ Parity 검증  │    │ 만 교체             │
└──────────────┘    └──────────────┘    └────────────────────┘
```

**MC Coin Bot은 이 패턴에 정확히 부합** — Phase 1-3(VBT) → Phase 4-5(EDA) → Phase 5-C/6(Live)

### 6.2 Event-Driven vs Vectorized: 현재 합의

> "For validating a strategy destined for production, the correctness and realism afforded by an event-driven architecture are non-negotiable."
> — [Interactive Brokers Quant News](https://www.interactivebrokers.com/campus/ibkr-quant-news/)

| 용도 | 추천 | 이유 |
|------|------|------|
| 파라미터 sweep (수천 조합) | VBT | 100-1000x 속도 우위 |
| 최종 전략 검증 | EDA | Lookahead bias 구조적 방지 |
| Live trading | EDA | 동일 코드 재사용 |
| 연구/프로토타이핑 | VBT | 빠른 실험 사이클 |

### 6.3 프레임워크 동향 (2025-2026)

#### Tier 1: Production-Grade

**NautilusTrader** — 2026년 가장 주목할 프레임워크
- Rust 코어 + Python API, nanosecond resolution
- **Backtest ↔ Live 코드 변경 제로** (동일 Strategy 클래스)
- Redis-backed state persistence, Docker 배포
- Multi-venue, multi-asset
- [GitHub](https://github.com/nautechsystems/nautilus_trader) | [Docs](https://nautilustrader.io/docs/latest/)

**VectorBT PRO** — 연구/탐색용 최강
- 1,000,000 orders in 70-100ms (Apple M1)
- VBT 1.2.0 (2025.10): tick-level resolution, slippage model (Binance 대비 0.3% 오차)
- Vectorized + event-driven callback 이중 모드
- [Features](https://vectorbt.pro/features/overview/)

#### Tier 2: Crypto-Specific

**Freqtrade** — 오픈소스 크립토 트레이딩 봇
- Docker Compose 배포, Telegram 통합
- strategy backtesting + hyperopt (parameter optimization)
- [Docker Quickstart](https://www.freqtrade.io/en/stable/docker_quickstart/)

**Hummingbot** — 시장 조성(Market Making) 특화
- CEX + DEX 통합 (Hyperliquid 포함)
- [Documentation](https://hummingbot.org/)

#### MC Coin Bot의 위치

MC Coin Bot의 자체 EDA는 NautilusTrader와 유사한 설계 패턴을 따르면서도, **프레임워크 전환 비용 없이 완전한 커스터마이징이 가능**한 장점이 있다. 규모가 커지면 NautilusTrader 참고 포인트(MessageBus, State Persistence)를 차용할 수 있다.

### 6.4 Crypto-Specific 고려사항

#### Funding Rate 모델링 (필수)

Perpetual futures의 funding rate는 8시간 주기로 long/short 간 이전되는 비용이다:
- `funding_payment = position_size × funding_rate`
- LONG + positive rate = 비용 지불 / SHORT + positive rate = 수익
- 장기 포지션일수록 funding 누적 영향 커짐

MC Coin Bot의 `CostModel`에 이미 포함되어 있으나, EDA 레벨에서 `FundingRateEvent`로 **이벤트화**하면 더 정밀한 시뮬레이션 가능.

#### DEX 급성장 (2026 트렌드)

- **Hyperliquid**: OI $9B+, 주문 확인 < 1초, [Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) 제공
- DEX perps 일일 거래량 $10B+ (DeFiLlama 기준)
- Delta-neutral funding rate arbitrage: CEX + DEX 양쪽에서 활발

#### Liquidation 모델링

2025-2026년 extreme events가 margin 메커니즘에 전례 없는 stress test 적용. Backtest에 **liquidation price 시뮬레이션** 추가 시 더 현실적:
```python
# Isolated margin: liq_price = entry * (1 - 1/leverage)
# Cross margin: 전체 계정 잔고 기반
```

### 6.5 Risk Management 패턴

#### Sequencer-Centric Design (2025 신규 트렌드)

24/7 크립토 시장 + 실시간 리스크 요구로, **Event Sequencer 중심 설계**가 sprawling microservices를 대체하는 추세:
- 모든 이벤트가 단일 sequencer를 통과 → 순서 보장 + 원자성
- MC Coin Bot의 `EventBus.flush()` 패턴이 이 원칙과 부합

#### Real-Time Volatility Alert

[Evinquo (2026.01)](https://www.globenewswire.com/news-release/2026/01/27/3226205/): 실시간 변동성 알림 + stress-testing이 프로덕션 리스크 관리의 새로운 표준으로 부상.

#### 3-Layer Defense (MC Coin Bot이 이미 구현)

```
PM (Position Sizing, SL, TS, Rebalance)
  → RM (Leverage Check, System SL, Circuit Breaker)
    → OMS (Idempotent Execution, Kill Switch)
```

### 6.6 OMS 멱등성 패턴

```python
# Client-generated unique ID 기반 멱등성
async def submit_order(order: OrderRequestEvent) -> FillEvent:
    # 1. client_order_id로 기존 주문 조회
    if order.client_order_id in self._processed_orders:
        return self._processed_orders[order.client_order_id]  # 멱등 반환

    # 2. 새 주문 실행
    fill = await self._executor.execute(order)

    # 3. 결과 저장
    self._processed_orders[order.client_order_id] = fill
    return fill
```

**Best Practice:** idempotency TTL을 최대 재시도 기간에 맞춤 (트레이딩: 24h).

### 6.7 AI/ML 통합 트렌드

#### Regime Detection (HMM 기반)

2025-2026년 가장 검증된 방법: **Hidden Markov Model로 시장 regime 탐지**

```
MarketData → RegimeDetector(HMM) → RegimeEvent → StrategyEngine
                                                     ↓
                                              regime에 따라:
                                              - Trending → TSMOM 활성화
                                              - Mean-reverting → BB-RSI 활성화
                                              - High-vol → position size 축소
```

- [HMM + RL for Portfolio Management (2025)](https://www.cloud-conf.net/datasec/2025/)
- [Multi-Model Ensemble-HMM Voting Framework](https://www.aimspress.com/article/id/69045d2fba35de34708adb5d)

#### LLM-Driven Strategy Discovery

- [QuantEvolve](https://arxiv.org/abs/2510.18569): Multi-agent LLM 프레임워크로 전략 탐색 자동화
- Quality-diversity optimization + hypothesis-driven 전략 생성
- 아직 실험 단계이지만, 2027년 이후 VBT 파라미터 sweep을 대체할 가능성

### 6.8 Fill Model 다양화 (NautilusTrader 참고)

NautilusTrader는 8가지 fill model을 제공하며, 이는 EDA 백테스트의 현실성을 높이는 핵심 요소이다:

| Model | 특징 | 적합 시나리오 |
|-------|------|-------------|
| **BestPriceFillModel** | 무제한 유동성, best price 체결 | 기본 백테스트 |
| **OneTickSlippageFillModel** | 1-tick slippage 고정 | 보수적 추정 |
| **ThreeTierFillModel** | 50/30/20 비율 3단계 분배 | 중형 주문 |
| **VolumeSensitiveFillModel** | 최근 거래량 기반 유동성 적응 | 대형 주문 |
| **LimitOrderPartialFillModel** | 가격 접촉당 최대 N계약 부분 체결 | 지정가 주문 |

**Bar Processing 순서:** 각 bar는 Open→High→Low→Close 4개 price point로 분해 (adaptive ordering으로 ~75-85% 정확도).

**Crypto-Specific Fill 수치:**
- ETH Maker/Taker fee: 0.02%/0.04%
- 평균 slippage: 0.05%
- 글로벌 주문 실패율: 1.354% (Binance/Bybit 실측, [Taylor & Francis 2025](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2515933))

**MC Coin Bot 현재:** `BacktestExecutor`는 Next-Open fill (BestPrice 패턴). 향후 `VolumeSensitiveFillModel` 추가 시 대형 주문의 시장 충격을 반영 가능.

### 6.9 인프라: Docker 배포

#### Docker Compose 패턴 (리테일/소규모)

```yaml
services:
  trading-bot:
    image: mc-coin-bot:latest
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file: .env

  redis:
    image: redis:7-alpine
    restart: always
```

#### 모니터링 핵심 지표

| 지표 | 중요도 | 설명 |
|------|:------:|------|
| P&L (실시간) | **P0** | 누적/일별 손익 |
| Order fill/rejection rate | **P0** | 체결률, 거부 원인 |
| Signal → Fill latency | P1 | 시그널 생성 ~ 체결 시간 |
| Position drift | P1 | target vs actual 포지션 차이 |
| WebSocket 상태 | P1 | 연결/재연결/끊김 |
| Circuit breaker 활성화 | **P0** | 발동 이벤트 즉시 알림 |

---

## 7. 참고 프레임워크 비교

### 7.1 아키텍처 비교

| 프레임워크 | Event Bus | Strategy | Portfolio | Execution | 언어 |
|-----------|-----------|----------|-----------|-----------|------|
| **MC Coin Bot** | asyncio.Queue (in-process) | BaseStrategy wrapper | PM + RM + OMS | BacktestExecutor / LiveExecutor | Python |
| **NautilusTrader** | MessageBus (in-process) | Strategy trait | Portfolio + Risk | ExecutionClient trait | Rust+Python |
| **Barter-rs** | Tokio channels | SignalGenerator trait | MarketUpdater + OrderGenerator | ExecutionClient | Rust |
| **QuantStart** | Event Queue (sync) | Strategy ABC | Portfolio Handler | ExecutionHandler ABC | Python |
| **Freqtrade** | Direct call (no bus) | IStrategy | Wallet | Exchange wrapper | Python |

### 7.2 MC Coin Bot vs NautilusTrader

| 관점 | MC Coin Bot | NautilusTrader |
|------|-------------|----------------|
| **Event Bus** | asyncio.Queue + flush() | MessageBus (Rust, zero-copy) |
| **State** | PM 내 dict-based | Redis-backed persistence |
| **Performance** | Python-native (적당) | Rust 코어 (초고속) |
| **유연성** | 완전한 커스터마이징 | 프레임워크 규칙 준수 필요 |
| **학습 비용** | 낮음 (자체 코드) | 높음 (프레임워크 학습) |
| **적합 시나리오** | 리테일/중소규모 | 기관급/HFT |

---

## 8. MC Coin Bot 개선 로드맵

### 8.1 단기 (P0 — Parity 확보)

| 작업 | 현재 상태 | 필요 작업 | 예상 영향 |
|------|----------|----------|----------|
| Signal dedup 제거 | **완료** (Phase 4.5) | 검증 | TSMOM 거래 수 정상화 |
| PM per-bar rebalancing | **완료** (Phase 4.5) | 실전 검증 | Vol-target 리밸런싱 복원 |
| Batch order processing | **완료** (Phase 5-B) | Multi-asset parity test | 8-asset EW 결과 검증 |
| 실전 전략 parity test | 부분 완료 | TSMOM/Breakout/BB-RSI 재검증 | Parity 확보 확인 |

### 8.2 중기 (P1 — Paper Trading 준비)

| 작업 | 설명 |
|------|------|
| Shadow Mode 구현 | 실시간 WebSocket + 시그널 로깅 (체결 없음) |
| LiveDataFeed | CCXT Pro WebSocket → BarEvent 변환 |
| FundingRateEvent | 8시간 주기 funding rate 이벤트화 |
| Discord 알림 | Signal/Fill/Error/Circuit Breaker 알림 |
| Docker Compose 배포 | healthcheck + restart policy |

### 8.3 장기 (P2 — 고도화)

| 작업 | 설명 |
|------|------|
| Regime Detection (HMM) | 시장 상태 감지 → 전략 선택 |
| Prometheus + Grafana | P&L, latency, fill rate 모니터링 |
| Paper → Canary 전환 | Binance testnet → 소액 실거래 |
| EventBus → Redis Streams | 멀티 프로세스 확장 (필요 시) |
| ML Alpha Signal | LLM/ML 기반 alpha factor 통합 |

---

## 9. 체크리스트

### 9.1 VBT → EDA 전환 전 확인

- [ ] 전략의 `generate_signals()` 코드가 VBT와 EDA에서 **동일**한가?
- [ ] StrategyEngine이 매 bar SignalEvent를 발행하는가? (dedup 없음)
- [ ] PM의 rebalance threshold가 VBT 설정과 일치하는가?
- [ ] 체결 가격 차이 (Close vs Next-Open) 를 이해하고 있는가?
- [ ] Equity 계산이 `cash + long_notional - short_notional` 인가?
- [ ] `await bus.flush()` 로 bar별 이벤트 체인 완결을 보장하는가?

### 9.2 Parity Test 전 확인

- [ ] 동일 데이터 (Silver layer, 동일 기간)를 사용하는가?
- [ ] 동일 PortfolioManagerConfig를 사용하는가?
- [ ] 동일 CostModel을 사용하는가?
- [ ] 초기 자본이 동일한가?
- [ ] Multi-asset의 경우 asset_weights가 동일한가?

### 9.3 Paper Trading 전환 전 확인

- [ ] 통제 전략 parity: 수익률 ±0.5pp, 거래 수 ±5%
- [ ] 실전 전략 parity: 수익률 부호 일치, 거래 수 0.5x~2.0x
- [ ] Stop-loss / Trailing stop 작동 확인
- [ ] Circuit breaker 작동 확인
- [ ] 72시간 연속 운영 시 메모리 누수 없음
- [ ] WebSocket 재연결 자동 복구

---

## 10. 참고 자료

### 아키텍처 & 설계

- [QuantStart: Event-Driven Backtesting with Python Part I-VIII](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/)
- [NautilusTrader Documentation](https://nautilustrader.io/docs/latest/concepts/overview/)
- [Barter-rs: Rust Event-Driven Trading Framework](https://github.com/barter-rs/barter-rs)
- [Advanced Event-Driven Architectures for Ultra-Low-Latency Trading (ResearchGate)](https://www.researchgate.net/publication/390338700)

### Vectorized vs Event-Driven

- [IBKR: A Practical Breakdown of Vector-Based vs. Event-Based Backtesting](https://www.interactivebrokers.com/campus/ibkr-quant-news/a-practical-breakdown-of-vector-based-vs-event-based-backtesting/)
- [Why Event-Driven Backtesting Is the Only Path to Profitable Real-Time Trading](https://medium.com/@anon.quant/why-event-driven-backtesting-is-the-only-path-to-profitable-real-time-trading-61e643286036)
- [Battle-Tested Backtesters: VectorBT, Zipline, Backtrader](https://medium.com/@trading.dude/battle-tested-backtesters-comparing-vectorbt-zipline-and-backtrader)

### Fill Model & Execution

- [NautilusTrader Backtesting (Fill Models)](https://nautilustrader.io/docs/latest/concepts/backtesting/)
- [Talos Market Impact Model (TMI)](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs)
- [Latency and Slippage on Bybit/Binance (Taylor & Francis 2025)](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2515933)
- [QuantConnect Fill Models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts)

### Walk-Forward & Overfitting Prevention

- [Walk-Forward Analysis Deep Dive (IBKR)](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)
- [CPCV Walk-Forward Validation (arXiv 2512.12924)](https://arxiv.org/html/2512.12924v1)
- [Backtest Overfitting Comparison (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- [Walk-Forward Optimization (QuantInsti)](https://blog.quantinsti.com/walk-forward-optimization-introduction/)

### Risk Management & OMS

- [From Silos to Sequencers: Core Trading Architectures Being Rewritten](https://a-teaminsight.com/blog/from-silos-to-sequencers-why-core-trading-architectures-are-being-rewritten-for-24-7-markets/)
- [Real-Time Financial Risk Management (Confluent)](https://www.confluent.io/blog/real-time-financial-risk-management/)
- [Idempotency and Ordering in Event-Driven Systems (CockroachDB)](https://www.cockroachlabs.com/blog/idempotency-and-ordering-in-event-driven-systems/)
- [Idempotency's Role in Financial Services](https://www.cockroachlabs.com/blog/idempotency-in-finance/)

### Async Event Bus Patterns

- [Building Scalable Async Event System (AlgoMart, Jan 2026)](https://medium.com/algomart/building-a-scalable-asynchronous-event-system-in-python-79e3258455df)
- [bubus: Production-Ready Event Bus (GitHub)](https://github.com/browser-use/bubus)
- [Architecture Patterns with Python - Events & Message Bus (O'Reilly)](https://www.oreilly.com/library/view/architecture-patterns-with/9781492052197/ch08.html)

### Signal Fidelity & Data Consistency

- [From Backtesting to Live: Consistent Indicator Data (Medium)](https://medium.com/@mariamhov/from-backtesting-to-live-trading-how-consistent-indicator-data-improves-strategy-performance-7639949bb791)
- [From Backtest to Live with VectorBT (Medium)](https://medium.com/@samuel.tinnerholm/from-backtest-to-live-going-live-with-vectorbt-in-2025-step-by-step-guide-681ff5e3376e)

### Crypto-Specific

- [2025 Crypto Derivatives Market Report (Bitget)](https://www.bitget.com/academy/2025-crypto-derivatives-market-cex-defi-cme-liquidations-2026-outlook)
- [Designing Funding Rates for Perpetual Futures (arXiv)](https://arxiv.org/html/2506.08573v1)
- [Hyperliquid Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk)
- [CoinAPI Historical Data for Perpetual Futures](https://www.coinapi.io/blog/historical-data-for-perpetual-futures)

### ML & Regime Detection

- [Generating Alpha: Hybrid AI-Driven Trading (ComSIA 2026)](https://arxiv.org/html/2601.19504v1)
- [QuantEvolve: Multi-Agent Evolutionary Framework](https://arxiv.org/abs/2510.18569)
- [Market Regime Detection with HMM (QuantInsti)](https://blog.quantinsti.com/regime-adaptive-trading-python/)
- [Multi-Model Ensemble-HMM Voting Framework](https://www.aimspress.com/article/id/69045d2fba35de34708adb5d)

### Infrastructure

- [Freqtrade Docker Quickstart](https://www.freqtrade.io/en/stable/docker_quickstart/)
- [Docker Compose Restart Policies and Healthchecks](https://blog.justanotheruptime.com/posts/2025_07_07_docker_compose_restart_policies_and_healthchecks/)
- [Kubernetes Observability Trends 2026](https://www.usdsi.org/data-science-insights/kubernetes-observability-and-monitoring-trends-in-2026)

### 프레임워크

- [NautilusTrader GitHub](https://github.com/nautechsystems/nautilus_trader)
- [VectorBT PRO](https://vectorbt.pro/)
- [Freqtrade](https://www.freqtrade.io/)
- [Hummingbot](https://hummingbot.org/)
- [Top 21 Python Trading Tools (2026)](https://analyzingalpha.com/python-trading-tools)

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-07 | 초기 작성 — VBT→EDA 전환 가이드, 2026 트렌드, 설계 원칙, 체크리스트 |
