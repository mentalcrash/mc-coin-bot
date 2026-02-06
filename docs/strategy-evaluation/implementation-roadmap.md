# Implementation Roadmap

전략 확정(8-asset EW TSMOM, Sharpe 2.41) 이후 실거래까지의 구현 로드맵.

> **현재 위치:** Phase 1~4 완료 → Phase 5 진입 예정

---

## Phase 개요

```
Phase 1          Phase 2              Phase 3           Phase 4          Phase 5          Phase 6
단일에셋          멀티에셋             고급 검증          EDA 시스템       Dry Run          Live
백테스트          백테스트 확장         IS/OOS/WFA/CPCV   이벤트 기반       Paper Trading     실거래
✅ 완료           ✅ 완료              ✅ 완료            ✅ 완료          ← 현재           예정
```

---

## Phase 1: 단일에셋 백테스트 (완료)

- VectorBT 기반 백테스트 엔진
- TSMOM/Breakout/Donchian/BB-RSI 4개 전략
- 파라미터 스윕, QuantStats 리포트
- Numba 최적화 (stop-loss, trailing-stop, rebalance)

---

## Phase 2: 멀티에셋 백테스트 확장 (완료)

> **상태:** ✅ 완료 (2026-02-06)

### 2.1 구현 결과

단일 심볼 → 8-asset Equal-Weight 포트폴리오를 VectorBT `cash_sharing=True`로 통합 백테스트.

#### 구현된 컴포넌트

| 컴포넌트 | 파일 | 설명 |
|---------|------|------|
| `MultiSymbolData` | `src/data/market_data.py` | 멀티심볼 데이터 컨테이너 (dataclass) |
| `MarketDataService.get_multi()` | `src/data/service.py` | Silver 데이터 일괄 로드 |
| `MultiAssetBacktestRequest` | `src/backtest/request.py` | 멀티에셋 백테스트 요청 DTO |
| `BacktestEngine.run_multi()` | `src/backtest/engine.py` | 멀티에셋 백테스트 실행 |
| `BacktestEngine.run_multi_with_returns()` | `src/backtest/engine.py` | + 수익률 시리즈 반환 |
| `BacktestEngine.run_multi_validated()` | `src/backtest/engine.py` | + 검증 결합 실행 |
| `MultiAssetConfig` | `src/models/backtest.py` | 결과 설정 모델 |
| `MultiAssetBacktestResult` | `src/models/backtest.py` | 결과 모델 (포트폴리오 + 심볼별) |
| `_apply_pm_rules_to_weights()` | `src/backtest/engine.py` | PM 규칙 모듈 레벨 함수 추출 |
| CLI `run-multi` | `src/cli/backtest.py` | 멀티에셋 백테스트 커맨드 |

#### 핵심 설계 결정

1. **전략 심볼별 독립 실행**: `strategy.run(df)` 인터페이스 변경 없음
2. **VectorBT `cash_sharing=True` + `group_by=True`**: 진정한 포트폴리오 시뮬레이션
3. **`from_orders` + `targetpercent`**: strength × asset_weight로 자본 배분
4. **PM 규칙 재사용**: 기존 Numba stop-loss/trailing-stop/rebalance를 심볼별 적용
5. **CAGR 수동 계산**: VBT `cash_sharing` 모드에서 `Annualized Return [%]` 미제공 → `_compute_cagr()` 추가

### 2.2 8-Asset EW 통합 백테스트 결과 (2020-2025)

| 지표 | Multi-Asset Portfolio | 개별 평균 | 개선 |
|------|:---:|:---:|:---:|
| **CAGR** | +57.95% | +36.05% | +61% |
| **Sharpe** | 1.57 | 1.08 | +45% |
| **MDD** | 19.43% | 39.8% | **-51% 감소** |
| **Sortino** | 2.98 | - | - |
| **Calmar** | 2.98 | - | - |
| **Profit Factor** | 2.00 | - | - |
| **Total Trades** | 1,123 | - | - |

> **참고:** 개별 스윕(EW 평균 방식, Sharpe 2.41)과 통합 백테스트(VBT cash_sharing, Sharpe 1.57)의 Sharpe 차이는 정상.
> 스윕은 각 에셋 독립 시뮬레이션 후 수익률 평균이고, 통합은 실제 자본 공유 시뮬레이션이므로
> 거래비용/슬리피지/자본 제약이 반영됨. 방향성(분산 효과, MDD 감소)은 일치.

#### 심볼별 기여도

| Symbol | Individual CAGR | Individual Sharpe | Contribution |
|--------|:---:|:---:|:---:|
| BTC/USDT | +51.8% | 0.84 | +46.0% |
| ETH/USDT | +68.2% | 0.82 | +65.3% |
| BNB/USDT | +99.3% | 1.15 | +79.1% |
| SOL/USDT | +87.6% | 0.77 | +95.1% |
| DOGE/USDT | +98.2% | 0.48 | +134.8% |
| LINK/USDT | +37.8% | 0.35 | +67.7% |
| ADA/USDT | +47.9% | 0.47 | +67.9% |
| AVAX/USDT | +21.5% | 0.19 | +61.4% |

### 2.3 테스트 현황

- `tests/backtest/test_multi_asset.py`: 12 tests (MultiSymbolData, Request, Config, Result)
- 전체: 191/191 passed, ruff 0 errors, pyright 0 errors

---

## Phase 3: 고급 검증 — 과적합 방지 (완료)

> **상태:** ✅ 완료 (2026-02-06)

### 3.1 구현된 검증 체계

3단계 Tiered Validation 시스템 구현. 단일에셋/멀티에셋 모두 지원.

#### 구현된 컴포넌트

| 컴포넌트 | 파일 | 설명 |
|---------|------|------|
| `split_multi_is_oos()` | `src/backtest/validation/splitters.py` | 멀티에셋 IS/OOS 분할 |
| `split_multi_walk_forward()` | `src/backtest/validation/splitters.py` | 멀티에셋 Walk-Forward 분할 |
| `split_multi_cpcv()` | `src/backtest/validation/splitters.py` | 멀티에셋 CPCV 분할 (purge+embargo) |
| `deflated_sharpe_ratio()` | `src/backtest/validation/deflated_sharpe.py` | Bailey & Lopez de Prado DSR |
| `probabilistic_sharpe_ratio()` | `src/backtest/validation/deflated_sharpe.py` | PSR 계산 |
| `expected_max_sharpe()` | `src/backtest/validation/deflated_sharpe.py` | 다중 테스트 기대 최대 Sharpe |
| `calculate_pbo()` | `src/backtest/validation/pbo.py` | PBO (순위 기반) |
| `calculate_pbo_logit()` | `src/backtest/validation/pbo.py` | PBO (로짓 기반) |
| `TieredValidator.validate_multi()` | `src/backtest/validation/validator.py` | 멀티에셋 3단계 검증 |
| `generate_validation_report()` | `src/backtest/validation/report.py` | 검증 결과 텍스트 리포트 |
| CLI `validate` | `src/cli/backtest.py` | 검증 커맨드 |

### 3.2 검증 3단계

#### QUICK: IS/OOS Split

```
|← ───── In-Sample (70%) ─────→|← ── OOS (30%) ──→|
|  2020-01          2024-04    |  2024-04   2025-12 |
```

- `split_multi_is_oos()`: 모든 심볼에 동일 시간 경계 적용
- 판정: OOS Sharpe > 0.5, 성과 열화 < 30%

#### MILESTONE: Walk-Forward Analysis

```
Fold 1: |───Train───|─Test─|
Fold 2:     |───Train───|─Test─|
Fold 3:         |───Train───|─Test─|
```

- `split_multi_walk_forward()`: expanding/rolling window 지원
- 판정: OOS Sharpe > 1.0, Consistency > 70%, Sharpe Decay < 30%

#### FINAL: CPCV + DSR + PBO + Monte Carlo

```
N개 블록 → C(N,k) 조합으로 독립적 경로 생성
각 경로에서 purge + embargo 적용 후 평가
```

- `split_multi_cpcv()`: 조합론적 교차 검증
- `deflated_sharpe_ratio()`: 다중 테스트 보정 (n_trials 기반)
- `calculate_pbo()`: 과적합 확률 추정
- 판정: PBO < 0.30, DSR > 0.5

### 3.3 검증 판정 기준 (상수)

| 기준 | 상수 | 통과 | 주의 | 실패 |
|------|------|------|------|------|
| IS/OOS 성과 열화 | `MULTI_QUICK_MAX_DEGRADATION` | < 30% | 30~50% | > 50% |
| WFA OOS Sharpe | `MULTI_WFA_MIN_OOS_SHARPE` | > 1.0 | 0.5~1.0 | < 0.5 |
| WFA Consistency | `MULTI_WFA_MIN_CONSISTENCY` | > 70% | 50~70% | < 50% |
| CPCV PBO | `MULTI_CPCV_MAX_PBO` | < 0.30 | 0.30~0.50 | > 0.50 |
| Deflated Sharpe | `MULTI_DEFLATED_SHARPE_MIN` | > 0.5 | - | < 0.5 |

### 3.4 테스트 현황

- `tests/backtest/validation/test_deflated_sharpe.py`: 14 tests (DSR/PSR/E[max(SR)])
- `tests/backtest/validation/test_pbo.py`: 9 tests (PBO rank/logit)
- `tests/backtest/validation/test_multi_splitters.py`: 10 tests (IS/OOS, WF, CPCV 분할)
- 전체: 191/191 passed, ruff 0 errors, pyright 0 errors

### 3.5 CLI 사용법

```bash
# QUICK 검증 (IS/OOS)
uv run python -m src.cli.backtest validate -m quick

# MILESTONE 검증 (Walk-Forward)
uv run python -m src.cli.backtest validate -m milestone --symbols BTC/USDT,ETH/USDT

# FINAL 검증 (CPCV + DSR + PBO)
uv run python -m src.cli.backtest validate -m final -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

---

## Phase 4: EDA 시스템 (이벤트 기반 아키텍처) — ✅ 완료

> **상태:** ✅ 완료 (2026-02-06)

### 4.1 왜 이벤트 기반인가

벡터화 백테스트는 전체 데이터를 한 번에 처리 — 빠르지만 실거래와 구조가 다름.
EDA 시스템은 **실거래와 동일한 코드 경로**로 백테스트를 실행하여 backtest-live parity를 보장.

```
                    Vectorized Backtest          EDA Backtest
─────────────────────────────────────────────────────────────
데이터              전체 DataFrame 한번에          바(bar) 단위 이벤트
시그널 생성          전체 Series 한번에            각 바마다 개별 생성
주문 체결           VectorBT 내부 시뮬레이션       OMS 시뮬레이터
리스크 체크          사후 분석                     실시간 체크 (사전)
코드 재사용          별도 라이브 구현 필요          라이브와 동일 코드
속도               매우 빠름 (Numba)              느림 (이벤트 오버헤드)
용도               파라미터 탐색, 빠른 실험        최종 검증, 라이브 전환
```

### 4.2 구현 결과

#### 구현된 컴포넌트

| 컴포넌트 | 파일 | 설명 |
|---------|------|------|
| **이벤트 모델** | `src/core/events.py` | Flat Pydantic frozen models, 11개 이벤트 타입, `AnyEvent` union |
| **EventBus** | `src/core/event_bus.py` | async Queue(bounded), backpressure, JSONL audit, handler error isolation |
| **EDA Config** | `src/models/eda.py` | `ExecutionMode(StrEnum)`, `EDAConfig`, `EDABacktestConfig` |
| **HistoricalDataFeed** | `src/eda/data_feed.py` | Silver 데이터 → BarEvent 순차 발행, 멀티심볼 correlation_id 동기화 |
| **StrategyEngine** | `src/eda/strategy_engine.py` | BaseStrategy 래퍼 (Adapter), bar-by-bar, 시그널 dedup |
| **EDAPortfolioManager** | `src/eda/portfolio_manager.py` | Signal→Order, Fill→Position, mark-to-market, SL/TS, equity 계산 |
| **EDARiskManager** | `src/eda/risk_manager.py` | Pre-trade validation, system stop-loss, CircuitBreaker |
| **OMS** | `src/eda/oms.py` | 멱등성(client_order_id), Executor 라우팅, CB 전량 청산 |
| **BacktestExecutor** | `src/eda/executors.py` | Open price fill, CostModel fee |
| **ShadowExecutor** | `src/eda/executors.py` | 로깅만 수행 (Phase 5용 스텁) |
| **AnalyticsEngine** | `src/eda/analytics.py` | Equity curve, TradeRecord, PerformanceMetrics |
| **EDARunner** | `src/eda/runner.py` | 컴포넌트 조립, asyncio 오케스트레이션 |
| CLI `eda run` | `src/cli/eda.py` | 단일 심볼 EDA 백테스트 커맨드 |

#### 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                    EventBus                          │
│  publish(event) → async Queue → dispatch handlers    │
└──────┬──────┬──────┬──────┬──────┬──────┬───────────┘
       │      │      │      │      │      │
  ┌────▼──┐ ┌─▼───┐ ┌▼────┐ ┌▼───┐ ┌▼───┐ ┌▼────────┐
  │Data   │ │Strat │ │ PM  │ │ RM │ │OMS │ │Analytics│
  │Feed   │ │Engine│ │     │ │    │ │    │ │Engine   │
  └───────┘ └──────┘ └─────┘ └────┘ └────┘ └─────────┘
```

#### 이벤트 흐름

```
BarEvent → StrategyEngine → SignalEvent → PM → OrderRequestEvent
   → RM (validated=True) → OMS → Executor → FillEvent
   → PM (position update) → BalanceUpdateEvent → Analytics
```

#### 핵심 설계 결정

1. **Flat Pydantic models**: `BaseEvent` 상속 대신 독립 frozen models — pyright Literal override 이슈 회피
2. **`type AnyEvent = BarEvent | SignalEvent | ...`** union type으로 dispatch
3. **Adapter 패턴**: 기존 `BaseStrategy.run()` 인터페이스 변경 없이 bar-by-bar 래핑
4. **Backpressure 정책**: BAR/HEARTBEAT → stale 드롭 / SIGNAL/FILL → 절대 드롭 금지
5. **멱등성**: client_order_id set으로 중복 주문 방지
6. **Correlation ID chain**: Bar → Signal → Order → Fill 인과관계 추적
7. **Executor Protocol**: BacktestExecutor / ShadowExecutor / (향후 PaperExecutor, LiveExecutor) 교체 가능
8. **Equity 정확성**: `total_equity = cash + long_notional - short_notional` (이중 계산 방지)
9. **Position Stop-Loss**: Intrabar (high/low) 또는 close 기반, `entry * (1 ± stop_loss_pct)` 비교
10. **Trailing Stop**: ATR(14) SMA 기반, peak/trough 추적 후 `close < peak - atr * multiplier` 판정
11. **매 Bar BalanceUpdateEvent**: RM의 실시간 drawdown 추적 지원

### 4.3 Vectorized vs EDA 백테스트 병행 전략

둘 다 유지. 용도가 다름:

| 용도 | 엔진 | 이유 |
|------|------|------|
| 파라미터 탐색/스윕 | Vectorized (VectorBT) | 속도 (수백~수천 배 빠름) |
| IS/OOS, WFA, CPCV | Vectorized (VectorBT) | 대량 반복 필요 |
| 최종 검증 | EDA | 실거래와 동일 코드 경로 |
| Paper/Shadow/Live | EDA | 실시간 이벤트 처리 |

### 4.4 VBT vs EDA Parity 검증

동일 데이터/전략/설정으로 VBT `BacktestEngine`과 `EDARunner`를 비교한 Parity Test 결과:

| 검증 항목 | 결과 |
|-----------|------|
| 양쪽 모두 결과 생성 | ✅ 통과 |
| 수익률 부호(양/음) 일치 | ✅ 통과 |
| Warmup 미달 시 거래 0 | ✅ 통과 |

#### Parity 수치 비교 (BTC/USDT, 2024-01 ~ 2025-12)

| 지표 | VBT | EDA | 비고 |
|------|:---:|:---:|------|
| **Total Return** | 2.77% | 2.72% | 근접 (차이 0.05pp) |
| **Sharpe Ratio** | 0.37 | 0.49 | 체결 방식 차이 |
| **MDD** | 12.32% | 12.34% | 거의 동일 |
| **Total Trades** | 21 | 20 | 1건 차이 |

> **참고:** 벡터화(VBT)와 이벤트 기반(EDA)은 체결 방식이 다르므로 수치가 정확히 일치하지는 않음.
> VBT는 시그널 시점의 close 기준, EDA는 다음 바의 open 기준으로 체결.
> 핵심은 **방향성(수익/손실 부호)과 규모(order of magnitude)**가 일치하는 것.

#### Equity 계산 버그 수정 (Sharpe 3.0 → 0.49)

초기 EDA에서 Sharpe 3.03이라는 비현실적 수치가 나왔으며, 근본 원인은 **equity 이중 계산 버그**였음:
- 버그: `total_equity = cash + notional + unrealized` — notional에 이미 unrealized 포함
- 수정: `total_equity = cash + long_notional - short_notional`
- 영향: equity 과대평가 → leverage 과소평가 → 과도한 포지션 → Sharpe 부풀림

### 4.5 테스트 현황

| 테스트 파일 | 테스트 수 | 설명 |
|------------|:---:|------|
| `tests/core/test_events.py` | 17 | 이벤트 생성, frozen, JSON 직렬화 |
| `tests/core/test_event_bus.py` | 12 | subscribe, publish, backpressure, JSONL |
| `tests/models/test_eda.py` | 5 | EDAConfig, ExecutionMode |
| `tests/eda/test_data_feed.py` | 8 | 발행 수, 필드 정확성, 멀티심볼 동기화 |
| `tests/eda/test_strategy_engine.py` | 8 | warmup, dedup, 멀티심볼 독립 버퍼 |
| `tests/eda/test_portfolio_manager.py` | 26 | Signal→Order, Fill→Position, equity, SL/TS, balance update |
| `tests/eda/test_risk_manager.py` | 10 | validation, circuit breaker, peak equity |
| `tests/eda/test_oms.py` | 5 | validated order, 멱등성, CB close |
| `tests/eda/test_executors.py` | 6 | open price fill, fee, multi symbol |
| `tests/eda/test_analytics.py` | 8 | equity curve, trade, metrics |
| `tests/eda/test_runner.py` | 4 | e2e 단일/멀티, analytics, short data |
| `tests/eda/test_parity.py` | 8 | VBT vs EDA parity (수익률/부호/거래수/warmup) |
| **합계** | **126** | ruff 0, pyright 0 |

- 전체 프로젝트: 322/322 passed

### 4.6 CLI 사용법

```bash
# EDA 백테스트 실행
python main.py eda run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31

# 옵션 지정
python main.py eda run tsmom BTC/USDT --capital 50000 --leverage 2.0 --timeframe 1d
```

---

## Phase 5: Dry Run (Paper Trading)

### 5.1 3단계 실거래 전 검증

```
Stage 1: Shadow Mode (2~4주)
  └─ 실시간 데이터 수신 + 시그널 생성 + 로깅 (주문 X)
  └─ 목적: 시그널이 백테스트와 일관되는지 확인

Stage 2: Paper Trading (1~3개월)
  └─ Binance Testnet으로 가상 주문 실행
  └─ 목적: OMS/PM/RM 파이프라인 end-to-end 검증

Stage 3: Canary Live (1~2개월)
  └─ 최소 자본(포트폴리오의 5~10%)으로 실거래
  └─ 목적: 실제 슬리피지, 체결 지연, API 안정성 확인
```

### 5.2 Shadow Mode 상세

```python
class ShadowExecutor:
    """주문을 보내지 않고, 시장가로 체결되었을 것이라 가정하여 기록"""

    async def submit(self, order: OrderRequestEvent) -> FillEvent:
        # 실제 시장 데이터의 현재가로 가상 체결
        current_price = await self.market_data.get_price(order.symbol)
        slippage = self._estimate_slippage(order)

        fill = FillEvent(
            symbol=order.symbol,
            fill_price=current_price + slippage,
            fill_quantity=order.quantity,
            commission=self._calc_commission(order),
            is_simulated=True,
        )

        # 로깅 — 나중에 실제 시장과 비교
        self.shadow_log.record(order, fill)
        return fill
```

### 5.3 Paper Trading 체크리스트

| 체크 항목 | 기준 |
|----------|------|
| WebSocket 연결 안정성 | 24시간 무중단 (자동 재연결) |
| 시그널 일관성 | Shadow vs 백테스트 시그널 일치율 > 95% |
| 주문 실행 성공률 | > 99% (Testnet 기준) |
| 리밸런싱 정확도 | 목표 비중 대비 오차 < 2% |
| 리스크 체크 동작 | max_leverage, system_stop_loss 정상 트리거 |
| 에러 핸들링 | API 오류, 타임아웃, rate limit 정상 처리 |
| 알림 시스템 | Discord 알림 정상 수신 |
| 로그 완결성 | 모든 이벤트가 감사 로그에 기록 |

### 5.4 작업 목록

| # | 작업 | 예상 난이도 |
|---|------|-----------|
| 1 | `ShadowExecutor` 구현 | 중간 |
| 2 | `PaperExecutor` — Binance Testnet 연동 | 중간 |
| 3 | WebSocket 실시간 데이터 피드 (`ccxt.pro`) | 높음 |
| 4 | 자동 재연결 + heartbeat 모니터링 | 중간 |
| 5 | Shadow vs Backtest 시그널 비교 도구 | 중간 |
| 6 | Paper Trading 성과 리포트 생성 | 낮음 |
| 7 | Discord 알림 통합 (주문, 에러, 일간 리포트) | 낮음 |

---

## Phase 6: Live Trading

### 6.1 Go-Live 기준

| 기준 | 임계값 |
|------|-------|
| Paper Trading 기간 | 최소 1개월 |
| Paper Sharpe | > 0 (시장 상황에 따라 유연하게) |
| 시스템 가동률 | > 99.5% |
| 시그널 일관성 | Shadow vs Backtest > 90% |
| 치명적 버그 | 0건 |

### 6.2 Canary Deployment 전략

```
Week 1-2:  전체 자본의 5%로 실거래 (8에셋 × 최소 단위)
Week 3-4:  문제 없으면 10%로 증액
Month 2:   20%로 증액
Month 3+:  50% → 100% 점진적 확대
```

### 6.3 운영 안전장치

```python
# 시스템 레벨 안전장치
class LiveSafetyConfig(BaseModel):
    max_daily_loss_pct: float = 0.05     # 일 최대 손실 5% → 전 포지션 청산
    max_drawdown_pct: float = 0.15       # MDD 15% → 시스템 중단
    max_order_size_usd: Decimal           # 단일 주문 최대 금액
    max_open_positions: int = 8           # 최대 동시 포지션 수
    kill_switch: bool = False             # 긴급 중단 스위치
    allowed_symbols: list[str]            # 화이트리스트
```

### 6.4 작업 목록

| # | 작업 | 예상 난이도 |
|---|------|-----------|
| 1 | `LiveExecutor` — 실거래 주문 실행 | 높음 |
| 2 | 안전장치 시스템 (`LiveSafetyConfig`) | 높음 |
| 3 | Kill switch (Discord 명령 또는 API) | 중간 |
| 4 | Canary 자본 관리 로직 | 중간 |
| 5 | 장애 복구 — 미체결 주문 정리, 포지션 동기화 | 높음 |

---

## Phase 7: 모니터링 & 분석

### 7.1 모니터링 스택

```
┌─────────────────────────────────────────────┐
│              Streamlit Dashboard              │
│  전략 성과 분석, PnL 곡선, 파라미터 탐색       │
│  (개발/리서치 단계부터 사용)                    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Prometheus + Grafana               │
│  시스템 헬스, 주문 latency, API 상태           │
│  (Paper Trading 단계부터 사용)                 │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Discord Alerts                   │
│  주문 체결, 에러, 일간 리포트, Kill Switch      │
│  (Shadow 단계부터 사용)                        │
└─────────────────────────────────────────────┘
```

### 7.2 메트릭 체계

#### A. 전략 성과 메트릭 (Streamlit)

| 메트릭 | 갱신 주기 | 용도 |
|--------|---------|------|
| 일간/주간/월간 PnL | 일봉 마감 시 | 수익성 추적 |
| Rolling Sharpe (30d, 90d) | 일간 | 성과 안정성 |
| Current Drawdown | 실시간 | 리스크 모니터링 |
| 심볼별 수익 기여도 | 일간 | 포트폴리오 분해 |
| 백테스트 vs 실거래 괴리 | 주간 | 전략 유효성 확인 |
| 시그널 빈도 | 일간 | 전략 행동 변화 감지 |

#### B. 시스템 운영 메트릭 (Grafana)

| 메트릭 | 알림 임계값 | 용도 |
|--------|-----------|------|
| WebSocket 연결 상태 | 연결 끊김 > 30초 | 데이터 피드 안정성 |
| 주문 응답 시간 (p50, p99) | p99 > 5초 | 체결 품질 |
| API Rate Limit 사용률 | > 80% | 요청 최적화 |
| 프로세스 메모리 | > 2GB | 메모리 누수 감지 |
| 이벤트 처리 지연 | > 1초 | 파이프라인 병목 |

#### C. Discord 알림 체계

| 이벤트 | 알림 레벨 | 내용 |
|--------|---------|------|
| 주문 체결 | INFO | 심볼, 방향, 수량, 가격 |
| 리밸런싱 | INFO | 변경된 포지션 목록 |
| 일간 리포트 | INFO | PnL, 포지션, Sharpe |
| 시스템 에러 | WARN | 에러 메시지, 스택트레이스 |
| 안전장치 트리거 | CRITICAL | max_loss, kill_switch |
| 연결 끊김 | CRITICAL | WebSocket, API 장애 |

### 7.3 작업 목록

| # | 작업 | 도입 시점 |
|---|------|---------|
| 1 | Streamlit 대시보드 — 백테스트 결과 시각화 | Phase 2부터 |
| 2 | Streamlit — 멀티에셋 포트폴리오 분석 뷰 | Phase 2 |
| 3 | Prometheus 메트릭 노출 (`prometheus_client`) | Phase 5 |
| 4 | Grafana 대시보드 템플릿 | Phase 5 |
| 5 | Discord 알림 시스템 확장 | Phase 5 |
| 6 | 백테스트 vs 실거래 괴리 분석 도구 | Phase 6 |
| 7 | 일간/주간 성과 자동 리포트 | Phase 6 |

---

## 기술 스택 결정 (2026년 기준 트렌드 반영)

### 유지하는 것

| 기술 | 이유 |
|------|------|
| **VectorBT** | 파라미터 스윕/벡터화 백테스트에서 대체 불가 성능 |
| **Pydantic V2** | 타입 안전성, Config/DTO 표준 |
| **CCXT Pro** | 거래소 통합 (WebSocket + REST) |
| **Loguru** | 구조화 로깅 |
| **Numba** | Numba-optimized 전략 함수 (이미 구현됨) |

### 새로 도입하는 것

| 기술 | Phase | 이유 |
|------|-------|------|
| **asyncio TaskGroup** | Phase 4 | 구조적 동시성 (Python 3.13 기본) |
| **Streamlit** | Phase 2 | Python-native 대시보드, 빠른 개발 |
| **Prometheus + Grafana** | Phase 5 | 프로덕션 시스템 모니터링 표준 |
| **Polars** (선택적) | Phase 2+ | Silver 데이터 로딩 10x 가속, `.to_pandas()` 호환 |

### 도입하지 않는 것

| 기술 | 이유 |
|------|------|
| **NautilusTrader** | 프레임워크 전환 비용 > 이점. 자체 EDA가 유연성 높음 |
| **MLflow** | 현 단계에서 과도한 인프라. 전략 수가 적음 (1개) |
| **Event Sourcing DB** | EventBus + 로컬 이벤트 로그로 충분. 규모가 커지면 재검토 |
| **Free-threaded Python** | 2026년 기준 NumPy/pandas 생태계 미성숙. 2027+ 재검토 |

---

## 전체 타임라인 (예상)

```
Phase 2: 멀티에셋 백테스트 ─────────┐
Phase 3: 고급 검증 (CPCV/WFA) ──────┤  ← 병렬 가능
Phase 7-A: Streamlit 대시보드 ──────┘

Phase 4: EDA 시스템 ─────────────────── ✅ 완료

Phase 5: Dry Run ───────────────────── ← 현재 (Phase 4 완료)
Phase 7-B: Grafana 모니터링 ────────── ← Phase 5와 병행

Phase 6: Live Trading ─────────────── ← Phase 5 검증 통과 후
Phase 7-C: 운영 분석 도구 ──────────── ← Phase 6과 병행
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-02-06 | 초기 로드맵 작성 — Phase 2~7 설계 |
| 2026-02-06 | **Phase 2 완료** — 멀티에셋 백테스트 (`run_multi`, `run_multi_with_returns`, `run_multi_validated`) |
| 2026-02-06 | **Phase 3 완료** — 고급 검증 (IS/OOS, WFA, CPCV, DSR, PBO, 검증 리포트) |
| 2026-02-06 | 8-asset EW 통합 백테스트 검증 — Sharpe 1.57, CAGR +57.95%, MDD 19.43% |
| 2026-02-06 | CAGR 수동 계산 추가 — VBT `cash_sharing` 모드 `Annualized Return [%]` 미제공 대응 |
| 2026-02-06 | **Phase 4 완료** — EDA 시스템 (EventBus, 11 컴포넌트, 110 tests, CLI `eda run`) |
| 2026-02-06 | **Phase 4 강화** — Equity 이중 계산 버그 수정, Position Stop-Loss/Trailing Stop 구현, 매 Bar BalanceUpdate, Parity 재검증 (VBT 2.77% vs EDA 2.72%), 126 tests |
