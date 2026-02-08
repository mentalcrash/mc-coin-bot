# EDA vs VBT Backtest Comparison Report

**날짜:** 2026-02-07
**목적:** EDA 백테스트 시스템의 신뢰성 검증 및 페이퍼 트레이딩 준비도 평가

---

## 1. 테스트 환경

| 항목 | 값 |
|------|---|
| 기간 | 2024-01-01 ~ 2025-12-31 (731 bars) |
| 초기 자본 | $10,000 |
| 최대 레버리지 | 2.0x |
| 비용 모델 | Binance 기본 (0.22% RT) |
| 데이터 | Silver layer (gap-filled, validated) |
| 테스트 전략 | TSMOM, Adaptive-Breakout, Donchian, BB-RSI |
| 테스트 자산 | BTC/USDT, ETH/USDT, SOL/USDT |

---

## 2. Parity Test 결과 (통제된 환경)

SimpleMomentum 전략 + 합성 데이터 (365일 상승 트렌드)로 양쪽 엔진 비교:

| Metric | VBT | EDA | Delta |
|--------|:---:|:---:|:-----:|
| **Total Return** | 2.77% | 2.72% | -0.05pp |
| **Sharpe Ratio** | 0.37 | 0.49 | +0.12 |
| **Max Drawdown** | 9.29% | 6.05% | -3.24pp |
| **Win Rate** | 55.0% | 45.0% | -10.0pp |
| **Total Trades** | 21 | 20 | -1 |

**판정: PASS** - 수익률 차이 0.05pp, 거래 수 1건 차이. 통제된 조건에서 parity 확보.

---

## 3. 실전 전략 비교 (BTC/USDT 2024-2025)

### 3.1 전략별 비교표

| Strategy | Engine | Return | Sharpe | Trades | MDD | Win Rate |
|----------|--------|-------:|-------:|-------:|----:|--------:|
| **TSMOM** | VBT | +47.21% | 0.82 | 95 | 28.23% | 53.2% |
| | EDA | -45.37% | -2.61 | **3** | 47.06% | 33.3% |
| **A-Breakout** | VBT | +16.22% | 0.65 | 23 | 13.07% | 52.2% |
| | EDA | **+632.32%** | 2.07 | 20 | 9.49% | 85.0% |
| **Donchian** | VBT | +17.98% | 0.44 | 35 | 30.99% | 65.7% |
| | EDA | +5.84% | 0.24 | **9** | 11.39% | 44.4% |
| **BB-RSI** | VBT | -2.80% | -0.30 | 117 | 7.44% | 50.9% |
| | EDA | -11.05% | -0.81 | 62 | 6.92% | 77.4% |

### 3.2 멀티 심볼 TSMOM 비교

| Symbol | Engine | Return | Sharpe | Trades | MDD |
|--------|--------|-------:|-------:|-------:|----:|
| **BTC/USDT** | VBT | +47.21% | 0.82 | 95 | 28.23% |
| | EDA | -45.37% | -2.61 | 3 | 47.06% |
| **ETH/USDT** | VBT | +76.46% | 1.12 | 61 | 26.27% |
| | EDA | +34.05% | 4.37 | 1 | 13.70% |
| **SOL/USDT** | VBT | +25.46% | 0.53 | 67 | 30.52% |
| | EDA | -14.24% | -0.74 | 1 | 10.78% |

---

## 4. 차이 원인 분석

### 4.1 거래 수 불일치 (핵심 문제)

**TSMOM**: EDA 3건 vs VBT 95건 (32배 차이)

**원인: StrategyEngine의 시그널 중복제거가 과도함**

```
EDA StrategyEngine 로직:
  1. 매 bar마다 strategy.run() 호출 → 시그널 전체 배열 생성
  2. 최신 시그널(direction, strength)만 추출
  3. 이전 시그널과 비교: direction 변화 또는 strength 변화(>1e-8)가 있을 때만 SignalEvent 발행
  4. TSMOM은 한번 LONG 진입하면 오래 유지 → direction 변화 거의 없음
  5. 결과: 2년간 3번만 시그널 변화 감지
```

```
VBT BacktestEngine 로직:
  1. 전체 시계열에 대해 벡터화된 포지션 사이징 (vol-target)
  2. 매 bar마다 변동성 기반 포지션 크기 재계산
  3. 리밸런싱 임계값 초과 시 포지션 조정 → 거래 발생
  4. 결과: 변동성 변화에 따라 빈번한 포지션 크기 조정
```

**VBT는 "포지션 사이징 조정"을 거래로 카운트하지만, EDA는 "시그널 변화"만 거래로 카운트.**

### 4.2 Adaptive-Breakout 632% 수익률 이상치

| 지표 | VBT | EDA | 비고 |
|------|-----|-----|------|
| 거래 수 | 23 | 20 | 유사 |
| 수익률 | +16.22% | +632.32% | **39배 차이** |

**원인 분석:**
- 거래 수는 유사 → 시그널 생성 자체는 정상
- EDA는 수익을 재투자(복리 효과)하여 equity 기반 포지션 사이징
- VBT는 고정 자본 기반이거나 다른 사이징 로직 적용
- 추가: EDA의 stop-loss가 빈번하게 발동 후 재진입 패턴이 유리하게 작용

### 4.3 BB-RSI (가장 유사한 결과)

| 지표 | VBT | EDA | 비율 |
|------|-----|-----|------|
| Return | -2.80% | -11.05% | 방향 일치 |
| Sharpe | -0.30 | -0.81 | 방향 일치 |
| Trades | 117 | 62 | 0.53x |
| MDD | 7.44% | 6.92% | 유사 |

**BB-RSI가 가장 유사한 이유:** 빈번한 시그널 변화(매수/매도 교대)로 EDA의 시그널 중복제거가 덜 영향을 줌.

### 4.4 체결 가격 차이 (구조적)

| 항목 | VBT | EDA |
|------|-----|-----|
| 체결 시점 | 시그널 bar의 close | **다음 bar의 open** |
| Look-ahead bias | 가능성 있음 | **방지됨** |
| 영향 | 약간 낙관적 | 보수적 (현실적) |

EDA의 next-open 체결은 라이브 트레이딩에 더 현실적이며, 이것 자체는 **장점**.

---

## 5. EDA 시스템 건전성 평가

### 5.1 통과 항목

| # | 항목 | 상태 | 근거 |
|---|------|:----:|------|
| 1 | 전체 테스트 | **PASS** | 335 tests 전부 통과 (EDA 95건 포함) |
| 2 | EventBus 안정성 | **PASS** | Bounded queue, flush(), backpressure 정상 |
| 3 | 이벤트 체인 완결성 | **PASS** | BAR→Signal→Order→Fill→Balance 전체 체인 동작 |
| 4 | Risk 컨트롤 | **PASS** | Stop-loss, trailing stop, leverage capping, circuit breaker 정상 |
| 5 | 멱등성 (OMS) | **PASS** | client_order_id 중복 방지 동작 확인 |
| 6 | Look-ahead bias 방지 | **PASS** | Next-open fill 정상 |
| 7 | 통제 환경 parity | **PASS** | SimpleMomentum: return 0.05pp, trades ±1 |
| 8 | 코드 품질 | **PASS** | ruff + pyright 0 errors |

### 5.2 미통과 항목

| # | 항목 | 상태 | 근거 |
|---|------|:----:|------|
| 1 | TSMOM 거래 수 parity | **FAIL** | EDA 3건 vs VBT 95건 (32배) |
| 2 | Breakout 수익률 parity | **FAIL** | EDA +632% vs VBT +16% (39배) |
| 3 | Donchian 거래 수 parity | **FAIL** | EDA 9건 vs VBT 35건 (4배) |
| 4 | Vol-target 리밸런싱 | **FAIL** | EDA PM에 bar별 포지션 크기 재조정 로직 없음 |

---

## 6. 페이퍼 트레이딩 준비도 평가

### 6.1 판정: NOT READY

EDA 인프라(EventBus, 이벤트 체인, 리스크 컨트롤)는 견고하지만,
**실전 전략과의 parity가 확보되지 않아** 페이퍼 트레이딩으로 전환하기에는 부족합니다.

### 6.2 필수 수정사항 (Paper Trading 전 해결 필요)

| 우선순위 | 작업 | 영향도 | 난이도 |
|:--------:|------|:------:|:------:|
| **P0** | StrategyEngine vol-target 리밸런싱 지원 | TSMOM 거래 수 정상화 | 중간 |
| **P0** | PM bar별 포지션 크기 재조정 | VBT와 동일한 포지션 사이징 | 높음 |
| **P1** | 실전 전략 parity 테스트 추가 | TSMOM/Breakout/Donchian 검증 | 중간 |
| **P1** | Breakout 수익률 이상치 조사 | 포지션 사이징/복리 로직 검증 | 중간 |
| **P2** | 멀티에셋 EDA CLI 지원 | run-multi 커맨드 | 낮음 |

### 6.3 수정 방안

#### P0-1: StrategyEngine 개선

현재 문제: `strength` 변화가 `1e-8` 이하면 시그널 중복제거 → vol-target 리밸런싱 누락

```
해결 방안 A: StrategyEngine에서 매 bar SignalEvent 발행 (dedup 제거)
  - 장점: 단순, VBT와 동일한 빈도
  - 단점: 이벤트 폭증, 성능 저하

해결 방안 B: PM에서 독립적 vol-target 리밸런싱 (권장)
  - 매 bar마다 PM이 현재 volatility 기반으로 target weight 재계산
  - rebalance_threshold 초과 시 자체적으로 OrderRequest 발행
  - 장점: StrategyEngine 변경 없음, 라이브와 동일 패턴
  - 단점: PM 복잡도 증가
```

#### P0-2: 실전 전략 Parity 테스트

```
현재: SimpleMomentum (통제 전략)만 parity 검증
목표: TSMOM, Breakout, Donchian, BB-RSI 각각에 대해
  - 거래 수 비율: 0.5x ~ 2.0x
  - 수익률 방향: 부호 일치
  - 수익률 규모: 0.2x ~ 5.0x
```

---

## 7. 현재 사용 가능한 시나리오

EDA 시스템이 현재 상태로 유효한 용도:

| 시나리오 | 적합성 | 사유 |
|----------|:------:|------|
| BB-RSI 전략 백테스트 | **적합** | 거래 수/방향 유사, parity 합리적 |
| Breakout 전략 시그널 검증 | **부분 적합** | 거래 수 유사하나 수익률 이상치 주의 |
| 이벤트 체인 / 리스크 로직 검증 | **적합** | 인프라 자체는 건전 |
| TSMOM vol-target 전략 | **부적합** | 거래 수 32배 차이, parity 미확보 |
| 페이퍼/라이브 트레이딩 | **부적합** | 포지션 사이징 로직 불일치 |

---

## 8. 다음 단계 로드맵

```
현재 상태 (Phase 4 완료)
    │
    ├── [P0] PM vol-target 리밸런싱 구현
    ├── [P0] 실전 전략 parity 검증 강화
    │
    ├── Parity 확보 확인
    │       │
    │       ├── [Phase 5-A] Shadow Mode (1~2주)
    │       │     실시간 WebSocket + 시그널 로깅만
    │       │
    │       ├── [Phase 5-B] Paper Trading (2~4주)
    │       │     실시간 데이터 + 시뮬레이션 체결
    │       │
    │       ├── [Phase 5-C] Canary ($100~500, 2~4주)
    │       │     실제 Binance 소액 주문
    │       │
    │       └── [Phase 6] Full Live
    │             전체 자본 투입
    │
    └── VBT는 계속 파라미터 스윕/검증용으로 병행 사용
```

---

## 9. 결론

### EDA 시스템 강점
- 이벤트 기반 아키텍처 설계 견고 (EventBus, flush, backpressure)
- 335개 테스트 전부 통과 (95개 EDA 전용 테스트 포함)
- Look-ahead bias 방지 (next-open fill)
- 리스크 컨트롤 3단계 방어 (PM → RM → OMS)
- 라이브 전환 시 동일 코드 재사용 가능 설계

### 해결 필요 사항
- Vol-target 기반 포지션 리밸런싱 로직 부재
- 실전 전략과의 parity 미확보 (TSMOM, Breakout, Donchian)
- 멀티에셋 EDA CLI 미구현

### 최종 판정
**EDA 인프라는 건전하지만, 포지션 사이징 parity 해결 전까지 페이퍼 트레이딩 전환 보류.**
P0 수정사항 해결 후 재검증 필요.

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-07 | 초기 작성 — EDA vs VBT 전략별 비교, parity 분석, 페이퍼 트레이딩 준비도 평가 |
