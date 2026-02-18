# Strategy Pipeline

전략 아이디어부터 실전 배포까지 **7단계 Phase**를 순차 통과하는 검증 파이프라인.
각 Phase에서 FAIL 시 즉시 RETIRED. 103개 전략 중 **2개 ACTIVE** (CTREND, Anchor-Mom).

---

## Phase Overview

```text
Phase 1         Phase 2           Phase 3          Phase 4
Alpha Research → Implementation → Code Audit     → Backtest
(발굴·설계)      (코드 구현)       (코드 검증)       (성과 검증)

     Phase 5           Phase 6              Phase 7
  → Robustness      → Deep Validation    → Live Readiness
    (강건성 검증)      (통계 검증)           (실전 준비)
```

### Status Lifecycle

```text
CANDIDATE ──[P2 완료]──→ IMPLEMENTED ──[P4 첫 PASS]──→ TESTING ──[P7 PASS]──→ ACTIVE
     │                       │                            │
     └── FAIL ──→ RETIRED    └── FAIL ──→ RETIRED         └── FAIL ──→ RETIRED
```

---

## Phase 1: Alpha Research (발굴 · 설계)

데이터 분석을 통한 알파 소스 발견 + 전략 설계.

**스킬**: `/p1-research`

### 데이터 기반 발굴 프로세스

1. **데이터 탐색** -- 카탈로그(75 datasets, 53 indicators) 분석, 시그널-리턴 Rank IC 계산
1. **레짐 조건부 분석** -- TRENDING/RANGING/VOLATILE별 IC 분해
1. **기존 전략 직교성** -- ACTIVE 전략과의 상관 < 0.5 확인
1. **전략 설계** -- 진입/청산 로직, 리스크 파라미터, 레짐 적응 방식 설계

### 알파 카테고리 (6종)

| Category | 데이터 소스 | 예시 |
|----------|-----------|------|
| Trend | OHLCV, Indicators | 모멘텀, 트렌드 강도 |
| Mean-Reversion | Bollinger, Z-Score | 과매도 반전, 변동성 수축→확장 |
| Sentiment | Fear&Greed, LS Ratio, Funding | 군중 심리 역행 |
| Flow | On-chain (MVRV, Stablecoin, TVL) | 자금 흐름, 스마트머니 |
| Macro | FRED, yfinance, CoinGecko | 달러·금리·유동성 레짐 |
| Derivatives | OI, Liquidation, Options | 포지셔닝 불균형 |

### PASS 기준

| 항목 | 기준 |
|------|------|
| Scorecard | >= 21/30 (6항목 x 5점) |
| Rank IC | \|IC\| > 0.02 |
| Regime IC | 2/3 레짐에서 양수 |
| Active 상관 | < 0.5 |

> Scorecard 항목: 경제적 논거 고유성, IC 사전 검증, 카테고리 성공률, 레짐 독립성, 앙상블 기여도, 수용 용량

---

## Phase 2: Implementation (코드 구현)

Phase 1 설계를 4-file 구조로 구현.

**스킬**: `/p2-implement`

### 4-File 구조

```text
src/strategy/{name}/
├── config.py          # Pydantic config + from_params()
├── preprocessor.py    # 지표 계산 + 데이터 전처리
├── signal.py          # 시그널 생성 로직
├── strategy.py        # BaseStrategy 상속 + @register()
└── __init__.py
```

### PASS 기준

| 항목 | 기준 |
|------|------|
| 4-file 구조 | 필수 5파일 존재 |
| 테스트 | pytest 0 failures |
| Lint | Ruff + Pyright 0 errors |
| Registry | `@register("name")` 확인 |
| P1 설계 반영 | `deviations_from_design` 문서화 |

---

## Phase 3: Code Audit (코드 검증)

구현된 코드의 정합성 검증. 백테스트 전 필수.

**스킬**: `/p3-audit`

### Critical Checks (C1~C7, 전부 PASS 필수)

| 코드 | 항목 | 검증 내용 |
|------|------|----------|
| C1 | Look-Ahead Bias | close[t] → signal[t] → open[t+1] 체결 |
| C2 | Data Leakage | IS/OOS 경계 넘어 학습 없음 |
| C3 | Survivorship Bias | 백테스트 기간 전체에 존재하는 에셋만 |
| C4 | Signal Vectorization | pandas 인덱스 정렬, NaN 전파 |
| C5 | Position Sizing | vol-target, leverage cap, 0-div 방지 |
| C6 | Cost Model | CostModel 파라미터 전달 확인 |
| C7 | Entry/Exit Logic | 동시 long+short 불가, 동일 bar 진입+청산 금지 |

### Warning Checks (W1~W5)

| 코드 | 항목 | 설명 |
|------|------|------|
| W1 | Warmup Period | 지표 lookback 대비 충분한 warmup |
| W2 | Parameter Count | Trades/Params > 10 |
| W3 | Regime Concentration | 특정 레짐 수익 집중도 |
| W4 | Turnover | 연 200회 이상이면 비용 민감도 경고 |
| W5 | Active Correlation | 기존 전략과 상관 0.7 이하 |

---

## Phase 4: Backtest (성과 검증)

단일에셋 백테스트 + IS/OOS 과적합 검증.

**스킬**: `/p4-backtest`

### 검증 범위

- **단일에셋**: 5개 에셋 x 6년 (2020~2025) 개별 백테스트
- **IS/OOS**: 70/30 분할, Sharpe decay 측정
- **비용 민감도**: 기본 4bps → 6/8bps 시뮬레이션, break-even fee 계산
- **레짐 분해**: 에셋별 TRENDING/RANGING/VOLATILE 성과 분석

### PASS 기준

| 항목 | 기준 |
|------|------|
| Best Asset Sharpe | > 1.0 |
| Best Asset CAGR | > 20% |
| Best Asset MDD | < 40% |
| Trades | > 50 |
| OOS Sharpe | >= 0.3 |
| Sharpe Decay | < 50% |
| Break-even Fee | > 8 bps |

**Immediate Fail**: MDD > 50%, 모든 에셋 Sharpe < 0, Break-even Fee < 6 bps

---

## Phase 5: Robustness (강건성 검증)

파라미터 최적화 + 안정성 검증.

**스킬**: `/p5-robustness`

### 검증 항목

1. **Optuna TPE 최적화** -- 200 trials, IS/OOS 분할 최적화
1. **파라미터 고원 분석** -- Sharpe > 0인 파라미터 조합 비율
1. **±20% 안정성** -- 핵심 파라미터 ±20% 변경 시 Sharpe 부호 유지

### PASS 기준

| 항목 | 기준 |
|------|------|
| 최적화 완료 | 정상 종료 |
| Plateau | >= 60% |
| ±20% 안정성 | 모든 핵심 파라미터에서 Sharpe > 0 |

---

## Phase 6: Deep Validation (통계 검증)

과적합 여부를 통계적으로 검증.

**스킬**: `/p6-validation`

### 검증 방법론

| 방법 | 설명 |
|------|------|
| WFA | Walk-Forward Analysis (5-fold), OOS Sharpe + Decay + Consistency |
| CPCV | Combinatorially Purged Cross-Validation |
| PBO | Probability of Backtest Overfitting |
| DSR | Deflated Sharpe Ratio (다중 테스트 보정) |
| Monte Carlo | 1000 시뮬레이션, p-value 검증 |

### PASS 기준

| 항목 | 기준 |
|------|------|
| WFA OOS Sharpe | >= 0.5 |
| WFA Decay | < 40% |
| WFA Consistency | >= 60% |
| PBO | Path A: < 40% \| Path B: < 80% + 모든 fold OOS 양수 + MC p < 0.05 |
| DSR | > 0.95 |
| MC p-value | < 0.05 |

---

## Phase 7: Live Readiness (실전 준비)

VBT vs EDA 정합성 + 라이브 인프라 검증.

**스킬**: `/p7-live`

### 검증 항목

1. **EDA Parity** -- VBT 백테스트와 EDA 이벤트 기반 백테스트의 수익 정합성
1. **라이브 준비 점검** -- EventBus flush, Executor order, Deferred execution, Circuit breaker
1. **배포 설정 생성** -- config YAML, 모니터링 알림 룰

### PASS 기준

| 항목 | 기준 |
|------|------|
| 수익 부호 일치 | VBT/EDA 동일 |
| 수익률 편차 | < 20% |
| 거래 수 비율 | 0.5x ~ 2.0x |
| L1/L2/L3/L6/L7 | 모든 Critical PASS |
| 배포 설정 | config YAML 생성 완료 |

---

## Strategy YAML Schema (v2)

모든 전략은 `strategies/{name}.yaml`에 Phase 결과를 누적 기록합니다.

```yaml
version: 2                              # 스키마 버전
meta:
  name: "ctrend"                        # kebab-case registry key
  display_name: "CTREND"
  category: "ML 앙상블"                  # 전략 유형
  timeframe: "1D"
  short_mode: "FULL"                    # DISABLED | HEDGE_ONLY | FULL
  status: "ACTIVE"                      # CANDIDATE | IMPLEMENTED | TESTING | ACTIVE | RETIRED
  created_at: "2026-02-10"
  economic_rationale: "..."

parameters:                             # 최종 파라미터 (P5 최적화 후 갱신)
  training_window: 68
  prediction_horizon: 12

phases:                                 # Phase 1~7 검증 결과
  P1: { status: PASS, date: "...", details: { score: 22, max_score: 30 } }
  P3: { status: PASS, date: "...", details: { C1: PASS, ... } }
  P4: { status: PASS, date: "...", details: { ... } }
  # ...

asset_performance:                      # P4 완료 후 에셋별 성과
  - { symbol: SOL/USDT, sharpe: 2.05, cagr: 97.8, mdd: 27.7, trades: 288 }

decisions:                              # 의사결정 이력
  - { date: "...", phase: P1, verdict: PASS, rationale: "22/30점" }
```

---

## CLI Commands

```bash
uv run mcbot pipeline status            # 현황 요약
uv run mcbot pipeline table             # 전체 Phase 진행도
uv run mcbot pipeline report            # 상세 대시보드
uv run mcbot pipeline show ctrend       # 전략 상세
uv run mcbot pipeline list              # 전략 목록 (필터 지원)

uv run mcbot pipeline create            # 신규 전략 YAML 생성
uv run mcbot pipeline record            # Phase 결과 기록
uv run mcbot pipeline update-status     # 상태 변경
```

---

## Criteria File

Phase별 상세 평가 기준은 [`gates/phase-criteria.yaml`](../gates/phase-criteria.yaml)에서 관리됩니다.
코드 모델은 `src/pipeline/models.py`의 `PhaseId`, `PhaseResult`, `StrategyRecord`를 참조.

---

## 핵심 교훈 (87개 전략 검증에서 도출)

103개 전략 평가 과정에서 축적된 교훈을 `lessons/*.yaml`로 구조화 관리합니다.

```bash
uv run mcbot pipeline lessons-list                      # 전체 교훈 목록
uv run mcbot pipeline lessons-list -c strategy-design   # 카테고리 필터
uv run mcbot pipeline lessons-show 1                    # 교훈 상세
```

### 구조적 발견

- **4H 단일지표 = 전량 RETIRED**: 거래비용이 edge를 잠식, OOS 붕괴 빈번
- **1D 앙상블만 G5 도달**: CTREND Sharpe 2.05, 8-asset EW Sharpe 1.57
- **SL/TS가 핵심**: Trailing Stop 3.0x ATR이 MDD 방어의 핵심 요소
- **비용 민감도**: break-even fee > 8bps 미달 시 실전 운용 불가
