# 전략 스코어카드: CTREND

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | CTREND (`ctrend`) |
| **유형** | ML 앙상블 (Technical aggregate) |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 2.05) |
| **2nd Asset** | BTC/USDT (Sharpe 1.70) |
| **3rd Asset** | BNB/USDT (Sharpe 1.59) |
| **경제적 논거** | 다수 기술적 지표의 앙상블 합산으로 개별 지표 노이즈를 상쇄하고 공통 방향성 추출 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **SOL/USDT** | **2.05** | +97.8% | -27.7% | 288 | 1.60 | +3072.9% | 0.11 |
| 2 | BTC/USDT | 1.70 | +72.5% | -31.7% | 304 | 1.71 | +2749.8% | 0.11 |
| 3 | BNB/USDT | 1.59 | +73.0% | -33.9% | 347 | 1.59 | -2410.8% | 0.12 |
| 4 | ETH/USDT | 1.53 | +64.1% | -41.5% | 321 | 1.37 | +400.8% | 0.14 |
| 5 | DOGE/USDT | 0.83 | +29.9% | -64.6% | 365 | 1.44 | -5469.0% | 0.04 |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 2.05 | > 1.0 | PASS |
| CAGR | +97.8% | > 20% | PASS |
| MDD | -27.7% | < 40% | PASS |
| Trades | 288 | > 50 | PASS |
| Win Rate | 57.1% | > 45% | PASS |
| Sortino | 3.31 | > 1.5 | PASS |
| Calmar | 3.53 | > 1.0 | PASS |
| Profit Factor | 1.60 | > 1.3 | PASS |
| Alpha (vs BTC B&H) | +3072.9% | > 0% | PASS |
| Beta (vs BTC) | 0.11 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 22/30점
G1 백테스트  [PASS] SOL/USDT Sharpe 2.05, CAGR +97.8%, MDD -27.7%
G2 IS/OOS    [PASS] OOS Sharpe 1.78, Decay 33.7%
G3 파라미터  [PASS] 4/4 파라미터 고원 + ±20% 안정
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 3 상세 (파라미터 안정성, SOL/USDT 1D)

| 파라미터 | 기본값 | Best Sharpe | 고원 | 고원 범위 | ±20% Sharpe | 판정 |
|---------|--------|-------------|:---:|----------|-------------|:---:|
| training_window | 252 | 2.10 | 9/10 | 126~350 | 1.93~2.05 | PASS |
| prediction_horizon | 5 | 2.60 | 3/10 | 7~14 | 1.78~2.07 | PASS |
| alpha | 0.5 | 2.32 | 9/10 | 0.1~1.0 | 1.77~2.05 | PASS |
| vol_target | 0.35 | 2.06 | 10/10 | 0.15~0.60 | 2.05~2.06 | PASS |

**판정**: **PASS** (4/4 파라미터 통과)

**분석**:
- **training_window**: 126~350 전 범위에서 Sharpe 1.75~2.10 — **넓은 고원**. 400에서도 1.51로 양수 유지
- **prediction_horizon**: 3~21 범위에서 Sharpe 1.76~2.60 — horizon이 길수록(7~14일) 성과 향상. 1~2일은 노이즈(0.53)
- **alpha**: 0.1~1.0 거의 전체 범위에서 Sharpe 1.77~2.32 — **L1/L2 비율에 둔감** (앙상블 로버스트성)
- **vol_target**: 0.15~0.60 전체 범위에서 Sharpe 2.03~2.06 — **거의 완전 평탄** (리스크 스케일링만 변경)

**핵심 관찰**:
1. vol_target은 Sharpe에 거의 영향 없음 (순수 레버리지 스케일링). CAGR만 39%→187%로 변동
2. alpha(ElasticNet 정규화)가 전 범위에서 안정 — 28개 feature 앙상블의 일반화 능력 확인
3. prediction_horizon 1~2일은 명확히 열등 — 일봉 전략에서 5일+ 예측이 적합
4. 40회 백테스트, 207.5초 소요

### Gate 2 상세 (IS/OOS 70/30, 5-coin EW Portfolio)

| 지표 | IS (70%) | OOS (30%) | 기준 | 판정 |
|------|----------|-----------|------|------|
| Sharpe | 2.69 | 1.78 | OOS > 0.3 | PASS |
| Decay | — | 33.7% | < 50% | PASS |
| Consistency | — | 100% | — | — |
| Overfit Prob | — | 20.2% | — | Low |

**판정**: **PASS**
- OOS Sharpe 1.78 >> 0.3 (기준의 5.9배)
- Decay 33.7% < 50%
- Consistency 100%, Overfit Probability 20.2% (양호)
- IS→OOS 성과 감소가 제한적이며, OOS에서도 매우 높은 Sharpe 유지

### Gate 1 상세

- **5개 에셋 중 3개 PASS**: SOL, BTC, BNB 모두 Sharpe > 1.5, CAGR > 70%
- ETH는 MDD 41.5%로 기준 소폭 초과 (WATCH)
- DOGE는 MDD 64.6%로 FAIL
- **전 에셋 양의 수익**: 최저 DOGE +29.9% CAGR
- **낮은 Beta (0.04~0.14)**: BTC 대비 독립적 수익 구조

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 22/30점 |
| 2026-02-10 | G1 | PASS | SOL/USDT Sharpe 2.05, CAGR +97.8%, MDD -27.7%, 288 trades |
| 2026-02-10 | G2 | PASS | OOS Sharpe 1.78, Decay 33.7%, Overfit Prob 20.2% |
| 2026-02-10 | G3 | PASS | 4/4 파라미터 고원 + ±20% 안정. vol_target 거의 평탄, alpha 전 범위 안정 |
