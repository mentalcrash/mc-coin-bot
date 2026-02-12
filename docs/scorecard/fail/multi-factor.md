# 전략 스코어카드: Multi-Factor

> 자동 생성 | 평가 기준: [dashboard.md](../../strategy/dashboard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Multi-Factor (`multi-factor`) |
| **유형** | 멀티팩터 (직교 alpha 결합) |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 1.22) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.74) |
| **3rd Asset** | ETH/USDT (Sharpe 0.39) |
| **경제적 논거** | 모멘텀/변동성/추세 등 직교 팩터 결합으로 단일 팩터 대비 안정적 alpha |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **SOL/USDT** | **1.22** | +48.4% | -40.3% | 206 | 1.51 | -3050.3% | 0.07 |
| 2 | DOGE/USDT | 0.74 | +27.4% | -49.6% | 242 | 1.30 | -5529.5% | 0.00 |
| 3 | ETH/USDT | 0.39 | +8.1% | -54.1% | 242 | 1.11 | -2050.6% | 0.03 |
| 4 | BNB/USDT | 0.23 | +1.4% | -58.3% | 238 | 1.10 | -6125.9% | 0.06 |
| 5 | BTC/USDT | 0.18 | -0.2% | -74.7% | 232 | 1.09 | -1076.4% | 0.04 |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.22 | > 1.0 | PASS |
| CAGR | +48.4% | > 20% | PASS |
| MDD | -40.3% | < 40% | **FAIL** (40.3% > 40%) |
| Trades | 206 | > 50 | PASS |
| Win Rate | 51.2% | > 45% | PASS |
| Sortino | 1.88 | > 1.5 | PASS |
| Calmar | 1.20 | > 1.0 | PASS |
| Profit Factor | 1.51 | > 1.3 | PASS |
| Beta (vs BTC) | 0.07 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 23/30점
G1 백테스트  [WATCH] SOL/USDT Sharpe 1.22, CAGR +48.4%, MDD -40.3% (MDD 기준 소폭 초과)
G2 IS/OOS    [FAIL] OOS Sharpe 0.17, Decay 83.3%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 2 상세 (IS/OOS 70/30, 5-coin EW Portfolio)

| 지표 | IS (70%) | OOS (30%) | 기준 | 판정 |
|------|----------|-----------|------|------|
| Sharpe | 1.02 | 0.17 | OOS > 0.3 | **FAIL** |
| Decay | — | 83.3% | < 50% | **FAIL** |
| Consistency | — | 0% | — | — |
| Overfit Prob | — | 90.0% | — | Very High |

**판정**: **FAIL**

- OOS Sharpe 0.17 < 0.3 (기준 미달)
- Decay 83.3% >> 50% (IS 성과의 5/6 소멸)
- Overfit Probability 90% (과적합 매우 강력)
- G1에서 이미 MDD 기준 초과 (WATCH) + G2 FAIL → 폐기 대상

### Gate 1 상세

- **SOL/USDT WATCH**: Sharpe 1.22, CAGR +48.4% 우수하나 MDD 40.3%가 기준 40% 소폭 초과
- DOGE WATCH (Sharpe 0.74)
- BTC/ETH/BNB 모두 MDD > 50%로 FAIL
- **MDD 개선 여지**: PM 파라미터 조정(trailing stop, leverage cap)으로 MDD 감소 가능성
- Gate 2 진행 권고: MDD 소폭 초과이므로 IS/OOS 검증까지 진행 후 최종 판단

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 23/30점 |
| 2026-02-10 | G1 | WATCH | SOL/USDT Sharpe 1.22, CAGR +48.4%, MDD -40.3% (MDD 기준 소폭 초과) |
| 2026-02-10 | G2 | FAIL | OOS Sharpe 0.17, Decay 83.3%, Overfit Prob 90% |
