# 전략 스코어카드: XSMOM

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | XSMOM (`xsmom`) |
| **유형** | 크로스섹셔널 모멘텀 (Behavioral herding) |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 1.34) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.80) |
| **3rd Asset** | BTC/USDT (Sharpe 0.55) |
| **경제적 논거** | 행동 편향(herding)에 의한 가격 추세 지속성. 수익률 모멘텀으로 방향 포착 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **SOL/USDT** | **1.34** | +61.2% | -31.6% | 101 | 2.38 | -2180.6% | 0.07 |
| 2 | DOGE/USDT | 0.80 | +33.1% | -49.5% | 130 | 1.62 | -5351.2% | 0.12 |
| 3 | BTC/USDT | 0.55 | +14.7% | -52.3% | 119 | 1.30 | -871.4% | 0.04 |
| 4 | ETH/USDT | 0.55 | +14.8% | -71.9% | 131 | 1.22 | -1950.2% | 0.08 |
| 5 | BNB/USDT | 0.40 | +8.5% | -64.6% | 145 | 1.20 | -6043.3% | 0.15 |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.34 | > 1.0 | PASS |
| CAGR | +61.2% | > 20% | PASS |
| MDD | -31.6% | < 40% | PASS |
| Trades | 101 | > 50 | PASS |
| Win Rate | 51.0% | > 45% | PASS |
| Sortino | 1.97 | > 1.5 | PASS |
| Calmar | 1.94 | > 1.0 | PASS |
| Profit Factor | 2.38 | > 1.3 | PASS |
| Beta (vs BTC) | 0.07 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G1 백테스트  [PASS] SOL/USDT Sharpe 1.34, CAGR +61.2%, MDD -31.6%
G2 IS/OOS    [FAIL] OOS Sharpe 0.30, Decay 76.6%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 2 상세 (IS/OOS 70/30, 5-coin EW Portfolio)

| 지표 | IS (70%) | OOS (30%) | 기준 | 판정 |
|------|----------|-----------|------|------|
| Sharpe | 1.30 | 0.30 | OOS > 0.3 | borderline |
| Decay | — | 76.6% | < 50% | **FAIL** |
| Consistency | — | 0% | — | — |
| Overfit Prob | — | 86.0% | — | High |

**판정**: **FAIL**
- OOS Sharpe 0.30은 기준 0.3과 동일 (borderline)
- Decay 76.6% >> 50% (IS 성과의 3/4 이상 소멸)
- Overfit Probability 86% (과적합 강력 의심)
- IS 기간의 높은 성과가 OOS에서 재현되지 않음

### Gate 1 상세

- **SOL/USDT만 PASS**: 유일하게 전 기준 충족
- DOGE는 WATCH (Sharpe 0.80, MDD 49.5%)
- BTC/ETH/BNB는 MDD > 50%로 FAIL
- **Profit Factor 2.38 주목**: 5개 전략 중 최고 수준
- **낮은 Beta (0.04~0.15)**: BTC 대비 거의 독립적

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 24/30점 |
| 2026-02-10 | G1 | PASS | SOL/USDT Sharpe 1.34, CAGR +61.2%, MDD -31.6%, 101 trades |
| 2026-02-10 | G2 | FAIL | OOS Sharpe 0.30, Decay 76.6%, Overfit Prob 86% |
