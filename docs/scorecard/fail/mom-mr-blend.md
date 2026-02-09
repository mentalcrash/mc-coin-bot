# 전략 스코어카드: Mom-MR Blend

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Mom-MR Blend (`mom-mr-blend`) |
| **유형** | 하이브리드 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | ETH/USDT (Sharpe 0.48) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.36) |
| **경제적 논거** | 모멘텀과 평균회귀를 동일 비중으로 혼합하면 레짐 변화에 대한 로버스트성이 향상된다는 가설. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **ETH/USDT** | **0.48** | 6.66% | -29.16% | 99 | 1.29 |
| 2 | DOGE/USDT | 0.36 | 11.85% | -36.24% | 89 | 1.39 |
| 3 | BTC/USDT | 0.09 | 0.30% | -27.43% | 86 | 1.08 |
| 4 | SOL/USDT | -0.13 | -2.60% | -26.42% | 71 | 0.89 |
| 5 | BNB/USDT | -0.21 | -4.45% | -52.36% | 108 | 0.86 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.48 | > 1.0 | FAIL |
| MDD | -29.16% | < 40% | PASS |
| Trades | 99 | > 50 | PASS |
| Win Rate | 50.51% | > 45% | — |
| Sortino | 0.26 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 23/30점
G1 백테스트  [WATCH] Sharpe 0.48, MDD 29.16%
G2 IS/OOS    [FAIL] OOS Sharpe -0.10, Decay 109.1%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 1.12, OOS Sharpe -0.10, Decay 109.1%
  - 실패 사유: OOS Sharpe (-0.10) < 0.3; Sharpe Decay (109.1%) >= 50%; OOS Return (-5.9%) <= 0%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 23/30점 |
| 2026-02-09 | G1 | WATCH | ETH/USDT Sharpe 0.48 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe -0.10, Decay 109.1% |
