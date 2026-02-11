# 전략 스코어카드: Vol Regime

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Vol Regime (`vol-regime`) |
| **유형** | 레짐 전환 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | ETH/USDT (Sharpe 1.41) |
| **2nd Asset** | SOL/USDT (Sharpe 0.89) |
| **경제적 논거** | 시장 변동성 수준에 따라 최적 전략 파라미터가 다르며, 자동 전환이 성과를 개선한다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **ETH/USDT** | **1.41** | 57.60% | -32.34% | 298 | 1.83 |
| 2 | SOL/USDT | 0.89 | 29.80% | -44.61% | 281 | 1.28 |
| 3 | BNB/USDT | 0.81 | 26.47% | -50.54% | 379 | 1.39 |
| 4 | DOGE/USDT | 0.77 | 35.21% | -72.02% | 335 | 1.44 |
| 5 | BTC/USDT | 0.55 | 13.94% | -54.36% | 343 | 1.17 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.41 | > 1.0 | PASS |
| MDD | -32.34% | < 40% | PASS |
| Trades | 298 | > 50 | PASS |
| Win Rate | 65.66% | > 45% | — |
| Sortino | 2.13 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 23/30점
G1 백테스트  [PASS] Sharpe 1.41, MDD 32.34%
G2 IS/OOS    [FAIL] OOS Sharpe 0.37, Decay 77.3%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 1.65, OOS Sharpe 0.37, Decay 77.3%

- 실패 사유: Sharpe Decay (77.3%) >= 50%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 23/30점 |
| 2026-02-09 | G1 | PASS | ETH/USDT Sharpe 1.41 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe 0.37, Decay 77.3% |
