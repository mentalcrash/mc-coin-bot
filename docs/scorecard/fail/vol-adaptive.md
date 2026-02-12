# 전략 스코어카드: Vol-Adaptive

> 자동 생성 | 평가 기준: [dashboard.md](../../strategy/dashboard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Vol-Adaptive (`vol-adaptive`) |
| **유형** | 추세추종 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 1.08) |
| **2nd Asset** | BNB/USDT (Sharpe 0.83) |
| **경제적 논거** | 여러 지표의 동시 확인(confluence)은 허위 시그널을 줄이고, ATR 사이징은 변동성에 적응한다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **SOL/USDT** | **1.08** | 31.45% | -44.88% | 57 | 1.86 |
| 2 | BNB/USDT | 0.83 | 23.99% | -41.96% | 82 | 2.10 |
| 3 | BTC/USDT | 0.72 | 16.72% | -30.49% | 75 | 1.66 |
| 4 | DOGE/USDT | 0.66 | 33.89% | -44.69% | 78 | 1.86 |
| 5 | ETH/USDT | 0.58 | 13.24% | -48.50% | 85 | 1.49 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.08 | > 1.0 | PASS |
| MDD | -44.88% | < 40% | FAIL |
| Trades | 57 | > 50 | PASS |
| Win Rate | 43.86% | > 45% | — |
| Sortino | 1.07 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 22/30점
G1 백테스트  [PASS] Sharpe 1.08, MDD 44.88%
G2 IS/OOS    [FAIL] OOS Sharpe -0.97, Decay 155.9%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 1.74, OOS Sharpe -0.97, Decay 155.9%

- 실패 사유: OOS Sharpe (-0.97) < 0.3; Sharpe Decay (155.9%) >= 50%; OOS Return (-32.7%) <= 0%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 22/30점 |
| 2026-02-09 | G1 | PASS | SOL/USDT Sharpe 1.08 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe -0.97, Decay 155.9% |
