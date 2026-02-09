# 전략 스코어카드: ADX Regime

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | ADX Regime (`adx-regime`) |
| **유형** | 하이브리드 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 0.94) |
| **2nd Asset** | ETH/USDT (Sharpe 0.81) |
| **경제적 논거** | ADX가 높은 구간(강추세)에서는 모멘텀이, 낮은 구간(횡보)에서는 평균회귀가 유효하다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **SOL/USDT** | **0.94** | 23.06% | -33.86% | 331 | 1.52 |
| 2 | ETH/USDT | 0.81 | 19.64% | -34.58% | 349 | 1.62 |
| 3 | DOGE/USDT | 0.69 | 21.32% | -45.05% | 362 | 1.36 |
| 4 | BTC/USDT | 0.55 | 11.44% | -36.77% | 425 | 1.34 |
| 5 | BNB/USDT | 0.49 | 10.38% | -53.93% | 449 | 1.32 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.94 | > 1.0 | FAIL |
| MDD | -33.86% | < 40% | PASS |
| Trades | 331 | > 50 | PASS |
| Win Rate | 52.73% | > 45% | — |
| Sortino | 1.29 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G1 백테스트  [WATCH] Sharpe 0.94, MDD 33.86%
G2 IS/OOS    [FAIL] OOS Sharpe -0.68, Decay 146.3%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 1.48, OOS Sharpe -0.68, Decay 146.3%
  - 실패 사유: OOS Sharpe (-0.68) < 0.3; Sharpe Decay (146.3%) >= 50%; OOS Return (-26.8%) <= 0%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 24/30점 |
| 2026-02-09 | G1 | WATCH | SOL/USDT Sharpe 0.94 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe -0.68, Decay 146.3% |
