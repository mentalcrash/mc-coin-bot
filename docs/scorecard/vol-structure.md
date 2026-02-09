# 전략 스코어카드: Vol Structure

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Vol Structure (`vol-structure`) |
| **유형** | 레짐 전환 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 1.18) |
| **2nd Asset** | ETH/USDT (Sharpe 0.55) |
| **경제적 논거** | 단기/장기 변동성 비율은 시장 레짐 전환의 선행 지표로, 구조적 변화를 조기에 포착한다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **SOL/USDT** | **1.18** | 31.77% | -28.30% | 268 | 1.89 |
| 2 | ETH/USDT | 0.55 | 11.92% | -57.18% | 354 | 1.25 |
| 3 | DOGE/USDT | 0.50 | 12.04% | -61.42% | 297 | 1.39 |
| 4 | BNB/USDT | 0.40 | 7.85% | -65.68% | 428 | 1.19 |
| 5 | BTC/USDT | 0.38 | 6.77% | -49.06% | 389 | 1.19 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.18 | > 1.0 | PASS |
| MDD | -28.30% | < 40% | PASS |
| Trades | 268 | > 50 | PASS |
| Win Rate | 55.81% | > 45% | — |
| Sortino | 1.82 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G1 백테스트  [PASS] Sharpe 1.18, MDD 28.30%
G2 IS/OOS    [FAIL] OOS Sharpe -0.23, Decay -28.3%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe -0.33, OOS Sharpe -0.23, Decay -28.3%
  - 실패 사유: OOS Sharpe (-0.23) < 0.5

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 24/30점 |
| 2026-02-09 | G1 | PASS | SOL/USDT Sharpe 1.18 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe -0.23, Decay -28.3% |
