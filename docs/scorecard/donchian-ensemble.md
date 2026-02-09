# 전략 스코어카드: Donchian Ensemble

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Donchian Ensemble (`donchian-ensemble`) |
| **유형** | 추세추종 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | ETH/USDT (Sharpe 0.99) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.92) |
| **경제적 논거** | 다양한 lookback의 평균은 특정 기간에 대한 과적합을 방지하고, 앙상블 효과로 안정성을 높인다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **ETH/USDT** | **0.99** | 10.79% | -9.73% | 181 | 1.91 |
| 2 | DOGE/USDT | 0.92 | 19.11% | -22.00% | 115 | 2.53 |
| 3 | BNB/USDT | 0.68 | 8.88% | -13.57% | 185 | 1.72 |
| 4 | BTC/USDT | 0.66 | 7.84% | -13.96% | 185 | 1.51 |
| 5 | SOL/USDT | 0.54 | 6.30% | -25.29% | 132 | 1.59 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.99 | > 1.0 | FAIL |
| MDD | -9.73% | < 40% | PASS |
| Trades | 181 | > 50 | PASS |
| Win Rate | 43.09% | > 45% | — |
| Sortino | 1.06 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 25/30점
G1 백테스트  [WATCH] Sharpe 0.99, MDD 9.73%
G2 IS/OOS    [PASS] OOS Sharpe 0.99, Decay 1.7%
G3 파라미터  [PASS] 2/2 고원 + ±20% 안정
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (PASS): IS Sharpe 1.01, OOS Sharpe 0.99, Decay 1.7%, OOS Return +13.4%

**Gate 3** (PASS): 2개 핵심 파라미터 모두 고원 존재 + ±20% Sharpe 부호 유지
- `vol_target`: 고원 10개 (0.2~0.55), ±20% Sharpe 0.95~0.99 — 전 범위 안정
- `atr_period`: 고원 10개 (8~25), ±20% Sharpe 0.95~1.00 — 전 범위 안정

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 25/30점 |
| 2026-02-09 | G1 | WATCH | ETH/USDT Sharpe 0.99 |
| 2026-02-09 | G2 | PASS | OOS Sharpe 0.99, Decay 1.7% |
| 2026-02-09 | G3 | PASS | 2/2 파라미터 고원+안정 |
