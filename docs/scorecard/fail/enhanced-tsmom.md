# 전략 스코어카드: Enhanced VW-TSMOM

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Enhanced VW-TSMOM (`enhanced-tsmom`) |
| **유형** | 추세추종 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | BTC/USDT (Sharpe 1.22) |
| **2nd Asset** | DOGE/USDT (Sharpe 1.17) |
| **경제적 논거** | 거래량 증가와 함께하는 가격 움직임은 더 높은 지속성을 가지며, 허위 돌파를 필터링한다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **BTC/USDT** | **1.22** | 39.05% | -37.52% | 171 | 1.86 |
| 2 | DOGE/USDT | 1.17 | 57.82% | -28.48% | 174 | 2.21 |
| 3 | ETH/USDT | 1.13 | 36.40% | -34.38% | 157 | 1.99 |
| 4 | SOL/USDT | 1.13 | 42.61% | -42.73% | 138 | 1.62 |
| 5 | BNB/USDT | 1.02 | 34.11% | -38.23% | 184 | 1.81 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.22 | > 1.0 | PASS |
| MDD | -37.52% | < 40% | PASS |
| Trades | 171 | > 50 | PASS |
| Win Rate | 52.94% | > 45% | — |
| Sortino | 1.73 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G1 백테스트  [PASS] Sharpe 1.22, MDD 37.52%
G2 IS/OOS    [FAIL] OOS Sharpe 0.25, Decay 85.2%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 1.67, OOS Sharpe 0.25, Decay 85.2%
  - 실패 사유: OOS Sharpe (0.25) < 0.3; Sharpe Decay (85.2%) >= 50%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 24/30점 |
| 2026-02-09 | G1 | PASS | BTC/USDT Sharpe 1.22 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe 0.25, Decay 85.2% |
