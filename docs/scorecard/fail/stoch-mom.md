# 전략 스코어카드: Stochastic Momentum

> 자동 생성 | 평가 기준: [dashboard.md](../../strategy/dashboard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Stochastic Momentum (`stoch-mom`) |
| **유형** | 하이브리드 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 0.94) |
| **2nd Asset** | BTC/USDT (Sharpe 0.93) |
| **경제적 논거** | 스토캐스틱 크로스오버는 단기 모멘텀 변화를, SMA 필터는 전체 추세 방향을 확인한다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **SOL/USDT** | **0.94** | 7.05% | -8.99% | 176 | 1.63 |
| 2 | BTC/USDT | 0.93 | 6.58% | -12.74% | 247 | 1.59 |
| 3 | DOGE/USDT | 0.76 | 9.52% | -12.49% | 187 | 1.80 |
| 4 | ETH/USDT | 0.75 | 4.49% | -13.13% | 227 | 1.42 |
| 5 | BNB/USDT | 0.15 | 0.84% | -12.23% | 250 | 1.10 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.94 | > 1.0 | FAIL |
| MDD | -8.99% | < 40% | PASS |
| Trades | 176 | > 50 | PASS |
| Win Rate | 45.45% | > 45% | — |
| Sortino | 0.89 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 21/30점
G1 백테스트  [WATCH] Sharpe 0.94, MDD 8.99%
G2 IS/OOS    [FAIL] OOS Sharpe -0.34, Decay 124.9%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 1.37, OOS Sharpe -0.34, Decay 124.9%

- 실패 사유: OOS Sharpe (-0.34) < 0.3; Sharpe Decay (124.9%) >= 50%; OOS Return (-3.4%) <= 0%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 21/30점 |
| 2026-02-09 | G1 | WATCH | SOL/USDT Sharpe 0.94 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe -0.34, Decay 124.9% |
