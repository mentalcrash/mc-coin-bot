# 전략 스코어카드: GK Breakout

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | GK Breakout (`gk-breakout`) |
| **유형** | 변동성 돌파 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | DOGE/USDT (Sharpe 0.77) |
| **2nd Asset** | BNB/USDT (Sharpe 0.59) |
| **경제적 논거** | 변동성 압축(squeeze)은 에너지 축적을 의미하며, 해소 시 강한 방향성 움직임이 나타난다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **DOGE/USDT** | **0.77** | 13.12% | -23.15% | 53 | 2.17 |
| 2 | BNB/USDT | 0.59 | 6.74% | -20.59% | 47 | 1.78 |
| 3 | BTC/USDT | 0.25 | 2.37% | -24.36% | 54 | 1.25 |
| 4 | ETH/USDT | 0.24 | 2.07% | -17.52% | 53 | 1.23 |
| 5 | SOL/USDT | -0.04 | -0.74% | -23.15% | 44 | 0.96 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.77 | > 1.0 | FAIL |
| MDD | -23.15% | < 40% | PASS |
| Trades | 53 | > 50 | PASS |
| Win Rate | 54.72% | > 45% | — |
| Sortino | 0.85 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 23/30점
G1 백테스트  [WATCH] Sharpe 0.77, MDD 23.15%
G2 IS/OOS    [FAIL] OOS Sharpe 0.39, Decay 59.0%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 0.96, OOS Sharpe 0.39, Decay 59.0%

- 실패 사유: Sharpe Decay (59.0%) >= 50%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 23/30점 |
| 2026-02-09 | G1 | WATCH | DOGE/USDT Sharpe 0.77 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe 0.39, Decay 59.0% |
