# 전략 스코어카드: Donchian Channel

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Donchian Channel (`donchian`) |
| **유형** | 추세추종 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 1.01) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.98) |
| **경제적 논거** | 신고가/신저가 돌파는 추세 시작의 강한 시그널이며, 터틀 트레이딩으로 검증되었다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **SOL/USDT** | **1.01** | 30.50% | -50.01% | 97 | 2.18 |
| 2 | DOGE/USDT | 0.98 | 35.73% | -42.73% | 126 | 2.25 |
| 3 | BNB/USDT | 0.93 | 29.14% | -34.55% | 147 | 2.21 |
| 4 | ETH/USDT | 0.88 | 25.17% | -38.38% | 157 | 1.63 |
| 5 | BTC/USDT | 0.78 | 20.77% | -47.70% | 134 | 1.64 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.01 | > 1.0 | PASS |
| MDD | -50.01% | < 40% | FAIL |
| Trades | 97 | > 50 | PASS |
| Win Rate | 70.10% | > 45% | — |
| Sortino | 0.98 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 25/30점
G1 백테스트  [PASS] Sharpe 1.01, MDD 50.01%
G2 IS/OOS    [FAIL] OOS Sharpe 0.12, Decay 91.1%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (FAIL): IS Sharpe 1.30, OOS Sharpe 0.12, Decay 91.1%

- 실패 사유: OOS Sharpe (0.12) < 0.3; Sharpe Decay (91.1%) >= 50%; OOS Return (-0.6%) <= 0%

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 25/30점 |
| 2026-02-09 | G1 | PASS | SOL/USDT Sharpe 1.01 |
| 2026-02-09 | G2 | FAIL | OOS Sharpe 0.12, Decay 91.1% |
