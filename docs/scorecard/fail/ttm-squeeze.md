# 전략 스코어카드: TTM Squeeze

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | TTM Squeeze (`ttm-squeeze`) |
| **유형** | 변동성 돌파 |
| **타임프레임** | 1D |
| **상태** | `폐기` |
| **Best Asset** | BTC/USDT (Sharpe 0.94) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.83) |
| **경제적 논거** | BB가 KC 안에 들어가면 변동성 압축, 밖으로 나오면 폭발적 움직임이 시작된다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **BTC/USDT** | **0.94** | 17.67% | -37.67% | 43 | 2.97 |
| 2 | DOGE/USDT | 0.83 | 23.19% | -37.69% | 50 | 2.15 |
| 3 | BNB/USDT | 0.59 | 12.07% | -31.45% | 51 | 1.89 |
| 4 | SOL/USDT | 0.55 | 8.54% | -31.43% | 27 | 1.95 |
| 5 | ETH/USDT | 0.17 | 1.39% | -39.09% | 41 | 1.23 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.94 | > 1.0 | FAIL |
| MDD | -37.67% | < 40% | PASS |
| Trades | 43 | > 50 | FAIL |
| Win Rate | 72.09% | > 45% | — |
| Sortino | 0.57 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 23/30점
G1 백테스트  [WATCH] Sharpe 0.94, MDD 37.67%
G2 IS/OOS    [PASS] OOS Sharpe 0.58, Decay 45.6%
G3 파라미터  [FAIL] bb_period, kc_mult 고원 부재
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (PASS): IS Sharpe 1.06, OOS Sharpe 0.58, Decay 45.6%, OOS Return +12.1%

**Gate 3** (FAIL): bb_period, kc_mult 파라미터에서 고원 부재
- `bb_period`: 고원 2개만 (18~20), bb_period=22에서 Sharpe 0.94→0.53 급락, bb_period=30에서 음수
- `kc_mult`: 고원 2개만 (1.8~2.0), kc_mult=1.0에서 거래 0건, 1.2~1.3에서 Sharpe 0.14~0.19
- `vol_target`: PASS — 고원 10개 (0.2~0.55), ±20% Sharpe 0.93~0.95

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 23/30점 |
| 2026-02-09 | G1 | WATCH | BTC/USDT Sharpe 0.94 |
| 2026-02-09 | G2 | PASS | OOS Sharpe 0.58, Decay 45.6% |
| 2026-02-09 | G3 | FAIL | bb_period, kc_mult 고원 부재 — 폐기 |
