# 전략 스코어카드: MAX-MIN

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | MAX-MIN (`max-min`) |
| **유형** | 하이브리드 |
| **타임프레임** | 1D |
| **상태** | `폐기` (Gate 4 FAIL) |
| **Best Asset** | DOGE/USDT (Sharpe 0.82) |
| **2nd Asset** | BNB/USDT (Sharpe 0.73) |
| **경제적 논거** | 신고가 돌파는 추세 지속을, 신저가 매수는 과매도 반등을 포착하여 시장 레짐에 적응한다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **DOGE/USDT** | **0.82** | 15.28% | -13.86% | 181 | 1.96 |
| 2 | BNB/USDT | 0.73 | 6.28% | -11.71% | 226 | 1.48 |
| 3 | ETH/USDT | 0.70 | 5.08% | -15.17% | 230 | 1.38 |
| 4 | BTC/USDT | 0.59 | 4.96% | -14.91% | 227 | 1.35 |
| 5 | SOL/USDT | 0.55 | 4.73% | -13.41% | 209 | 1.36 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.82 | > 1.0 | FAIL |
| MDD | -13.86% | < 40% | PASS |
| Trades | 181 | > 50 | PASS |
| Win Rate | 50.83% | > 45% | — |
| Sortino | 0.97 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 22/30점
G1 백테스트  [WATCH] Sharpe 0.82, MDD 13.86%
G2 IS/OOS    [PASS] OOS Sharpe 0.80, Decay 6.2%
G3 파라미터  [PASS] 3/3 고원 + ±20% 안정
G4 심층검증  [FAIL] WFA OOS 0.47 (<0.5), Fold 2 OOS -0.34
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (PASS): IS Sharpe 0.85, OOS Sharpe 0.80, Decay 6.2%, OOS Return +10.4%

**Gate 3** (PASS): 3개 핵심 파라미터 모두 고원 존재 + ±20% Sharpe 부호 유지

- `lookback`: 고원 11개 (5~20), ±20% Sharpe 0.77~0.83 — 전 범위 안정
- `vol_target`: 고원 6개 (0.2~0.5), ±20% Sharpe 0.76~0.91
- `max_weight`: 고원 7개 (0.3~0.7), ±20% Sharpe 0.78~0.94 — 가중치에 둔감

**Gate 4** (FAIL): WFA OOS Sharpe 기준 미달, 최근 기간 음의 OOS

WFA (Walk-Forward, 5-fold expanding window):

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| OOS Sharpe (avg) | 0.47 | >= 0.5 | **FAIL** |
| Sharpe Decay | 38.4% | < 40% | PASS |
| Consistency | 66.7% | >= 60% | PASS |
| Overfit Probability | 36.4% | — | 보통 |

| Fold | IS Sharpe | OOS Sharpe | Decay |
|------|-----------|------------|-------|
| 0 | 0.779 | 0.768 | 1.4% |
| 1 | 0.740 | 0.970 | -31.2% |
| 2 | 0.748 | -0.342 | 145.7% |

CPCV (10-fold):

| 지표 | 값 |
|------|---|
| OOS Sharpe (avg) | 0.67 |
| Sharpe Decay | -9.4% |
| Consistency | 60% |
| MC p-value | 0.000 |

> 실패 사유: WFA OOS Sharpe 0.47 < 0.5 기준 미달. Fold 2 (최근 기간)에서 OOS Sharpe -0.34로 급락. CPCV에서는 OOS 0.67로 양호하나 WFA 기준 미충족.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 22/30점 |
| 2026-02-09 | G1 | WATCH | DOGE/USDT Sharpe 0.82 |
| 2026-02-09 | G2 | PASS | OOS Sharpe 0.80, Decay 6.2% |
| 2026-02-09 | G3 | PASS | 3/3 파라미터 고원+안정 |
| 2026-02-09 | G4 | FAIL | WFA OOS 0.47 (<0.5), Fold 2 OOS -0.34 |
