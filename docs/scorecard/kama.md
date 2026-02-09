# 전략 스코어카드: KAMA

> 자동 생성 | 평가 기준: [strategy-evaluation-standard.md](../strategy-evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | KAMA (`kama`) |
| **유형** | 추세추종 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | DOGE/USDT (Sharpe 1.14) |
| **2nd Asset** | ETH/USDT (Sharpe 0.79) |
| **경제적 논거** | 적응형 이동평균은 추세 시장에서는 빠르게, 횡보 시장에서는 느리게 반응하여 whipsaw를 줄인다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **DOGE/USDT** | **1.14** | 35.82% | -13.25% | 117 | 2.72 |
| 2 | ETH/USDT | 0.79 | 11.33% | -16.33% | 164 | 1.51 |
| 3 | SOL/USDT | 0.75 | 8.93% | -22.00% | 150 | 1.46 |
| 4 | BTC/USDT | 0.55 | 7.44% | -22.20% | 145 | 1.37 |
| 5 | BNB/USDT | 0.54 | 7.57% | -20.12% | 141 | 1.35 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.14 | > 1.0 | PASS |
| MDD | -13.25% | < 40% | PASS |
| Trades | 117 | > 50 | PASS |
| Win Rate | 54.31% | > 45% | — |
| Sortino | 2.02 | > 1.5 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 22/30점
G1 백테스트  [PASS] Sharpe 1.14, MDD 13.25%
G2 IS/OOS    [PASS] OOS Sharpe 1.01, Decay 21.8%
G3 파라미터  [PASS] 3/3 고원 + ±20% 안정
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (PASS): IS Sharpe 1.30, OOS Sharpe 1.01, Decay 21.8%, OOS Return +31.8%

**Gate 3** (PASS): 3개 핵심 파라미터 모두 고원 존재 + ±20% Sharpe 부호 유지
- `er_lookback`: 고원 7개 (5~12), ±20% Sharpe 0.97~1.15
- `slow_period`: 고원 10개 (20~50), ±20% Sharpe 1.10~1.16 — 매우 안정
- `vol_target`: 고원 8개 (0.2~0.5), ±20% Sharpe 1.05~1.14

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 22/30점 |
| 2026-02-09 | G1 | PASS | DOGE/USDT Sharpe 1.14 |
| 2026-02-09 | G2 | PASS | OOS Sharpe 1.01, Decay 21.8% |
| 2026-02-09 | G3 | PASS | 3/3 파라미터 고원+안정 |
