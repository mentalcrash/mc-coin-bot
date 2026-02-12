# 전략 스코어카드: Donchian Ensemble

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Donchian Ensemble (`donchian-ensemble`) |
| **유형** | 추세추종 |
| **타임프레임** | 1D |
| **상태** | `폐기` |
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

### 벤치마크 비교 (vs Buy & Hold)

| | Return | CAGR | MDD | Sharpe | Calmar | Beta |
|---|---:|---:|---:|---:|---:|---:|
| **Donchian Ensemble** | **+85.0%** | **+10.8%** | **-9.7%** | **0.99** | **1.11** | 0.04 |
| ETH Buy & Hold | +2,172% | +68.3% | -79% | 1.05 | 0.86 | — |
| BTC Buy & Hold | +1,117% | +51.7% | -77% | 0.99 | 0.67 | 1.00 |

- **Jensen Alpha** (vs BTC B&H): **+8.7%** (CAGR 10.8% - 0.04 x 51.7%)
- **Beta**: 0.04 (시장 거의 무관, 독립적 수익원)
- **Calmar**: 1.11 vs BTC 0.67 — **MDD 대비 수익 효율 67% 우위**
- **MDD**: -9.7% vs ETH -79% — **낙폭 1/8 수준**

> 절대 수익은 B&H 대비 현저히 낮으나, Beta 0.04로 시장 방향과 거의 무관한 독립 alpha.
> 2022년 하락장(-79% ETH)에서도 MDD -10% 이내를 유지하는 방어적 전략.

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 25/30점
G1 백테스트  [FAIL] CAGR 10.8% < 20% 최소 기준
G2 IS/OOS    [PASS] OOS Sharpe 0.99, Decay 1.7%
G3 파라미터  [PASS] 2/2 고원 + ±20% 안정
G4 심층검증  [PASS] WFA OOS 0.81, Decay 13%, Consistency 100% / CPCV OOS 1.02
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (PASS): IS Sharpe 1.01, OOS Sharpe 0.99, Decay 1.7%, OOS Return +13.4%

**Gate 3** (PASS): 2개 핵심 파라미터 모두 고원 존재 + ±20% Sharpe 부호 유지

- `vol_target`: 고원 10개 (0.2~0.55), ±20% Sharpe 0.95~0.99 — 전 범위 안정
- `atr_period`: 고원 10개 (8~25), ±20% Sharpe 0.95~1.00 — 전 범위 안정

**Gate 4** (PASS): WFA + CPCV 심층 검증 통과

WFA (Walk-Forward, 5-fold expanding window):

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| OOS Sharpe (avg) | 0.81 | >= 0.5 | PASS |
| Sharpe Decay | 13.1% | < 40% | PASS |
| Consistency | 100% | >= 60% | PASS |
| Overfit Probability | 7.8% | — | 양호 |

| Fold | IS Sharpe | OOS Sharpe | Decay |
|------|-----------|------------|-------|
| 0 | 0.990 | 0.827 | 16.5% |
| 1 | 0.923 | 0.989 | -7.2% |
| 2 | 0.874 | 0.607 | 30.5% |

CPCV (10-fold Combinatorial Purged Cross-Validation):

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| OOS Sharpe (avg) | 1.02 | — | 우수 (IS 0.92 초과) |
| Sharpe Decay | -10.5% | — | 역감쇠 (OOS > IS) |
| Consistency | 70% | — | 양호 |
| MC p-value | 0.000 | < 0.05 | PASS |
| MC 95% CI | [0.75, 1.25] | — | 하한 > 0 |

> 특기사항: CPCV OOS Sharpe(1.02)가 IS(0.92)보다 높음 — 과적합 징후 없음. 3개 Fold 모두 양의 OOS, 10-fold 중 7개 양의 OOS.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 25/30점 |
| 2026-02-09 | G1 | WATCH | ETH/USDT Sharpe 0.99 |
| 2026-02-09 | G2 | PASS | OOS Sharpe 0.99, Decay 1.7% |
| 2026-02-09 | G3 | PASS | 2/2 파라미터 고원+안정 |
| 2026-02-09 | G4 | PASS | WFA Decay 13%+Consistency 100%, CPCV OOS 1.02 |
| 2026-02-09 | G1 | FAIL | CAGR +10.8% < 20% 최소 기준 미달 (기준 변경에 의한 소급 적용) |
