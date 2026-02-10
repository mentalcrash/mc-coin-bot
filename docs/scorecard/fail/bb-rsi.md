# 전략 스코어카드: BB-RSI

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | BB-RSI (`bb-rsi`) |
| **유형** | 평균회귀 |
| **타임프레임** | 1D |
| **상태** | `폐기` |
| **Best Asset** | SOL/USDT (Sharpe 0.53) |
| **2nd Asset** | ETH/USDT (Sharpe 0.30) |
| **경제적 논거** | 볼린저밴드 상/하한에서의 가격 복귀는 과매수/과매도 심리의 회복에 의해 발생한다. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF |
|------|------|--------|------|-----|--------|------|
| **1** | **SOL/USDT** | **0.53** | 4.12% | -15.07% | 196 | 1.53 |
| 2 | ETH/USDT | 0.30 | 1.35% | -14.86% | 218 | 1.27 |
| 3 | BNB/USDT | 0.09 | 0.43% | -27.72% | 257 | 1.13 |
| 4 | BTC/USDT | 0.08 | 0.29% | -12.26% | 270 | 1.11 |
| 5 | DOGE/USDT | -0.10 | -0.72% | -19.56% | 205 | 0.96 |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.53 | > 1.0 | FAIL |
| MDD | -15.07% | < 40% | PASS |
| Trades | 196 | > 50 | PASS |
| Win Rate | 51.79% | > 45% | — |
| Sortino | 0.55 | > 1.5 | — |

### 벤치마크 비교 (vs Buy & Hold)

| | Return | CAGR | MDD | Sharpe | Calmar | Beta |
|---|---:|---:|---:|---:|---:|---:|
| **BB-RSI** | **+30.7%** | **+4.6%** | **-14.0%** | **0.59** | **0.33** | 0.05 |
| SOL Buy & Hold | +4,274% | +87.7% | -96% | 1.12 | 0.91 | — |
| BTC Buy & Hold | +1,117% | +51.7% | -77% | 0.99 | 0.67 | 1.00 |

- **Jensen Alpha** (vs BTC B&H): **+2.0%** (CAGR 4.6% - 0.05 x 51.7%)
- **Beta**: 0.05 (시장 거의 무관, 독립적 수익원)
- **MDD**: -14% vs SOL -96% — **낙폭 1/7 수준**

> 절대 수익은 B&H 대비 현저히 낮으나, Beta 0.05로 시장 방향과 거의 무관한 독립 alpha.
> 평균회귀 전략으로서 추세추종(Donchian)과 낮은 상관 기대 — 포트폴리오 분산 효과.
> 다만 Sharpe 0.59, CAGR 4.6%는 단독 운용 시 매력 부족. 앙상블 편입 시 가치.

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 23/30점
G1 백테스트  [FAIL] CAGR 4.6% < 20% 최소 기준
G2 IS/OOS    [PASS] OOS Sharpe 0.59, Decay 2.2%
G3 파라미터  [PASS] 4/4 고원 + ±20% 안정
G4 심층검증  [PASS] WFA OOS 0.82, Decay 2%, Consistency 67% / CPCV OOS 0.52
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

**Gate 2** (PASS): IS Sharpe 0.61, OOS Sharpe 0.59, Decay 2.2%, OOS Return +5.3%

**Gate 3** (PASS): 4개 핵심 파라미터 모두 고원 존재 + ±20% Sharpe 부호 유지
- `bb_period`: 고원 6개 (16~30), ±20% Sharpe 0.43~0.55
- `rsi_period`: 고원 8개 (8~20), ±20% Sharpe 0.52~0.56 — 매우 안정
- `vol_target`: 고원 7개 (0.1~0.3), ±20% Sharpe 0.52~0.55
- `bb_weight`: 고원 6개 (0.3~0.8), ±20% Sharpe 0.48~0.54 — 가중치에 둔감

**Gate 4** (PASS): WFA + CPCV 심층 검증 통과

WFA (Walk-Forward, 5-fold expanding window):

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| OOS Sharpe (avg) | 0.82 | >= 0.5 | PASS |
| Sharpe Decay | 1.6% | < 40% | PASS |
| Consistency | 66.7% | >= 60% | PASS |
| Overfit Probability | 14.3% | — | 양호 |

| Fold | IS Sharpe | OOS Sharpe | Decay |
|------|-----------|------------|-------|
| 0 | 0.823 | 1.237 | -50.3% |
| 1 | 0.838 | 0.835 | 0.4% |
| 2 | 0.835 | 0.386 | 53.8% |

CPCV (10-fold Combinatorial Purged Cross-Validation):

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| OOS Sharpe (avg) | 0.52 | — | 양호 |
| Sharpe Decay | 15.5% | — | 낮은 감쇠 |
| Consistency | 70% | — | 양호 |
| MC p-value | 0.001 | < 0.05 | PASS |
| MC 95% CI | [0.17, 0.83] | — | 하한 > 0 |

> 특기사항: WFA Decay 1.6%로 4개 전략 중 가장 낮음 — IS/OOS 거의 동일 성과. CPCV 10-fold 중 7개 양의 OOS. 평균회귀 전략으로서 안정적.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-09 | G0 | PASS | 23/30점 |
| 2026-02-09 | G1 | WATCH | SOL/USDT Sharpe 0.53 |
| 2026-02-09 | G2 | PASS | OOS Sharpe 0.59, Decay 2.2% |
| 2026-02-09 | G3 | PASS | 4/4 파라미터 고원+안정 |
| 2026-02-09 | G4 | PASS | WFA Decay 1.6%+OOS 0.82, CPCV OOS 0.52+MC p<0.01 |
| 2026-02-09 | G1 | FAIL | CAGR +4.6% < 20% 최소 기준 미달 (기준 변경에 의한 소급 적용) |
