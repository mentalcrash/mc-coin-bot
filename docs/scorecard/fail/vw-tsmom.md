# 전략 스코어카드: VW-TSMOM

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | VW-TSMOM (`vw-tsmom`) |
| **유형** | 추세추종 (Volume-weighted momentum) |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | SOL/USDT (Sharpe 0.91) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.87) |
| **3rd Asset** | ETH/USDT (Sharpe 0.60) |
| **경제적 논거** | 거래량 가중 모멘텀으로 가격 추세의 확신도를 반영, 노이즈 필터링 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **SOL/USDT** | **0.91** | +32.8% | -49.6% | 240 | 1.39 | -3704.2% | 0.11 |
| 2 | DOGE/USDT | 0.87 | +39.0% | -50.8% | 225 | 1.80 | -5158.3% | 0.02 |
| 3 | ETH/USDT | 0.60 | +16.3% | -57.6% | 243 | 1.23 | -1935.9% | 0.10 |
| 4 | BNB/USDT | 0.51 | +13.2% | -62.2% | 301 | 1.24 | -5982.5% | 0.15 |
| 5 | BTC/USDT | 0.38 | +7.4% | -61.0% | 243 | 1.16 | -1001.1% | 0.10 |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.91 | > 1.0 | **FAIL** |
| CAGR | +32.8% | > 20% | PASS |
| MDD | -49.6% | < 40% | **FAIL** |
| Trades | 240 | > 50 | PASS |
| Win Rate | 41.8% | > 45% | FAIL |
| Sortino | 1.45 | > 1.5 | FAIL |
| Calmar | 0.66 | > 1.0 | FAIL |
| Profit Factor | 1.39 | > 1.3 | PASS |
| Beta (vs BTC) | 0.11 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 21/30점
G1 백테스트  [WATCH] SOL/USDT Sharpe 0.91, CAGR +32.8%, MDD -49.6%
G2 IS/OOS    [FAIL] OOS Sharpe 0.12, Decay 92.1%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 2 상세 (IS/OOS 70/30, 5-coin EW Portfolio)

| 지표 | IS (70%) | OOS (30%) | 기준 | 판정 |
|------|----------|-----------|------|------|
| Sharpe | 1.57 | 0.12 | OOS > 0.3 | **FAIL** |
| Decay | — | 92.1% | < 50% | **FAIL** |
| Consistency | — | 0% | — | — |
| Overfit Prob | — | 95.3% | — | Very High |

**판정**: **FAIL**

- OOS Sharpe 0.12 << 0.3 (기준 대비 60% 미달)
- Decay 92.1% >> 50% (IS 성과의 92% 소멸)
- Overfit Probability 95.3% (사실상 확정 과적합)
- IS Sharpe 1.57이 높았으나 OOS에서 거의 전부 소멸 → 폐기 대상

### Gate 1 상세

- **전 에셋 WATCH 이하**: Best SOL도 Sharpe 0.91 (< 1.0), MDD 49.6% (> 40%)
- DOGE는 CAGR +39.0%로 높지만 MDD 50.8%
- **낮은 Win Rate (41.8%)**: 추세추종 전략으로 허용 범위이나 보조 기준 미달
- **높은 MDD**: 전 에셋 49~62% — PM 파라미터 조정 필요
- Gate 2 진행 보류: PM 최적화 후 재평가 권고

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 21/30점 |
| 2026-02-10 | G1 | WATCH | SOL/USDT Sharpe 0.91, MDD -49.6% (Sharpe/MDD 기준 미달) |
| 2026-02-10 | G2 | FAIL | OOS Sharpe 0.12, Decay 92.1%, Overfit Prob 95.3% |
