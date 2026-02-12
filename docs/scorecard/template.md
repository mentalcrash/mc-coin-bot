# 전략 스코어카드: {전략명}

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | {DisplayName} (`{name}`) |
| **유형** | 추세추종 / 평균회귀 / 레짐전환 / ... |
| **타임프레임** | 1D |
| **상태** | `검증중` / `배포` / `은퇴` / `폐기` |
| **Best Asset** | {SYMBOL} (Sharpe {X.XX}) |
| **2nd Asset** | {SYMBOL} (Sharpe {X.XX}) |
| **경제적 논거** | 1~2문장 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **{BEST}** | **X.XX** | X.X% | -X.X% | N | X.XX | X.X% | X.XX |
| 2 | {2ND} | X.XX | X.X% | -X.X% | N | X.XX | X.X% | X.XX |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | X.XX | > 1.0 | PASS/FAIL |
| MDD | -X.X% | < 40% | PASS/FAIL |
| Trades | N | > 50 | PASS/FAIL |
| Alpha (vs BTC B&H) | X.X% | > 0% | — |
| Beta (vs BTC) | X.XX | < 0.5 | — |
| Win Rate | X.X% | > 45% | — |
| Sortino | X.XX | > 1.5 | — |
| Calmar | X.XX | > 1.0 | — |
| Tail Ratio | X.XX | > 1.0 | — |
| Recovery Factor | X.XX | > 3.0 | — |
| Expectancy ($) | X.XX | > 0 | — |
| Avg Trade Duration | N bars | — | — |
| Max Consec. Losses | N | < 10 | — |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] XX/30점
G1 백테스트  [PASS] Sharpe X.XX, MDD X.X%
G2 IS/OOS    [    ] OOS Sharpe ____, Decay ___%
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

> Gate 2 이후의 상세 결과는 해당 Gate 완료 시 추가한다.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| YYYY-MM-DD | G0 | PASS | XX/30점 |
| YYYY-MM-DD | G1 | PASS | {SYMBOL} Sharpe X.XX |
