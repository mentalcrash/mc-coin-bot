# 전략 스코어카드: AC-Regime

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | AC-Regime (`ac-regime`) |
| **유형** | 레짐전환 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | PENDING |
| **2nd Asset** | PENDING |
| **경제적 논거** | Returns의 serial correlation 부호로 regime을 분류. 양수 AC → trending (정보 점진적 반영), 음수 AC → mean-reverting (과잉반응 후 복귀). |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| — | PENDING | — | — | — | — | — | — | — |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | — | > 1.0 | PENDING |
| MDD | — | < 40% | PENDING |
| Trades | — | > 50 | PENDING |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 27/30점
G1 백테스트  [    ]
G2 IS/OOS    [    ]
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
G6 모의거래  [    ]
G7 실전배포  [    ]
```

### Gate 상세 (완료된 Gate만 기록)

> Gate 1 이후의 상세 결과는 해당 Gate 완료 시 추가한다.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 27/30점 — 경제적 논거 4, 참신성 5, 데이터 5, 구현 4, 용량 4, 레짐독립 5 |
