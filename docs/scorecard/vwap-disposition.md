# 전략 스코어카드: VWAP-Disposition

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | VWAP-Disposition (`vwap-disposition`) |
| **유형** | 행동재무학 |
| **타임프레임** | 4H |
| **상태** | `검증중` |
| **Best Asset** | PENDING |
| **2nd Asset** | PENDING |
| **경제적 논거** | Rolling VWAP를 시장 참여자의 평균 취득가(cost basis)로 사용. Capital Gains Overhang(CGO)에 따른 disposition effect로 매도/매수 압력 예측. BTC에서 2017년 이후 disposition effect 유의미 증가 확인. |

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
G0A 아이디어  [PASS] 23/30점
G0B 코드감사  [    ]
G1  백테스트  [    ]
G2  IS/OOS   [    ]
G3  파라미터  [    ]
G4  심층검증  [    ]
G5  EDA검증  [    ]
G6  모의거래  [    ]
G7  실전배포  [    ]
```

### Gate 상세

#### G0A 아이디어 검증 (2026-02-10)

| 항목 | 점수 |
|------|:----:|
| 경제적 논거 | 4/5 |
| 참신성 | 5/5 |
| 데이터 확보 | 4/5 |
| 구현 복잡도 | 4/5 |
| 용량 수용 | 3/5 |
| 레짐 독립성 | 3/5 |
| **합계** | **23/30** |

**핵심 근거:**
- 행동재무학 카테고리 완전 미탐색 — 기존 전략과 메커니즘 근본적 차이
- BTC에서 disposition effect 유의미 실증 (Schatzmann 2023, Digital Finance)
- On-chain MVRV > 3.5 = 고점, < 1.0 = 저점 패턴을 VWAP proxy로 OHLCV 구현
- CGO 극단값 + Volume 확인으로 capitulation/profit-taking 포착
- ShortMode FULL — 차익실현 압력 SHORT이 핵심 에지

> **다음 단계**: G0B 코드 감사

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 23/30점 — 경제적 논거 4, 참신성 5, 데이터 4, 구현 4, 용량 3, 레짐독립 3 |
