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
G0A 아이디어  [PASS] 27/30점
G0B 코드감사  [PASS] Critical 0, High 0 (수정완료), Medium 4
G1  백테스트  [    ]
G2  IS/OOS   [    ]
G3  파라미터  [    ]
G4  심층검증  [    ]
G5  EDA검증  [    ]
G6  모의거래  [    ]
G7  실전배포  [    ]
```

### Gate 상세

#### G0B 코드 감사 (2026-02-10)

**종합 등급: A** (HIGH 이슈 전수 수정 완료)

| 항목 | 점수 |
|------|:----:|
| 데이터 무결성 | 10/10 |
| 시그널 로직 | 10/10 |
| 실행 현실성 | 9/10 |
| 리스크 관리 | 9/10 |
| 코드 품질 | 9/10 |

**수정 완료된 이슈:**
- [H-001] ~~AC 공식 `var(x)` 단일 분모~~ → `sqrt(var(x)*var(x_lag))` + `.clip(-1.0, 1.0)` 적용 완료
- [H-002] ~~HEDGE_ONLY drawdown shift(1) 미적용~~ → `df["drawdown"].shift(1)` 적용 완료
- [M-002] ~~warmup_periods에 ac_lag 미반영~~ → `ac_window + ac_lag` 반영 완료

**잘된 점:**
- shift(1) 규칙 준수 (ac_rho, sig_bound, mom_direction, vol_scalar, drawdown 모두 shift)
- Pearson autocorrelation 정확 구현 (sqrt 분모 + clip)
- 벡터화 연산, 루프 없음
- Bartlett significance bound로 통계적 필터링
- NaN fillna(0) 처리, 0 나눗셈 방어

> **다음 단계**: G1 단일에셋 백테스트 (5코인 x 6년)

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 27/30점 — 경제적 논거 4, 참신성 5, 데이터 5, 구현 4, 용량 4, 레짐독립 5 |
| 2026-02-10 | G0B | PASS | Critical 0개. HIGH 2건 발견 → 전수 수정 완료 (AC sqrt 분모, drawdown shift, warmup fix) |
