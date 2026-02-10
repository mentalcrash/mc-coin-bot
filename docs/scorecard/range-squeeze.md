# 전략 스코어카드: Range-Squeeze

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Range-Squeeze (`range-squeeze`) |
| **유형** | 변동성/돌파 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | PENDING |
| **2nd Asset** | PENDING |
| **경제적 논거** | NR7 패턴 + range ratio squeeze 후 vol compression이 해소되며 발생하는 방향성 breakout을 포착. Crypto의 높은 vol로 squeeze-breakout 사이클이 뚜렷. |

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
G0A 아이디어  [PASS] 24/30점
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
| 시그널 로직 | 9/10 |
| 실행 현실성 | 9/10 |
| 리스크 관리 | 8/10 |
| 코드 품질 | 9/10 |

**수정 완료된 이슈:**
- [H-001] ~~NR 패턴 floating-point `==` 비교~~ → `np.isclose(rtol=1e-9)` 적용 완료
- [H-002] ~~`daily_range` dirty data 가드 없음~~ → `.clip(lower=0)` 적용 완료
- [H-003] vol_scalar 무제한 — PM 클램핑에 의존 (전략 레벨 추가 방어 불요, 아키텍처 설계대로)
- [추가] HEDGE_ONLY drawdown shift(1) 미적용 → `df["drawdown"].shift(1)` 적용 완료

**잘된 점:**
- shift(1) 규칙 올바르게 적용 (is_nr, range_ratio, vol_scalar, drawdown 모두 shift)
- NR7 + range_ratio squeeze 이중 감지 로직
- avg_range `.replace(0, np.nan)` 0 나눗셈 방어
- 벡터화 연산, 깔끔한 4파일 분리 구조

> **다음 단계**: G1 단일에셋 백테스트 (5코인 x 6년)

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 — 경제적 논거 4, 참신성 3, 데이터 5, 구현 5, 용량 4, 레짐독립 3 |
| 2026-02-10 | G0B | PASS | Critical 0개. HIGH 3건 발견 → 2건 수정 완료 (isclose, clip), 1건 아키텍처 의존 (vol_scalar PM 클램핑) |
