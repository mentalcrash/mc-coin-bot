# 전략 스코어카드: VPIN-Flow

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | VPIN-Flow (`vpin-flow`) |
| **유형** | 마이크로스트럭처 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | PENDING |
| **2nd Asset** | PENDING |
| **경제적 논거** | BVC로 buy/sell volume을 근사하고 VPIN으로 정보거래 확률을 측정. 고독성(high VPIN) 시 informed trading 방향을 추종하여 대형 가격 변동을 사전 포착. |

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
G0A 아이디어  [PASS] 22/30점
G0B 코드감사  [PASS] Critical 0 (수정완료), High 0, Medium 3
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

**종합 등급: A** (CRITICAL/HIGH 이슈 전수 수정 완료)

| 항목 | 점수 |
|------|:----:|
| 데이터 무결성 | 10/10 |
| 시그널 로직 | 9/10 |
| 실행 현실성 | 9/10 |
| 리스크 관리 | 9/10 |
| 코드 품질 | 9/10 |

**수정 완료된 이슈:**
- [C-001] ~~HEDGE_ONLY drawdown shift(1) 미적용~~ → `df["drawdown"].shift(1)` 적용 완료 (look-ahead bias 제거)
- [H-001] Hedge 모드 `suppress_short`과 `active_short` 상호 배타성 — 로직상 보장됨 (short_mask & ~hedge = suppress, short_mask & hedge = active)
- [H-002] shift(1) 회귀 테스트 — 기존 테스트에서 shift 적용 검증 커버

**잘된 점:**
- 주요 indicator (vpin, flow_direction, vol_scalar, drawdown) shift(1) 전수 적용
- BVC `high - low + 1e-10` epsilon 방어
- VPIN [0,1] 범위 보장 (|imbalance|/volume)
- norm.cdf BVC → buy_pct [0,1] 보장

> **다음 단계**: G1 단일에셋 백테스트 (5코인 x 6년)

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 22/30점 — 경제적 논거 4, 참신성 5, 데이터 3, 구현 3, 용량 3, 레짐독립 4 |
| 2026-02-10 | G0B | ~~FAIL~~ → **PASS** | Critical 1건 발견 → 수정 완료 (drawdown shift(1) 적용). 재감사 통과 |
