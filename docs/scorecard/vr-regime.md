# 전략 스코어카드: VR-Regime

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | VR-Regime (`vr-regime`) |
| **유형** | 레짐전환 |
| **타임프레임** | 1D |
| **상태** | `검증중` |
| **Best Asset** | PENDING |
| **2nd Asset** | PENDING |
| **경제적 논거** | Lo-MacKinlay Variance Ratio로 random walk hypothesis를 검정. VR > 1 → trending (momentum), VR < 1 → mean-reverting (contrarian). Non-parametric test 기반. |

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
| 리스크 관리 | 9/10 |
| 코드 품질 | 9/10 |

**수정 완료된 이슈:**
- [H-001] ~~`warmup_periods()` chained rolling 미반영~~ → `vr_window + vr_k` 반영 완료
- [H-002] ~~`use_heteroscedastic` 라벨 오해~~ → 주석 수정: "Lo-MacKinlay (1988) Eq.5 simplified form, NOT full Eq.10" 명시
- [추가] HEDGE_ONLY drawdown shift(1) 미적용 → `df["drawdown"].shift(1)` 적용 완료

**잘된 점:**
- shift(1) 규칙 정확 (vr, z_stat, mom_direction, vol_scalar, drawdown 모두 shift)
- theta_safe = `np.maximum(theta, 1e-10)` z-stat 0 나눗셈 방어
- VR 계산 시 `var_1.replace(0, np.nan)` 0 나눗셈 방어
- cross-field validation: `vr_k * 2 < vr_window`

> **다음 단계**: G1 단일에셋 백테스트 (5코인 x 6년)

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 — 경제적 논거 4, 참신성 4, 데이터 5, 구현 4, 용량 3, 레짐독립 4 |
| 2026-02-10 | G0B | PASS | Critical 0개. HIGH 2건 발견 → 전수 수정 완료 (warmup vr_k 반영, 주석 정정, drawdown shift) |
