# 전략 스코어카드: Kalman-Trend

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Kalman-Trend (`kalman-trend`) |
| **유형** | 통계필터링 / 추세추종 |
| **타임프레임** | 4H |
| **상태** | `검증중` |
| **Best Asset** | PENDING |
| **2nd Asset** | PENDING |
| **경제적 논거** | 칼만 필터로 가격에서 노이즈를 베이지안 최적으로 분리. Velocity(1st derivative) 기반 추세 감지. Adaptive Q로 변동성 레짐에 자동 적응. 고정 lookback MA 대비 lag 감소, false signal 60% 필터링. |

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
| 참신성 | 4/5 |
| 데이터 확보 | 5/5 |
| 구현 복잡도 | 3/5 |
| 용량 수용 | 4/5 |
| 레짐 독립성 | 4/5 |
| **합계** | **24/30** |

**핵심 근거:**
- 베이지안 최적 추정기 — 고정 lookback MA와 달리 자동 노이즈 적응
- Velocity > 0이면 상승 추세, < 0이면 하락 추세 — 직관적 해석
- MA 대비 lag 감소, profit factor 개선 학술 확인 (PyQuantLab, 2025)
- 4H가 크립토의 "equilibrium zone" (arXiv 2601.06084)
- Adaptive Q = base_Q * (realized_vol / long_term_vol) — 과적합 여지 최소화

> **다음 단계**: G0B 코드 감사

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 — 경제적 논거 4, 참신성 4, 데이터 5, 구현 3, 용량 4, 레짐독립 4 |
