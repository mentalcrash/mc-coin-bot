# 전략 스코어카드: Kalman-Trend

> 자동 생성 | 평가 기준: [dashboard.md](../../strategy/dashboard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Kalman-Trend (`kalman-trend`) |
| **유형** | 통계필터링 / 추세추종 |
| **타임프레임** | 1D |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | SOL/USDT (Sharpe 0.78) |
| **2nd Asset** | BNB/USDT (Sharpe 0.61) |
| **경제적 논거** | 칼만 필터로 가격에서 노이즈를 베이지안 최적으로 분리. Velocity(1st derivative) 기반 추세 감지. Adaptive Q로 변동성 레짐에 자동 적응. 고정 lookback MA 대비 lag 감소, false signal 60% 필터링. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **SOL/USDT** | **0.78** | +19.4% | -30.0% | 138 | 1.36 | -4052.8% | 0.03 |
| 2 | BNB/USDT | 0.61 | +16.1% | -57.5% | 222 | 1.30 | -5964.8% | 0.11 |
| 3 | ETH/USDT | 0.39 | +7.9% | -41.7% | 273 | 1.14 | -2057.8% | 0.08 |
| 4 | BTC/USDT | 0.18 | +0.1% | -64.5% | 281 | 1.09 | -1077.5% | 0.09 |
| 5 | DOGE/USDT | 0.00 | 0.0% | 0.0% | 0 | N/A | 0.0% | N/A |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.78 | > 1.0 | **FAIL** |
| CAGR | +19.4% | > 20% | **FAIL** |
| MDD | -30.0% | < 40% | PASS |
| Trades | 138 | > 50 | PASS |
| Win Rate | 45.7% | — | — |
| Sortino | 0.98 | — | — |
| Calmar | 0.65 | — | — |
| Profit Factor | 1.36 | — | — |
| Alpha (vs BTC B&H) | -4052.8% | — | — |
| Beta (vs BTC) | 0.03 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 24/30점
G0B 코드감사  [PASS] 7/7 Critical 결함 0개
G1  백테스트  [FAIL] SOL/USDT Sharpe 0.78, CAGR +19.4%, MDD -30.0%
G2  IS/OOS   [    ] — (G1 FAIL로 미진행)
G3  파라미터  [    ] — (G1 FAIL로 미진행)
G4  심층검증  [    ] — (G1 FAIL로 미진행)
G5  EDA검증  [    ] — (G1 FAIL로 미진행)
G6  모의거래  [    ] — (G1 FAIL로 미진행)
G7  실전배포  [    ] — (G1 FAIL로 미진행)
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

#### G0B 코드 감사 (2026-02-10)

- Critical C1-C7 결함 0개 PASS
- W1 (Kalman filter loop 예외 인정), W2 (params=5, loop 정당성 문서화 완료)

#### Gate 1 상세 (5코인 x 6년 백테스트)

**판정**: **FAIL** — 전 에셋 Sharpe < 1.0, Best CAGR +19.4% < 20%

**FAIL 사유**:

1. **Sharpe 전 에셋 미달**: Best 0.78 (SOL), 전 에셋 0.00~0.78 범위. 기준 1.0 미달
2. **CAGR 경계 미달**: Best +19.4% (SOL) — 20% 기준에 0.6%p 미달. 근소하나 규정상 FAIL
3. **DOGE 거래 0건**: vel_threshold 0.5가 DOGE의 1D 데이터에서 도달 불가 — 완전한 신호 부재
4. **BTC MDD -64.5%**: 기준(40%) 대폭 초과, 칼만 필터의 lag가 급락에 무방비

**퀀트 해석**:

- SOL에서 MDD -30.0%은 양호하나, Sharpe 0.78은 CTREND 2.05 대비 38% 수준
- SOL/BNB에서 부분적 edge 존재 (높은 변동성 에셋에서 Kalman smoothing 효과)
- BTC/ETH에서 edge 미약 — 추세가 약하거나 whipsaw 구간에서 velocity 신호 반복 역전
- DOGE 거래 0건은 vel_threshold 파라미터가 에셋별 변동성에 미적응한 구조적 결함
- HEDGE_ONLY 모드 — 2022 약세장 Short 기회 제한. FULL이었으면 SOL Sharpe 개선 가능성
- 교훈: 칼만 필터의 학술적 최적성이 크립토 1D 백테스트에서 MA 대비 뚜렷한 우위 없음. velocity 신호의 threshold 설정이 에셋 변동성에 민감

**CTREND 비교**:

| 지표 | CTREND Best (SOL) | Kalman-Trend Best (SOL) |
|------|-------------------|------------------------|
| Sharpe | 2.05 | 0.78 |
| CAGR | +97.8% | +19.4% |
| MDD | -27.7% | -30.0% |
| Trades | 288 | 138 |

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 — 경제적 논거 4, 참신성 4, 데이터 5, 구현 3, 용량 4, 레짐독립 4 |
| 2026-02-10 | G0B | PASS | 7/7 Critical 결함 0개, W1 (Kalman loop 예외), W2 (params=5) |
| 2026-02-10 | G1 | **FAIL** | 전 에셋 Sharpe < 1.0 (Best 0.78 SOL). CAGR +19.4% < 20%. DOGE 거래 0건. BTC MDD -64.5% |
