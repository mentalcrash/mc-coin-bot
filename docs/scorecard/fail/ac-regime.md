# 전략 스코어카드: AC-Regime

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | AC-Regime (`ac-regime`) |
| **유형** | 레짐전환 |
| **타임프레임** | 1D |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | ETH/USDT (Sharpe 0.08) |
| **2nd Asset** | SOL/USDT (Sharpe -0.04) |
| **경제적 논거** | Returns의 serial correlation 부호로 regime을 분류. 양수 AC → trending (정보 점진적 반영), 음수 AC → mean-reverting (과잉반응 후 복귀). |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| 1 | ETH/USDT | 0.08 | +0.3% | -25.9% | 28 | 1.09 | -2168.2% | 0.01 |
| 2 | SOL/USDT | -0.04 | -1.2% | -18.7% | 48 | 0.93 | -4283.2% | 0.02 |
| 3 | DOGE/USDT | -0.12 | -2.6% | -41.6% | 46 | 0.81 | -5997.0% | 0.01 |
| 4 | BTC/USDT | -0.39 | -5.1% | -42.4% | 34 | 0.54 | -1153.7% | 0.00 |
| 5 | BNB/USDT | -0.51 | -5.1% | -32.7% | 44 | 0.60 | -6210.7% | 0.01 |

### Best Asset 핵심 지표 (ETH/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.08 | > 1.0 | **FAIL** |
| CAGR | +0.3% | > 20% | **FAIL** |
| MDD | -25.9% | < 40% | PASS |
| Trades | 28 | > 50 | **FAIL** |
| Win Rate | 46.4% | > 45% | PASS |
| Sortino | 0.02 | > 1.5 | **FAIL** |
| Calmar | 0.01 | > 1.0 | **FAIL** |
| Profit Factor | 1.09 | > 1.3 | **FAIL** |
| Alpha (vs BTC B&H) | -2168.2% | > 0% | **FAIL** |
| Beta (vs BTC) | 0.01 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 27/30점
G0B 코드감사  [PASS] Critical 0, High 0 (수정완료), Medium 4
G1  백테스트  [FAIL] 전 에셋 Sharpe 음수/0 근접, Best ETH 0.08 << 1.0
G2  IS/OOS   [    ] (G1 FAIL로 미실행)
G3  파라미터  [    ] (G1 FAIL로 미실행)
G4  심층검증  [    ] (G1 FAIL로 미실행)
G5  EDA검증  [    ] (G1 FAIL로 미실행)
G6  모의거래  [    ] (G1 FAIL로 미실행)
G7  실전배포  [    ] (G1 FAIL로 미실행)
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

#### G1 단일에셋 백테스트 (2026-02-10)

**판정: FAIL** — 전 에셋 음수/0 근접 Sharpe. Best Asset(ETH) Sharpe 0.08, CAGR +0.3%.

**즉시 폐기 조건 해당**: 4/5 에셋 Sharpe 음수 (BTC -0.39, BNB -0.51, SOL -0.04, DOGE -0.12). 유일한 양수인 ETH도 0.08로 사실상 0.

**CTREND 비교**:

| 지표 | CTREND Best (SOL) | AC-Regime Best (ETH) | 차이 |
|------|-------------------|---------------------|------|
| Sharpe | 2.05 | 0.08 | -96% |
| CAGR | +97.8% | +0.3% | -100% |
| MDD | -27.7% | -25.9% | +6% |
| Trades | 288 | 28 | -90% |

**퀀트 해석**:

1. **AC(autocorrelation) 시그널 약함**: rolling autocorrelation은 returns의 serial dependence를 측정하나, 1D 크립토 시장에서 AC 부호 변환만으로는 유효한 트레이딩 시그널을 생성하지 못함. 전 에셋 Profit Factor < 1.1 (무작위 수준)

2. **거래 수 극소 (28~48)**: Bartlett significance bound가 시그널 대부분을 필터링하여 연간 5~8회 수준. 충분한 거래 표본 미확보

3. **Beta ~ 0 + Alpha 음수**: BTC 대비 완전 독립적이나(Beta 0.00~0.02), 독립성이 곧 알파를 의미하지 않음. BTC B&H 대비 전 에셋에서 크게 열등

4. **MDD는 양호하나 의미 없음**: 포지션 사이즈 자체가 작고(vol-target 보수적), 수익이 없으므로 MDD가 낮은 것은 단순히 "거래를 거의 안 했기 때문"

5. **에셋 순서 패턴**: ETH > SOL > DOGE > BTC > BNB. 전형적인 추세추종 패턴(SOL > BTC)과 다르며, 이는 전략이 추세를 잡지 못한다는 방증

6. **전략 한계**: 일봉 autocorrelation 기반 레짐 전환은 이론적 논거(information diffusion vs overreaction)가 명확하나, 실증적으로 크립토 시장의 AC 구조가 안정적이지 않아 백테스트에서 양의 기대값을 생성하지 못함

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 27/30점 — 경제적 논거 4, 참신성 5, 데이터 5, 구현 4, 용량 4, 레짐독립 5 |
| 2026-02-10 | G0B | PASS | Critical 0개. HIGH 2건 발견 → 전수 수정 완료 (AC sqrt 분모, drawdown shift, warmup fix) |
| 2026-02-10 | G1 | **FAIL** | 4/5 에셋 Sharpe 음수. Best ETH 0.08 << 1.0. CAGR +0.3% << 20%. Trades 28 << 50. AC 시그널이 크립토 1D에서 무효 |
