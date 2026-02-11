# 전략 스코어카드: Liq-Momentum

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Liq-Momentum (`liq-momentum`) |
| **유형** | Liquidity-Adjusted Momentum |
| **타임프레임** | 1H |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | N/A (전 에셋 Sharpe 음수) |
| **경제적 논거** | Amihud illiquidity + relative volume으로 유동성 상태를 분류하고, 저유동성 환경에서 모멘텀 conviction을 확대하여 정보 비대칭 프리미엄을 포착 |

---

## 성과 요약 (6년, 2020-2025, 1H)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate | Alpha | Beta |
|------|------|--------|------|-----|--------|------|----------|-------|------|
| 1 | DOGE/USDT | -3.07 | -75.5% | -100.0% | 11810 | 0.76 | 42.5% | -6134.0% | 0.05 |
| 2 | SOL/USDT | -3.13 | -70.0% | -99.9% | 9472 | 0.83 | 43.9% | -4377.8% | 0.02 |
| 3 | BNB/USDT | -4.90 | -84.3% | -100.0% | 10839 | 0.82 | 40.6% | -6295.0% | 0.02 |
| 4 | ETH/USDT | -5.02 | -85.3% | -100.0% | 11256 | 0.80 | 40.9% | -2305.4% | 0.02 |
| 5 | BTC/USDT | -6.48 | -90.3% | -100.0% | 10588 | 0.71 | 37.6% | -1232.3% | 0.04 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS]
G0B 코드검증 [PASS] 44 tests, shift(1) 준수, vectorized ops, frozen config
G1 백테스트  [FAIL] 전 에셋 Sharpe 음수 (-3.07 ~ -6.48), MDD ~100%. 즉시 폐기.
G2 IS/OOS    [    ]
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
```

### Gate 1 상세

**판정**: **FAIL** (즉시 폐기 -- 전 에셋 Sharpe 음수 + MDD ~100%)

**즉시 폐기 사유**:

- 즉시 폐기 조건 #1 해당: **전 5개 에셋에서 MDD > 50%** (99.9% ~ 100.0% -- 자본 전소)
- 즉시 폐기 조건 #2 해당: **전 5개 에셋에서 Sharpe 음수** (-3.07 ~ -6.48)
- Profit Factor 전 에셋 < 1.0 (0.71 ~ 0.83)
- CAGR 전 에셋 극단적 음수 (-70.0% ~ -90.3%)

**근본 원인 분석**:

1. **과다 거래 (극단적)**: 6년간 9,472~11,810건 (연평균 1,579~1,968건). 1H bar 기준으로도 거의 매 bar 진입/청산을 반복. 거래 비용 (편도 ~0.11%) 적용 시 연 174~217건의 비용 drag = 연 19~24%의 거래비용 부담.

2. **유동성 상태 분류의 노이즈**: Amihud illiquidity + relative volume의 1H 단위 분류가 너무 빈번하게 상태 전환을 발생시킴. 저유동성(low_liq_multiplier=1.5x) 확대가 잘못된 방향 진입을 증폭시키는 역효과.

3. **Momentum 12H lookback의 한계**: `mom_lookback=12` (12시간)은 크립토 1H 노이즈 대비 너무 짧음. 단기 모멘텀 신호가 noise-dominated되어 예측력 부재.

4. **Weekend multiplier 1.2x의 역효과**: 크립토는 주말 유동성이 낮아 변동성이 높지만, 이를 conviction 확대(1.2x)로 처리하면 whipsaw 손실이 증폭.

5. **Session Breakout과 동일 패턴**: 1H timeframe + 빈번한 거래 + 낮은 Win Rate (~38~44%) = 비용 구조적 문제. Larry-VB 교훈과 일치 -- 짧은 holding period의 거래비용 drag는 극복 불가.

**CTREND 비교**:

| 지표 | CTREND Best (SOL) | Liq-Momentum Best (DOGE) |
|------|-------------------|--------------------------|
| Sharpe | 2.05 | -3.07 |
| CAGR | +97.8% | -75.5% |
| MDD | -27.7% | -100.0% |
| Trades | 288 | 11810 |

**교훈**: 1H 유동성 기반 모멘텀 전략은 크립토 시장에서 edge가 없음. Amihud illiquidity는 equity 시장의 미시구조에서 유래한 지표로, 24/7 크립토 시장에서는 유동성 상태 전환이 너무 빈번하여 의미 있는 분류가 불가능. 모멘텀 lookback을 크게 늘려도 (예: 168H = 7일) 근본적으로 1H bar의 noise 문제와 거래비용 drag를 해결하기 어려움.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0B | PASS | 44 tests PASS, shift(1) lookahead 방지, vectorized ops, Pydantic frozen config, model_validator |
| 2026-02-10 | G1 | FAIL | 전 에셋 Sharpe 음수 (-3.07~-6.48), MDD ~100%, 즉시 폐기 조건 #1+#2 해당 |
