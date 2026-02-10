# 전략 상황판 (Strategy Dashboard)

> 44개 전략의 평가 현황과 검증 기준을 한눈에 파악하는 문서.
> 개별 스코어카드는 [docs/scorecard/](../scorecard/)에, 상세 평가 기준은 [전략 평가 표준](evaluation-standard.md)에 있다.

---

## 평가 파이프라인

```
Gate 0A → Gate 0B → Gate 1 → Gate 2 → Gate 3 → Gate 4 → Gate 5 → Gate 6 → Gate 7
아이디어   코드검증  백테스트  IS/OOS   파라미터  심층검증   EDA     Paper   실전배포
```

### Gate별 통과 기준

| Gate | 검증 | 핵심 기준 | CLI |
|:----:|------|----------|-----|
| **0A** | 아이디어 검증 | 6항목 합계 >= 18/30 | — |
| **0B** | 코드 품질 검증 | Critical 7항목 결함 0개 | `/quant-code-audit` |
| **1** | 단일에셋 백테스트 (5코인 x 6년) | Sharpe > 1.0, CAGR > 20%, MDD < 40%, Trades > 50 | `run {config}` |
| **2** | IS/OOS 70/30 | OOS Sharpe >= 0.3, Decay < 50% | `validate -m quick` |
| **3** | 파라미터 안정성 | 고원 존재, ±20% Sharpe 부호 유지 | `sweep {config}` |
| **4** | WFA + CPCV + PBO + DSR | WFA OOS >= 0.5, PBO < 40%, DSR > 0.95 | `validate -m milestone/final` |
| **5** | EDA Parity | VBT vs EDA 수익 부호 일치, 편차 < 20% | `eda run {config}` |
| **6** | Paper Trading (2주+) | 시그널 일치 > 90%, 무중단 | `eda run-live` |
| **7** | 실전 배포 | 3개월 이동 Sharpe > 0.3 | — |

### 비용 모델

| 항목 | 값 | 항목 | 값 |
|------|---:|------|---:|
| Maker Fee | 0.02% | Slippage | 0.05% |
| Taker Fee | 0.04% | Funding (8h) | 0.01% |
| Market Impact | 0.02% | **편도 합계** | **~0.11%** |

---

## 현재 전략 현황 (46개)

### 활성 전략 (1개, Gate 5 완료)

| 전략 | Best Asset | TF | Sharpe | CAGR | MDD | Trades | G0 | G1 | G2 | G3 | G4 | G5 | 비고 |
|------|-----------|-----|--------|------|-----|--------|:--:|:--:|:--:|:--:|:--:|:--:|------|
| [**CTREND**](scorecard/ctrend.md) | SOL/USDT | 1D | 2.05 | +97.8% | -27.7% | 288 | P | P | P | P | **F** | P | PBO 60% > 40% |

> G5 EDA Parity PASS: EDA Sharpe 2.82 vs VBT 2.05, 수익 부호 일치.
> G4 PBO 60% FAIL이나, 전 CPCV fold OOS Sharpe 양수 (0.49~2.79). MC p=0.000 PASS.
> **다음 단계**: Paper Trading (G6) 2주+ 실시간 검증.

### 검증중 전략 (0개)

> 검증 대기 전략 없음. 새 전략 발굴 필요 (`/strategy-discovery`).

### PENDING 전략 (2개, 데이터 부재)

| 전략 | G0 점수 | 차단 사유 |
|------|---------|----------|
| [**Funding Carry**](scorecard/funding-carry.md) | 25/30 | `funding_rate` 데이터 수집 필요 |
| [**Copula Pairs**](scorecard/copula-pairs.md) | 20/30 | `pair_close` 데이터 구성 필요 |

### 폐기 전략 (38개)

#### Gate 4 실패 — WFA 심층검증

| 전략 | Sharpe | WFA OOS | WFA Decay | 사유 |
|------|--------|---------|-----------|------|
| [KAMA](scorecard/fail/kama.md) | 1.14 | 0.56 | 56.3% | Decay 56%, Fold 2 OOS -0.06 |
| [MAX-MIN](scorecard/fail/max-min.md) | 0.82 | 0.47 | 38.4% | OOS 0.47 < 0.5, Fold 2 OOS -0.34 |

#### Gate 3 실패 — 파라미터 불안정

| 전략 | Sharpe | 사유 |
|------|--------|------|
| [TTM Squeeze](scorecard/fail/ttm-squeeze.md) | 0.94 | bb_period ±2 Sharpe 급락, kc_mult 1.0 거래 0건 |

#### Gate 2 실패 — IS/OOS 과적합 (16개)

| 전략 | Sharpe | OOS Sharpe | Decay | 전략 | Sharpe | OOS Sharpe | Decay |
|------|--------|-----------|-------|------|--------|-----------|-------|
| [XSMOM](scorecard/fail/xsmom.md) | 1.34 | 0.30 | 76.6% | [TSMOM](scorecard/fail/tsmom.md) | 1.33 | 0.19 | 87.2% |
| [Multi-Factor](scorecard/fail/multi-factor.md) | 1.22 | 0.17 | 83.3% | [Enhanced TSMOM](scorecard/fail/enhanced-tsmom.md) | 1.22 | 0.25 | 85.2% |
| [Vol Regime](scorecard/fail/vol-regime.md) | 1.41 | 0.37 | 77.3% | [Vol Structure](scorecard/fail/vol-structure.md) | 1.18 | 0.59 | 57.2% |
| [Vol-Adaptive](scorecard/fail/vol-adaptive.md) | 1.08 | -0.97 | 155.9% | [Donchian](scorecard/fail/donchian.md) | 1.01 | 0.12 | 91.1% |
| [VW-TSMOM](scorecard/fail/vw-tsmom.md) | 0.91 | 0.12 | 92.1% | [ADX Regime](scorecard/fail/adx-regime.md) | 0.94 | -0.68 | 146.3% |
| [Stoch-Mom](scorecard/fail/stoch-mom.md) | 0.94 | -0.34 | 124.9% | [GK Breakout](scorecard/fail/gk-breakout.md) | 0.77 | 0.39 | 59.0% |
| [MTF-MACD](scorecard/fail/mtf-macd.md) | 0.76 | 0.21 | 78.1% | [HMM Regime](scorecard/fail/hmm-regime.md) | 0.75 | -0.66 | 162.9% |
| [Adaptive Breakout](scorecard/fail/adaptive-breakout.md) | 0.54 | -0.68 | 201.1% | [Mom-MR Blend](scorecard/fail/mom-mr-blend.md) | 0.48 | -0.10 | 109.1% |

#### Gate 1 실패 — Sharpe/CAGR 미달

| 전략 | Sharpe | CAGR | 사유 |
|------|--------|------|------|
| [VWAP-Disposition](scorecard/fail/vwap-disposition.md) | 0.96 | +30.4% | Sharpe 0.96 < 1.0 (0.04 미달). DOGE MDD -622% (FULL short 파산) |
| [Donchian Ensemble](scorecard/fail/donchian-ensemble.md) | 0.99 | +10.8% | CAGR < 20% |
| [Kalman-Trend](scorecard/fail/kalman-trend.md) | 0.78 | +19.4% | Sharpe < 1.0, CAGR -0.6%p 미달. DOGE 거래 0건 |
| [BB-RSI](scorecard/fail/bb-rsi.md) | 0.59 | +4.6% | CAGR < 20% |
| [Entropy-Switch](scorecard/fail/entropy-switch.md) | 0.52 | +11.1% | 전 에셋 Sharpe < 1.0, MDD -47.9%. Alpha 부재 |

#### Gate 1 실패 — 전 에셋 Sharpe 음수/0 근접

| 전략 | Sharpe | CAGR | 사유 |
|------|--------|------|------|
| [Range-Squeeze](scorecard/fail/range-squeeze.md) | 0.33 | +3.8% | 전 에셋 Sharpe < 1.0, 3/5 에셋 음수, squeeze-breakout edge 부재 |
| [AC-Regime](scorecard/fail/ac-regime.md) | 0.08 | +0.3% | 4/5 에셋 Sharpe 음수, AC 시그널 무효 |
| [VR-Regime](scorecard/fail/vr-regime.md) | 0.17 | +0.8% | 전 에셋 Sharpe < 1.0, 극소 거래 (2~30건), significance_z 과도 → 시그널 부재 |
| [VPIN-Flow](scorecard/fail/vpin-flow.md) | 0.00 | 0.0% | 전 에셋 거래 0건, VPIN threshold 0.7이 1D 데이터에서 도달 불가 (max 0.45) |
| [Session-Breakout](scorecard/fail/session-breakout.md) | -1.67 | -29.9% | 전 에셋 Sharpe 음수 (-1.67~-3.49), MDD 88~97%. 크립토 24/7 시장에서 session breakout edge 부재 |
| [Liq-Momentum](scorecard/fail/liq-momentum.md) | -3.07 | -75.5% | 전 에셋 Sharpe 음수 (-3.07~-6.48), MDD ~100%. 1H Amihud + 12H momentum = noise-dominated + 과다거래 |
| [Flow-Imbalance](scorecard/fail/flow-imbalance.md) | -0.12 | -2.0% | 전 에셋 Sharpe 음수 (-0.12~-1.63), BNB MDD -54.2%. BVC 근사의 1H에서도 flow 방향성 예측력 부재 |
| [Hour-Season](scorecard/fail/hour-season.md) | -1.01 | -12.8% | 전 에셋 Sharpe 음수 (-1.01~-4.46), MDD 45~78%. 크립토 intraday 계절성(hour-of-day t-stat) edge 부재 |

#### Gate 1 실패 — 구조적 결함 / 코드 삭제

| 전략 | Sharpe | 사유 |
|------|--------|------|
| [HAR Vol](scorecard/fail/har-vol.md) | -0.23 | 전 에셋 Sharpe 음수, MDD 78~97% |
| [Larry VB](scorecard/fail/larry-vb.md) | 0.15 | 1-bar hold 비용 구조 (연 12.5% drag) |
| [Overnight](scorecard/fail/overnight.md) | 0.00 | 1H TF 데이터 부족 |
| [Z-Score MR](scorecard/fail/zscore-mr.md) | -0.02 | 통계적 무의미 |
| [RSI Crossover](scorecard/fail/rsi-crossover.md) | -0.16 | RSI 단순 크로스오버 |
| [Hurst/ER Regime](scorecard/fail/hurst-regime.md) | 0.24 | Hurst 추정 노이즈 |
| [Risk Momentum](scorecard/fail/risk-mom.md) | 0.77 | TSMOM 높은 상관, 차별화 부족 |

---

## 핵심 교훈

| # | 교훈 |
|---|------|
| 1 | **앙상블 > 단일지표**: ML 앙상블(CTREND)의 낮은 Decay(33.7%)가 단일 팩터 전략 대비 일반화 우수 |
| 2 | **IS Sharpe ≠ 실전 성과**: Gate 1 PASS 전략 24개 중 Gate 4까지 도달한 전략은 4개뿐 |
| 3 | **SOL/USDT = Best Asset**: 높은 변동성 + 추세 지속성이 모멘텀/앙상블 전략에 유리 |
| 4 | **CAGR > 20% 필터의 위력**: 안정적이나 수익 낮은 전략 (Donchian +10.8%, BB-RSI +4.6%) 조기 제거 |
| 5 | **PBO FAIL ≠ 즉시 폐기**: 전 fold OOS 양수 + MC p=0.000이면 실시간 검증이 합리적 다음 단계 |
| 6 | **다양성이 알파**: 단일코인 < 멀티에셋, 단일지표 < 앙상블 |
| 7 | **데이터 해상도 = 전략 해상도**: 마이크로스트럭처 전략(VPIN)은 tick/volume bar 데이터 필수. 1D OHLCV에서 BVC 근사는 정보 손실 심각 |
| 8 | **G0A 이론 ≠ G1 실전**: AC-Regime(27/30), VR-Regime(24/30) 등 G0A 고득점이 G1 백테스트 성과를 보장하지 않음. 39개 전략 중 G5 도달 1개(2.6%) |
| 9 | **통계적 검정 전략의 한계**: significance threshold (z=1.96)가 일봉 데이터에서 거래 빈도를 극단 제한 (BTC 6년간 2건). 학술적 엄밀성 ≠ 실용성 |
| 10 | **밈코인 FULL Short = 구조적 자살**: VWAP-Disposition DOGE MDD -622%. ShortMode.FULL + 밈코인 급등 = 치명적. DOGE 제외 시 SOL Sharpe 0.96 |
| 11 | **칼만 필터 ≠ 알파**: 학술적 최적 노이즈 분리가 크립토 1D에서 MA 대비 우위 없음. vel_threshold가 에셋 변동성에 미적응 (DOGE 거래 0건) |
| 12 | **레짐 필터의 양면성**: Entropy 필터가 거래 중단→리스크 감소에는 기여하나, alpha 생성 메커니즘 없으면 수익도 함께 감소. 43개 전략 중 G5 도달 여전히 1개(2.3%) |
| 13 | **FX Session Edge ≠ Crypto Edge**: Asian session breakout은 FX 시장의 institutional flow 시간대 분리에 기반. 크립토 24/7 시장에서는 session 분리가 구조적으로 약하며, 1H breakout 시그널은 false breakout 비율이 60%+ (Win Rate 32~38%). G0A 27/30점이 G1에서 전 에셋 Sharpe -1.67~-3.49 |
| 14 | **Amihud Illiquidity ≠ Crypto Alpha**: Equity 미시구조 지표(Amihud)를 24/7 크립토에 1H 단위 적용 시, 유동성 상태 전환이 과빈번 → conviction 확대가 whipsaw 증폭. 연 1,600~2,000건 과다거래 × Win Rate 38~44% = 구조적 손실. Mom lookback 12H은 noise-dominated |
| 15 | **BVC 근사의 TF 불변 한계**: OHLCV 기반 BVC(close-low)/(high-low)는 1D(VPIN-Flow, 거래 0건)→1H(Flow-Imbalance, 전 에셋 Sharpe 음수)로 해상도를 올려도 flow 방향성 예측 불가. 강한 OFI(>0.6) 진입 후 mean reversion이 더 빈번 — 정보거래자 이익 실현 패턴. Microstructure alpha에는 L2 order book 또는 tick data 필수 |
| 16 | **Intraday 계절성 비정상성**: 30일 rolling hour-of-day t-stat은 noise를 유의미한 패턴으로 오인. BTC(-4.46)가 최악 = 효율적 시장에서 계절성 즉시 차익거래. 원논문(Vojtko 2023)의 고정 시간대 vs 동적 시간대 선택의 차이 — 동적 선택이 과적합 증가. 4종 1H 전략 전멸 (session-breakout, liq-momentum, flow-imbalance, hour-season) |
