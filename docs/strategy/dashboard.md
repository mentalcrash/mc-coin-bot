# 전략 상황판 (Strategy Dashboard)

> 35개 전략의 평가 현황과 검증 기준을 한눈에 파악하는 문서.
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

## 현재 전략 현황 (35개)

### 활성 전략 (1개, Gate 4 완료)

| 전략 | Best Asset | TF | Sharpe | CAGR | MDD | Trades | G0 | G1 | G2 | G3 | G4 | 비고 |
|------|-----------|-----|--------|------|-----|--------|:--:|:--:|:--:|:--:|:--:|------|
| [**CTREND**](scorecard/ctrend.md) | SOL/USDT | 1D | 2.05 | +97.8% | -27.7% | 288 | P | P | P | P | **F** | PBO 60% > 40% |

> WFA PASS (OOS Sharpe 1.49, Decay 39%), MC p=0.000 PASS.
> PBO 60%로 G4 FAIL이나, 전 CPCV fold OOS Sharpe 양수 (0.49~2.79).
> **다음 단계**: EDA Parity + Paper Trading에서 실시간 검증 권고.

### 검증중 전략 (4개, G0A PASS → G0B 대기)

| 전략 | 유형 | G0A 점수 | G0B | G1 | 다음 단계 |
|------|------|:--------:|:---:|:--:|----------|
| [**AC-Regime**](../scorecard/ac-regime.md) | 레짐전환 | 27/30 | — | — | `/quant-code-audit` → 백테스트 |
| [**Range-Squeeze**](../scorecard/range-squeeze.md) | 변동성/돌파 | 24/30 | — | — | `/quant-code-audit` → 백테스트 |
| [**VR-Regime**](../scorecard/vr-regime.md) | 레짐전환 | 24/30 | — | — | `/quant-code-audit` → 백테스트 |
| [**VPIN-Flow**](../scorecard/vpin-flow.md) | 마이크로스트럭처 | 22/30 | — | — | `/quant-code-audit` → 백테스트 |

> 4개 전략 코드 구현 완료 (2026-02-10). 182 tests PASS, pyright 0 errors.
> **다음 단계**: G0B 코드 감사 → G1 단일에셋 백테스트 (5코인 x 6년).

### PENDING 전략 (2개, 데이터 부재)

| 전략 | G0 점수 | 차단 사유 |
|------|---------|----------|
| [**Funding Carry**](scorecard/funding-carry.md) | 25/30 | `funding_rate` 데이터 수집 필요 |
| [**Copula Pairs**](scorecard/copula-pairs.md) | 20/30 | `pair_close` 데이터 구성 필요 |

### 폐기 전략 (28개)

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

#### Gate 1 실패 — CAGR 미달

| 전략 | Sharpe | CAGR | 사유 |
|------|--------|------|------|
| [Donchian Ensemble](scorecard/fail/donchian-ensemble.md) | 0.99 | +10.8% | CAGR < 20% |
| [BB-RSI](scorecard/fail/bb-rsi.md) | 0.59 | +4.6% | CAGR < 20% |

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
