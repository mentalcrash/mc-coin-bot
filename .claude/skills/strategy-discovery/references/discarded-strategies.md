# Discarded Strategies — 폐기 전략 목록 (2026-02-10)

이 파일은 strategy-discovery 스킬에서 중복 회피를 위해 참조한다.
**동일 접근법 재시도는 금지**하며, 차별화 포인트를 명시해야 한다.

## Gate 1 실패 — 구조적 결함

| 전략 | Registry | 핵심 지표 | 실패 사유 |
|------|----------|----------|----------|
| HAR Vol | `har-vol` | HAR volatility model | 전 에셋 Sharpe 음수, MDD 78~97%, 과다거래 1200+ |

## Gate 1 실패 — CAGR 미달

| 전략 | Registry | IS Sharpe | CAGR | 실패 사유 |
|------|----------|-----------|------|----------|
| Donchian Ensemble | `donchian-ensemble` | 0.99 | +10.8% | CAGR < 20% 기준 미달 |
| BB-RSI | `bb-rsi` | 0.59 | +4.6% | CAGR < 20% 기준 미달 |

## Gate 2 실패 — IS/OOS 과적합 (Decay > 50%)

| 전략 | Registry | IS Sharpe | OOS Sharpe | Decay | 핵심 교훈 |
|------|----------|-----------|-----------|-------|----------|
| Adaptive Breakout | `breakout` | 0.54 | -0.68 | 201% | 최악의 Decay, 노이즈에 과적합 |
| HMM Regime | `hmm-regime` | 0.75 | -0.66 | 163% | HMM 수렴 불안정 |
| Vol-Adaptive | `vol-adaptive` | 1.08 | -0.97 | 156% | EMA+RSI+ADX+ATR 과다 지표 |
| ADX Regime | `adx-regime` | 0.94 | -0.68 | 146% | OOS Return -26.8% |
| Stoch Momentum | `stoch-mom` | 0.94 | -0.34 | 125% | 포트폴리오 분산 효과 부재 |
| Mom-MR Blend | `mom-mr-blend` | 0.48 | -0.10 | 109% | 동일 TF에서 Mom+MR alpha 상쇄 |
| VW-TSMOM | `vw-tsmom` | 0.91 | 0.12 | 92% | Overfit Prob 95.3% |
| Donchian Channel | `donchian` | 1.01 | 0.12 | 91% | 단일 Donchian 과적합 |
| TSMOM | `tsmom` | 1.33 | 0.19 | 87% | 1세대 주력이었으나 OOS 실패 |
| Enhanced TSMOM | `enhanced-tsmom` | 1.22 | 0.25 | 85% | 강화해도 Decay 개선 미미 |
| Multi-Factor | `multi-factor` | 1.22 | 0.17 | 83% | Overfit Prob 90%, 팩터 과다 |
| MTF-MACD | `mtf-macd` | 0.76 | 0.21 | 78% | 연 5건 거래, 신호 빈도 부족 |
| Vol Regime | `vol-regime` | 1.41 | 0.37 | 77% | IS→OOS 미유지 |
| XSMOM | `xsmom` | 1.34 | 0.30 | 77% | 횡단면 모멘텀, OOS borderline |
| GK Breakout | `gk-breakout` | 0.77 | 0.39 | 59% | Garman-Klass, Decay 59% |
| Vol Structure | `vol-structure` | 1.18 | 0.59 | 57% | Short/long vol ratio, Decay 57% |

## Gate 3 실패 — 파라미터 불안정

| 전략 | Registry | 실패 사유 |
|------|----------|----------|
| TTM Squeeze | `ttm-squeeze` | bb_period, kc_mult에 고원 부재. +/-2 변경 시 Sharpe 급락 |

## Gate 4 실패 — WFA 심층검증

| 전략 | Registry | WFA Decay | OOS | 실패 사유 |
|------|----------|-----------|-----|----------|
| KAMA | `kama` | 56% | Fold 2: -0.06 | WFA Decay 56%, 개별 Fold 실패 |
| MAX-MIN | `max-min` | — | 0.47 | WFA OOS < 0.5, Fold 2: -0.34 |

## 코드 삭제됨 (Gate 1 이전 폐기)

| 전략 | 폐기 사유 |
|------|----------|
| Larry VB | 1-bar hold 비용 구조 (연 125건 x 0.1% = 12.5% drag) |
| Overnight | 1H TF 데이터 부족 + 계절성 불안정 |
| Z-Score MR | 단일 z-score 평균회귀, 낮은 Sharpe |
| RSI Crossover | RSI 단순 크로스오버, 통계적 무의미 |
| Hurst/ER Regime | Hurst exponent 추정 노이즈 |
| Risk Momentum | TSMOM과 높은 상관, 차별화 부족 |

## PENDING (데이터 부재)

| 전략 | Registry | 필요 데이터 |
|------|----------|------------|
| Funding Carry | `funding-carry` | funding_rate 시계열 |
| Copula Pairs | `copula-pairs` | pair_close 데이터 |

---

## 폐기에서 얻은 핵심 교훈 (10가지)

1. **단일 지표 < 앙상블**: 다중 lookback/다중 소스 결합이 일반화 성능 향상
2. **동일 TF에서 반대 전략 블렌딩 = alpha 상쇄**: Mom + MR 결합은 실패
3. **1-bar hold = 비용 사망**: 거래 빈도가 비용을 초과하는 구조
4. **IS Sharpe > 1.0이어도 OOS 재현 안 되면 무의미**: 26개 중 G2 통과 1개
5. **Decay 50% 미만이 핵심**: CTREND 33.7% vs 나머지 57~201%
6. **ML 앙상블(CTREND)이 단일 팩터 대비 일반화 우수**
7. **SOL/USDT가 전 전략 Best Asset**: 높은 변동성 + 추세 지속성
8. **파라미터 고원 없으면 FAIL**: 소수점 변경에 결과 급변 = 과적합
9. **연 5건 미만 거래 = 통계적 무의미**: 충분한 빈도 확보 필수
10. **Regime Detection 전략 5개 전멸**: 레짐 감지 자체가 전략이 되면 과적합 쉬움
