# Discarded Strategies — 폐기 전략 목록 (2026-02-10)

이 파일은 strategy-discovery 스킬에서 중복 회피를 위해 참조한다.
**동일 접근법 재시도는 금지**하며, 차별화 포인트를 명시해야 한다.

> 최신 현황은 [docs/strategy/dashboard.md](../../../../docs/strategy/dashboard.md) 참조.
> 총 46개 전략 중 **활성 1개 (CTREND)**, **폐기 45개**.

---

## Gate 4 실패 — WFA 심층검증 (2개)

| 전략 | Registry | WFA Decay | OOS | 실패 사유 |
|------|----------|-----------|-----|----------|
| KAMA | `kama` | 56% | Fold 2: -0.06 | WFA Decay 56%, 개별 Fold 실패 |
| MAX-MIN | `max-min` | — | 0.47 | WFA OOS < 0.5, Fold 2: -0.34 |

## Gate 3 실패 — 파라미터 불안정 (1개)

| 전략 | Registry | 실패 사유 |
|------|----------|----------|
| TTM Squeeze | `ttm-squeeze` | bb_period, kc_mult에 고원 부재. +/-2 변경 시 Sharpe 급락 |

## Gate 2 실패 — IS/OOS 과적합 (16개)

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

## Gate 1 실패 — Sharpe/CAGR 미달 (5개)

| 전략 | Registry | Sharpe | CAGR | 실패 사유 |
|------|----------|--------|------|----------|
| VWAP-Disposition | `vwap-disposition` | 0.96 | +30.4% | Sharpe 0.04 미달. DOGE FULL Short MDD -622% (밈코인 구조적 자살) |
| Donchian Ensemble | `donchian-ensemble` | 0.99 | +10.8% | CAGR < 20% |
| Kalman-Trend | `kalman-trend` | 0.78 | +19.4% | Sharpe < 1.0, CAGR -0.6%p 미달. DOGE 거래 0건 |
| BB-RSI | `bb-rsi` | 0.59 | +4.6% | CAGR < 20% |
| Entropy-Switch | `entropy-switch` | 0.52 | +11.1% | 전 에셋 Sharpe < 1.0. 레짐 필터만으로 alpha 생성 불가 |

## Gate 1 실패 — 전 에셋 Sharpe 음수/0 근접 (12개)

### 1D 전략 (4개)

| 전략 | Registry | Best Sharpe | 실패 사유 |
|------|----------|-------------|----------|
| Range-Squeeze | `range-squeeze` | 0.33 | 3/5 에셋 Sharpe 음수, NR7 squeeze-breakout edge 부재 |
| AC-Regime | `ac-regime` | 0.08 | 4/5 에셋 Sharpe 음수, rolling autocorrelation 시그널 무효 |
| VR-Regime | `vr-regime` | 0.17 | 극소 거래 (BTC 6년간 2건), significance_z 과도 → 시그널 부재 |
| VPIN-Flow | `vpin-flow` | 0.00 | 전 에셋 거래 0건, VPIN threshold 0.7이 1D 데이터에서 도달 불가 (max 0.45) |

### 1H 전략 — Tier 5 전멸 (4개)

| 전략 | Registry | Best Sharpe | MDD | Trades/yr | 실패 사유 |
|------|----------|-------------|-----|-----------|----------|
| Session-Breakout | `session-breakout` | -1.67 | -88% | ~280 | FX session edge ≠ crypto 24/7. False breakout 60%+, Win Rate 32~38% |
| Liq-Momentum | `liq-momentum` | -3.07 | -100% | ~1,700 | Amihud 1H 과빈번 전환, 12H momentum = noise. 비용 drag 19~24%/yr |
| Flow-Imbalance | `flow-imbalance` | -0.12 | -27% | ~54 | BVC 근사로 flow 방향 예측 불가. OFI>0.6 진입 후 mean reversion |
| Hour-Season | `hour-season` | -1.01 | -45% | ~190 | 30일 rolling t-stat이 noise 과적합. BTC(-4.46) 최악 = 효율적 시장에서 계절성 즉시 차익거래 |

## Gate 1 실패 — 데이터 부재 (2개)

| 전략 | Registry | G0 점수 | 실패 사유 |
|------|----------|---------|----------|
| Funding Carry | `funding-carry` | 25/30 | funding_rate 데이터 부재, 인프라 미구축 → 폐기 |
| Copula Pairs | `copula-pairs` | 20/30 | pair_close 데이터 부재, 인프라 미구축 → 폐기 |

## Gate 1 실패 — 구조적 결함 / 코드 삭제 (7개)

| 전략 | Registry | 폐기 사유 |
|------|----------|----------|
| HAR Vol | `har-vol` | 전 에셋 Sharpe 음수, MDD 78~97%, 과다거래 1200+ |
| Larry VB | `larry-vb` | 1-bar hold 비용 구조 (연 125건 x 0.1% = 12.5% drag) |
| Overnight | `overnight` | 1H TF 데이터 부족 + 계절성 불안정 |
| Z-Score MR | `zscore-mr` | 단일 z-score 평균회귀, 낮은 Sharpe |
| RSI Crossover | `rsi-crossover` | RSI 단순 크로스오버, 통계적 무의미 |
| Hurst/ER Regime | `hurst-regime` | Hurst exponent 추정 노이즈 |
| Risk Momentum | `risk-mom` | TSMOM과 높은 상관, 차별화 부족 |

---

## 폐기에서 얻은 핵심 교훈 (16가지)

> 최신 교훈은 [dashboard.md](../../../../docs/strategy/dashboard.md) "핵심 교훈" 섹션 참조.

1. **앙상블 > 단일지표**: ML 앙상블(CTREND)의 낮은 Decay(33.7%)가 단일 팩터 전략 대비 일반화 우수
2. **IS Sharpe ≠ 실전 성과**: Gate 1 PASS 전략 24개 중 Gate 4까지 도달한 전략은 4개뿐
3. **SOL/USDT = Best Asset**: 높은 변동성 + 추세 지속성이 모멘텀/앙상블 전략에 유리
4. **CAGR > 20% 필터의 위력**: 안정적이나 수익 낮은 전략 조기 제거
5. **PBO FAIL ≠ 즉시 폐기**: 전 fold OOS 양수 + MC p=0.000이면 실시간 검증 합리적
6. **다양성이 알파**: 단일코인 < 멀티에셋, 단일지표 < 앙상블
7. **데이터 해상도 = 전략 해상도**: 마이크로스트럭처 전략(VPIN)은 tick/volume bar 데이터 필수
8. **G0A 이론 ≠ G1 실전**: G0A 고득점이 G1 성과를 보장하지 않음. 46개 중 G5 도달 1개(2.2%)
9. **통계적 검정 전략의 한계**: significance threshold가 거래 빈도를 극단 제한 (BTC 6년간 2건)
10. **밈코인 FULL Short = 구조적 자살**: DOGE MDD -622%. ShortMode.FULL + 밈코인 급등 = 치명적
11. **칼만 필터 ≠ 알파**: 학술적 최적 노이즈 분리가 크립토 1D에서 MA 대비 우위 없음
12. **레짐 필터의 양면성**: 레짐 필터만으로는 alpha 생성 메커니즘 부재
13. **FX Session Edge ≠ Crypto Edge**: 크립토 24/7 시장에서 session 분리 구조적으로 약함
14. **Amihud Illiquidity ≠ Crypto Alpha**: 1H Amihud conviction 확대가 whipsaw 증폭
15. **BVC 근사의 TF 불변 한계**: OHLCV BVC는 1D→1H 해상도 올려도 flow 방향 예측 불가. L2 data 필수
16. **Intraday 계절성 비정상성**: hour-of-day t-stat은 noise를 과적합. 4종 1H 전략 전멸

### 실패 패턴 요약 (빠른 참조)

| 패턴 | 해당 전략 | 결론 |
|------|----------|------|
| 단일 지표 trend-following | TSMOM, Enhanced, VW, Donchian, KAMA, MAX-MIN | OOS Decay 56~92% |
| 동일 TF Mom+MR 블렌딩 | Mom-MR Blend | Alpha 상쇄 |
| 1-bar hold | Larry VB | 비용 > 수익 |
| 레짐 감지 = 전략 | ADX, HMM, Vol-Adaptive, Hurst, Entropy, AC, VR | 과적합 or 시그널 부재 |
| OHLCV 기반 microstructure | VPIN-Flow, Flow-Imbalance | BVC 정밀도 부족, L2 필수 |
| 전통금융→크립토 전이 | Session-Breakout, Liq-Momentum, Hour-Season | 시장 구조 차이로 edge 소멸 |
| 데이터 부재 전략 | Funding Carry, Copula Pairs | 인프라 미구축 → 무기한 보류 = 폐기 |
| 밈코인 FULL Short | VWAP-Disposition | DOGE 급등 시 MDD 무한대 |
