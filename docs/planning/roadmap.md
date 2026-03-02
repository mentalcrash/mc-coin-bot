# 개선 로드맵

미착수 개선 항목 요약. 완료 항목은 정식 문서로 이동 완료.

**갱신일**: 2026-03-01

---

## 현황 요약

| 지표 | 값 |
|------|-----|
| 전략 총 시도 | 185개 |
| ACTIVE | 4개 (Anchor-Mom, Donch-Multi, Tri-Channel, Dual-Mom) |
| RETIRED | 181개 (97.8%) |
| 확정 TF | 12H 단일 (4H/6H/8H/1D 전멸) |
| 포트폴리오 Sharpe | 1.10 (Orchestrator v5.1) |
| 고갈 확인 | 1D OHLCV, 4H/8H TF, 대안데이터 단독 alpha, ML, Carry |

---

## Tier 1: Portfolio-Level Alpha (최우선)

> 개별 전략 alpha가 고갈 상태. **기존 3 전략의 조합/배분 최적화**가 가장 높은 ROI.
> 근거: AdaptiveTrend (arXiv 2602.11708) Sharpe 2.41, HRP (Lopez de Prado) tail risk 개선

### A. Orchestrator Dynamic Weighting

| # | 항목 | 핵심 효과 | 난이도 | 기대 Sharpe 개선 | 상태 |
|---|------|----------|--------|-----------------|------|
| ~~A1~~ | ~~Rolling Sharpe-Based Weighting~~ | ~~최근 성과 기반 pod 비중 동적 조정~~ | ~~낮~~ | ~~+0.15~0.30~~ | ✅ 완료 |
| A2 | Inverse-Volatility Weighting | 저변동 전략에 자본 집중 | 낮 | +0.10~0.20 | ✅ 완료 (기존 InvVol) |
| A3 | HRP (Hierarchical Risk Parity) | Clustering 기반 상관 구조 활용 배분 | 중 | +0.15~0.25 | 미착수 |
| ~~A4~~ | ~~Drawdown-Based De-Risking~~ | ~~Pod DD > threshold → weight 자동 축소~~ | ~~낮~~ | ~~+0.05~0.15 (MDD 방어)~~ | ✅ 완료 |

**설계 포인트:**

- `Orchestrator._compute_pod_weights()` 신규 메서드
- 현재 static weight → Rolling 30-bar Sharpe EMA decay로 교체
- Inverse-vol: `w_i = (1/σ_i) / Σ(1/σ_j)`, lookback 60 bars (12H × 60 = 30일)
- HRP: scikit-learn AgglomerativeClustering + recursive bisection
- DD de-risk: pod DD > 10% → weight 50% 축소, DD > 20% → weight 0%

**검증 방법:**

- Orchestrator VBT 백테스트로 static vs dynamic weight 비교
- Walk-forward 30-bar window, OOS 검증

### B. Signal Consensus Meta-Layer

| # | 항목 | 핵심 효과 | 난이도 | 기대 Sharpe 개선 |
|---|------|----------|--------|-----------------|
| B1 | Signal Voting Filter | 2/3 합의 시 full, 1/3 시 half | 낮 | +0.10~0.20 |
| B2 | Regime-Conditional Rotation | HMM 2-state → bull/bear별 전략 강조 | 중 | +0.15~0.30 |
| B3 | Rolling Performance Rotation | EMA decay 기반 전략 가중치 자동 조정 | 낮-중 | +0.10~0.25 |

**설계 포인트:**

- B1: 동일 심볼이 아니므로 **방향 합의** (long/short/flat 투표)
  - 3 pod 중 2+ long → portfolio risk budget 100%
  - 혼재 → 50%, 전부 flat → 0%
- B2: BTC 수익률 기반 2-state HMM
  - Trending → donch-multi/tri-channel 강조 (momentum-friendly)
  - Ranging → anchor-mom 강조 또는 전체 축소
  - **주의**: standalone HMM 전략은 RETIRED, 하지만 meta-level regime는 다른 용도
- B3: 각 pod의 최근 60-bar 성과로 weight = `softmax(rolling_sharpe * temperature)`

**검증 방법:**

- 2년 백테스트 (2024-2025) static vs consensus 비교
- Regime 오분류 sensitivity analysis

### 권장 순서

```text
Phase 1 (3일): A1 Rolling Sharpe Weight + A4 DD De-Risking
Phase 2 (3일): B1 Signal Voting + B3 Performance Rotation
Phase 3 (5일): A3 HRP + B2 HMM Regime (실험적)
```

---

## Tier 1.5: Dynamic Asset Surveillance Backtest (높은 우선순위)

> 라이브 Surveillance(동적 에셋 교체)를 백테스트에서 재현.
> 고정 심볼 백테스트의 survivorship bias 제거 + 라이브-백테스트 parity 확보.
> 상세: [`dynamic-asset-surveillance-backtest.md`](dynamic-asset-surveillance-backtest.md)

| Phase | 항목 | 핵심 효과 | 난이도 | 상태 |
|-------|------|----------|--------|------|
| 1 | Wide Universe 데이터 수집 (~40 에셋 1m) | 후보군 풀 확보 | 중 (API 시간) | 미착수 |
| 2 | BacktestSurveillanceSimulator 구현 | 7일 rolling volume 기반 동적 교체 시뮬레이션 | 중 | 미착수 |
| 3 | 검증 (Parity + Survivorship Bias 정량화) | 고정 vs 동적 비교, 멀티에셋 범용성 | 중 | 미착수 |
| 4 | 라이브 Surveillance 활성화 | pinned_symbols: false 전환 | 낮 | 미착수 |

**설계 포인트:**

- `BacktestSurveillanceSimulator`: 1D quote_volume summary → 7D rolling → `ScanResult` 생성
- `OrchestratedRunner`: 기존 `on_universe_update()` 경로 재사용 (라이브와 동일 코드)
- 메모리 최적화: 운용 에셋만 1m 상주, 후보군은 volume matrix만 (~1MB)
- 전략 범용성 검증: 3 전략 × 40 에셋 매트릭스에서 평균 Sharpe > 0.5 확인 필수

**전제 조건:**

- Wide Universe 1m 데이터 수집 완료 (에셋당 2~4GB, 총 ~150GB)
- 멀티에셋 범용 P4 통과 (특정 에셋 의존 → 범용 검증)

---

## Tier 2: 미탐색 알파 소스 (중간 우선순위)

### C. 6H Timeframe 탐색

> AdaptiveTrend 논문 (arXiv 2602.11708): **6H interval에서 Sharpe 2.41** (OOS 2022-2024).
> 12H의 절반 주기로, 4H 사망 패턴과 12H 성공 사이의 sweet spot 가능성.

| # | 항목 | 핵심 효과 | 난이도 |
|---|------|----------|--------|
| C1 | Tri-Channel 6H 백테스트 | 기존 최고 전략의 TF 변환 테스트 | 낮 |
| C2 | Donch-Multi 6H 백테스트 | 3-scale Donchian의 6H 적용 | 낮 |
| C3 | 비용 분석 | 6H 거래 빈도 vs cost drag 사전 검증 | 낮 |

**판단 기준:**

- 6H 거래 횟수 < 300/년 → cost drag 허용 (4H는 500+ → 전멸 패턴)
- IS Sharpe > 1.0 AND OOS Decay < 30% → Phase 4-7 진행
- **1회 VBT sweep으로 go/no-go 결정** (정보 가치 대비 비용 극히 낮음)

**위험 요인:**

- 4H 사망 패턴 (cost erosion) 재현 가능
- 교훈 #090: 6H cost-channel-6h는 150-230회 거래에서도 CAGR 13.4% < 20% 기준 미달
- 기존 전략 구조가 6H에서 다른 파라미터 최적점을 가질 수 있음

### D. Cross-Asset Momentum Rotation

> Cross-sectional momentum factor가 crypto 수익률의 25% 설명
> (Quantitative Finance 2023, 3900+ coins 대상)

| # | 항목 | 핵심 효과 | 난이도 |
|---|------|----------|--------|
| D1 | 에셋 간 상대 모멘텀 Ranking | Top N 에셋에 동적 배분 | 중 |
| D2 | Lead-Lag 구조 활용 | BTC→ALT 선행 관계 signal | 중 |
| D3 | Correlation Breakdown Alert | 상관 구조 변화 시 포지션 조정 | 중 |

**설계 포인트:**

- D1: 12H rolling 30-bar return ranking → top 3 에셋 EW 배분
  - Dual-Mom (INCUBATION)이 유사 접근 → P1 결과 우선 확인
- D2: BTC 12H return → ALT 다음 bar return 예측 (Granger causality 검정)
- D3: rolling 30-bar correlation matrix → eigenvalue ratio 급변 시 risk-off

**전제 조건:**

- Dual-Mom P1 결과 확인 후 확장 여부 결정
- Orchestrator가 Pod 간 심볼 비중복을 강제하므로, rotation 시 Pod 재구성 필요

### E. Funding Rate Risk Overlay

> 교훈 #085: FR crowd filter가 유효 추세까지 억제 → 독립 signal 부적합.
> 하지만 **extreme FR에서 position sizing 축소**는 risk management로 유효.

| # | 항목 | 핵심 효과 | 난이도 |
|---|------|----------|--------|
| E1 | FR Extreme Position Scale | FR > 0.05% → size 50% 축소 | 낮 |
| E2 | OI Spike Alert | 90일 95th percentile 초과 → 경고 | 낮 |
| E3 | FR + OI + Low Vol 3중 경고 | 3조건 동시 → 즉시 축소 | 낮 |

**이전 로드맵 #4+#5와 통합** — `DerivativesRiskMonitor` 클래스:

- FR 등급: 정상(<0.03%) / 경고(0.03~0.05%) / 위험(0.05~0.10%) / 극단(>0.10%)
- OI 등급: 90일 percentile 80/90/95th + 24h 변화율
- **alpha 생성이 아닌 tail risk 방어 관점**

---

## Tier 3: 실행 품질 개선 (기존 로드맵)

> 원본: execution-and-risk-improvements.md (삭제됨)
> 완료 항목: SmartExecutor → [`docs/architecture/smart-executor.md`](../architecture/smart-executor.md)

| # | 항목 | 핵심 효과 | 난이도 | 상태 |
|---|------|----------|--------|------|
| ~~1~~ | ~~Limit Order~~ | ~~비용 40-50% 절감~~ | ~~중간~~ | ✅ 완료 |
| F1 | 동적 슬리피지 모델 | 백테스트 정확도 향상 | 쉬움 | 미착수 |
| F2 | Alpha Decay 모니터링 | 전략 수명 관리 | 쉬움 | 미착수 |
| F3 | Multi-Pod Execution Stagger | 동시 rebalancing 시 timing 분산 | 쉬움 | 미착수 |

**F1 동적 슬리피지**: `dynamic_slippage = base × asset_factor × vol_factor`

- BTC 0.7x, DOGE 1.5x, vol 0.5x~3.0x 범위

**F2 Alpha Decay**: `AlphaDecayMonitor` — 30/60/90일 Rolling Sharpe 기반 건강 상태

- 건강(>0.5) / 경고(0~0.5) / 위험(<0, 30d) / 사망(<0, 60d)
- Discord 알림 연동

**F3 Execution Stagger**: 4-pod 동시 rebalancing 방지

- Pod 간 1-2분 간격으로 order 전송 → market impact 분산

---

## Tier 4: 고갈 확인 영역 (비추천)

> 아래 영역은 충분한 검증을 거쳐 **추가 탐색 가치 없음** 확인됨.

### 고갈 근거 요약

| 영역 | 시도 수 | 결론 | 핵심 교훈 |
|------|--------|------|----------|
| **1D OHLCV** | 92개 | 검색공간 완전 고갈 | 1D 추세 성숙 → alpha 부재 |
| **4H/8H TF** | 50+개 | 구조적 비용 벽 + TF 한계 | #087, #089, #090 |
| **ML 전략** | 4개 | Look-ahead bias 극복 불가 | CTREND/GBTrend/Vol-Term 전멸 |
| **대안데이터 단독** | 20+개 | On-chain/Deriv/TradeFlow alpha 0 | #088, #095, #096 |
| **Carry/Basis** | 8+개 | 2025 기준 carry alpha 구조적 소멸 | BIS #1087: 2025 음수 전환 |
| **Sentiment (F&G)** | 5+개 | Standalone alpha 부재 | fg_asym_mom 등 전멸 |
| **Calendar Anomaly** | 10+개 | 시간-비정상성 과적합 | #097 weekend-mom Decay 130% |
| **Regime Filter** | 15+개 | 필터가 유효 추세까지 억제 | #085 FR crowd filter |
| **극저빈도 FSM** | 8+개 | 직렬 조건 확률 곱셈 → 통계 무의미 | #044, #092 |
| **Cross-Exchange Arb** | - | Binance 단일 거래소 제약 | 구현 복잡도 대비 효과 낮음 |

### 문헌 기반 비추천 근거

**Carry/Basis**: BIS Working Paper #1087 (2024) — 전체 기간 Sharpe 6.45이나
2024부터 4.06으로 하락, **2025 음수 전환**. Spot ETF 승인 후 carry 36% 감소,
CME에서 97% 감소. Binance BTC carry Sharpe 음수.

**Volatility Structure**: Bitcoin VRP는 equity와 **반대 부호** (arXiv 2410.15195).
equity VRP 전략 그대로 적용 불가. `vrp_regime_trend`, `vol_structure_trend_12h` 등 이미 RETIRED.

**Sentiment**: 2025년 구조적 호재(ETF, 규제 완화)에도 sentiment-price disconnect 발생
(CryptoSlate 2025). 5개 F&G 전략 전멸.

---

## Taker Buy Base Volume 인프라

> 원본: taker-buy-volume-infra.md (삭제됨)
> **우선순위 하향**: 교훈 #096에서 Trade Flow 독립 alpha 부재 확인.
> 리스크 관리 보조로만 활용 가치 있으므로, Tier 2-E (FR Risk Overlay)에 통합 검토.

| Phase | 항목 | 상태 |
|-------|------|------|
| ~~1~~ | ~~데이터 모델 + 지표 (`taker_cvd`, `taker_buy_ratio`)~~ | ✅ 완료 |
| 2 | API 캡처 (`publicGetKlines` column 9) | 보류 |
| 3 | Storage 파이프라인 (Bronze/Silver) | 보류 |
| 4 | EDA 통합 (BarEvent, CandleAggregator) | 보류 |
| 5 | 데이터 재수집 (5 에셋) | 보류 |
| 6 | CVD 전략 통합 | 보류 |

---

## 전체 실행 로드맵

```text
Sprint 1 (1주):  A1 Rolling Sharpe Weight + A4 DD De-Risking + F2 Alpha Decay  ← ✅ 완료
Sprint 2 (2~3일): Wide Universe 데이터 수집 (~40 에셋 1m Parquet)
Sprint 3 (1주):  BacktestSurveillanceSimulator 구현 + 검증
Sprint 4 (1주):  B1 Signal Voting + B3 Performance Rotation + E1 FR Scale
Sprint 5 (1주):  C1-C3 6H TF 탐색 (1회 VBT sweep → go/no-go)
Sprint 6 (1주):  A3 HRP + B2 HMM Regime (실험적)
Sprint 7 (1주):  D1-D3 Cross-Asset Rotation (Dual-Mom P1 결과 기반)
Sprint 8 (1주):  F1 동적 슬리피지 + F3 Execution Stagger
```

### 기대 효과

| 현재 | Sprint 1-2 후 | Sprint 3-4 후 |
|------|--------------|--------------|
| Sharpe 1.10 | 1.30~1.50 | 1.40~1.70 |
| MDD 16.3% | 12~14% | 10~13% |
| 전략 수 4 | 4 | 4~5 (6H 성공 시) |

---

## 참고 문헌

- [AdaptiveTrend: Systematic Trend-Following (2025, arXiv 2602.11708)](https://arxiv.org/abs/2602.11708)
- [BIS Working Paper #1087: Crypto Carry (2024)](https://www.bis.org/publ/work1087.pdf)
- [Cross-Sectional Momentum in Crypto (2023, Quantitative Finance)](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2269999)
- [HRP for Portfolio Allocation (2025, arXiv 2509.03712)](https://arxiv.org/pdf/2509.03712)
- [Bitcoin VRP Risk Premia (2024, arXiv 2410.15195)](https://arxiv.org/pdf/2410.15195)
- [Trading Games: Cointegrated Pairs (2025, J. Futures Markets)](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70018)
- [Deep Learning VWAP Execution (2025, arXiv 2502.13722)](https://arxiv.org/html/2502.13722v1)
- [Funding Rate Predictability (2025, SSRN 5576424)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424)
- [Regime Switching Forecasting (2025, Digital Finance)](https://link.springer.com/article/10.1007/s42521-024-00123-2)
