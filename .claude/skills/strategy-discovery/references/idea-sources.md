# Idea Sources — 알파 소싱 카탈로그

전략 아이디어의 출처와 경제적 논거를 체계적으로 정리한다.
각 카테고리는 "왜 이 edge가 존재하는가"에 대한 경제적 설명을 포함한다.

---

## 1. Microstructure (시장 미시구조)

OHLCV에서 파생 가능한 미시구조 시그널. 외부 데이터 불필요.

### 1-A. Order Flow Imbalance (주문 흐름 불균형) — OHLCV 접근 실패 확인

```
경제적 논거: 대규모 매수/매도 압력은 가격에 선행한다.
             정보 거래자(informed trader)의 활동이 가격 변동을 유발.
지표:
  - Volume Imbalance: buy_vol / total_vol (OHLCV에서 추정 가능)
  - Tick Rule: close > open → buy dominant
  - VPIN (Volume-Synchronized Probability of Informed Trading)
    → 일정 volume bucket 기준으로 buy/sell 분류
참고 논문:
  - Easley, D. et al. "Microstructure in Crypto Markets" (Cornell, 2024)
  - "Explainable Patterns in Cryptocurrency Microstructure" (arXiv:2602.00776)
구현 난이도: 중간 (OHLCV에서 volume 분류 필요)
TF 적합: 4H, 1D

⚠️ 실패 기록 (2026-02-10):
  - VPIN-Flow (1D): 전 에셋 거래 0건. VPIN threshold 도달 불가.
  - Flow-Imbalance (1H): 전 에셋 Sharpe 음수. BVC 방향 예측 불가.
  → OHLCV 기반 BVC 근사는 TF에 관계없이 microstructure alpha 불충분.
  → 이 카테고리는 L2 order book 또는 tick data 확보 시에만 재시도 가능.
```

### 1-B. Roll Measure (Bid-Ask Spread 추정)

```
경제적 논거: 높은 스프레드 = 낮은 유동성 = 가격 충격 위험 증가.
             유동성 감소 후 평균회귀 기회 발생.
지표:
  - Roll = 2 * sqrt(-cov(delta_p[t], delta_p[t-1]))
  - 음의 자기상관이 없으면 스프레드 = 0으로 설정
구현 난이도: 낮음 (close 가격만 필요)
TF 적합: 1D
```

### 1-C. Amihud Illiquidity — 1H 크립토 실패 확인

```
경제적 논거: 비유동성 프리미엄 — 유동성이 낮을수록 기대 수익률 높음.
             유동성 감소 국면에서 진입 → 유동성 회복 시 이익.
지표:
  - Amihud = |return| / volume (일별)
  - 이동 평균으로 smoothing
구현 난이도: 낮음 (OHLCV 직접)
TF 적합: 1D (1H에서 실패 확인)

⚠️ 실패 기록 (2026-02-10):
  - Liq-Momentum (1H): Sharpe -3.07~-6.48, MDD ~100%, 연 ~1,700건 과다거래.
  - 1H Amihud 유동성 상태 전환이 과빈번 → conviction 확대가 whipsaw 증폭.
  → 1D에서 재시도 가능하나, momentum+liquidity 조합 자체의 효과성 의문.
```

---

## 2. Volatility (변동성)

### 2-A. Volatility Risk Premium (변동성 리스크 프리미엄)

```
경제적 논거: 옵션 시장에서 implied vol > realized vol (공포 프리미엄).
             크립토에서는 OHLCV 기반 realized vol의 term structure로 대체.
지표:
  - Short-term RV (5일) vs Long-term RV (30일) 비율
  - Parkinson volatility (high-low 기반, 더 정밀)
  - Garman-Klass volatility (OHLC 4가지 사용)
  - Yang-Zhang volatility (overnight gap 포함)
구현 난이도: 낮음~중간
TF 적합: 1D, 4H
주의: GK Breakout이 Decay 59%로 실패. 단순 breakout이 아닌
      vol structure 기반 시그널이 필요.
```

### 2-B. Realized Volatility Term Structure

```
경제적 논거: 단기 vol > 장기 vol = 시장 스트레스 (backwardation).
             정상: 단기 < 장기 (contango). 역전 시 평균회귀 기회.
지표:
  - RV_5d / RV_30d 비율
  - RV_10d / RV_60d 비율
  - Z-score of vol ratio (정규화)
구현 난이도: 낮음
TF 적합: 1D
```

### 2-C. Intraday Volatility Pattern — Session/Seasonality 실패 확인

```
경제적 논거: 크립토는 24/7 거래. 시간대별 유동성 차이 → 변동성 U-shape.
             아시아/유럽/미국 세션 전환 시 변동성 급증.
지표:
  - High-Low range / Close (일내 변동성)
  - 연속 n일 range 축소 후 확대 (squeeze → expansion)
구현 난이도: 낮음
TF 적합: 4H, 1H

⚠️ 실패 기록 (2026-02-10):
  - Session-Breakout (1H): Sharpe -1.67~-3.49. Asian session range breakout은
    크립토 24/7 시장에서 구조적 edge 부재. FX session 분리 ≠ 크립토.
  - Hour-Season (1H): Sharpe -1.01~-4.46. Hour-of-day t-stat이 noise 과적합.
    BTC(-4.46) 최악 = 효율적 시장에서 계절성 즉시 차익거래.
  → Intraday session/seasonality 접근은 크립토에서 재시도 비권장.
```

---

## 3. Carry / Yield (캐리)

### 3-A. Funding Rate Carry — 폐기 (데이터 부재)

```
경제적 논거: 영구선물 funding rate는 레버리지 수요를 반영.
             양(+) funding = 과열 → 숏 캐리 수익.
             음(-) funding = 공포 → 롱 캐리 수익.
지표:
  - Funding rate 8h 이동 평균
  - Funding rate z-score
상태: 폐기 (funding_rate 데이터 미확보, 인프라 투자 미진행)
TF 적합: 8H, 1D
```

### 3-B. Basis Spread (Spot-Futures)

```
경제적 논거: 선물 프리미엄은 시장 심리를 반영.
             높은 basis = 과열 (contango) → mean reversion.
             음의 basis = 공포 (backwardation) → 롱 기회.
지표:
  - (futures_price - spot_price) / spot_price
  - Annualized basis
상태: 데이터 확보 필요 (futures OHLCV)
TF 적합: 1D
```

---

## 4. Information Theory (정보 이론)

### 4-A. Approximate Entropy (ApEn)

```
경제적 논거: 낮은 엔트로피 = 예측 가능한 패턴 = 추세 지속.
             높은 엔트로피 = 무작위 = 추세 전환 가능.
지표:
  - ApEn(m=2, r=0.2*std) of returns
  - Sample Entropy (SampEn) — 더 안정적
구현 난이도: 중간 (numba 최적화 필요)
TF 적합: 1D, 4H
```

### 4-B. Transfer Entropy (Lead-Lag)

```
경제적 논거: BTC → ALT 정보 전달 지연. BTC 움직임이 ALT에 선행.
             단일 에셋에서: 가격 → 거래량 정보 흐름 분석.
지표:
  - TE(price → volume) vs TE(volume → price)
  - Net TE = TE(V→P) - TE(P→V)
구현 난이도: 높음
TF 적합: 1D
주의: 단일 에셋 제약 — 자기 자신의 price↔volume 관계로 제한
```

---

## 5. Behavioral Finance (행동 재무학)

### 5-A. Disposition Effect (처분 효과)

```
경제적 논거: 투자자는 이익은 빨리, 손실은 늦게 실현.
             → 상승 후 과도한 매도압력 → delayed momentum.
             → 하락 후 holding → 급락 시 panic selling.
지표:
  - Capital Gains Overhang: (price - reference_price) / reference_price
  - Reference price = volume-weighted average (n일)
구현 난이도: 중간
TF 적합: 1D, 4H
```

### 5-B. Anchoring (앵커링)

```
경제적 논거: 투자자는 과거 고점/저점에 앵커링.
             심리적 지지/저항선 → 가격 반등/돌파 시 과잉 반응.
지표:
  - Distance from 52-week high/low
  - Price relative to n-day VWAP
  - Proximity to round numbers ($50K, $100K)
구현 난이도: 낮음
TF 적합: 1D
```

---

## 6. Cross-Asset Signal → 단일 에셋 적용

단일 에셋 제약 내에서 cross-asset 정보를 활용하는 방법.

### 6-A. BTC Dominance Signal

```
경제적 논거: BTC dominance 상승 = risk-off (ALT 약세).
             BTC dominance 하락 = risk-on (ALT 강세).
지표:
  - BTC/USDT price로 market regime 추론
  - BTC 변동성으로 ALT 포지션 조절
구현 난이도: 중간 (BTC 데이터를 auxiliary input으로)
TF 적합: 1D
주의: preprocessor에서 BTC 가격을 보조 입력으로 사용 가능
      (현재 인프라에서 지원 여부 확인 필요)
```

---

## 7. 최신 학술 트렌드 (2025-2026)

### 7-A. GT-Score (Golden Ticket Score)

```
출처: MDPI JRFM, 2026 / arXiv:2602.00080
GitHub: shep-analytics/gt_score
수식: GT-Score = (mu * ln(z) * r^2) / sigma_d
  mu      = 평균 수익 (성과)
  ln(z)   = Z-score 로그 (통계적 유의성, 과대 지배 방지)
  r^2     = R-squared (수익 일관성, outlier 전략 패널티)
  sigma_d = 하방 편차 (하방 리스크만 패널티)
결과: WFA Generalization Ratio 98% 향상 (0.185→0.365)
적용: 파라미터 최적화 시 Sharpe 대신 GT-Score를 objective로 사용.
      CPCV는 과적합 사후 감지, GT-Score는 사전 방지. 보완적 사용 권장.
```

### 7-B. AlphaEval: 백테스트 없는 Alpha 평가

```
출처: arXiv:2508.13174, 2025
개념: 5차원 alpha 품질 평가 프레임워크.
  1. Predictive Power (PPS) = beta*IC + (1-beta)*RankIC
  2. Temporal Stability (RRE) = Relative Rank Entropy
  3. Robustness (PFS) = Perturbation Fidelity Score
  4. Financial Logic = 경제적 해석 가능성
  5. Diversity (DH) = Eigenvalue-based diversity entropy
결과: Composite score 기반 포트폴리오가 IC-only 대비 수익/리스크 모두 우월.
적용: Gate 0 단계에서 IC + AlphaEval로 시그널 품질 사전 필터링.
```

### 7-C. QuantEvolve Framework

```
출처: arXiv:2510.18569, 2025
개념: 4개 전문 에이전트(Research/Coding/Evaluation/Data)의 진화적 루프.
      Quality-Diversity: Sharpe/Sortino/MDD/전략유형별 feature map으로 다양성 유지.
      Composite Score = SR + IR + MDD.
적용: 아이디어 생성 단계에서 "다양성"을 체계적으로 추구.
```

### 7-D. Human-in-the-Loop Quant

```
출처: Medium, Aymane Boutbati, 2025
개념: AI 생성 아이디어 + 인간의 경제적 논거 검증.
      Shadow Execution → Feature Distribution Monitoring → Kill Switch.
      과적합 방지: 인간이 "왜 작동하는가?"를 설명 못하면 폐기.
적용: 이 스킬의 Gate 0 스코어카드가 이 패턴을 따름.
```

### 7-E. 크립토 Microstructure 실증 (2026)

```
출처: arXiv:2602.00776, 2026.02
핵심: 5개 암호화폐에서 cross-asset 안정적 패턴 확인.
  - Order Flow Imbalance: tick size 큰 자산에서 강력한 단조 효과
  - VPIN: AUC > 0.55 (전 암호화폐), informed trading 탐지
  - Roll Measure: 주요 예측자 (유동성 proxy)
  - VWAP-to-Mid Deviation: 비대칭 효과, 미시구조 회귀
실전: 2025.10 flash crash에서 order book imbalance 기반 수익 달성.
```

### 7-F. Funding Rate Alpha (2025 실적)

```
출처: Gate.com Research, 1Token Quant Index
실적: 2025년 funding rate 연 수익률 19.26% (전년 14.39% 대비 +34%)
      Cross-platform arbitrage로 추가 3-5% 연간 수익.
      자본 유입 전년 대비 215% 증가.
주의: funding_rate 데이터 확보 필요 (현재 PENDING 상태).
```

---

## 미탐색 영역 우선순위 (2026-02-10 갱신)

> 46개 전략 시도, 45개 폐기 경험을 반영한 우선순위.
> Microstructure/Session/Seasonality/Liquidity 영역은 실패 확인됨.

```
1순위: Behavioral Finance (Disposition Effect, Anchoring)
       — 경제적 논거 강함, 미시도, OHLCV 직접 구현 가능
       — 처분 효과: 크립토 retail 지배 시장에서 효과 극대화 가능성

2순위: ML 앙상블 변형 (CTREND 외 다른 모델/피처셋)
       — 유일한 성공 패턴(CTREND) 확장
       — Random Forest, XGBoost, 다른 feature 조합 탐색
       — 주의: CTREND와 높은 상관 시 포트폴리오 가치 제한

3순위: Information Theory (Transfer Entropy, Sample Entropy)
       — 참신성 최고, 완전 미시도 영역
       — 구현 복잡도 중~높, numba 최적화 필요

4순위: Cross-Asset Signal (BTC Dominance → 단일에셋 변환)
       — BTC regime으로 ALT 포지션 조절
       — 보조 입력 인프라 필요

5순위: Vol Term Structure (RV_short/RV_long ratio)
       — 외부 데이터 불필요, 구현 쉬움
       — 단, Vol Structure/GK Breakout 실패 경험 → 단순 vol 비율 전략은 부족.
         추가 시그널 소스 결합 필수

⛔ 재시도 비권장 영역:
   - OHLCV 기반 Microstructure (BVC/OFI/VPIN) — L2 data 없이 불가
   - FX Session Decomposition — 크립토 24/7 비적합
   - Hour-of-day Seasonality — noise 과적합
   - Amihud 1H Liquidity Regime — 과빈번 전환
   - Carry/Pairs — 데이터 인프라 부재
```
