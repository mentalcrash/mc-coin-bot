---
name: p1-g0a-discover
description: >
  체계적 알파 리서치 워크플로우로 새 트레이딩 전략 아이디어를 발굴하고 검증한다.
  타임프레임별 적합 전략 탐색, Long/Short/Hedge 모드 평가, 경제적 논거 검증,
  폐기 전략 회피, 단일 에셋 전용.
  사용 시점: (1) 새 전략 아이디어가 필요할 때,
  (2) "전략 발굴", "전략 탐색", "알파 리서치", "새 전략 찾기" 요청 시,
  (3) 특정 타임프레임/시장 조건에 맞는 전략을 찾을 때,
  (4) 학술 논문이나 리서치에서 전략을 포팅할 때,
  (5) 기존 전략 포트폴리오의 다양성을 높이고 싶을 때.
argument-hint: <timeframe or theme>
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - WebSearch
  - WebFetch
---

# Strategy Discovery — 체계적 알파 리서치

## 역할

**시니어 크립토 퀀트 리서처**로서 행동한다.
핵심 판단 기준: "이 전략에 경제적 논거가 있고, 과적합 없이 전 시장환경에서 작동하는가?"

## 핵심 원칙

1. **경제적 논거 우선**: 통계적 패턴만으로는 불충분. "왜 이 edge가 존재하는가?"를 설명할 수 있어야 한다
2. **참신성 추구**: 이미 폐기된 45개 전략과 차별화. 동일 지표 조합 재시도 금지
3. **전 시장환경 대응**: 특정 레짐(상승장/하락장)에만 작동하는 전략 지양
4. **단일 에셋 전용**: 멀티에셋/횡단면 전략은 범위 밖 (포트폴리오는 PM이 처리)
5. **Long/Short 다양성**: DISABLED/HEDGE_ONLY/FULL 모든 ShortMode를 검토
6. **크립토 네이티브 edge**: 전통금융 전략의 단순 포팅은 위험. 크립토 24/7 시장 특성에 맞는 edge 필요 (교훈 #13~#16)
7. **CTREND 상관 최소화**: 유일한 활성 전략과 낮은 상관이 포트폴리오 가치 극대화

## 워크플로우 (7단계)

### Step 0: 컨텍스트 수집

시작 전 반드시 다음을 확인한다:

```
1. 파이프라인 현황 확인 (필수 — CLI 사용)
   uv run mcbot pipeline status    # 상태별 카운트 (ACTIVE/RETIRED/CANDIDATE 등)
   uv run mcbot pipeline table     # 전체 전략 Gate 진행도

2. 교훈 데이터 참조 (필수 — 실패 반복 방지)
   uv run mcbot pipeline lessons-list              # 전체 교훈 목록
   uv run mcbot pipeline lessons-list --tf {TF}    # 타겟 TF 관련 교훈 필터
   uv run mcbot pipeline lessons-list -c strategy-design   # 전략 설계 교훈
   uv run mcbot pipeline lessons-list -c market-structure  # 시장 구조 교훈
   # → 교훈에서 명시된 안티패턴/실패 유형을 아이디어 생성 시 반드시 회피

3. 타겟 타임프레임 확인 (미지정 시 사용자에게 질문)
   - 1D (일봉): 가장 안정적, 비용 효율적. 프로젝트 주력. 유일한 G5 PASS가 1D
   - 4H: 중간 빈도. 비용과 신호 밸런스
   - 1H: 높은 빈도. Tier 5 4종 전멸 (교훈 #13~#16). 극히 신중하게 접근
   - 1m→aggregation: EDA 전용 (CandleAggregator 활용)

4. 현재 포트폴리오 구성 확인
   uv run mcbot pipeline list --status ACTIVE   # 활성 전략
   # G5 도달까지: G1 통과율 ~50%, G2 통과율 ~20%, G4 통과율 ~5%

5. 폐기 전략 실패 패턴 확인 (필수)
   uv run mcbot pipeline list --status RETIRED   # YAML 기반 동적 조회
   references/discarded-strategies.md의 "실패 패턴 요약" 섹션 참조
   동일 접근법 재시도 금지
```

### Step 1: 아이디어 생성 — IC 기반 시그널 탐색

학술 논문, 시장 미시구조, 행동 재무학에서 아이디어를 도출한다.

#### 1-A. 아이디어 소싱 채널 (우선순위)

```
1. 학술 논문 (SSRN, arXiv, Journal of Financial Economics)
   - 크립토 특화 최신 논문 검색
   - "왜 작동하는가?"의 경제적 설명이 있는 논문만

2. 시장 미시구조 (Microstructure)
   - Order book imbalance, VPIN, Roll measure
   - Funding rate dynamics, liquidation cascades
   - Basis spread (spot-perp), contango/backwardation

3. 변동성 구조 (Volatility Surface)
   - Realized vs implied vol spread
   - Vol-of-vol, term structure slope
   - Intraday volatility patterns (U-shape, overnight gap)

4. 정보 이론 (Information Theory)
   - Transfer entropy (lead-lag between assets)
   - Mutual information for feature selection
   - Approximate entropy for regime detection

5. 행동 재무학 (Behavioral Finance)
   - Disposition effect → delayed mean reversion
   - Anchoring bias → support/resistance persistence
   - Herding → momentum continuation

6. 대안 데이터 시그널 (Alternative Data)
   - On-chain: whale flow, exchange net flow
   - Social sentiment: fear/greed aggregation
   - Funding rate carry (PENDING 상태 — 데이터 확보 시)
```

참고: [references/idea-sources.md](references/idea-sources.md)

#### 1-B. 시그널 품질 사전 검증 (IC + AlphaEval)

아이디어가 전략으로 발전하기 전에, 시그널의 예측력을 검증한다.

**IC (Information Coefficient) — 필수:**

```python
# IC = rank_correlation(signal[t], forward_return[t+1])
# 기준:
#   |IC| > 0.02: 유의미한 시그널
#   |IC| > 0.05: 강한 시그널
#   IC의 t-stat > 2.0: 통계적 유의
#   IR = mean(IC) / std(IC) > 0.5: 안정적
```

**AlphaEval 5차원 — 권장 (arXiv:2508.13174):**

```
1. Predictive Power (PPS): IC + RankIC 결합
2. Temporal Stability (RRE): Rolling IC의 순위 엔트로피 → 시간 안정성
3. Robustness (PFS): 입력 노이즈 섭동 후 시그널 유지율
4. Financial Logic: 경제적 논거 설명 가능성 (Gate 0 [1]과 연동)
5. Diversity (DH): 기존 전략 시그널과의 독립성 (상관 eigenvalue 분석)
```

**IC 검증 실패 시 → 전략 개발 진행하지 않는다.**

### Step 2: 아이디어 평가 — Gate 0 스코어카드

6가지 기준을 각각 1~5점으로 평가한다.

```
══════════════════════════════════════════════════════
  GATE 0: IDEA VIABILITY SCORECARD
  전략: [이름]  |  타임프레임: [TF]
══════════════════════════════════════════════════════

  [1] 경제적 논거 (Economic Rationale)        : _/5
      "왜 이 edge가 존재하는가?"
      1=없음, 3=가설수준, 5=학술논문+실증

  [2] 참신성 (Novelty)                        : _/5
      기존 26개 전략과 차별화 정도
      1=동일지표, 3=새조합, 5=새카테고리

  [3] 데이터 확보 (Data Availability)          : _/5
      OHLCV만으로 구현 가능?
      1=외부API필수, 3=파생계산, 5=OHLCV직접

  [4] 구현 복잡도 (Implementation Complexity)  : _/5
      4-파일 구조에 맞는가?
      1=인프라변경필요, 3=중간, 5=직관적

  [5] 용량 수용 (Capacity)                     : _/5
      단일에셋에서 충분한 거래 빈도?
      1=월2건미만, 3=주1-2건, 5=일1건+

  [6] 레짐 독립성 (Regime Independence)        : _/5
      전 시장환경에서 작동?
      1=특정레짐전용, 3=2/3레짐, 5=전레짐

──────────────────────────────────────────────────────
  TOTAL: __/30
  판정: PASS (>=18) / WATCH (12-17) / FAIL (<12)
══════════════════════════════════════════════════════
```

**PASS (18점 이상)인 경우에만 Step 3으로 진행한다.**

### Step 3: 중복 검사 — 폐기 전략 회피 + 교훈 검증

[references/discarded-strategies.md](references/discarded-strategies.md)의 **"실패 패턴 요약"** 테이블과 **교훈 데이터**를 함께 확인한다:

```
0. 교훈 데이터 매칭 (최우선 — 프로그래매틱 검색)
   # 아이디어 관련 교훈이 있는지 확인
   uv run mcbot pipeline lessons-list --tf {TF}        # 타겟 TF 교훈
   uv run mcbot pipeline lessons-list -s {관련전략}     # 유사 전략 교훈
   uv run mcbot pipeline lessons-list -t {키워드}       # 태그 검색
   # 교훈이 명시하는 안티패턴과 아이디어가 일치하면 → 즉시 폐기 또는 수정
   # 예: lessons-list --tf 1H → "FX Session ≠ Crypto", "BVC 근사 한계" 등 확인

1. 실패 패턴 매칭 (빠른 체크)
   - 단일 지표 trend-following? → Decay 56~92% (TSMOM 외 6개 전멸)
   - 레짐 감지 = 전략? → ADX/HMM/Hurst/AC/VR 등 7개 전멸
   - OHLCV 기반 microstructure? → BVC 정밀도 부족 (VPIN-Flow, Flow-Imbalance)
   - 전통금융→크립토 단순 전이? → Session/Amihud/Seasonality 4종 전멸
   - 동일 TF에서 Mom+MR 블렌딩? → alpha 상쇄
   - 1-bar hold? → 비용 > 수익
   - 밈코인 FULL Short? → MDD 무한대 위험

2. 동일 핵심 지표 사용 여부 확인
   - 예: RSI 단독 → RSI Crossover 폐기됨
   - 예: Donchian 단독 → Donchian 폐기됨 (Decay 91%)
   - 예: BVC/OFI/VPIN → Flow-Imbalance, VPIN-Flow 폐기됨 (OHLCV 한계)
   - 예: Amihud → Liq-Momentum 폐기됨 (1H 과빈번 전환)
   - 예: Hour-of-day t-stat → Hour-Season 폐기됨 (noise 과적합)

3. 차별화 포인트 명시
   - 어떤 점이 폐기 전략과 다른가?
   - 새로운 데이터 소스, 새로운 수학적 접근, 또는 근본적으로 다른 edge가 있는가?
   - "파라미터 조정"이나 "필터 추가"만으로는 차별화 불충분
   - 관련 교훈 번호를 명시하여 왜 이번에는 다른지 설명

→ 차별화 불충분 시 아이디어를 수정하거나 폐기한다.
```

### Step 4: 전략 설계 — ShortMode + TF 적합성

#### 4-A. ShortMode 결정 매트릭스

```
                    | 추세추종     | 평균회귀     | 변동성      | 미시구조   |
─────────────────---|-------------|-------------|------------|-----------|
Long-Only (0)       | 상승장 최적  | 고저 반등    | Vol 매수    | Bid-side  |
Hedge-Only (1)      | MDD 방어    | 양방향 진입  | Straddle형  | 양측 유동성|
Full Short (2)      | 하락장 수익  | 평균회귀 원래| Vol 매도    | Ask-side  |
─────────────────---|-------------|-------------|------------|-----------|
권장                | HEDGE_ONLY  | FULL        | HEDGE_ONLY | FULL      |
```

#### 4-B. 타임프레임 적합성

```
TF    | 적합 전략 유형           | 비용 영향 | 주의점
──────|------------------------|----------|──────────────────
1D    | Trend, Vol Structure   | 최소     | 거래 빈도 월 2건 이상 필수
4H    | Mean Reversion, Micro  | 중간     | 비용 < 수익의 30%
1H    | HF Mean Rev, Scalping  | 높음     | Maker-only 또는 낮은 빈도
1m    | Intrabar SL/TS 전용    | N/A      | 전략 자체가 아닌 PM 보조
```

#### 4-C. 설계 문서 작성

다음 항목을 정리한다:

```
1. 전략 이름 (kebab-case): ___
2. 핵심 가설: ___
3. 경제적 논거: ___
4. 사용 지표: [목록]
5. 시그널 생성 로직 (수식): ___
6. ShortMode: DISABLED / HEDGE_ONLY / FULL
7. 타임프레임: 1D / 4H / 1H
8. 예상 거래 빈도: ___/년
9. 예상 Sharpe 범위: ___
10. CTREND와의 상관관계 예측: 낮음/중간/높음
```

**CTREND와 낮은 상관관계가 예상될수록 포트폴리오 가치가 높다.**

### Step 4.5: YAML 파이프라인 등록

Gate 0 PASS인 아이디어를 `pipeline create` CLI로 YAML에 등록한다.

```bash
uv run mcbot pipeline create {strategy-name} \
  --display-name "{DisplayName}" \
  --category "{Category}" \
  --timeframe "{TF}" \
  --short-mode "{HEDGE_ONLY|FULL|DISABLED}" \
  --status CANDIDATE
```

> YAML이 Single Source of Truth. 별도 임시 파일 불필요.

**핵심 가설**: (1-2문장)

**경제적 논거**: (왜 이 edge가 존재하는가)

**사용 지표**: (목록)

**시그널 생성 로직**:
(수식 또는 의사코드)

**CTREND 상관 예측**: 낮음 / 중간 / 높음

**예상 거래 빈도**: __건/년

**차별화 포인트**: (폐기 전략 대비 어떤 점이 다른가)

**출처**: (논문/URL/독자적)

**Gate 0 상세 점수**:
- 경제적 논거: _/5
- 참신성: _/5
- 데이터 확보: _/5
- 구현 복잡도: _/5
- 용량 수용: _/5
- 레짐 독립성: _/5

---
```

#### 상태 전이

```
🔵 후보 → 사용자 승인 → 🟡 구현중 → Gate 1+ → ✅ 구현완료 / ❌ 폐기
                ↘ 사용자 거부 → ❌ 폐기 (사유 기록)
```

### Step 4.6: YAML 메타데이터 생성 (Gate 0A PASS 시 필수)

Gate 0 PASS인 아이디어는 `pipeline create` CLI로 YAML을 생성한다:

```bash
uv run mcbot pipeline create {registry-name} \
  --display-name "{Display Name}" \
  --category "{카테고리}" \
  --timeframe {TF} \
  --short-mode {SHORT_MODE} \
  --rationale "{경제적 논거}" \
  --g0a-score {점수}
```

- `strategies/{registry-name}.yaml` 생성 (status: CANDIDATE, G0A: PASS)
- 생성 후 Dashboard 갱신: `uv run mcbot pipeline report`

### Step 5: 구현 위임

설계가 완료되고 **사용자 승인**을 받으면 구현을 진행한다.

```
구현 시 주의사항:
- preprocessor.py: shift() 금지, vectorized ops only
- signal.py: shift(1) 필수, ShortMode 3가지 분기
- strategy.py: @register(), from_params(), recommended_config()
- warmup_periods(): 사용 지표 중 max(window) + 여유분
```

### Step 6: 백테스트 실행 + 해석

#### 6-A. 표준 백테스트 실행

```bash
# 5개 에셋, 6년 데이터 (Gate 1 표준)
for symbol in BTC/USDT ETH/USDT BNB/USDT SOL/USDT DOGE/USDT; do
  uv run mcbot backtest run {strategy-name} $symbol \
    --start 2020-01-01 --end 2025-12-31
done
```

#### 6-B. Gate 1 판정 기준 (Best Asset 기준)

```
PASS: Sharpe > 1.0 AND CAGR > 20% AND MDD < 40% AND 거래 50건+
WATCH: 0.5 <= Sharpe <= 1.0, 또는 25% <= MDD <= 40%
FAIL: 총수익 음수 + 거래 20건 미만, 또는 MDD > 50%
```

#### 6-C. 결과 해석 위임

`/backtest-interpret` 스킬로 상세 해석을 위임한다.

### Step 7: 반복 또는 종료 결정

```
Gate 1 PASS → Gate 2 IS/OOS 진행 (별도 세션)
Gate 1 WATCH → 파라미터 조정 후 재실행 (최대 2회)
Gate 1 FAIL → 아이디어 폐기, Step 1로 복귀
```

**파라미터 최적화 시 GT-Score 활용 권장** (arXiv:2602.00080):

```
GT-Score = (mu * ln(z) * r^2) / sigma_d
  mu      = 평균 수익 (성과)
  ln(z)   = Z-score 로그 (통계적 유의성, 과대 지배 방지)
  r^2     = R-squared (수익 일관성, outlier 전략 패널티)
  sigma_d = 하방 편차 (하방 리스크만 패널티)

→ Sharpe 대신 GT-Score를 sweep objective로 사용 시
  Walk-Forward Generalization Ratio 98% 향상 (0.185→0.365)
```

## 아이디어 발상 보조 도구

사용자가 아이디어가 없을 때, 다음 프롬프트로 탐색을 돕는다:

### 카테고리별 커버리지 확인

```
현재 포트폴리오 커버리지를 동적으로 확인:
  uv run mcbot pipeline status         # 전체 상태 카운트
  uv run mcbot pipeline list --status RETIRED   # 폐기 전략 카테고리 확인
  uv run mcbot pipeline list --status ACTIVE    # 활성 전략 확인

미탐색 영역 (판단 영역 — 폐기 전략 카테고리와 대조하여 확인):
  - Behavioral Finance — Disposition effect, anchoring
  - Information-Theoretic — Transfer entropy, mutual information
  - Cross-Asset Signal — BTC dominance → 단일에셋 신호 변환
  - ML 앙상블 변형 — CTREND 외 다른 ML 접근

위 목록은 발견된 미탐색 영역 가이드이며, pipeline list로 확인한 폐기 목록과 대조하여 최신 상태를 반영한다.
```

### 최신 학술 리서치 탐색

아이디어가 필요할 때 다음을 WebSearch로 조사한다:

```
검색 쿼리 예시:
  - "cryptocurrency trading strategy {year} SSRN"
  - "crypto microstructure alpha order book"
  - "volatility premium cryptocurrency futures"
  - "behavioral finance crypto disposition effect"
  - "information theory transfer entropy crypto"
  - "{specific-indicator} trading strategy backtest"
```

## 출력 형식

### 1. 콘솔 리포트 (사용자에게 표시)

```
══════════════════════════════════════════════════════
  STRATEGY DISCOVERY REPORT
  타임프레임: [TF]  |  세션: [날짜]
══════════════════════════════════════════════════════

  아이디어 #1: [이름]
  ──────────────────────────────────────────────────
  카테고리     : [Microstructure / Vol / Carry / ...]
  핵심 가설    : [1-2문장]
  경제적 논거  : [왜 이 edge가 존재하는가]
  참신성       : [기존 전략과 차별화 포인트]
  사용 지표    : [목록]
  ShortMode    : [DISABLED / HEDGE_ONLY / FULL]
  예상 빈도    : [거래/년]
  Gate 0 점수  : [__/30]
  레짐 독립성  : [전 레짐 / 2-3 레짐]
  CTREND 상관  : [낮음 / 중간 / 높음]
  출처         : [논문/블로그/독자적]
  ──────────────────────────────────────────────────

  아이디어 #2: ...

══════════════════════════════════════════════════════
  권장 액션
──────────────────────────────────────────────────────
  1순위: [아이디어 #X] → 구현 진행
  근거: [선택 이유]
  📄 YAML 등록: strategies/{name}.yaml (pipeline create)
══════════════════════════════════════════════════════
```

### 2. 후보 문서 기록 (자동)

Gate 0 PASS 아이디어는 **자동으로** `pipeline create` CLI로 YAML에 등록한다.
Step 4.5의 절차를 따른다.

## 안티패턴 — 반드시 피해야 할 것

```
1. 지표 수프 (Indicator Soup)
   - 5개 이상 지표 조합 → 과적합 확률 급증
   - 예외: CTREND처럼 ML 앙상블은 체계적 차원 축소 필요

2. 동일 TF에서 반대 전략 혼합
   - Momentum + Mean Reversion 블렌딩 → alpha 상쇄
   - Mom-MR Blend FAIL의 교훈

3. 1-bar hold 전략
   - 진입 → 즉시 청산 → 비용이 수익 초과
   - Larry-VB FAIL: 연 125건 x 0.1% = 12.5% drag

4. 단일 지표 의존
   - RSI만, MACD만, Donchian만 → 대부분 과적합
   - 최소 2개 독립 시그널 소스 결합 필요

5. 특정 레짐 타겟
   - "상승장에서만 작동" → 전체 기간 Sharpe 저하
   - 레짐 필터는 보조 역할로만 (시그널 강도 조절)

6. IS Sharpe > 3.0 추구
   - 과적합의 강력한 신호
   - 현실적 목표: IS Sharpe 1.0~2.0

7. 전통금융 전략의 무비판적 크립토 포팅 (신규)
   - FX session decomposition → 크립토 24/7에서 edge 소멸
   - Equity microstructure (Amihud) → 1H 크립토에서 과빈번 전환
   - Intraday seasonality → noise를 패턴으로 오인
   - 반드시 "크립토 시장에서 왜 작동하는가?"를 검증

8. OHLCV로 microstructure alpha 추구 (신규)
   - BVC 근사 (close-low)/(high-low)는 1D→1H 모두 불충분
   - VPIN-Flow (1D, 거래 0건), Flow-Imbalance (1H, Sharpe 음수) 연속 실패
   - L2 order book 또는 tick data 없이는 진정한 flow 시그널 불가

9. 레짐 감지 자체를 전략으로 사용 (신규)
   - ADX, HMM, Hurst, AC, VR, Entropy 등 7개 전략 전멸
   - 레짐 감지는 "필터/오버레이"로만 사용, 독립 alpha 소스 아님
```

## 체크리스트

완료 전 확인:

- [ ] **교훈 데이터 확인됨** (`lessons-list`로 관련 TF/카테고리/전략 교훈 검토)
- [ ] Gate 0 스코어카드 작성됨 (18점 이상)
- [ ] 폐기 전략과 중복 없음 확인 (교훈 + discarded-strategies 교차 검증)
- [ ] 경제적 논거 1문단 이상 작성됨
- [ ] ShortMode 3가지 중 근거 있는 선택
- [ ] TF 적합성 확인 (비용 영향 포함)
- [ ] CTREND와의 예상 상관관계 평가됨
- [ ] 설계 문서 10개 항목 모두 작성됨
- [ ] **`pipeline create` 실행하여 YAML 생성됨** (Gate 0A PASS 시 필수)
- [ ] **`pipeline report` 실행하여 dashboard 갱신됨**
