---
name: p1-research
description: >
  새 트레이딩 전략 아이디어 발굴 + Phase 1 검증 (IC/스코어카드/YAML 등록).
  사용 시점: 전략 발굴, 알파 리서치, 전략 탐색 요청 시.
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
판단 기준: "이 전략에 경제적 논거가 있고, 과적합 없이 전 시장환경에서 작동하는가?"

## 핵심 원칙

### A. 전략 철학 (Strategy Philosophy)

1. **비효율 원천 우선 (Inefficiency-First Thinking)**:
   전략보다 "왜 이 비효율이 존재하는가"를 먼저 생각한다.
   단순히 차트 패턴을 찾는 것이 아니라, 시장 미시구조, 행동재무학적 편향,
   정보 비대칭 같은 **경제적 논리가 있는 비효율**을 찾는다.
   "이 전략이 왜 돈을 벌 수 있는지" 설명할 수 없으면 과적합이다.

   비효율 원천 5가지 프레임워크:
   - **행동 편향**: 처분 효과, 앵커링, 군집 행동, 과잉 반응/과소 반응
   - **정보 비대칭**: Informed trader 활동, 온체인 whale flow, 거래소 간 자금 흐름
   - **구조적 제약**: 펀딩레이트 메커니즘, 레버리지 청산 캐스케이드, 마진 콜
   - **유동성 프리미엄**: 비유동성 보상, 거래소 간 스프레드, 시간대별 유동성 편차
   - **리스크 프리미엄**: 변동성 리스크 프리미엄, 캐리, 보험 판매자 역할

1. **약한 알파의 포트폴리오 결합 (Weak Alpha Portfolio)**:
   하나의 "성배" 전략을 찾지 않는다. Sharpe 0.3~0.5인 약한 시그널이라도
   상관관계가 낮으면 포트폴리오로 결합 시 전체 Sharpe가 크게 상승한다.
   `Sharpe_portfolio ≈ sqrt(N) * avg(Sharpe) * sqrt(1 - avg(rho))`
   → 단독 Sharpe 1.0+ 추구보다, 상관 < 0.3인 Sharpe 0.5 전략 3개가 낫다.

1. **대안 데이터 차별화 (Alternative Data Edge)**:
   가격/거래량 데이터만으로는 이미 너무 많은 참여자가 같은 것을 본다.
   온체인 데이터(whale wallet, DEX 유동성, 거래소 간 자금 흐름),
   Derivatives(펀딩레이트, OI), 소셜 센티먼트 등 차별화된 데이터를 우선한다.
   — 현재 인프라: DerivativesDataService (FR/OI/LS Ratio/Taker Ratio)
   — On-chain: OnchainDataService (22 datasets + 12 CoinMetrics, oc_*auto-enrich in EDA)
   — 주요 oc_* 컬럼: oc_mvrv, oc_flow_in_ex_usd, oc_flow_out_ex_usd,
     oc_stablecoin_total_circulating_usd, oc_tvl_usd, oc_fear_greed,
     oc_adractcnt, oc_txcnt, oc_mktcap_usd, oc_supply 등
   — Sentiment: Fear & Greed Index (oc_fear_greed, 2018~)

1. **거래 비용의 극사실적 모델링 (Realistic Cost Modeling)**:
   백테스트에서 연 50% 수익이 나와도 실제 거래 비용, 슬리피지,
   마켓 임팩트를 반영하면 마이너스가 되는 전략이 허다하다.
   특히 크립토는 스프레드가 넓고 유동성이 불안정하다.
   - **비용 체크**: 연간 거래 비용 = 거래 수 x 2(왕복) x 수수료율 x 평균 포지션
   - **비용 비율**: 거래 비용 / 총수익 < 30% (초과 시 edge 잠식)
   - **슬리피지**: 대형코인 0.01~0.03%, 소형코인 0.05~0.1% 추가

### B. 운영 원칙 (Operational Principles)

1. **참신성 추구**: 폐기된 45개 전략과 차별화. 동일 지표 조합 재시도 금지
1. **전 시장환경 대응**: 특정 레짐 전용 지양. RegimeService 적응적 대응 권장
1. **단일 에셋 전용**: 멀티에셋/횡단면은 범위 밖 (PM이 처리)
1. **적극적 Long/Short 활용 (One-way Mode)**: 거래소는 One-way Mode(심볼당 단일 방향, netting).
   동일 심볼 Long+Short 동시 보유 불가. 포지션 전환은 flat 거치거나 방향 flip.
   ShortMode.FULL/HEDGE_ONLY를 기본 검토하며, 하락장 alpha를 적극 추구한다.
   DISABLED는 명확한 근거가 있을 때만 선택
1. **크립토 네이티브 edge**: 전통금융 단순 포팅 위험 (교훈 #13~#16)
1. **CTREND 상관 최소화**: 유일한 활성 전략과 낮은 상관이 포트폴리오 가치 극대화
1. **RegimeService 활용**: 공유 레짐 인프라로 적응형 설계 가능
1. **앙상블 기여도 관점**: 단독 Sharpe 0.5+라도 낮은 상관 + 독립 alpha면 앙상블로 Sharpe 0.8~1.0 달성 가능
1. **On-chain 데이터 우선 탐색**: OHLCV-only 전략보다 edge 감쇠 느림 (수월~수분기).
    22개 데이터셋 + 12개 CoinMetrics 인프라 완비, 전략 0개 사용 중 → 최우선 탐색 영역.
    idea-sources.md § 6.5에 구체적 전략 후보 (Stablecoin Flow, MVRV, Exchange Flow 등)

## 워크플로우 (7단계)

### Step 0: 컨텍스트 수집

시작 전 반드시 다음을 확인한다:

```
1. 파이프라인 현황 확인 (필수 — CLI 사용)
   uv run mcbot pipeline status    # 상태별 카운트 (ACTIVE/RETIRED/CANDIDATE 등)
   uv run mcbot pipeline table     # 전체 전략 Phase 진행도

2. 교훈 데이터 참조 (필수 — 실패 반복 방지)
   uv run mcbot pipeline lessons-list              # 전체 교훈 목록
   uv run mcbot pipeline lessons-list --tf {TF}    # 타겟 TF 관련 교훈 필터
   uv run mcbot pipeline lessons-list -c strategy-design   # 전략 설계 교훈
   uv run mcbot pipeline lessons-list -c market-structure  # 시장 구조 교훈
   # → 교훈에서 명시된 안티패턴/실패 유형을 아이디어 생성 시 반드시 회피

3. 타겟 타임프레임 확인 (미지정 시 사용자에게 질문)
   - 1D (일봉): 가장 안정적, 비용 효율적. 프로젝트 주력. 유일한 P7 PASS가 1D
   - 4H: 중간 빈도. 비용과 신호 밸런스
   - 1H: 높은 빈도. Tier 5 4종 전멸 (교훈 #13~#16). 극히 신중하게 접근
   - 1m→aggregation: EDA 전용 (CandleAggregator 활용)

4. 현재 포트폴리오 구성 확인
   uv run mcbot pipeline list --status ACTIVE   # 활성 전략
   # P7 도달까지: P4 통과율 ~50%, P4 IS/OOS 통과율 ~20%, P6 통과율 ~5%

5. 폐기 전략 실패 패턴 확인 (필수)
   uv run mcbot pipeline list --status RETIRED   # YAML 기반 동적 조회
   references/discarded-strategies.md의 "실패 패턴 요약" 섹션 참조
   동일 접근법 재시도 금지

6. On-chain/Sentiment 데이터 활용 검토 (권장 — 미탐색 블루오션)
   uv run mcbot ingest onchain info              # 22개 데이터셋 인벤토리
   # → 전략 아이디어에 on-chain 데이터를 포함할 수 있는지 우선 검토
   # → idea-sources.md § 6.5 참조: Stablecoin Flow, MVRV, Exchange Flow,
   #   Fear&Greed, Network Activity 등 구체적 전략 후보 목록
   # → 현재 전략 0개 사용 중 — OHLCV-only 전략보다 edge 감쇠 느림
```

#### 0-Z. 탐색 모드 선택

| 모드 | 언제 선택 | 다음 Step |
|------|----------|----------|
| **S (Single)** | 새로운 단일 전략 발굴 | Step 1 (기존 워크플로우) |
| **E (Ensemble)** | 기존 전략 N개 조합 탐색 | Step 1E |
| **S→E (Hybrid)** | 새 전략 발굴 후 앙상블 편입 | Step 1 → Step 5.5E |

판단 기준:

- 활성/후보 전략 2개+ → E 모드 검토 권장
- "앙상블"/"조합"/"포트폴리오" 키워드 → E 모드
- 신규 아이디어 탐색 → S 모드 (기본값)

---

### Step 1: 아이디어 생성 — 비효율 원천 → IC 검증

**전략 기법이 아닌 "시장 비효율"에서 출발한다.**

#### 1-0. 비효율 원천 분석 (필수 — 아이디어 생성 전)

아이디어를 찾기 전에, 먼저 어떤 비효율을 타겟으로 할지 결정한다:

```
질문 체크리스트 (3개 이상 명확한 답변 필수):
  ✓ 이 비효율은 왜 존재하는가? (행동 편향 / 정보 비대칭 / 구조적 제약 / 유동성 / 리스크)
  ✓ 누가 이 비효율의 반대편에 서는가? (noise trader / hedger / 규제 제약 참여자)
  ✓ 왜 이 비효율이 아직 차익거래되지 않았는가? (자본 제약 / 실행 비용 / 정보 지연)
  ✓ 이 비효율은 시장 구조 변화로 사라질 수 있는가? (지속성 평가)
  ✓ 크립토 시장에서 이 비효율이 전통 금융보다 강한가/약한가?

→ 3개 미만 답변 시 아이디어 생성 진행하지 않는다.
→ 비효율 원천이 명확할수록 과적합 위험이 낮아진다.
```

#### 1-A. 아이디어 소싱 채널 (우선순위)

| # | 채널 | 설명 |
|---|------|------|
| 1 | 학술 논문 (SSRN, arXiv, JFE) | 크립토 특화 + 경제적 설명 있는 논문 |
| 2 | 시장 미시구조 | Order book imbalance, VPIN, funding rate, basis spread |
| 3 | 변동성 구조 | Realized vs implied vol, VoV, term structure |
| 4 | 정보 이론 | Transfer entropy, mutual information, approximate entropy |
| 5 | 행동 재무학 | Disposition effect, anchoring, herding |
| 6 | 대안 데이터 | ✅ On-chain 22개 (stablecoin flow, TVL, DEX vol, fear&greed 등) |
| 7 | Derivatives | ✅ FR(백테스팅 O), OI/LS/Taker(30일 제한), EDA auto-enrich |

상세: [references/idea-sources.md](references/idea-sources.md)

#### 1-B. 시그널 품질 사전 검증 (IC + AlphaEval)

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
4. Financial Logic: 경제적 논거 설명 가능성 (Phase 1 [1]과 연동)
5. Diversity (DH): 기존 전략 시그널과의 독립성 (상관 eigenvalue 분석)
```

**IC 검증 실패 시 → 전략 개발 진행하지 않는다.**

### Step 2: 아이디어 평가 — Phase 1 스코어카드

6가지 기준을 각각 1~5점으로 평가한다.

```
══════════════════════════════════════════════════════
  PHASE 1: IDEA VIABILITY SCORECARD
  전략: [이름]  |  타임프레임: [TF]
══════════════════════════════════════════════════════

  [1] 비효율 원천 (Inefficiency Source)         : _/5
      "왜 이 비효율이 존재하고, 왜 사라지지 않는가?"
      1=설명불가, 2=패턴관찰만, 3=가설수준,
      4=경제적논거+반대편 식별, 5=학술실증+지속성근거

  [2] 참신성 (Novelty)                        : _/5
      기존 26개 전략과 차별화 정도
      1=동일지표, 3=새조합, 5=새카테고리

  [3] 데이터 확보 (Data Availability)          : _/5
      1=외부API필수(미구축), 2=유료API필요, 3=파생계산,
      4=Derivatives/On-chain(Silver 구축완료), 5=OHLCV직접

  [4] 구현 복잡도 (Implementation Complexity)  : _/5
      1=인프라변경필요, 3=중간, 5=직관적

  [5] 용량 수용 (Capacity)                     : _/5
      1=월2건미만, 3=주1-2건, 5=일1건+

  [6] 레짐 적응성 (Regime Adaptability)         : _/5
      1=특정레짐전용(무적응), 3=2/3레짐 or 적응형, 5=전레짐 or 확률가중적응

──────────────────────────────────────────────────────
  TOTAL: __/30
  판정: PASS (>=18) / WATCH (12-17) / FAIL (<12)

  [선택] 앙상블 기여도 평가
  기존 활성 전략과의 상관: 낮음(<0.3) / 중간(0.3~0.6) / 높음(>0.6)
  → 상관 < 0.3이면 단독 Sharpe 0.5+ 수준에서도 앙상블 PASS 고려
══════════════════════════════════════════════════════
```

**PASS (18점 이상)인 경우에만 Step 3으로 진행한다.**

### Step 3: 중복 검사 — 폐기 전략 회피 + 교훈 검증

[references/discarded-strategies.md](references/discarded-strategies.md)와 교훈 데이터를 확인한다:

```
0. 교훈 데이터 매칭 (최우선)
   uv run mcbot pipeline lessons-list --tf {TF}
   uv run mcbot pipeline lessons-list -s {관련전략}
   uv run mcbot pipeline lessons-list -t {키워드}
   # 교훈 안티패턴과 일치 → 즉시 폐기 또는 수정

1. 실패 패턴 매칭
   - 단일 지표 trend-following? → Decay 56~92%
   - 레짐 감지 = 전략? → 7개 전멸
   - OHLCV microstructure? → BVC 정밀도 부족
   - 전통금융→크립토 단순 전이? → 4종 전멸
   - 동일 TF Mom+MR 블렌딩? → alpha 상쇄
   - 1-bar hold? → 비용 > 수익

2. 동일 핵심 지표 사용 여부 확인
   - RSI/Donchian/BVC/VPIN/Amihud/Hour-of-day 등 폐기 확인

3. 차별화 포인트 명시
   - 새 데이터 소스, 새 수학적 접근, 근본적으로 다른 edge
   - "파라미터 조정"/"필터 추가"만으로는 불충분
   - 관련 교훈 번호 명시

→ 차별화 불충분 시 수정하거나 폐기.
```

### Step 4: 전략 설계 — ShortMode + TF + 레짐 적응

#### 4-0. RegimeService 활용 설계

공유 RegimeService가 StrategyEngine을 통해 6개 레짐 컬럼을 자동 주입한다.

| 패턴 | 설명 | 적합 상황 |
|------|------|----------|
| A. 확률 가중 | 레짐 확률로 파라미터 연속 조절 | 부드러운 전환 필요 시 |
| B. 조건부 필터 | 특정 레짐에서 시그널 활성화/비활성화 | 명확한 On/Off 로직 |
| C. 방향 가중 | trend_direction/strength로 시그널 가중 | 추세 방향 활용 시 |

**올바른 vs 잘못된 사용:**

- OK: 기존 alpha에 레짐을 오버레이/필터로 적용 (사이징/강도 조절)
- OK: `regime_service=None` 시 기본 동작 유지 (backward compatible)
- NG: 레짐 전환 자체를 매매 시그널로 사용 (7개 전략 전멸)
- NG: 레짐 없이 시그널이 생성되지 않는 구조

상세: [references/regime-design.md](references/regime-design.md)

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

> **One-way Mode 제약:**
>
> - 동일 심볼 Long+Short 동시 보유 불가 (netting — 반대 주문은 기존 포지션 축소/청산)
> - HEDGE_ONLY = 수비적 숏 허용 (동시 보유 아님, 롱 청산 후 숏 진입)
> - 시그널 상태: +1(롱) / 0(플랫) / -1(숏) — 혼합 상태 없음
> - 기본 권장: FULL 또는 HEDGE_ONLY (DISABLED는 명확한 근거 시만)
> - 밈코인(DOGE 등) FULL Short 주의 (교훈 #10: MDD -622%)

#### 4-B. 타임프레임 적합성

```
TF    | 적합 전략 유형           | 비용 영향 | 주의점
──────|------------------------|----------|──────────────────
1D    | Trend, Vol Structure   | 최소     | 거래 빈도 월 2건 이상 필수
4H    | Mean Reversion, Micro  | 중간     | 비용 < 수익의 30%
1H    | HF Mean Rev, Scalping  | 높음     | Maker-only 또는 낮은 빈도
1m    | Intrabar SL/TS 전용    | N/A      | 전략 자체가 아닌 PM 보조
```

#### 4-C. 사용 가능한 지표 라이브러리

`src/market/indicators/` 패키지 53개 공유 지표. 중복 구현 금지.
전체 목록: `src/market/indicators/__init__.py` 참조.

#### 4-D. 설계 문서 작성

다음 항목을 정리한다:

```
1. 전략 이름 (kebab-case)     2. 핵심 가설
3. 비효율 원천 (행동편향/정보비대칭/구조적제약/유동성/리스크 중 택)
4. 비효율 지속 이유 (왜 차익거래되지 않는가)
5. 경제적 논거                6. 사용 지표 (라이브러리에서 선택)
7. 시그널 생성 로직 (수식)    8. ShortMode: DISABLED/HEDGE_ONLY/FULL
9. 타임프레임: 1D/4H/1H      10. 예상 거래 빈도 (건/년)
11. 예상 Sharpe 범위          12. CTREND 상관 예측: 낮음/중간/높음
13. 비용 추정: 연간 거래비용 / 예상 총수익 비율 (< 30% 필수)
14. 레짐 활용: 없음/패턴A/B/C (활용 시 컬럼+파라미터 명시)
15. 데이터 요구사항: OHLCV only / Derivatives / On-chain / Sentiment
    (On-chain 주요 컬럼: oc_mvrv, oc_flow_in_ex_usd, oc_flow_out_ex_usd,
     oc_stablecoin_total_circulating_usd, oc_tvl_usd, oc_fear_greed,
     oc_adractcnt, oc_txcnt, oc_mktcap_usd — 전체 목록: docs/data-collection.md)
16. 백테스팅 데이터 가용성: 전 기간/30일 제한/미확보
    (On-chain: 2018~2020+ 소스별 상이, publication lag T+1 자동 적용)
17. 앙상블 활용: 단독/서브 전략 후보/앙상블 전용
```

**CTREND와 낮은 상관관계가 예상될수록 포트폴리오 가치가 높다.**

### Step 4.5–4.6: YAML 파이프라인 등록 (Phase 1 PASS 시 필수)

Phase 1 PASS인 아이디어를 `pipeline create` CLI로 YAML에 등록한다.

```bash
uv run mcbot pipeline create {strategy-name} \
  --display-name "{DisplayName}" \
  --category "{Category}" \
  --timeframe {TF} \
  --short-mode {SHORT_MODE} \
  --rationale "{경제적 논거}" \
  --p1-score {점수} \
  --status CANDIDATE
```

- `strategies/{strategy-name}.yaml` 생성 (status: CANDIDATE, P1: PASS)
- YAML이 Single Source of Truth. 별도 임시 파일 불필요
- 생성 후 Dashboard 갱신: `uv run mcbot pipeline report`

**상태 전이**: 후보 → 사용자 승인 → 구현중 → Phase 4+ → 구현완료 / 폐기

---

## E Mode: 앙상블 탐색 워크플로우

> S Mode에서 돌아온 경우 (S→E Hybrid) Step 5.5E로 바로 이동.

### Step 1E: 서브전략 후보 풀 구성

```
1. 파이프라인에서 후보 수집
   uv run mcbot pipeline list --status ACTIVE
   uv run mcbot pipeline list --status CANDIDATE

2. 쌍별 상관 행렬 분석 (수익률 시계열)
   - 백테스트 결과의 daily_returns 사용
   - |상관| < 0.5 쌍이 분산 효과 핵심

3. 후보 적격 조건
   - Registry 등록 완료 (@register)
   - 단독 Sharpe >= 0.5 (P4 기준 하한)
   - 기존 후보와 평균 상관 < 0.5
   - from_params() 구현 (EnsembleStrategy가 호출)

4. 유형 구분
   - 메타 앙상블: EnsembleStrategy + 여러 BaseStrategy (이 모드의 대상)
   - 내부 앙상블: 단일 전략 내부 다중 TF/파라미터 (CTREND/Donchian 패턴)
```

### Step 2E: 앙상블 Phase 1E 스코어카드

```
══════════════════════════════════════════════════════
  PHASE 1E: ENSEMBLE VIABILITY SCORECARD
  앙상블: [이름]  |  서브전략 수: [N]
══════════════════════════════════════════════════════

  [1] 분산 효과 (Diversification)              : _/5
      서브전략 간 평균 상관
      >0.7=1, 0.5~0.7=2, 0.3~0.5=3, 0.1~0.3=4, <0.1=5

  [2] 서브전략 품질 (Sub-Strategy Quality)      : _/5
      평균 단독 Sharpe
      <0.3=1, 0.3~0.5=2, 0.5~0.8=3, 0.8~1.0=4, >1.0=5

  [3] 결합 기대 효과 (Expected Improvement)     : _/5
      Sharpe 향상 예측
      악화=1, 0~10%=2, 10~20%=3, 20~30%=4, 30%+=5

  [4] Aggregation 적합성 (Method Fit)           : _/5
      서브전략 특성 부합도
      (ensemble-guide.md 선택 가이드 참조)

  [5] 구현 복잡도 (Implementation Readiness)    : _/5
      전원 등록+테스트=5, 일부 미구현=3, 대부분 미구현=1

  [6] 운영 안정성 (Operational Stability)       : _/5
      TF/ShortMode/warmup 호환성
      전원 동일 TF+호환=5, 조정 필요=3, 비호환=1

──────────────────────────────────────────────────────
  TOTAL: __/30
  판정: PASS (>=18) / WATCH (12-17) / FAIL (<12)

  ★ 킬러 지표: 평균 쌍별 상관 >= 0.6 → 즉시 FAIL
══════════════════════════════════════════════════════
```

**PASS (18점 이상)인 경우에만 Step 3E으로 진행한다.**

### Step 3E: 앙상블 안티패턴 검사

[references/ensemble-guide.md](references/ensemble-guide.md)의 AP1~AP8 체크:

| # | 안티패턴 | 탐지 방법 |
|---|---------|----------|
| AP1 | 동질성 함정 | 전원 동일 카테고리 → 상관 > 0.6 |
| AP2 | 이중 Vol Scaling | 서브전략 vol_target + 앙상블 vol_target 중복 |
| AP3 | 과적합 전략 세탁 | P4 IS/OOS FAIL 전략을 앙상블로 구제 시도 |
| AP4 | warmup 불일치 | max(sub_warmup) > 앙상블 데이터 시작 |
| AP5 | ShortMode 충돌 | 서브전략간 DISABLED/FULL 혼재 시 방향 상쇄 |
| AP6 | 과다 서브전략 | N > 5 → 관리 복잡도 + 개별 기여도 희석 |
| AP7 | TF 불일치 | 서브전략간 TF 상이 → CandleAggregator 미지원 |
| AP8 | 백테스트 기간 불일치 | 최신 전략 데이터 < 3년 → 검증 불충분 |

기존 앙상블 시도 확인:

```
uv run mcbot pipeline lessons-list -t ensemble
# Donchian Ensemble P4 FAIL 등 기존 실패 사례 반드시 참조
```

위반 항목 없을 시 Step 4E 진행.

### Step 4E: EnsembleConfig 결정

```
1. 서브전략 리스트 결정
   - SubStrategySpec: name (registry명), params (dict), weight (기본 1.0)
   - 최소 2개, 권장 3~4개, 최대 5개

2. Aggregation 방법 선택 (ensemble-guide.md 플로우차트)
   - 2개 → EW or InvVol
   - 3개 → 성과편차 작으면 EW, 크면 InvVol, 합의 중요하면 MajVote
   - 4개+ → 레짐 적응 원하면 StratMom, 아니면 InvVol

3. 공통 파라미터
   - vol_target: 0.05~1.0 (기본 0.35). 서브전략에 vol_target 있으면 이중 적용 주의!
   - short_mode: DISABLED(0) / FULL(2). HEDGE_ONLY 미지원
   - vol_window: 5~252 (기본 30)
   - annualization_factor: 365 (일봉 크립토)

4. Aggregation별 추가 파라미터
   - inverse_volatility: vol_lookback (5~504, 기본 63)
   - strategy_momentum: momentum_lookback (10~504, 기본 126), top_n (1~N)
   - majority_vote: min_agreement (0.0~1.0, 기본 0.5)

5. 이론적 Sharpe 예측
   Sharpe_ens ≈ sqrt(N) * avg(Sharpe_i) * sqrt(1 - avg(rho))
   N=서브전략 수, avg(rho)=평균 상관
```

### Step 4.5E: 앙상블 YAML 등록

`pipeline create`는 메타 앙상블 미지원 → 수동으로 `strategies/ens-{name}.yaml` 작성.

```yaml
# strategies/ens-{name}.yaml
name: ens-{name}
display_name: "{DisplayName} Ensemble"
status: CANDIDATE
category: "메타 앙상블"
timeframe: "1D"
short_mode: 0  # DISABLED=0, FULL=2

rationale: |
  {서브전략 조합 근거 + 분산 효과 설명}

parameters:
  strategy_name: ensemble
  sub_strategies:
    - name: "{sub1}"
      params: {}
      weight: 1.0
    - name: "{sub2}"
      params: {}
      weight: 1.0
  aggregation: "equal_weight"
  vol_target: 0.35

phases:
  P1: {score: _, status: PASS, date: "YYYY-MM-DD"}
  P1E: {score: _, status: PASS, date: "YYYY-MM-DD"}

meta:
  category: "메타 앙상블"
  created: "YYYY-MM-DD"
  author: "claude"
```

### Step 5.5E: 앙상블 편입 평가 (S→E Hybrid 전용)

S Mode에서 발굴한 새 전략의 앙상블 편입을 평가한다:

```
편입 조건:
  - 단독 Sharpe >= 0.5
  - 기존 서브전략과 상관 < 0.3
  - 앙상블 Sharpe 개선 확인 (백테스트)
  - from_params() + Registry 등록 완료

→ 조건 충족 시 기존 EnsembleConfig에 SubStrategySpec 추가
→ 미충족 시 단독 전략으로 운영
```

### Step 6E: 앙상블 Phase 4E 판정

```
══════════════════════════════════════════════════════
  PHASE 4E: ENSEMBLE BACKTEST SCORECARD
══════════════════════════════════════════════════════

  PASS 조건 (3개 모두 충족):
    ✓ Sharpe > max(서브전략 Best Sharpe)
    ✓ CAGR > 20%
    ✓ MDD < 35%

  핵심 원칙:
    최고 서브전략보다 나아야 앙상블 존재 의미 있음.
    "평균" 수준이면 → 최고 전략 단독 운영이 낫다.

  WATCH: Sharpe >= Best*0.9 AND MDD 개선 10%+
  FAIL: Sharpe < Best*0.9 OR MDD > 40%
══════════════════════════════════════════════════════
```

Phase 4E PASS → Phase 4 IS/OOS+ 진행 (별도 세션).
Phase 4E FAIL → Aggregation 방법 변경 또는 서브전략 교체 후 재시도 (최대 2회).

---

### Step 5: 구현 위임

설계가 완료되고 **사용자 승인**을 받으면 구현을 진행한다.

```
구현 시 주의사항:
- preprocessor.py: shift() 금지, vectorized ops only
- signal.py: shift(1) 필수, ShortMode 3가지 분기
- strategy.py: @register(), from_params(), recommended_config()
- warmup_periods(): 사용 지표 중 max(window) + 여유분
```

### Step 5.5: 앙상블 전략 구성 (선택)

단독 Sharpe < 1.0 + 기존 전략과 상관 < 0.3 → **E Mode (Step 1E~6E)**로 전환.
설계 가이드: [references/ensemble-guide.md](references/ensemble-guide.md)

### Step 6: 백테스트 실행 + 해석

#### 6-A. 표준 백테스트 실행

```bash
# 5개 에셋, 6년 데이터 (Phase 4 표준)
for symbol in BTC/USDT ETH/USDT BNB/USDT SOL/USDT DOGE/USDT; do
  uv run mcbot backtest run {strategy-name} $symbol \
    --start 2020-01-01 --end 2025-12-31
done
```

#### 6-B. Phase 4 판정 기준 (Best Asset 기준)

```
PASS: Sharpe > 1.0 AND CAGR > 20% AND MDD < 40% AND 거래 50건+
WATCH: 0.5 <= Sharpe <= 1.0, 또는 25% <= MDD <= 40%
FAIL: 총수익 음수 + 거래 20건 미만, 또는 MDD > 50%
```

#### 6-C. 결과 해석 위임

`/backtest-interpret` 스킬로 상세 해석을 위임한다.

### Step 7: 반복 또는 종료 결정

```
Phase 4 PASS → Phase 4 IS/OOS 진행 (별도 세션)
Phase 4 WATCH → 파라미터 조정 후 재실행 (최대 2회)
Phase 4 FAIL → 아이디어 폐기, Step 1로 복귀
```

**파라미터 최적화 시 GT-Score 활용 권장** (arXiv:2602.00080):
`GT-Score = (mu * ln(z) * r^2) / sigma_d` — Sharpe 대비 WF Generalization Ratio 98% 향상.

## 아이디어 발상 보조 도구

아이디어가 없을 때 활용:

```
# 포트폴리오 커버리지 확인
uv run mcbot pipeline status
uv run mcbot pipeline list --status RETIRED   # 폐기 카테고리 확인
uv run mcbot pipeline list --status ACTIVE    # 활성 전략 확인

# 미탐색 영역: Behavioral Finance, Information-Theoretic,
#   Cross-Asset Signal, ML 앙상블 변형
# → pipeline list 폐기 목록과 대조하여 최신 상태 반영
```

**WebSearch 쿼리 예시**: "cryptocurrency trading strategy {year} SSRN", "crypto microstructure alpha", "volatility premium cryptocurrency futures"

## 출력 형식

리포트 형식: [references/report-template.md](references/report-template.md) 참조.
Phase 1 PASS 아이디어는 `pipeline create` CLI로 YAML에 자동 등록 (Step 4.5).

## 안티패턴 — 반드시 피해야 할 것

| # | 패턴 | 교훈 |
|---|------|------|
| 1 | 지표 수프 (5개+) | 과적합 확률 급증 (ML 앙상블 제외) |
| 2 | 동일 TF Mom+MR 블렌딩 | alpha 상쇄 (Mom-MR Blend FAIL) |
| 3 | 1-bar hold / 고빈도 거래 | 비용 > 수익 (Larry-VB: 125건/년 x 0.1% = 12.5% drag) |
| 4 | 단일 지표 의존 | RSI/MACD/Donchian 단독 → 대부분 과적합 |
| 5 | 특정 레짐 타겟 | "상승장에서만" → 전체 기간 Sharpe 저하 |
| 6 | IS Sharpe > 3.0 추구 | 과적합 신호. 현실 목표: IS 1.0~2.0 |
| 7 | 전통금융 무비판 포팅 | FX session/Amihud/Seasonality 전멸 (교훈 #13~#16) |
| 8 | OHLCV microstructure | BVC 근사 불충분, L2 order book 필요 |
| 9 | 레짐 감지 = 전략 | ADX/HMM/Hurst/AC/VR 7개 전멸. 오버레이로만 사용 |
| 10 | 앙상블 동질성 함정 | 전원 동일 카테고리 → 상관 > 0.6 → 분산 효과 없음 |
| 11 | 과적합 전략 세탁 | P4 IS/OOS FAIL 전략을 앙상블로 구제 → 앙상블도 과적합 |
| 12 | 비효율 원천 불명 | "왜 돈을 버는지" 설명 불가 → 과적합. 비효율 원천 5가지 중 미해당 |
| 13 | 비용 비율 > 30% | edge가 거래 비용에 잠식됨. 빈도 감소 또는 TF 상향 필요 |
| 14 | OHLCV만으로 알파 기대 | 가격/거래량은 가장 빠르게 감쇠. Derivatives/On-chain/Sentiment 우선 (✅ 인프라 구축 완료) |

전체 목록: [references/discarded-strategies.md](references/discarded-strategies.md) + `uv run mcbot pipeline lessons-list`

## 체크리스트

- [ ] 비효율 원천 분석 완료 (Step 1-0 질문 3개+ 답변)
- [ ] 교훈 데이터 확인됨 (`lessons-list` TF/카테고리/전략)
- [ ] Phase 1 스코어카드 18점 이상
- [ ] 폐기 전략과 중복 없음 (교훈 + discarded-strategies)
- [ ] 비효율 원천 + 지속 이유 1문단 이상
- [ ] 비용 추정: 연간 거래비용/총수익 < 30%
- [ ] ShortMode + TF 적합성 확인
- [ ] CTREND 예상 상관관계 평가됨
- [ ] 지표 `src/market/indicators/`에서 선택
- [ ] 설계 문서 17개 항목 작성됨
- [ ] Derivatives 필요 시 Silver _deriv 가용성 확인
- [ ] On-chain/Sentiment 데이터 활용 여부 검토 (22개 데이터셋 가용)
- [ ] 앙상블 기여도 평가됨 (약한 alpha + 낮은 상관 → 편입 후보 판정)
- [ ] `pipeline create` → YAML 생성됨
- [ ] `pipeline report` → dashboard 갱신됨

### 앙상블 체크리스트 (E Mode)

- [ ] 서브전략 후보 풀 구성 (pipeline list로 수집)
- [ ] 쌍별 상관 행렬 분석 완료
- [ ] Phase 1E 스코어카드 18점+ (킬러: 평균 상관 < 0.6)
- [ ] AP1~AP8 안티패턴 위반 없음
- [ ] Aggregation 방법 선택 + 근거 명시
- [ ] EnsembleConfig 파라미터 결정
- [ ] 이론적 Sharpe 예측 산출
- [ ] `strategies/ens-{name}.yaml` 등록됨
- [ ] 백테스트 config YAML 작성됨
- [ ] Phase 4E 판정 완료 (Sharpe > Best 서브전략)
