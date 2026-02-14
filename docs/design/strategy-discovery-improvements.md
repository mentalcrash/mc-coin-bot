# 전략 발굴 메커니즘 개선 방안

> 작성일: 2026-02-14
> 상태: PROPOSAL (구현 전 사용자 승인 필요)

---

## 1. 현황 요약

### 1.1 Gate 파이프라인 구조

```
G0A(주관) → 구현 → G0B(체크) → G1(백테스트) → G2(IS/OOS) → G3(파라미터) → G4(통계) → G5(EDA)
```

### 1.2 성과 지표 (2026-02-14 기준)

| 지표 | 수치 |
|------|------|
| 총 전략 수 | ~85개 |
| ACTIVE | 2개 (CTREND, Anchor-Mom) |
| RETIRED | ~75개 |
| **성공률** | **~2.4%** (ACTIVE/총) |
| G1 FAIL 비율 | ~60% (전체 RETIRED 중) |
| G2 FAIL 비율 | ~25% |
| G4 FAIL 비율 | ~5% |

### 1.3 핵심 문제

G1 이후 검증 파이프라인(데이터 기반)은 견고하지만, **G0A 발굴 초기 필터링이 느슨**하여 구현 비용이 낭비되고 있다.

---

## 2. 식별된 약점 5가지

### 약점 1: G0A 점수 검증 로직 부재

**현상**: `pipeline create --g0a-score 15`로 18점 미만도 PASS 처리 가능.

**원인**: `src/cli/pipeline.py`의 `create` 커맨드에서 점수 검증 없이 무조건 `GateVerdict.PASS` 기록.

```python
# 현재 코드 (pipeline.py create 커맨드)
gates={
    GateId.G0A: GateResult(
        status=GateVerdict.PASS,  # ← 점수 무관하게 항상 PASS
        ...
    ),
},
```

**영향**: Gate 시스템의 첫 번째 방어선이 무력화됨.

**개선안**:
- `g0a_score < 18`이면 `GateVerdict.FAIL` 자동 판정
- 항목별 점수 입력 지원 (`--g0a-items '{"경제적_논거":4,"참신성":3,...}'`)
- 단일 항목 1점이면 경고 출력 (극단적 약점 존재)

---

### 약점 2: IC (Information Coefficient) 사전 검증 부재

**현상**: 지표의 예측력을 사전 검증하지 않고 바로 전략 구현에 들어감.

**영향**: 예측력 없는 지표 기반 전략을 구현 완료(4-file 구조 + 테스트)한 후 G1에서 FAIL → 개발 시간 낭비.

**RETIRED 전략 실패 패턴에서 근거**:
- G1 FAIL 60%의 주요 원인: "모든 에셋 Sharpe 음수 또는 0.5 미만" → 신호 자체에 Alpha 없음
- `accel-conv` (Sharpe -1.8), `liq-momentum` (Sharpe -4.9~-6.5) 등 구현 전 IC 검증으로 걸러낼 수 있었음

**개선안 — Gate 0A+ (IC Quick Check) 도입**:

```
G0A(주관) → [G0A+ IC Quick Check] → 구현 → G0B → G1 → ...
```

| 검증 항목 | 방법 | 기준 |
|----------|------|------|
| Rank IC | `spearmanr(indicator[t], return[t+1])` | \|IC\| > 0.02 |
| IC IR (IC의 Sharpe) | `mean(IC) / std(IC)` | \|IC IR\| > 0.1 |
| IC Decay | `IC[t] vs IC[t-60]` 상관 | 1년 내 부호 반전 없음 |
| Hit Rate | `sign(indicator) == sign(return)` 비율 | > 52% |

**구현 범위**:
- `src/backtest/ic_analyzer.py` — IC 계산 유틸
- CLI: `uv run mcbot backtest ic-check <indicator> <symbol>` — 구현 전 빠른 검증
- G0A YAML에 IC 결과 기록 필드 추가

**비용 대비 효과**: 구현(4 파일 + 테스트) 대비 IC 검증은 ~10분. G1 FAIL 60%의 절반만 사전 필터링해도 전략당 2~3시간 절약.

---

### 약점 3: 학술 근거 미기록

**현상**: 전략의 경제적 논거(economic_rationale)에 학술 논문 참조가 없음.

**현재 상태**:
- SSRN #4825389 (TSMOM) 하나만 코드에 기록
- 나머지 전략의 `economic_rationale`은 1줄 서술만 존재
  - 예: `"모멘텀 지속성"`, `"변동성 구조 불균형"`
- 왜 그 논거가 유효한지 검증/추적 불가

**영향**:
- 동일 논거의 전략 중복 발굴 (이미 실패한 논거를 반복)
- 경제적 논거의 사후 검증 불가
- 학습 축적 불가 (어떤 논거가 실제 Alpha로 이어지는지)

**개선안 — StrategyRecord YAML 필드 확장**:

```yaml
# strategies/example.yaml — 개선 후
meta:
  economic_rationale: "OBV 가속도가 가격 반전을 선행한다"
  rationale_references:
    - type: paper
      title: "On-Balance Volume and Stock Returns"
      source: "Journal of Finance, 2018"
      url: "https://doi.org/10.xxxx"
      relevance: "OBV의 예측력 통계적 검증"
    - type: lesson
      id: 15
      relevance: "BVC 근사의 TF 불변 한계"
    - type: prior_strategy
      name: "obv-momentum"
      outcome: "RETIRED at G1"
      relevance: "단순 OBV는 Alpha 없음, 가속도 필요"
  rationale_category: "flow-based"  # momentum | mean-reversion | flow-based | structural | behavioral
```

**카테고리 분류 (논거 중복 방지)**:

| 카테고리 | 설명 | 현재 전략 수 | 성공률 |
|----------|------|:-----------:|:-----:|
| `momentum` | 추세 지속성, 모멘텀 효과 | ~30개 | ~3% |
| `mean-reversion` | 평균 회귀, 과매수/과매도 | ~15개 | 0% |
| `flow-based` | 거래량, 자금 흐름 | ~10개 | 0% |
| `structural` | 시장 구조 (변동성, 상관관계) | ~15개 | 0% |
| `behavioral` | 행동 편향 (Disposition, Herding) | ~5개 | 0% |
| `ensemble` | 복합 신호 앙상블 | ~5개 | ~40% |
| `regime-adaptive` | 레짐 전환 적응 | ~5개 | 0% |

**핵심 인사이트**: momentum + ensemble만 성공. 다른 카테고리는 암호화폐 시장에서 아직 유효한 Alpha를 찾지 못함.

---

### 약점 4: RETIRED 실패 패턴 체계적 분석 부재

**현상**: 75개 RETIRED 전략의 실패 사유가 개별 YAML에 분산되어 있고, 종합 분석이 없음.

**영향**:
- 같은 실패 패턴 반복 (예: 단순 단일지표 모멘텀 → G1 FAIL 반복)
- 새 전략 발굴 시 기존 실패로부터 학습 불가

**실패 패턴 분석 결과 (20개 표본)**:

| 실패 패턴 | 비율 | 대표 전략 | 근본 원인 |
|----------|:----:|----------|----------|
| 신호 Alpha 부재 | 40% | accel-conv, entropy-switch | 지표 예측력 없음 (IC ≈ 0) |
| 과적합 (OOS 붕괴) | 25% | donchian, adaptive-breakout | 파라미터 IS에 과적합 |
| Timeframe 부적합 | 10% | liq-momentum, hour-season | 고주파 전략의 1D 백테스트 왜곡 |
| 단일 레짐 의존 | 10% | adx-regime, hurst-regime | 특정 시장 환경에서만 작동 |
| CAGR 기준 미달 | 10% | bb-rsi, kalman-trend | 거래 빈도 부족 |
| 기타 | 5% | – | 코드 버그, 데이터 오류 |

**개선안 — RETIRED 분석 대시보드**:

1. **자동 분류 CLI 커맨드**:
   ```bash
   uv run mcbot pipeline retired-analysis
   ```
   - Gate별 FAIL 분포
   - 실패 사유 카테고리별 집계
   - rationale_category별 성공/실패 비율

2. **발굴 시 자동 경고**:
   ```bash
   uv run mcbot pipeline create obv-accel --category flow-based
   # ⚠️ WARNING: flow-based 카테고리 10개 전략 중 0개 ACTIVE (성공률 0%)
   # ⚠️ 유사 RETIRED 전략: obv-momentum (G1 FAIL), flow-imbalance (G1 FAIL)
   ```

3. **Lessons 자동 연결**:
   - RETIRED 시 관련 Lesson 자동 생성/링크
   - `uv run mcbot pipeline retire <name> --lesson "단순 flow 지표는 Alpha 없음"`

---

### 약점 5: G0A 채점 기준의 변별력 부족

**현상**: 6항목 채점 기준이 너무 일반적이어서 대부분의 아이디어가 18점 이상을 받음.

**데이터 근거**:
- ~85개 전략 중 G0A FAIL은 0개 → 필터링 기능 없음
- 18/30 (60%) 통과 기준이 너무 낮음

**항목별 문제점**:

| 항목 | 문제 | 개선 방향 |
|------|------|----------|
| 경제적 논거 | "모멘텀"이면 3점 이상 → 차별화 불가 | 논거의 **고유성** 추가 평가 (기존 전략과 차별점) |
| 참신성 | 암호화폐 분야 논문 적어 대부분 4~5점 | **프로젝트 내 참신성** 기준 추가 (기존 RETIRED와 비교) |
| 데이터 확보성 | Binance API만 쓰므로 대부분 5점 | 차별력 없음 → **제거 또는 가중치 축소** 고려 |
| 구현 난이도 | 대부분 VectorBT 기반 3~5점 | 차별력 없음 → **제거 또는 가중치 축소** 고려 |
| 수용 용량 | 암호화폐 선물이면 대부분 3~4점 | 차별력 낮음 |
| 레짐 의존성 | 사전 검증 없이 주관 판단 | **레짐별 IC 검증**으로 객관화 가능 |

**개선안 — G0A v2 채점 기준**:

```yaml
# 제안: gates/criteria.yaml G0A 개선
scoring:
  pass_threshold: 21  # 18 → 21 (70%)
  max_total: 30
  items:
    - name: 경제적 논거 고유성
      description: >
        5=기존 RETIRED 전략과 완전히 다른 메커니즘,
        3=유사 메커니즘이지만 개선 포인트 명확,
        1=기존 RETIRED 전략과 동일 논거
    - name: IC 사전 검증
      description: >
        5=|Rank IC| > 0.05 확인됨,
        3=|Rank IC| > 0.02 확인됨,
        1=IC 미검증 또는 |IC| < 0.02
    - name: 카테고리 성공률
      description: >
        5=해당 카테고리 성공률 > 20%,
        3=0~20% 성공률이지만 차별화된 접근,
        1=성공률 0% 카테고리에 유사 접근
    - name: 레짐 독립성
      description: >
        5=3개 이상 레짐에서 IC 양수 확인,
        3=2개 레짐에서 양수,
        1=단일 레짐 또는 미검증
    - name: 앙상블 기여도
      description: >
        5=기존 ACTIVE 전략과 상관 < 0.3,
        3=상관 0.3~0.5,
        1=상관 > 0.5 (중복 Alpha)
    - name: 수용 용량
      description: "5=$1M+ 운용, 1=$100K 미만"
```

**핵심 변경점**:
- 차별력 없는 항목(데이터 확보성, 구현 난이도) → 차별력 있는 항목(IC 사전 검증, 카테고리 성공률, 앙상블 기여도)으로 교체
- 통과 기준 18점 → 21점 (70%) 상향
- 데이터 기반 항목 비율: 기존 0/6 → 4/6

---

## 3. 구현 우선순위

| 순위 | 개선 항목 | 난이도 | 기대 효과 | 비고 |
|:----:|----------|:------:|:--------:|------|
| **P0** | G0A 점수 검증 로직 | 낮음 | 중간 | `pipeline.py` 3줄 수정 |
| **P1** | 학술 근거 YAML 필드 확장 | 낮음 | 중간 | `models.py` + YAML 스키마 |
| **P2** | RETIRED 실패 패턴 분석 CLI | 중간 | 높음 | `pipeline.py` + 집계 로직 |
| **P3** | IC Quick Check 도입 | 중간 | **매우 높음** | `ic_analyzer.py` 신규 |
| **P4** | G0A v2 채점 기준 개편 | 높음 | 높음 | 기존 전략 재평가 필요 |

---

## 4. 기대 효과

### Before (현재)

```
100개 아이디어 → G0A 100개 PASS → 100개 구현 → G1에서 60개 FAIL → ... → 2개 ACTIVE
구현 낭비: ~60개 × 3시간 = ~180시간
```

### After (개선 후)

```
100개 아이디어 → G0A v2 60개 PASS → IC Check 30개 PASS → 30개 구현 → G1에서 10개 FAIL → ... → 3~5개 ACTIVE
구현 낭비: ~10개 × 3시간 = ~30시간 (83% 감소)
```

---

## 5. 참고: 현재 성공 전략의 공통 특성

ACTIVE 2개 전략(CTREND, Anchor-Mom)의 공통점:

| 특성 | CTREND | Anchor-Mom |
|------|--------|-----------|
| 카테고리 | ensemble | momentum |
| G0A 점수 | 22/30 | 21/30 |
| 지표 수 | 28개 (앙상블) | 복합 |
| Timeframe | 1D | 1D |
| ShortMode | HEDGE_ONLY | HEDGE_ONLY |
| Best Sharpe | 2.05 | 1.73 |
| OOS Decay | 33.7% | ~35% |

**성공 패턴**: 복합 신호 + 1D + HEDGE_ONLY + Decay < 40%

이 패턴에서 벗어나는 전략(단일 지표, 고주파, FULL Short)은 높은 RETIRED 확률을 보임.
