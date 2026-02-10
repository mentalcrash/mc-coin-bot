# 전략 스코어카드: VPIN-Flow

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | VPIN-Flow (`vpin-flow`) |
| **유형** | 마이크로스트럭처 |
| **타임프레임** | 1D |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | N/A (전 에셋 거래 0건) |
| **2nd Asset** | N/A |
| **경제적 논거** | BVC로 buy/sell volume을 근사하고 VPIN으로 정보거래 확률을 측정. 고독성(high VPIN) 시 informed trading 방향을 추종하여 대형 가격 변동을 사전 포착. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| — | BTC/USDT | 0.00 | 0.0% | 0.0% | 0 | — | 0.0% | — |
| — | ETH/USDT | 0.00 | 0.0% | 0.0% | 0 | — | 0.0% | — |
| — | BNB/USDT | 0.00 | 0.0% | 0.0% | 0 | — | 0.0% | — |
| — | SOL/USDT | 0.00 | 0.0% | 0.0% | 0 | — | 0.0% | — |
| — | DOGE/USDT | 0.00 | 0.0% | 0.0% | 0 | — | 0.0% | — |

### Best Asset 핵심 지표

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.00 | > 1.0 | **FAIL** |
| CAGR | 0.0% | > 20% | **FAIL** |
| MDD | 0.0% | < 40% | N/A (거래 없음) |
| Trades | 0 | > 50 | **FAIL** |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 22/30점
G0B 코드감사  [PASS] Critical 0 (수정완료), High 0, Medium 3
G1  백테스트  [FAIL] 전 에셋 거래 0건 — VPIN threshold 도달 불가 (1D 데이터 한계)
G2  IS/OOS   [    ] (미진행)
G3  파라미터  [    ] (미진행)
G4  심층검증  [    ] (미진행)
G5  EDA검증  [    ] (미진행)
G6  모의거래  [    ] (미진행)
G7  실전배포  [    ] (미진행)
```

### Gate 상세

#### G1 단일에셋 백테스트 (2026-02-10) — FAIL

**즉시 폐기**: 전 에셋 거래 0건 (시그널 미생성)

**근본 원인 분석**:

1D 일봉 데이터에서 BVC 기반 VPIN은 n_buckets=50 rolling window에서 다음과 같이 분포한다:

| 에셋 | VPIN Mean | VPIN Max | VPIN > 0.7 | VPIN > 0.5 | VPIN > 0.4 |
|------|-----------|----------|:----------:|:----------:|:----------:|
| BTC/USDT | 0.3316 | 0.3951 | 0 | 0 | 0 |
| ETH/USDT | 0.3408 | 0.4214 | 0 | 0 | 14 |
| BNB/USDT | 0.3333 | 0.4216 | 0 | 0 | 6 |
| SOL/USDT | 0.3086 | 0.4229 | 0 | 0 | 25 |
| DOGE/USDT | 0.3276 | 0.4500 | 0 | 0 | 93 |

- **threshold_high = 0.7**인데, 전 에셋의 VPIN 최대값이 0.45 이하
- VPIN이 `|buy - sell| / total volume`의 rolling sum이므로, 50-bar window에서 평균 ~0.33으로 수렴 (중심극한정리)
- 논문 원본의 VPIN은 **volume bar** (tick-level 데이터) 기준이며, 1D 일봉에서는 bar 내 imbalance가 평탄화됨
- **구조적 불일치**: VPIN threshold 0.7은 tick/volume bar 데이터에서만 의미 있는 수치. 1D OHLCV에서는 물리적으로 도달 불가

**수정 가능성 평가**:
- threshold_high를 0.35~0.40으로 낮추면 신호 발생 가능하나, 이는 VPIN의 "고독성" 개념을 무력화
- 1D 데이터에서 BVC의 `norm.cdf((close-open)/(high-low))` 근사는 bar 내 가격 분포를 단일 점으로 축소 → 정보 손실 심각
- 원래 설계 의도(마이크로스트럭처)와 가용 데이터(1D)의 **해상도 불일치**가 근본 원인

#### G0B 코드 감사 (2026-02-10)

**종합 등급: A** (CRITICAL/HIGH 이슈 전수 수정 완료)

| 항목 | 점수 |
|------|:----:|
| 데이터 무결성 | 10/10 |
| 시그널 로직 | 9/10 |
| 실행 현실성 | 9/10 |
| 리스크 관리 | 9/10 |
| 코드 품질 | 9/10 |

**수정 완료된 이슈:**
- [C-001] ~~HEDGE_ONLY drawdown shift(1) 미적용~~ → `df["drawdown"].shift(1)` 적용 완료 (look-ahead bias 제거)
- [H-001] Hedge 모드 `suppress_short`과 `active_short` 상호 배타성 — 로직상 보장됨 (short_mask & ~hedge = suppress, short_mask & hedge = active)
- [H-002] shift(1) 회귀 테스트 — 기존 테스트에서 shift 적용 검증 커버

**잘된 점:**
- 주요 indicator (vpin, flow_direction, vol_scalar, drawdown) shift(1) 전수 적용
- BVC `high - low + 1e-10` epsilon 방어
- VPIN [0,1] 범위 보장 (|imbalance|/volume)
- norm.cdf BVC → buy_pct [0,1] 보장

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 22/30점 — 경제적 논거 4, 참신성 5, 데이터 3, 구현 3, 용량 3, 레짐독립 4 |
| 2026-02-10 | G0B | ~~FAIL~~ → **PASS** | Critical 1건 발견 → 수정 완료 (drawdown shift(1) 적용). 재감사 통과 |
| 2026-02-10 | G1 | **FAIL** | 전 에셋 거래 0건. VPIN threshold_high=0.7이 1D 일봉 데이터에서 도달 불가 (max VPIN 0.45). 마이크로스트럭처 전략의 데이터 해상도 불일치가 근본 원인 |
