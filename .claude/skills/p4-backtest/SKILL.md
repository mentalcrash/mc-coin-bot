---
name: p4-backtest
description: >
  Phase 4 백테스트 검증 파이프라인 (단일에셋 + IS/OOS).
  사용 시점: P3 PASS 전략의 백테스트, "backtest" "pipeline" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name> [--from p4a|p4b]
---

# Phase 4: 백테스트 검증 (단일에셋 + IS/OOS)

## 역할

**시니어 퀀트 리서처 겸 검증 엔지니어**로서 행동한다.

핵심 원칙:

- 단순 threshold 비교가 아닌 **경제적 의미 해석** — 숫자 뒤의 이유를 찾는다
- FAIL 시 **구체적 사유 + 수정 방향** 제시 (단순 "FAIL" 판정 금지)
- Phase 간 **결과 일관성 추적** — P4A Sharpe -> P4B OOS Sharpe -> P5/P6 흐름 확인
- 모든 결과를 **CTREND 선례**와 비교하여 상대적 위치 파악

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ac-regime` |
| `--from p4a\|p4b` | X | 시작 Phase (기본: p4a). 이전 Phase 결과는 YAML에서 복원 | `--from p4b` |

---

## Step 0: Pre-flight Check

다음 4항목을 검증한 후 진행한다. **하나라도 실패하면 중단**.

### 0-1. 전략 디렉토리 존재

```bash
ls src/strategy/{name_underscore}/  # config.py, preprocessor.py, signal.py, strategy.py
```

> `name_underscore`: 하이픈 -> 언더스코어 (e.g., `ac-regime` -> `ac_regime`)

### 0-2. YAML 메타데이터 + P3 PASS

```bash
cat strategies/{strategy_name}.yaml  # phases.P3.status: PASS 확인
```

YAML 없으면 `uv run mcbot pipeline migrate`. P3 PASS 없으면 `/p3-audit` 먼저 실행.

### 0-3. Silver 데이터 존재

```bash
ls data/silver/BTC_USDT_1D.parquet data/silver/ETH_USDT_1D.parquet \
   data/silver/BNB_USDT_1D.parquet data/silver/SOL_USDT_1D.parquet \
   data/silver/DOGE_USDT_1D.parquet
```

### 0-4. --from 복원 (해당 시)

`--from p4b` 지정 시 YAML에서 이전 Phase 결과 복원:

| --from | 복원 대상 |
|--------|----------|
| `p4b` | P4A 결과 (Best Asset, Sharpe, CAGR) |

---

## Step 1: Phase 4A — 단일에셋 백테스트

### 실행

5개 코인 x 6년 (2020-01-01 ~ 2025-12-31) 백테스트:

```bash
uv run mcbot backtest run {strategy_name} {SYMBOL} \
  --start 2020-01-01 --end 2025-12-31 --capital 100000
```

5개 심볼: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `SOL/USDT`, `DOGE/USDT`

> 또는 `scripts/bulk_backtest.py` 패턴으로 Python API 직접 호출 (더 빠름).

### Derivatives 전략

`required_columns`에 OHLCV 외 컬럼이 있으면: _deriv 파일 확인, `include_derivatives=True` 필요 (CLI 미지원 -> 코드 직접 실행). Funding Rate만 전 기간 가능.

### 수집 지표

Sharpe/Sortino/Calmar, CAGR, MDD, Trades, Win Rate, Profit Factor, Alpha/Beta(vs BTC B&H) -- 모두 `metrics.*` / `benchmark.*`에서 수집.

### 판정 기준

**PASS 조건** (Best Asset 기준): Sharpe > 1.0, CAGR > 20%, MDD < 40%, Trades > 50

**즉시 폐기** (전 에셋 해당 시):

1. MDD > 50% — 전 에셋
1. Sharpe < 0 — 전 에셋
1. Trades < 20 + 수익 음수
1. 80%+ 단일 거래 의존

### 비용 민감도

연간 100회+ 에셋: 2배 비용 (편도 ~0.22%) 적용. Sharpe <= 0이면 **비용 민감 경고**.

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조.
핵심: Best Asset 순서 패턴 (SOL>BTC>BNB>ETH>DOGE가 추세추종 일반), Beta < 0.3이면 BTC 독립 알파.

### CTREND 비교 (참조점)

| 지표 | CTREND Best (SOL) | 현재 전략 Best |
|------|-------------------|---------------|
| Sharpe | 2.05 | ? |
| CAGR | +97.8% | ? |
| MDD | -27.7% | ? |
| Trades | 288 | ? |

### YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P4A --verdict PASS \
  --detail "sharpe={best_sharpe}" --detail "cagr={best_cagr}" \
  --rationale "{Best Asset} Sharpe X.XX, CAGR +XX.X%"
```

**FAIL -> Step F** | **PASS -> Step 2**

---

## Step 2: Phase 4B — IS/OOS 70/30

### 실행

Best Asset에 대해 IS/OOS 검증:

```bash
uv run mcbot backtest validate \
  -s {strategy_name} --symbols {best_asset} \
  -m quick -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| IS / OOS Sharpe | In-Sample / Out-of-Sample Sharpe |
| Decay (%) | `(1 - OOS/IS) x 100` |
| OOS Trades | OOS 구간 거래 수 |
| Consistency | OOS fold 양수 비율 |

### 판정 기준

| 조건 | 기준 | CTREND 참조 |
|------|------|------------|
| OOS Sharpe | >= 0.3 | 1.78 |
| Decay | < 50% | 33.7% |
| OOS Trades | >= 15 | -- |

상세: [references/phase-criteria.md](references/phase-criteria.md)

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조.
핵심: Decay < 20% 우수, 20-35% 양호, 35-50% 경계, > 50% FAIL. OOS Sharpe > P4A Sharpe x 0.5이면 양호.

**FAIL -> Step F** | **PASS -> /p5-robustness**

---

## Step F: 실패 처리

Phase FAIL 시 다음을 순차 실행한다.

### F-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase {PN} --verdict FAIL \
  --rationale "{구체적 FAIL 사유}"
```

> `pipeline record`가 FAIL 시 자동으로 status -> RETIRED 변경.

### F-2. 교훈 기록 (새로운 패턴 시)

기존 교훈 검색 후, 새로운 실패 패턴이면 추가:

```bash
uv run mcbot pipeline lessons-list -c {category} -t {관련키워드}
uv run mcbot pipeline lessons-add \
  --title "{패턴 한줄 요약}" --body "{구체적 사유}" \
  -c {category} -t {태그1} -t {태그2} -s {strategy_name} --tf {TF}
```

| FAIL 유형 | 카테고리 |
|----------|----------|
| 전 에셋 Sharpe 음수 | `strategy-design` |
| 특정 TF 구조적 실패 | `market-structure` |
| MDD 폭발 | `risk-management` |
| IS->OOS Decay > 50% | `pipeline-process` |
| OHLCV 데이터 한계 | `data-resolution` |

> 기존 교훈과 겹치면 추가하지 않는다.

### F-3. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](references/report-template.md) 참조.

---

## Step S: Phase 4B PASS -> 다음 단계: `/p5-robustness`

### S-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --phase P4B --verdict PASS \
  --detail "oos_sharpe={X.XX}" --detail "decay={XX}%" \
  --rationale "IS/OOS 검증 PASS — OOS Sharpe X.XX, Decay XX%"
```

상태 `TESTING` 유지 (Phase 5 Robustness 대기).

### S-2. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](references/report-template.md) 참조.

---

## 참조 문서

- [references/phase-criteria.md](references/phase-criteria.md) — Phase별 정량 기준 + CTREND 비교 + CLI 명령
- [references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) — 시니어 퀀트 해석 패턴
- [references/report-template.md](references/report-template.md) — 리포트 출력 형식
- `pipeline report` — 전략 상황판 (CLI)
- `_phase_runners.py` — Phase runner 구현체
