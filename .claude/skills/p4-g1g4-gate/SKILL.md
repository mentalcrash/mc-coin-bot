---
name: p4-g1g4-gate
description: >
  G1~G4 순차 검증 파이프라인 (백테스트/IS-OOS/파라미터안정성/WFA-CPCV-PBO-DSR).
  사용 시점: G0B PASS 전략의 Gate 검증, "gate" "pipeline" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name> [--from g1|g2|g2h|g3|g4]
---

# Gate Pipeline: G1~G4 순차 검증

## 역할

**시니어 퀀트 리서처 겸 검증 엔지니어**로서 행동한다.

핵심 원칙:
- 단순 threshold 비교가 아닌 **경제적 의미 해석** — 숫자 뒤의 이유를 찾는다
- FAIL 시 **구체적 사유 + 수정 방향** 제시 (단순 "FAIL" 판정 금지)
- Gate 간 **결과 일관성 추적** — G1 Sharpe -> G2 OOS Sharpe -> G4 WFA OOS 흐름 확인
- 모든 결과를 **CTREND 선례**와 비교하여 상대적 위치 파악

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ac-regime` |
| `--from gN` | X | 시작 Gate (기본: g1). 이전 Gate 결과는 YAML에서 복원 | `--from g2` |

---

## Step 0: Pre-flight Check

다음 5항목을 검증한 후 진행한다. **하나라도 실패하면 중단**.

### 0-1. 전략 디렉토리 존재

```bash
ls src/strategy/{name_underscore}/  # config.py, preprocessor.py, signal.py, strategy.py
```

> `name_underscore`: 하이픈 -> 언더스코어 (e.g., `ac-regime` -> `ac_regime`)

### 0-2. YAML 메타데이터 + G0B PASS

```bash
cat strategies/{strategy_name}.yaml  # gates.G0B.status: PASS 확인
```

YAML 없으면 `uv run mcbot pipeline migrate`. G0B PASS 없으면 `/p3-g0b-verify` 먼저 실행.

### 0-3. Silver 데이터 존재

```bash
ls data/silver/BTC_USDT_1D.parquet data/silver/ETH_USDT_1D.parquet \
   data/silver/BNB_USDT_1D.parquet data/silver/SOL_USDT_1D.parquet \
   data/silver/DOGE_USDT_1D.parquet
```

### 0-4. --from 복원 (해당 시)

`--from gN` 지정 시 YAML에서 이전 Gate 결과 복원:

| --from | 복원 대상 |
|--------|----------|
| `g2` | G1 결과 (Best Asset, Sharpe, CAGR) |
| `g2h` | G1 + G2 결과 (Best Asset, OOS Sharpe, Decay) |
| `g3` | G1 + G2 + G2H 결과. `results/gate2h_{strategy}.json` 존재 확인 |
| `g4` | G1 + G2 + G2H + G3 결과 |

### 0-5. Gate 3 스윕 소스 확인

Gate 3 스윕 범위는 두 소스 중 하나에서 제공:

1. **G2H 자동 생성 (권장)**: `results/gate2h_{strategy}.json`의 `g3_sweeps`
2. **수동 등록 (fallback)**: `src/cli/_gate_runners.py`의 `GATE3_STRATEGIES` dict

G2H를 실행하면 수동 등록 불필요. G2H 건너뛴 경우에만 수동 등록 (사용자 승인).

---

## Step 1: Gate 1 — 단일에셋 백테스트

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
2. Sharpe < 0 — 전 에셋
3. Trades < 20 + 수익 음수
4. 80%+ 단일 거래 의존

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
  --gate G1 --verdict PASS \
  --detail "sharpe={best_sharpe}" --detail "cagr={best_cagr}" \
  --rationale "{Best Asset} Sharpe X.XX, CAGR +XX.X%"
```

**FAIL -> Step F** | **PASS -> Step 2**

---

## Step 2: Gate 2 — IS/OOS 70/30

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

상세: [references/gate-criteria.md](references/gate-criteria.md)

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조.
핵심: Decay < 20% 우수, 20-35% 양호, 35-50% 경계, > 50% FAIL. OOS Sharpe > G1 Sharpe x 0.5이면 양호.

**FAIL -> Step F** | **PASS -> Step 2H**

---

## Step 2H: Gate 2H — 파라미터 최적화 (정보 전용)

> **Always PASS** — 정보 제공 목적. 과적합 방어는 G4에서 담당.

### 목적

1. 기본 파라미터 대비 Optuna TPE 최적화로 개선 가능성 확인
2. 최적 파라미터 중심으로 G3 sweep 범위 자동 생성 (수동 등록 불필요)
3. IS/OOS 분할 검증으로 과적합 경향 사전 파악

### 실행

Best Asset (G1 결과)에 대해 IS(70%)/OOS(30%) 분할 후 최적화:

```bash
uv run mcbot pipeline gate2h-run {strategy_name} --n-trials 100 --seed 42 --json
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| Default Sharpe (IS) | 기본 파라미터의 IS Sharpe |
| Best Sharpe (IS) | 최적 파라미터의 IS Sharpe |
| Improvement (%) | `(Best - Default) / |Default| × 100` |
| OOS Sharpe | 최적 파라미터의 OOS 검증 Sharpe (정보 전용) |
| Search Space | 최적화 대상 파라미터 수 |
| Trials | 완료/전체 trial 수 |

### 결과 해석

| Improvement | 해석 |
|-------------|------|
| < 5% | 기본 파라미터가 근최적 — 민감도 낮음 (좋은 신호) |
| 5~30% | 합리적 개선 — 정상 범위 |
| > 30% | IS 과적합 가능성. OOS Sharpe 확인 필수 |

OOS Sharpe 확인:
- OOS > 0, IS 대비 Decay < 50%: 양호
- OOS <= 0: IS 과적합 의심 — G3/G4에서 추가 검증 (G2H 자체는 PASS)

### 출력

| 파일 | 내용 |
|------|------|
| `results/gate2h_{strategy}.json` | 최적화 결과 + G3 sweep 범위 + top 10 trials |
| YAML `gates.G2H` | PASS + IS/OOS Sharpe + improvement (CLI 자동 갱신) |

### G3 연계

G2H CLI가 `results/gate2h_{strategy}.json`에 `g3_sweeps`를 자동 생성.
G3 CLI(`pipeline gate3-run`)가 이 파일을 자동 감지하여 sweep 범위로 사용.
**G2H 실행 후에는 G3 수동 스윕 등록 불필요.**

> YAML 갱신은 G2H CLI가 자동 처리 — 별도 `pipeline record` 불필요.

**Always PASS -> Step 3**

---

## Step 3: Gate 3 — 파라미터 안정성

### 사전 작업: 스윕 소스 확인

**G2H 결과 우선**: `results/gate2h_{strategy_name}.json` 존재 확인.

- **존재** (G2H 실행됨): G3 CLI가 자동으로 G2H sweep 범위 사용. 수동 등록 불필요.
- **미존재** (G2H 건너뜀): `src/cli/_gate_runners.py`의 `GATE3_STRATEGIES` dict에 수동 등록 (사용자 승인).
  핵심 파라미터 3~5개, ±20% + 넓은 범위 그리드. vol_target 필수, short_mode 제외.

### 실행

```bash
uv run mcbot pipeline gate3-run {strategy_name} --json
```

> G2H JSON 존재 시 최적화 sweep 범위 자동 사용. 미존재 시 GATE3_STRATEGIES fallback.

### 수집 지표 (파라미터별)

| 지표 | 설명 |
|------|------|
| Plateau 존재 | Best Sharpe의 80% 이상인 값 >= 3개 |
| Plateau Count / Range | 고원 값 수 / 범위 (min~max) |
| +/-20% Stable / Sharpe | 기본값 +/-20%에서 Sharpe 양수 유지 / min~max |

### 판정 기준

| 조건 | 기준 |
|------|------|
| 고원 존재 | **모든** 핵심 파라미터에서 60%+ (plateau_count/total >= 60%) |
| +/-20% 안정 | **모든** 핵심 파라미터에서 Sharpe 부호 양수 유지 |

**전체 PASS**: 모든 파라미터가 (고원 존재 AND +/-20% 안정)

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조.
핵심: 넓은 고원 = 로버스트, 좁은 봉우리 = 과적합 위험. vol_target Sharpe 불변은 정상 (레버리지 스케일링).

CTREND 비교: [references/gate-criteria.md](references/gate-criteria.md) 참조.

**FAIL -> Step F** | **PASS -> Step 4**

---

## Step 4: Gate 4 — 심층검증

### 실행 (2단계)

**Phase A: WFA (Walk-Forward Analysis)**

```bash
uv run mcbot backtest validate \
  -s {strategy_name} --symbols {best_asset} \
  -m milestone -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

**Phase B: CPCV + PBO + DSR + Monte Carlo**

```bash
uv run mcbot backtest validate \
  -s {strategy_name} --symbols {best_asset} \
  -m final -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

WFA: OOS Sharpe(평균), Decay, Consistency, Fold별 IS/OOS.
CPCV+PBO+DSR: CPCV 평균 OOS Sharpe, PBO(%), DSR(batch/all), MC p-value/95%CI.

### 판정 기준

| 조건 | 기준 | CTREND 참조 |
|------|------|------------|
| WFA OOS Sharpe | >= 0.5 | 1.49 |
| WFA Decay | < 40% | 39% |
| WFA Consistency | >= 60% | 67% |
| PBO | 이중 경로 (아래 참조) | 60% (경로 B PASS) |
| DSR (batch) | > 0.95 | 1.00 |
| MC p-value | < 0.05 | 0.000 |

상세: [references/gate-criteria.md](references/gate-criteria.md)

> **PBO 이중 경로**: 경로 A (PBO < 40%) 또는 경로 B (PBO < 80% AND CPCV 전 fold OOS 양수 AND MC p < 0.05).
> 경로 B는 파라미터 순위 역전이 있으나 기저 alpha가 견고한 전략을 구제한다.

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조.
핵심: WFA Decay vs G2 Decay 일관성 확인. 최근 fold OOS가 평균보다 낮으면 시장 적응 문제 경고.

CTREND 비교: [references/gate-criteria.md](references/gate-criteria.md) 참조.

**FAIL -> Step F** | **PASS -> Step S**

---

## Step F: 실패 처리

Gate FAIL 시 다음을 순차 실행한다.

### F-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --gate {GN} --verdict FAIL \
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

## Step S: 성공 처리 (G4 PASS)

### S-1. YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --gate G4 --verdict PASS \
  --detail "wfa_oos_sharpe={X.XX}" --detail "pbo={XX}" \
  --rationale "WFA/CPCV/PBO/DSR 모두 PASS"
```

상태 `TESTING` 유지 (G5 EDA Parity 대기).
G5 검증: **2년** (2024-01-01 ~ 2025-12-31), 구현 정합성 검증이므로 6년 불필요.

### S-2. Dashboard + 리포트

```bash
uv run mcbot pipeline report
```

리포트 형식: [references/report-template.md](references/report-template.md) 참조.

---

## 참조 문서

- [references/gate-criteria.md](references/gate-criteria.md) — Gate별 정량 기준 + CTREND 비교 + CLI 명령
- [references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) — 시니어 퀀트 해석 패턴
- [references/report-template.md](references/report-template.md) — 리포트 출력 형식
- `pipeline report` — 전략 상황판 (CLI)
