---
name: gate-pipeline
description: >
  G0B PASS 전략의 G1~G4 순차 검증 파이프라인. 시니어 퀀트 관점에서 백테스트(G1),
  IS/OOS(G2), 파라미터 안정성(G3), WFA/CPCV/PBO/DSR(G4)을 실행하고
  스코어카드/대시보드를 자동 갱신한다. FAIL 시 즉시 중단 + 폐기 처리.
  사용 시점: (1) G0B PASS 전략의 본격 검증,
  (2) "gate", "pipeline", "G1", "검증 파이프라인" 요청 시,
  (3) 여러 전략을 순차적으로 Gate 검증할 때.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name> [--from g1|g2|g3|g4]
---

# Gate Pipeline: G1~G4 순차 검증

## 역할

**시니어 퀀트 리서처 겸 검증 엔지니어**로서 행동한다.

핵심 원칙:

- 단순 threshold 비교가 아닌 **경제적 의미 해석** — 숫자 뒤의 이유를 찾는다
- FAIL 시 **구체적 사유 + 수정 방향** 제시 (단순 "FAIL" 판정 금지)
- Gate 간 **결과 일관성 추적** — G1 Sharpe → G2 OOS Sharpe → G4 WFA OOS 흐름 확인
- 모든 결과를 **CTREND 선례**와 비교하여 상대적 위치 파악

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ac-regime` |
| `--from gN` | X | 시작 Gate (기본: g1). 이전 Gate 결과는 스코어카드에서 복원 | `--from g2` |

---

## Step 0: Pre-flight Check

다음 5항목을 검증한 후 진행한다. **하나라도 실패하면 중단**.

### 0-1. 전략 디렉토리 존재

```bash
ls src/strategy/{name_underscore}/
# config.py, preprocessor.py, signal.py, strategy.py 존재 확인
```

`name_underscore`는 `strategy_name`의 하이픈을 언더스코어로 변환 (e.g., `ac-regime` → `ac_regime`).

### 0-2. 스코어카드 존재 + G0B PASS

```bash
# 스코어카드 파일 존재 확인
cat docs/scorecard/{strategy_name}.md
# G0B [PASS] 확인 — "G0B" 행에서 PASS 키워드 존재
```

G0B PASS가 없으면 중단: "G0B 미통과 전략입니다. `/verify-strategy`를 먼저 실행하세요."

### 0-3. Silver 데이터 존재

5개 코인의 Silver 데이터가 2020~2025 범위에 존재하는지 확인:

```bash
ls data/silver/BTC_USDT_1D.parquet data/silver/ETH_USDT_1D.parquet \
   data/silver/BNB_USDT_1D.parquet data/silver/SOL_USDT_1D.parquet \
   data/silver/DOGE_USDT_1D.parquet
```

### 0-4. --from 복원 (해당 시)

`--from g2` 등 지정 시, 스코어카드에서 이전 Gate 결과를 파싱하여 Best Asset 등 변수를 복원한다.

```
--from g2 → G1 결과에서 Best Asset 추출
--from g3 → G1 Best Asset + G2 OOS Sharpe 추출
--from g4 → G1 Best Asset + G2/G3 결과 추출
```

### 0-5. Gate 3 스윕 등록 확인

`scripts/gate3_param_sweep.py`의 STRATEGIES dict에 해당 전략이 등록되어 있는지 확인한다.
미등록이면 G3 단계에서 등록 작업을 수행한다 (사용자 승인 필요).

---

## Step 1: Gate 1 — 단일에셋 백테스트

### 실행

5개 코인 × 6년 (2020-01-01 ~ 2025-12-31) 백테스트를 실행한다.

```bash
# 각 심볼별 실행
uv run python -m src.cli.backtest run {strategy_name} {SYMBOL} \
  --start 2020-01-01 --end 2025-12-31 --capital 100000
```

5개 심볼: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `SOL/USDT`, `DOGE/USDT`

> 또는 `scripts/bulk_backtest.py` 패턴으로 Python API 직접 호출하여 한 번에 실행.
> 스크립트 방식이 더 빠르고 JSON으로 결과를 수집할 수 있다.

### 수집 지표

에셋별로 다음 지표를 수집하여 표로 정리:

| 지표 | 수집 위치 |
|------|----------|
| Sharpe Ratio | `metrics.sharpe_ratio` |
| CAGR (%) | `metrics.cagr` |
| MDD (%) | `metrics.max_drawdown` |
| Total Trades | `metrics.total_trades` |
| Profit Factor | `metrics.profit_factor` |
| Win Rate (%) | `metrics.win_rate` |
| Sortino Ratio | `metrics.sortino_ratio` |
| Calmar Ratio | `metrics.calmar_ratio` |
| Alpha (vs BTC B&H) | `benchmark.alpha` |
| Beta (vs BTC) | `benchmark.beta` |

### 판정 기준

**PASS 조건** (Best Asset 기준):

- Sharpe > 1.0
- CAGR > 20%
- MDD < 40%
- Trades > 50

**즉시 폐기 (전 에셋 해당 시)**:

1. MDD > 50% — 전 에셋에서 MDD 50% 초과
2. Sharpe < 0 — 전 에셋에서 음수 Sharpe
3. Trades < 20 + 수익 음수 — 극소 거래 + 손실
4. 80%+ 단일 거래 — 총 수익의 80% 이상이 단일 거래에서 발생

### 비용 민감도 체크

연간 거래 100회 이상인 에셋에 대해 2배 비용 시나리오 기록:

```
Cost = maker 0.04% + taker 0.08% + slippage 0.10%  (편도 ~0.22%)
```

2배 비용 Sharpe가 0 이하이면 **비용 민감 경고** 기록.

### 퀀트 해석 (단순 PASS/FAIL을 넘어서)

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조하여:

1. **Best Asset 패턴**: SOL > BTC > BNB > ETH > DOGE 순서가 일반적 (추세추종). 역순이면 전략 특성 분석
2. **DOGE 성과**: DOGE에서 양의 수익이면 범용성 높음. 최악이더라도 양수면 양호
3. **Beta 분석**: Beta < 0.3이면 BTC 독립적 알파. Beta > 0.5이면 BTC 방향성 의존
4. **거래 수 분포**: 에셋간 거래 수 편차가 크면 신호 생성 메커니즘 점검 필요
5. **MDD 패턴**: 전 에셋 MDD 유사 → 동일 레짐에서 동시 손실 (시스템 리스크)

### CTREND 비교 (참조점)

| 지표 | CTREND Best (SOL) | 현재 전략 Best |
|------|-------------------|---------------|
| Sharpe | 2.05 | ? |
| CAGR | +97.8% | ? |
| MDD | -27.7% | ? |
| Trades | 288 | ? |

### 문서 갱신

스코어카드에 다음 섹션 추가/갱신:

1. **성과 요약** 섹션: 에셋별 비교 표 (CTREND ctrend.md:24-31 패턴)
2. **Best Asset 핵심 지표** 표 (CTREND ctrend.md:34-46 패턴)
3. **Gate 진행 현황**: `G1 백테스트 [PASS/FAIL] {Best Asset} Sharpe X.XX, CAGR +XX.X%, MDD -XX.X%`
4. **Gate 1 상세** 섹션 추가 (CTREND ctrend.md:184-191 패턴)
5. **의사결정 기록** 행 추가

**FAIL → Step F (실패 처리)**
**PASS → Step 2**

---

## Step 2: Gate 2 — IS/OOS 70/30

### 실행

Best Asset에 대해 IS/OOS 검증을 실행한다:

```bash
uv run python -m src.cli.backtest validate \
  -s {strategy_name} \
  --symbols {best_asset} \
  -m quick \
  -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

| 지표 | 설명 |
|------|------|
| IS Sharpe | In-Sample Sharpe Ratio |
| OOS Sharpe | Out-of-Sample Sharpe Ratio |
| Decay (%) | `(1 - OOS/IS) × 100` |
| OOS Trades | OOS 구간 거래 수 |
| Consistency | OOS fold 양수 비율 |
| Overfit Probability | 과적합 확률 |

### 판정 기준

상세 기준: [references/gate-criteria.md](references/gate-criteria.md)

| 조건 | 기준 | CTREND 참조 |
|------|------|------------|
| OOS Sharpe | >= 0.3 | 1.78 |
| Decay | < 50% | 33.7% |
| OOS Trades | >= 15 | — |

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조:

1. **Decay 패턴 해석**:
   - < 20%: 우수 (일반화 강함)
   - 20-35%: 양호
   - 35-50%: 경계 (특정 파라미터에 의존)
   - > 50%: 과적합 (FAIL)

2. **G1 → G2 일관성**: G1 Best Asset Sharpe 대비 OOS Sharpe 관계 확인
   - OOS Sharpe > G1 Sharpe × 0.5 이면 양호
   - OOS Sharpe < G1 Sharpe × 0.3 이면 과적합 강력 의심

3. **역방향 과적합 탐지**: OOS > IS이면 (Decay 음수), 데이터 분할 문제 또는 레짐 특성 분석 필요

4. **OOS MDD 확인**: OOS 구간의 MDD가 IS 대비 2배 이상 증가하면 경고

### 문서 갱신

스코어카드에 다음 추가:

1. **Gate 2 상세** 섹션 (CTREND ctrend.md:169-182 패턴):

   ```markdown
   ### Gate 2 상세 (IS/OOS 70/30, {best_asset})

   | 지표 | IS (70%) | OOS (30%) | 기준 | 판정 |
   |------|----------|-----------|------|------|
   | Sharpe | X.XX | X.XX | OOS > 0.3 | PASS/FAIL |
   | Decay | — | XX.X% | < 50% | PASS/FAIL |
   ...
   ```

2. **Gate 진행 현황** 갱신: `G2 IS/OOS [PASS/FAIL] OOS Sharpe X.XX, Decay XX.X%`
3. **의사결정 기록** 행 추가

**FAIL → Step F (실패 처리)**
**PASS → Step 3**

---

## Step 3: Gate 3 — 파라미터 안정성

### 사전 작업: 스윕 등록 확인

`scripts/gate3_param_sweep.py`의 STRATEGIES dict에 해당 전략이 등록되어 있는지 확인한다.

**미등록 시** (사용자 승인 필요 — 코드 수정):

1. 전략의 `config.py`에서 핵심 파라미터 3~5개 식별
2. 각 파라미터의 baseline 값 (기본 Config 값) 확인
3. ±20% + 넓은 범위 그리드 정의 (총 8~10개 값)
4. STRATEGIES dict에 entry 추가:

```python
"{strategy_name}": {
    "best_asset": "{best_asset}",  # G1에서 확인된 Best Asset
    "baseline": {
        "param1": baseline_value,
        "param2": baseline_value,
        "vol_target": 0.XX,
        "short_mode": N,
    },
    "sweeps": {
        "param1": [v1, v2, ..., v10],  # ±20% 포함 + 넓은 범위
        "param2": [v1, v2, ..., v10],
        "vol_target": [0.15, 0.20, ..., 0.60],
    },
},
```

> **파라미터 선정 원칙**:
>
> - vol_target은 항상 포함 (모든 전략 공통)
> - 전략 핵심 로직 파라미터 2~3개 (lookback, threshold 등)
> - short_mode는 스윕 대상 아님 (이산값)

### 실행

```bash
uv run python scripts/gate3_param_sweep.py {strategy_name}
```

결과는 `results/gate3_param_sweep.json`에 저장되고 콘솔에 Rich 테이블로 출력된다.

### 수집 지표 (파라미터별)

| 지표 | 설명 |
|------|------|
| Plateau 존재 | Best Sharpe의 80% 이상인 값 >= 3개 |
| Plateau Count | 고원에 속하는 값 수 |
| Plateau Range | 고원 범위 (min~max) |
| ±20% Stable | 기본값 ±20% 범위에서 Sharpe 양수 유지 |
| ±20% Sharpe | ±20% 범위 Sharpe min~max |

### 판정 기준

| 조건 | 기준 |
|------|------|
| 고원 존재 | **모든** 핵심 파라미터에서 60%+ (plateau_count/total >= 60%) |
| ±20% 안정 | **모든** 핵심 파라미터에서 Sharpe 부호 양수 유지 |

**전체 PASS**: 모든 파라미터가 (고원 존재 AND ±20% 안정)

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조:

1. **고원 형태 분석**:
   - 넓은 고원: 전략이 로버스트 (CTREND의 alpha가 대표적)
   - 좁은 봉우리: 과적합 위험 (TTM Squeeze의 bb_period 패턴)
   - 단조 경사: 파라미터 방향성 존재 (prediction_horizon 패턴)
   - 절벽: 특정 값에서 급락 (kc_mult 1.0 거래 0건 패턴)

2. **vol_target 민감도**: Sharpe 거의 불변 → 순수 레버리지 스케일링 (정상). CAGR만 변동

3. **파라미터 간 교호작용**: 2개 파라미터가 동시에 같은 방향으로 성과 영향 시 주의

### CTREND 비교 (참조점)

| 파라미터 | CTREND 고원 | CTREND ±20% |
|---------|------------|-------------|
| training_window | 9/10 (126~350) | 1.93~2.05 |
| prediction_horizon | 3/10 | 1.78~2.07 |
| alpha | 9/10 (0.1~1.0) | 1.77~2.05 |
| vol_target | 10/10 (0.15~0.60) | 2.05~2.06 |

### 문서 갱신

스코어카드에 다음 추가:

1. **Gate 3 상세** 섹션 (CTREND ctrend.md:62-83 패턴):

   ```markdown
   ### Gate 3 상세 (파라미터 안정성, {best_asset} 1D)

   | 파라미터 | 기본값 | Best Sharpe | 고원 | 고원 범위 | ±20% Sharpe | 판정 |
   |---------|--------|-------------|:---:|----------|-------------|:---:|
   | param1 | XX | X.XX | N/M | min~max | X.XX~X.XX | PASS |
   ...
   ```

2. **분석** + **핵심 관찰** 서브섹션
3. **Gate 진행 현황** 갱신: `G3 파라미터 [PASS/FAIL] N/N 파라미터 고원 + ±20% 안정`
4. **의사결정 기록** 행 추가

**FAIL → Step F (실패 처리)**
**PASS → Step 4**

---

## Step 4: Gate 4 — 심층검증

### 실행 (2단계)

**Phase A: WFA (Walk-Forward Analysis)**

```bash
uv run python -m src.cli.backtest validate \
  -s {strategy_name} \
  --symbols {best_asset} \
  -m milestone \
  -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

**Phase B: CPCV + PBO + DSR + Monte Carlo**

```bash
uv run python -m src.cli.backtest validate \
  -s {strategy_name} \
  --symbols {best_asset} \
  -m final \
  -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

### 수집 지표

#### WFA

| 지표 | 설명 |
|------|------|
| WFA OOS Sharpe (평균) | Walk-Forward OOS Sharpe 평균 |
| WFA Sharpe Decay | IS → OOS Decay |
| WFA Consistency | OOS fold 양수 비율 |
| Fold별 IS/OOS Sharpe | 각 fold 상세 |

#### CPCV + PBO + DSR

| 지표 | 설명 |
|------|------|
| CPCV 평균 OOS Sharpe | C(5,2) 10-fold 평균 |
| PBO (%) | Probability of Backtest Overfitting |
| DSR (batch) | Deflated Sharpe Ratio (동일 배치 기준) |
| DSR (all) | Deflated Sharpe Ratio (전 전략 기준) |
| MC p-value | Monte Carlo p-value |
| MC 95% CI | Monte Carlo 95% 신뢰구간 |

### 판정 기준

상세 기준: [references/gate-criteria.md](references/gate-criteria.md)

| 조건 | 기준 | CTREND 참조 |
|------|------|------------|
| WFA OOS Sharpe | >= 0.5 | 1.49 |
| WFA Decay | < 40% | 39% |
| WFA Consistency | >= 60% | 67% |
| PBO | 이중 경로 (A: <40% / B: <80% + CPCV 전fold OOS>0 + MC p<0.05) | 60% (경로 B PASS) |
| DSR (batch) | > 0.95 | 1.00 |
| MC p-value | < 0.05 | 0.000 |

> **PBO 이중 경로**: 경로 A (PBO < 40%) 또는 경로 B (PBO < 80% AND CPCV 전 fold OOS 양수 AND MC p < 0.05).
> 경로 B는 파라미터 순위 역전이 있으나 기저 alpha가 견고한 전략을 구제한다.

### 퀀트 해석

[references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) 참조:

1. **PBO 해석**: 이중 경로로 판정한다.
   - 경로 A: PBO < 40% → PASS (과적합 위험 낮음)
   - 경로 B: PBO < 80% AND CPCV 전 fold OOS > 0 AND MC p < 0.05 → PASS (기저 alpha 견고)
   - PBO >= 80%: FAIL (경로 B 상한 초과)

2. **WFA Decay vs G2 Decay 일관성**: 두 값이 비슷하면 일관된 IS→OOS 감쇠 패턴. 차이가 크면 CV 방법론 민감도 주의

3. **DSR n_trials 민감도**: batch (동일 배치) vs cumulative (전 전략) 기준이 다름
   - batch PASS + all FAIL: 해당 배치 내에서는 유의, 전체 전략 대비로는 불확실
   - 둘 다 PASS: 강건한 결과

4. **최근 fold 성과**: WFA 마지막 fold (가장 최근 기간)의 OOS Sharpe가 평균보다 낮으면 최근 시장 적응 문제 경고

5. **MC 95% CI 하한**: CI 하한 > 0이면 Sharpe가 통계적으로 양수

### 문서 갱신

스코어카드에 다음 추가:

1. **Gate 4 상세** 섹션 (CTREND ctrend.md:85-146 패턴):
   - WFA 서브섹션: fold별 표 + 판정 표
   - CPCV 서브섹션: fold별 IS/OOS Sharpe 표
   - PBO/DSR/MC 서브섹션: 판정 표
   - Gate 4 종합 판정 + 분석 + 핵심 관찰
2. **Gate 진행 현황** 갱신
3. **의사결정 기록** 행 추가

**FAIL → Step F (실패 처리)**
**PASS → Step S (성공 처리)**

---

## Step F: 실패 처리

Gate FAIL 시 다음을 순차 실행한다.

### F-1. 스코어카드에 FAIL 기록

- 해당 Gate의 상세 결과 + FAIL 사유 기록
- 상태를 `폐기 (Gate N FAIL)` 로 변경
- 의사결정 기록에 FAIL 판정 + 근거 추가

### F-2. 스코어카드 이동

```bash
# fail/ 디렉토리 존재 확인
mkdir -p docs/scorecard/fail/

# 스코어카드를 fail/ 디렉토리로 이동
mv docs/scorecard/{strategy_name}.md docs/scorecard/fail/{strategy_name}.md
```

### F-3. Dashboard 갱신 (`docs/strategy/dashboard.md`)

1. **"검증중 전략" 테이블**에서 해당 전략 행 제거
2. **해당 Gate 실패 섹션**에 새 행 추가:
   - Gate 1 실패 → "Gate 1 실패" 섹션
   - Gate 2 실패 → "Gate 2 실패 — IS/OOS 과적합" 섹션
   - Gate 3 실패 → "Gate 3 실패 — 파라미터 불안정" 섹션
   - Gate 4 실패 → "Gate 4 실패 — WFA 심층검증" 섹션
3. 행 형식은 해당 섹션의 기존 행과 일치시킨다

### F-4. 최종 리포트 출력

Step 10의 리포트 형식으로 출력한다.

---

## Step S: 성공 처리 (G4 PASS)

### S-1. 스코어카드 완성

- Gate 4까지 전체 결과 기록 완료
- 상태는 `검증중` 유지 (G5 EDA Parity 대기)
- G5 검증 기간은 **2년** (2024-01-01 ~ 2025-12-31). 성능 평가가 아닌 구현 정합성 검증이므로 전체 6년 불필요.

### S-2. Dashboard 갱신 (`docs/strategy/dashboard.md`)

- "검증중 전략" 테이블의 해당 행에 Gate 진행 상태 갱신
- `G1` ~ `G4` 컬럼에 P/F 추가
- `다음 단계` → `G5 EDA Parity`

### S-3. 최종 리포트 출력

Step 10의 리포트 형식으로 출력한다.

---

## 리포트 출력 형식

모든 Gate 완료 후 (FAIL 또는 G4 PASS) 아래 형식으로 리포트를 출력한다:

```
============================================================
  GATE PIPELINE REPORT
  전략: {display_name} ({registry_key})
  실행일: {YYYY-MM-DD}
  범위: G{start} -> G{end}
  Best Asset: {symbol} (1D)
============================================================

  GATE 1: 단일에셋 백테스트
  Best Asset: {symbol} | Sharpe {X.XX} | CAGR {+XX.X%} | MDD {-XX.X%} | Trades {N}
  판정: [PASS / FAIL]
  {FAIL 사유 또는 주요 관찰}

------------------------------------------------------------

  GATE 2: IS/OOS 70/30
  OOS Sharpe: {X.XX} | Decay: {XX.X%} | OOS Trades: {N}
  판정: [PASS / FAIL]
  {해석}

------------------------------------------------------------

  GATE 3: 파라미터 안정성
  파라미터: {N}/{M} PASS
  {파라미터별 한줄 요약}
  판정: [PASS / FAIL]

------------------------------------------------------------

  GATE 4: 심층검증
  WFA OOS: {X.XX} | Decay: {XX.X%} | Consist: {XX%}
  PBO: {XX%} | DSR: {X.XX} | MC p: {X.XXX}
  판정: [PASS / FAIL]
  {해석}

============================================================
  PIPELINE SUMMARY
  최종 판정: [G4 PASS / G{N} FAIL]
  다음 단계: {G5 EDA Parity / 폐기}
  스코어카드: docs/scorecard/{path}.md (갱신 완료)
  대시보드:   docs/strategy/dashboard.md (갱신 완료)
============================================================
```

---

## 참조 문서

- [references/gate-criteria.md](references/gate-criteria.md) — Gate별 정량 기준 + CLI 명령 표
- [references/quant-interpretation-guide.md](references/quant-interpretation-guide.md) — 시니어 퀀트 해석 패턴
- [docs/scorecard/ctrend.md](../../../docs/scorecard/ctrend.md) — CTREND 스코어카드 (참조 선례)
- [docs/strategy/dashboard.md](../../../docs/strategy/dashboard.md) — 전략 상황판
- [docs/strategy/evaluation-standard.md](../../../docs/strategy/evaluation-standard.md) — 전략 평가 표준
