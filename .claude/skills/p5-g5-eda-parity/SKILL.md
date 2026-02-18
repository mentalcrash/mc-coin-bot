---
name: p5-g5-eda-parity
description: >
  G5 EDA Parity 검증 — VBT vs EDA 수익 정합성 + 라이브 준비 상태 점검.
  사용 시점: G4 PASS 전략의 EDA 검증, "G5" "EDA parity" 요청 시.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name> [--symbol SYMBOL] [--period 1y|2y]
---

# Gate 5: EDA Parity 검증

## 역할

**시니어 퀀트 시스템 엔지니어**로서 행동한다.

핵심 원칙:

- **정합성 우선**: VBT와 EDA 결과의 불일치 원인을 구조적으로 분석
- **라이브 관점**: "이 시스템으로 실거래해도 안전한가?" 판단
- **수치 근거**: 모든 판정에 구체적 수치 비교 동반
- **CTREND 선례 비교**: G5 통과 유일 전략과의 상대적 위치 파악

---

## Gate 5의 목적

Gate 5는 **성능 평가가 아닌 구현 정합성 검증**이다.

| 질문 | 기준 |
|------|------|
| VBT와 EDA가 같은 방향으로 수익을 내는가? | 수익 부호 일치 (핵심) |
| 절대 수익률의 괴리가 허용 범위인가? | 편차 < 20% |
| 거래 패턴이 근사한가? | 거래 수 비율 0.5x ~ 2.0x |
| EDA 코드가 라이브 전환에 안전한가? | 코드 레벨 7항목 점검 |

VBT = 이상적 벡터화 실행, EDA = 실전 시뮬레이션 (bar-by-bar). 결과 일치 시 EDA 실행 로직 신뢰.

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ctrend` |
| `--symbol` | X | 검증 심볼 (기본: YAML Best Asset) | `SOL/USDT` |
| `--period` | X | 검증 기간 (기본: `1y`) | `1y`, `2y` |

**기간**: 기본 1년 (2025-01-01~2026-01-01). 2년은 레짐 다양성 확보 (권장 최대). 속도: 1년 ~2-5분, 2년 ~4-10분.

**실행 모드**: `shift(-N)`, `forward_return`, `.fit()`, `ewm()` → `--fast` 필수. 순수 rolling → 둘 다 가능. intrabar SL/TS 검증 → standard.

---

## Step 0: Pre-flight Check

다음 7항목을 검증한 후 진행한다. **하나라도 실패하면 중단**.

### 0-1. 전략 디렉토리 존재

```bash
ls src/strategy/{name_underscore}/
# config.py, preprocessor.py, signal.py, strategy.py 존재 확인
```

### 0-2. YAML 메타데이터 + G4 PASS

```bash
cat strategies/{strategy_name}.yaml
# gates 섹션에서 G4: status: PASS 확인
```

YAML 없으면 `uv run mcbot pipeline create`로 생성. G4 PASS 없으면 중단.

### 0-3. Best Asset + TF 추출

YAML에서 추출: `best_asset`, `best_tf`, `g1_sharpe`, `g1_cagr`. `--symbol` 인수가 있으면 best_asset 덮어씀.

### 0-4. 1분봉 Silver 데이터 존재

```bash
ls data/silver/{symbol_underscore}_1m.parquet
# 없으면: uv run mcbot ingest pipeline {symbol} --timeframe 1m --year {years}
```

### 0-5. EDA Config 존재/생성

```bash
ls config/{strategy_name}*.yaml
```

없으면 생성 (사용자 승인 필요). 파일명: `config/{strategy_name}_g5_{period}.yaml`.
기존 config에서 기간만 조정. 필수 섹션: `backtest` (symbols, timeframe, start, end, capital), `strategy` (name, params), `portfolio` (max_leverage_cap, rebalance_threshold, system_stop_loss, use_trailing_stop, trailing_stop_atr_multiplier, cost_model).

### 0-6. max_leverage_cap 확인

YAML 또는 config에서 확인. Tier 7 등 커스텀 값 반드시 반영.

### 0-7. 실행 모드 결정

```bash
grep -rn "shift(-\|forward_return\|\.fit(\|ewm(" src/strategy/{name_underscore}/
```

감지되면 `fast_mode = True`.

---

## Step 1: VBT 벡터화 백테스트 실행

```bash
uv run mcbot backtest run {strategy_name} {best_asset} \
  --start {start_date} --end {end_date} --capital 100000
```

수집: `vbt_sharpe`, `vbt_cagr`, `vbt_return`, `vbt_mdd`, `vbt_trades`, `vbt_winrate`, `vbt_pf`.

> G5 검증 기간은 G1(6년)보다 짧으므로 VBT 지표가 G1과 다를 수 있다. **동일 기간 VBT vs EDA 비교**가 핵심.

---

## Step 2: EDA 이벤트 기반 백테스트 실행

```bash
# fast mode (forward_return/EWM 전략)
uv run mcbot eda run config/{strategy_name}_g5_{period}.yaml --fast

# standard mode (순수 rolling indicator 전략)
uv run mcbot eda run config/{strategy_name}_g5_{period}.yaml
```

수집: `eda_sharpe`, `eda_cagr`, `eda_return`, `eda_mdd`, `eda_trades`, `eda_winrate`, `eda_pf`.

실행 에러 → **즉시 FAIL**. 일반 원인: 1m 데이터 부재, precomputed_signals 오류, PM/RM config 불일치, 메모리 부족.

---

## Step 3: Parity 비교 분석

### 3-1. 핵심 지표 비교표

| 지표 | VBT | EDA | 편차 | 기준 | 판정 |
|------|-----|-----|------|------|------|
| **수익 부호** | +/- | +/- | — | 일치 필수 | PASS/FAIL |
| **Sharpe** | X.XX | X.XX | XX.X% | — | 참고 |
| **CAGR** | +XX.X% | +XX.X% | XX.X%p | 부호 일치 | PASS/FAIL |
| **Total Return** | +XX.X% | +XX.X% | XX.X% | 편차 < 20% | PASS/FAIL |
| **MDD** | XX.X% | XX.X% | XX.X%p | — | 참고 |
| **Trades** | N | N | X.Xx | 0.5x~2.0x | PASS/FAIL |
| **Win Rate** | XX.X% | XX.X% | XX.X%p | — | 참고 |
| **Profit Factor** | X.XX | X.XX | XX.X% | — | 참고 |

### 3-2. 편차 계산

```
수익 편차 (%) = |eda_return - vbt_return| / max(|vbt_return|, 1.0) * 100
거래 비율 = eda_trades / vbt_trades
```

### 3-3. 통과 기준

| 조건 | 기준 | 가중치 |
|------|------|--------|
| **수익 부호 일치** | VBT 양수 → EDA 양수, 또는 둘 다 음수 | **필수** (FAIL 시 즉시 중단) |
| **수익률 편차** | < **20%** | 필수 |
| **거래 수 비율** | 0.5x ~ 2.0x | 필수 |

> **CTREND 선례**: Trades 0.25x (PM threshold 구조적 필터링). 구조적 사유면 기준 미달도 PASS 가능.

### 3-4. 거래 수 괴리 분석

비율 0.5x 미만 또는 2.0x 초과 시, **원인 분석 후 구조적 사유면 PASS 가능**:

| 원인 | 구조적? | 판정 |
|------|:------:|------|
| PM `rebalance_threshold` 필터링 | O | PASS (CTREND 선례) |
| RM `max_order_size` 거부 | O | PASS (안전 기제) |
| 전략 로직 버그 / 데이터 불일치 | X | FAIL |
| EWM/rolling 초기화 차이 | △ | fast mode 재실행 후 재판정 |

---

## Step 4: 괴리 원인 심층 분석

편차 존재 시 (통과 여부 무관) 원인을 분석하여 기록한다.
상세 카탈로그: [references/parity-criteria.md](references/parity-criteria.md) 참조.

### VBT vs EDA 핵심 구조적 차이

| 차이점 | VBT | EDA |
|--------|-----|-----|
| 실행 방식 | 전체 벡터화 | bar-by-bar 이벤트 |
| 체결 가격 | 다음 bar close | 다음 TF bar open (deferred) |
| PM/RM | 없음 | rebalance_threshold, max_order_size |
| SL/TS | 벡터화 stop | ATR 기반 trailing stop |
| 비용 모델 | 정적 적용 | bar별 funding drag |

### 괴리 라이브 영향

| 괴리 유형 | 심각도 |
|----------|--------|
| EDA > VBT | 양호 (PM/RM 방어 효과) |
| EDA < VBT | 정상 (비용 현실 반영) |
| 부호 불일치 / 거래 0건 | **치명적** |
| MDD 급증 (EDA > VBT x 2) | **경고** |

---

## Step 5: 라이브 준비 상태 점검 (Code-Level)

EDA 코드의 라이브 안전성 7항목 점검. 상세 검증 패턴: [references/live-readiness-checklist.md](references/live-readiness-checklist.md) 참조.

| 항목 | PASS 기준 |
|------|----------|
| L1 EventBus Flush | TF bar 완성 시 `bus.flush()` 호출 |
| L2 Executor Handler 순서 | `executor_bar_handler` 첫 번째 등록 |
| L3 Deferred Execution | 두 Executor 동일 패턴 |
| L4 PM Batch Mode | `flush_pending_signals()` 존재 (멀티에셋) |
| L5 Position Reconciler | `_periodic_reconciliation()` 존재 |
| L6 Graceful Shutdown | SIGTERM handler + task cancellation |
| L7 Circuit Breaker | CB close 시 `pos.last_price` 설정 |

> L1/L2/L3/L6/L7 = Critical. L4/L5 = Non-critical (WARNING 허용).

---

## Step 6: 판정

### 6-1. Parity 판정

| 조건 | 결과 | 판정 |
|------|------|------|
| 수익 부호 일치 | ? | ? |
| 수익률 편차 < 20% | ? | ? |
| 거래 수 비율 0.5x~2.0x (또는 구조적 사유) | ? | ? |

**Parity 종합**: 3개 모두 PASS → **G5 Parity PASS**

### 6-2. Live Readiness 판정

| 항목 | 결과 |
|------|------|
| L1~L7 | ? |

**Live Readiness 종합**: 전 항목 PASS → **Live Ready**

### 6-3. Gate 5 최종 판정

| 판정 | 조건 |
|------|------|
| **PASS** | Parity PASS **AND** Live Readiness 전 항목 PASS |
| **CONDITIONAL PASS** | Parity PASS **AND** Live Readiness 1~2개 WARNING (non-critical) |
| **FAIL** | Parity FAIL **OR** Live Readiness critical 항목 FAIL |

---

## Step 7: 문서 갱신

### YAML 갱신 (필수)

```bash
uv run mcbot pipeline record {strategy_name} \
  --gate G5 --verdict PASS \
  --detail "eda_sharpe={X.XX}" --detail "vbt_sharpe={X.XX}" \
  --rationale "EDA Parity PASS. 수익 부호 일치, Sharpe 편차 XX%"
# PASS 시: store.update_status(name, StrategyStatus.ACTIVE)
```

### 진행 현황 + 의사결정 기록

```markdown
| {날짜} | G5 | PASS/FAIL | EDA Sharpe X.XX vs VBT X.XX. Trades N. Live Ready 7/7 |
```

### 교훈 기록 (FAIL 시, 새로운 패턴만)

```bash
uv run mcbot pipeline lessons-list --tag EDA
uv run mcbot pipeline lessons-add \
  --title "{괴리 원인}" --category {category} --tag EDA --tag G5
```

### Dashboard

```bash
uv run mcbot pipeline report
```

---

## Step 8: CTREND 비교

| 지표 | VBT | EDA | 편차 | 교훈 |
|------|-----|-----|------|------|
| Sharpe | 2.05 | 2.82 | +37.6% | PM threshold 비용 절감 |
| CAGR | +97.8% | +173.8% | +77.7% | 거래 감소 + TS 효과 |
| MDD | 27.7% | 19.8% | -28.5% | Trailing stop ATR 3.0x |
| Trades | 288 | 72 | -75.0% | rebalance_threshold 10% |

현재 전략과 비교: Sharpe 편차 방향, 거래 수 비율, MDD 개선 여부, 실행 모드.

---

## 리포트 출력

리포트 형식: [references/report-template.md](references/report-template.md) 참조.

---

## 참조 문서

- [references/parity-criteria.md](references/parity-criteria.md) — Parity 정량 기준 + 괴리 원인 카탈로그
- [references/live-readiness-checklist.md](references/live-readiness-checklist.md) — 라이브 준비 7항목 상세 검증 패턴
- [references/report-template.md](references/report-template.md) — G5 리포트 출력 형식
