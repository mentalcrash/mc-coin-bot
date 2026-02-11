---
name: eda-parity
description: >
  Gate 5 EDA Parity 검증. VBT 벡터화 백테스트와 EDA 이벤트 기반 백테스트의
  수익 정합성을 검증하고, 라이브 환경 전환 준비 상태를 확인한다.
  수익 부호 일치, 편차 < 20%, 거래 비율 0.5x~2.0x 기준으로 PASS/FAIL 판정.
  코드 레벨 라이브 준비 상태 점검(7항목)도 포함.
  사용 시점: (1) G4 PASS 전략의 EDA 검증,
  (2) "G5", "EDA parity", "EDA 검증", "라이브 준비" 요청 시,
  (3) VBT와 EDA 결과 비교가 필요할 때.
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

- **정합성 우선**: VBT와 EDA 결과의 불일치 원인을 구조적으로 분석한다
- **라이브 관점**: 백테스트 통과가 아닌, "이 시스템으로 실거래해도 안전한가?" 를 판단한다
- **수치 근거**: 모든 판정에 구체적 수치 비교를 동반한다
- **괴리 해석**: 단순 숫자 비교가 아닌, 괴리의 원인과 그것이 라이브에 미치는 영향을 설명한다
- **CTREND 선례 비교**: G5 통과 유일 전략인 CTREND와의 상대적 위치를 파악한다

---

## Gate 5의 목적

Gate 5는 **성능 평가가 아닌 구현 정합성 검증**이다.

| 질문 | 설명 |
|------|------|
| VBT와 EDA가 **같은 방향으로** 수익을 내는가? | 수익 부호 일치 (핵심) |
| 절대 수익률의 괴리가 허용 범위인가? | 편차 < 20% |
| 거래 패턴이 근사한가? | 거래 수 비율 0.5x ~ 2.0x |
| EDA 코드가 라이브 전환에 안전한가? | 코드 레벨 7항목 점검 |

VBT는 **이상적 벡터화 실행** (전 데이터 한 번에), EDA는 **실전 시뮬레이션** (bar-by-bar 이벤트 처리).
둘의 결과가 일치하면 EDA 시스템의 실행 로직을 신뢰할 수 있다.

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `ctrend` |
| `--symbol SYMBOL` | X | 검증 심볼 (기본: 스코어카드 Best Asset) | `SOL/USDT` |
| `--period` | X | 검증 기간 (기본: `1y`) | `1y`, `2y` |

### 검증 기간 결정 규칙

- **기본 1년** (2025-01-01 ~ 2026-01-01): 1m bar 리플레이 속도 고려. 1D TF 기준 ~525,600개 1m bar/year.
- **2년** (2024-01-01 ~ 2026-01-01): 상승장(2024) + 조정장(2025) 포함하여 레짐 다양성 확보. 권장 최대.
- **근거**: G5는 성능 평가가 아닌 구현 정합성 검증이므로 전체 6년 불필요. 1년이면 거래 20건+ 확보 가능 (1D TF 기준). 속도: 1년 ≈ 2-5분, 2년 ≈ 4-10분 (fast mode).

### 실행 모드 결정

| 전략 특성 | 모드 | 이유 |
|----------|------|------|
| `forward_return` 사용 (CTREND 등) | `--fast` | pre-computed signals로 edge effect 해결 |
| EWM feature 사용 | `--fast` 권장 | incremental buffer truncation 방지 |
| 순수 rolling indicator | `--fast` 또는 standard | 둘 다 가능 |
| intrabar SL/TS 검증 필요 | standard (fast 미사용) | 1m bar에서 SL/TS 발동 확인 |

> **판단 기준**: 전략 코드에서 `shift(-N)`, `forward_return`, `.fit()`, EWM feature가 있으면 → `--fast` 필수.
> 순수 rolling (SMA, RSI 등)만 사용하면 standard 모드도 가능하지만, 일관성 위해 fast 권장.

---

## Step 0: Pre-flight Check

다음 7항목을 검증한 후 진행한다. **하나라도 실패하면 중단**.

### 0-1. 전략 디렉토리 존재

```bash
ls src/strategy/{name_underscore}/
# config.py, preprocessor.py, signal.py, strategy.py 존재 확인
```

### 0-2. 스코어카드 존재 + G4 PASS

```bash
cat docs/scorecard/{strategy_name}.md
# Gate 진행 현황에서 G4 [PASS] 확인
```

G4 PASS가 없으면 중단: "G4 미통과 전략입니다. `/gate-pipeline`을 먼저 실행하세요."

### 0-3. Best Asset + TF 추출

스코어카드에서 다음 변수를 추출:

- `best_asset`: Best Asset 심볼 (예: `SOL/USDT`)
- `best_tf`: 타임프레임 (예: `1D`, `4h`)
- `g1_sharpe`: Gate 1 Sharpe (VBT 기준)
- `g1_cagr`: Gate 1 CAGR

`--symbol` 인수가 제공되면 best_asset을 덮어씀.

### 0-4. 1분봉 Silver 데이터 존재

```bash
# 심볼 변환: SOL/USDT → SOL_USDT
ls data/silver/{symbol_underscore}_1m.parquet
```

1m 데이터가 없으면:

```bash
# Bronze → Silver 1분봉 파이프라인 실행 안내
echo "1m data not found. Run:"
echo "  python main.py ingest pipeline {symbol} --timeframe 1m --year {years}"
```

### 0-5. YAML Config 존재/생성

G5용 YAML config 파일이 필요하다. 기존 config에서 기간만 조정:

```bash
# 기존 config 확인
ls config/{strategy_name}*.yaml
```

없으면 생성 (사용자 승인 필요). 기존 config 패턴 참조:

```yaml
backtest:
  symbols:
    - {best_asset}
  timeframe: "{best_tf}"
  start: "{start_date}"
  end: "{end_date}"
  capital: 100000.0

strategy:
  name: {strategy_name}
  params:
    # 스코어카드/기존 config에서 복사
    ...

portfolio:
  max_leverage_cap: 2.0
  rebalance_threshold: 0.10
  system_stop_loss: 0.10
  use_trailing_stop: true
  trailing_stop_atr_multiplier: 3.0
  cost_model:
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage: 0.0005
    funding_rate_8h: 0.0001
    market_impact: 0.0002
```

파일명: `config/{strategy_name}_g5_{period}.yaml` (예: `config/ctrend_g5_1y.yaml`)

### 0-6. max_leverage_cap 확인

전략 스코어카드 또는 config에서 `max_leverage_cap` 값을 확인한다.
Tier 7 전략 등 `max_leverage_cap` 커스텀 값이 있으면 반드시 반영.

### 0-7. 실행 모드 결정

전략 코드를 분석하여 fast_mode 사용 여부를 결정:

```bash
# forward_return, shift(-N), fit(), ewm() 등 감지
grep -rn "shift(-" src/strategy/{name_underscore}/
grep -rn "forward_return" src/strategy/{name_underscore}/
grep -rn "\.fit(" src/strategy/{name_underscore}/
grep -rn "ewm(" src/strategy/{name_underscore}/
```

감지되면 `fast_mode = True`. 없으면 전략 특성에 따라 결정.

---

## Step 1: VBT 벡터화 백테스트 실행

동일 기간 VBT 백테스트를 먼저 실행하여 기준값을 확보한다.

### 실행

```bash
uv run python -m src.cli.backtest run {strategy_name} {best_asset} \
  --start {start_date} --end {end_date} --capital 100000
```

### 수집 지표

| 지표 | 변수명 | 설명 |
|------|--------|------|
| Sharpe Ratio | `vbt_sharpe` | 위험조정 수익 |
| CAGR (%) | `vbt_cagr` | 복리 연평균 수익률 |
| Total Return (%) | `vbt_return` | 총 수익률 |
| MDD (%) | `vbt_mdd` | 최대 낙폭 |
| Total Trades | `vbt_trades` | 총 거래 수 |
| Win Rate (%) | `vbt_winrate` | 승률 |
| Profit Factor | `vbt_pf` | 총이익/총손실 |

> **주의**: G5 검증 기간은 G1 전체 기간(6년)보다 짧으므로, VBT 지표가 G1 스코어카드와 다를 수 있다.
> 이는 정상이며, **동일 기간 VBT vs EDA 비교**가 핵심이다.

---

## Step 2: EDA 이벤트 기반 백테스트 실행

### 실행

```bash
# fast mode (forward_return/EWM 전략)
uv run python main.py eda run config/{strategy_name}_g5_{period}.yaml --fast

# standard mode (순수 rolling indicator 전략)
uv run python main.py eda run config/{strategy_name}_g5_{period}.yaml
```

### 수집 지표

VBT와 동일 지표를 수집:

| 지표 | 변수명 |
|------|--------|
| Sharpe Ratio | `eda_sharpe` |
| CAGR (%) | `eda_cagr` |
| Total Return (%) | `eda_return` |
| MDD (%) | `eda_mdd` |
| Total Trades | `eda_trades` |
| Win Rate (%) | `eda_winrate` |
| Profit Factor | `eda_pf` |

### 실행 실패 시

EDA 실행이 에러로 실패하면 → **즉시 FAIL**. 에러 로그를 분석하여 원인 기록.

일반적 실패 원인:

| 원인 | 해결 |
|------|------|
| 1m 데이터 부재 | `python main.py ingest pipeline` 실행 |
| precomputed_signals 오류 | fast_mode 비활성화 또는 전략 코드 수정 |
| PM/RM config 불일치 | YAML config 검토 |
| 메모리 부족 | 기간 축소 (2y → 1y) |

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
수익 편차 (%) = |eda_return - vbt_return| / max(|vbt_return|, 1.0) × 100
거래 비율 = eda_trades / vbt_trades
```

> `vbt_return`이 0에 가까우면 (|vbt_return| < 1.0) 분모를 1.0으로 대체하여 극단 편차 방지.

### 3-3. 통과 기준

| 조건 | 기준 | 가중치 |
|------|------|--------|
| **수익 부호 일치** | VBT 양수 → EDA 양수, 또는 둘 다 음수 | **필수** (FAIL 시 즉시 중단) |
| **수익률 편차** | < **20%** | 필수 |
| **거래 수 비율** | 0.5x ~ 2.0x | 필수 |

> **CTREND 참조**: Sharpe +37.6% 편차, Trades -75.0% (0.25x). 거래 수 기준은 PM rebalance
> threshold 필터링으로 CTREND에서 이미 예외 확인됨. PM threshold에 의한 거래 감소는 구조적이며 허용.

### 3-4. 거래 수 괴리 분석

거래 수 비율이 0.5x 미만 또는 2.0x 초과일 경우, **원인 분석 후 구조적 사유면 PASS 가능**:

| 원인 | 구조적? | 판정 |
|------|:------:|------|
| PM `rebalance_threshold` 필터링 | O | PASS (CTREND 선례) |
| RM `max_order_size` 거부 | O | PASS (안전 기제) |
| 전략 로직 버그 (시그널 누락/중복) | X | FAIL |
| 데이터 불일치 (1m aggregation 오류) | X | FAIL |
| EWM/rolling 초기화 차이 | △ | fast mode로 재실행 후 재판정 |

---

## Step 4: 괴리 원인 심층 분석

수익률 편차가 존재하는 경우 (통과 여부와 무관하게), 원인을 분석하여 기록한다.

### 4-1. VBT vs EDA 구조적 차이

| 차이점 | VBT | EDA | 영향 |
|--------|-----|-----|------|
| **실행 방식** | 전체 벡터화 (한 번에) | bar-by-bar 이벤트 | 시그널 타이밍 미세 차이 |
| **체결 가격** | 다음 bar close | 다음 TF bar open (deferred) | 체결 가격 차이 |
| **PM 필터링** | 없음 | rebalance_threshold 적용 | 거래 수 감소 |
| **RM 검증** | 없음 | max_order_size, leverage 체크 | 일부 주문 거부 |
| **SL/TS** | 벡터화 stop | ATR 기반 trailing stop | MDD 차이 |
| **비용 모델** | 정적 적용 | bar별 funding drag | 수익률 미세 차이 |
| **시그널 계산** | 전체 데이터 | incremental/pre-computed | edge effect 가능 |

### 4-2. 괴리가 라이브에 미치는 영향

| 괴리 유형 | 라이브 영향 | 심각도 |
|----------|-----------|--------|
| EDA > VBT (수익 더 높음) | PM/RM 방어가 효과적 | **양호** |
| EDA < VBT (수익 더 낮음) | 비용/slippage 현실적 반영 | **정상** |
| 부호 불일치 | EDA 실행 로직 결함 | **치명적** |
| 거래 0건 (EDA) | 시그널 전달 문제 | **치명적** |
| MDD 급증 (EDA > VBT × 2) | SL/TS 로직 결함 | **경고** |

---

## Step 5: 라이브 준비 상태 점검 (Code-Level)

VBT-EDA parity 외에, **EDA 코드가 라이브 환경에서 안전하게 작동하는지** 7항목을 점검한다.

### L1: EventBus Flush 패턴

**검증**: DataFeed에서 `await bus.flush()` 호출이 TF bar 완성 시마다 존재하는가?

```bash
grep -n "flush" src/eda/data_feed.py src/eda/live_data_feed.py
```

- flush 없으면 → 파생 이벤트가 마지막 bar 가격으로만 체결 (치명적)
- **PASS 기준**: DataFeed의 TF bar 발행 직후 `await bus.flush()` 호출 확인

### L2: Executor Bar Handler 순서

**검증**: `executor_bar_handler`가 BAR 이벤트 핸들러 중 **첫 번째**로 등록되는가?

```bash
grep -n "subscribe.*BAR" src/eda/runner.py src/eda/live_runner.py
```

- 순서 잘못되면 → Strategy/PM이 이전 bar의 fill을 모르는 상태에서 실행
- **PASS 기준**: `executor_bar_handler` 등록이 다른 BAR handler보다 앞에 위치

### L3: Deferred Execution 일관성

**검증**: `BacktestExecutor`와 `LiveExecutor`가 동일한 deferred execution 패턴을 따르는가?

```bash
grep -n "pending\|deferred\|fill_pending" src/eda/executors.py
```

- 일반 주문(price=None): 다음 bar open에 체결
- SL/TS 주문(price set): 즉시 체결
- **PASS 기준**: 두 Executor 모두 동일 패턴 구현

### L4: PM Batch Mode 동작

**검증**: 멀티에셋 배치 처리에서 `flush_pending_signals()` 호출이 Runner에 존재하는가?

```bash
grep -n "flush_pending" src/eda/runner.py src/eda/live_runner.py
```

- 멀티에셋에서 flush 누락 → 시그널 버퍼에 남아 미실행
- 단일에셋이면 이 항목은 N/A

### L5: Position Reconciler 연동

**검증**: `LiveRunner`에서 PositionReconciler의 주기적 실행이 구현되어 있는가?

```bash
grep -n "reconcil" src/eda/live_runner.py
```

- reconciliation 없으면 → PM과 거래소 포지션 drift 감지 불가
- **PASS 기준**: `_periodic_reconciliation()` 또는 동등 메커니즘 존재

### L6: Graceful Shutdown

**검증**: LiveRunner에서 SIGTERM/SIGINT 처리 + partial candle flush가 구현되어 있는가?

```bash
grep -n "signal\|SIGTERM\|SIGINT\|shutdown\|cancel" src/eda/live_runner.py
```

- shutdown 미처리 → 미체결 주문 orphan, 포지션 미정리
- **PASS 기준**: signal handler + task cancellation + flush 존재

### L7: Circuit Breaker Close 즉시성

**검증**: CircuitBreaker 청산 주문에 `price`가 설정되어 deferred 방지되는가?

```bash
grep -n "circuit\|close_all\|close.*price" src/eda/oms.py src/eda/portfolio_manager.py
```

- CB close가 price=None → pending되어 다음 bar까지 청산 지연
- **PASS 기준**: CB 청산 시 `pos.last_price`로 price 설정 확인

---

## Step 6: 판정

### 6-1. Parity 판정

| 조건 | 결과 | 판정 |
|------|------|------|
| 수익 부호 일치 | ? | ? |
| 수익률 편차 < 20% | ? | ? |
| 거래 수 비율 0.5x~2.0x (또는 구조적 사유) | ? | ? |

**Parity 종합**: 3개 조건 모두 PASS → **G5 Parity PASS**

### 6-2. Live Readiness 판정

| 항목 | 결과 |
|------|------|
| L1 EventBus Flush | ? |
| L2 Executor Handler 순서 | ? |
| L3 Deferred Execution | ? |
| L4 PM Batch Mode | ? |
| L5 Position Reconciler | ? |
| L6 Graceful Shutdown | ? |
| L7 Circuit Breaker | ? |

**Live Readiness 종합**: 전 항목 PASS → **Live Ready**

### 6-3. Gate 5 최종 판정

| 판정 | 조건 |
|------|------|
| **PASS** | Parity PASS **AND** Live Readiness 전 항목 PASS |
| **CONDITIONAL PASS** | Parity PASS **AND** Live Readiness 1~2개 WARNING (non-critical) |
| **FAIL** | Parity FAIL **OR** Live Readiness critical 항목 FAIL |

> **CONDITIONAL PASS**: 스코어카드에 WARNING 항목을 기록하고, G6 Paper Trading 전에 수정 권고.

---

## Step 7: 문서 갱신

### 7-1. 스코어카드 갱신

스코어카드에 **Gate 5 상세** 섹션을 추가한다 (CTREND ctrend.md:152-172 패턴):

```markdown
### Gate 5 상세 (EDA Parity, {best_asset} {best_tf}, {fast_mode_label})

| 지표 | VBT | EDA | 편차 | 기준 | 판정 |
|------|-----|-----|------|------|------|
| Sharpe | X.XX | X.XX | +XX.X% | 수익 부호 일치 | **PASS** |
| CAGR | +XX.X% | +XX.X% | +XX.X% | 부호 일치 | **PASS** |
| MDD | XX.X% | XX.X% | XX.X% | — | 양호/주의 |
| Trades | N | N | -XX.X% | — | 주의/양호 |
| Win Rate | XX.X% | XX.X% | — | — | — |
| Profit Factor | X.XX | X.XX | — | — | — |

**Gate 5 판정**: **PASS/FAIL** ({사유})

**분석**:

- **{괴리 원인 1}**: ...
- **{괴리 원인 2}**: ...
- **실행 모드**: {fast mode / standard mode} ({이유})

**Live Readiness**: {7/7 PASS} — L1~L7 전 항목 통과

**참고**: {추가 관찰 사항}
```

### 7-2. Gate 진행 현황 갱신

```
G5 EDA검증   [PASS/FAIL] Sharpe X.XX, CAGR +XX.X%, 수익 부호 일치/불일치
```

### 7-3. 의사결정 기록 추가

```markdown
| {날짜} | G5 | PASS/FAIL | EDA Sharpe X.XX vs VBT X.XX. 수익 부호 일치. Trades N (PM threshold 필터링). Live Ready 7/7 |
```

### 7-4. Dashboard 갱신 (`docs/strategy/dashboard.md`)

**PASS 시**:

1. "검증중 전략" 테이블의 Gate 컬럼에 `G5` → `P` 추가
2. 비고에 "G5 EDA Parity PASS" 추가
3. "활성 전략" 섹션으로 이동
4. `> G5 EDA Parity PASS: ...` 설명 추가

**FAIL 시**:

1. "검증중 전략" 테이블의 `G5` → `F` 추가
2. 비고에 FAIL 사유 기록
3. **스코어카드 이동하지 않음** (G5 FAIL은 코드 수정 후 재시도 가능)

---

## Step 8: CTREND 비교

### CTREND G5 결과 (참조점)

| 지표 | VBT | EDA | 편차 |
|------|-----|-----|------|
| Sharpe | 2.05 | 2.82 | +37.6% |
| CAGR | +97.8% | +173.8% | +77.7% |
| MDD | 27.7% | 19.8% | -28.5% |
| Trades | 288 | 72 | -75.0% |

**CTREND 교훈**:

1. EDA가 VBT보다 높은 Sharpe/CAGR → PM/RM 방어가 비용 절감 효과
2. 거래 수 75% 감소 → rebalance_threshold 10%가 미세 조정 필터링
3. MDD 개선 → trailing stop (ATR 3.0x) 효과
4. fast mode 사용 → forward_return edge effect 해결

### 비교 포인트

현재 전략과 CTREND를 비교하여 다음을 확인:

| 비교 항목 | CTREND | 현재 전략 | 해석 |
|----------|--------|----------|------|
| Sharpe 편차 방향 | EDA > VBT | ? | EDA가 높으면 PM/RM 효과 |
| 거래 수 비율 | 0.25x | ? | PM threshold 영향도 |
| MDD 개선 여부 | EDA < VBT | ? | SL/TS 효과 |
| 실행 모드 | fast | ? | 전략 특성 반영 |

---

## 리포트 출력 형식

모든 검증 완료 후 아래 형식으로 리포트를 출력한다:

```
============================================================
  GATE 5: EDA PARITY VERIFICATION REPORT
  전략: {display_name} ({registry_key})
  실행일: {YYYY-MM-DD}
  검증 기간: {start} ~ {end} ({period})
  심볼: {best_asset} ({best_tf})
  실행 모드: {fast/standard}
============================================================

  PARITY COMPARISON
  ┌──────────────┬─────────┬─────────┬──────────┬──────────┐
  │ 지표         │   VBT   │   EDA   │   편차   │   판정   │
  ├──────────────┼─────────┼─────────┼──────────┼──────────┤
  │ Sharpe       │  X.XX   │  X.XX   │ +XX.X%   │   —      │
  │ CAGR         │ +XX.X%  │ +XX.X%  │ +XX.X%p  │  PASS    │
  │ Total Return │ +XX.X%  │ +XX.X%  │  XX.X%   │  PASS    │
  │ MDD          │  XX.X%  │  XX.X%  │  XX.X%p  │   —      │
  │ Trades       │   N     │   N     │  X.Xx    │  PASS    │
  │ Win Rate     │  XX.X%  │  XX.X%  │  XX.X%p  │   —      │
  │ Profit Factor│  X.XX   │  X.XX   │  XX.X%   │   —      │
  └──────────────┴─────────┴─────────┴──────────┴──────────┘

  수익 부호: [일치 / 불일치]
  수익률 편차: XX.X% [< 20% PASS / >= 20% FAIL]
  거래 수 비율: X.Xx [0.5x~2.0x PASS / 구조적 사유 PASS / FAIL]

  Parity 판정: [PASS / FAIL]

------------------------------------------------------------
  DISCREPANCY ANALYSIS
------------------------------------------------------------

  1. {괴리 원인 1}: {설명}
  2. {괴리 원인 2}: {설명}

------------------------------------------------------------
  LIVE READINESS CHECK (7/7)
------------------------------------------------------------

  [L1] EventBus Flush       : [PASS / FAIL]
  [L2] Executor Handler 순서 : [PASS / FAIL]
  [L3] Deferred Execution   : [PASS / FAIL]
  [L4] PM Batch Mode        : [PASS / N/A]
  [L5] Position Reconciler  : [PASS / FAIL]
  [L6] Graceful Shutdown    : [PASS / FAIL]
  [L7] Circuit Breaker      : [PASS / FAIL]

  Live Readiness: [PASS / WARNING / FAIL]

------------------------------------------------------------
  CTREND COMPARISON
------------------------------------------------------------

  │ 항목           │ CTREND  │ 현재 전략 │ 해석              │
  │ Sharpe 편차    │ +37.6%  │  XX.X%    │ {해석}            │
  │ 거래 수 비율   │ 0.25x   │  X.Xx     │ {해석}            │
  │ MDD 변화       │ -28.5%  │  XX.X%    │ {해석}            │

============================================================
  GATE 5 SUMMARY
  Parity: [PASS / FAIL]
  Live Readiness: [PASS / WARNING / FAIL]
  최종 판정: [PASS / CONDITIONAL PASS / FAIL]
  다음 단계: [G6 Paper Trading / 코드 수정 후 G5 재시도 / 폐기]
  스코어카드: docs/scorecard/{strategy_name}.md (갱신 완료)
  대시보드:   docs/strategy/dashboard.md (갱신 완료)
============================================================
```

---

## 참조 문서

- [references/parity-criteria.md](references/parity-criteria.md) — Parity 정량 기준 + 괴리 원인 카탈로그
- [references/live-readiness-checklist.md](references/live-readiness-checklist.md) — 라이브 준비 7항목 상세 검증 패턴
- [docs/scorecard/ctrend.md](../../../docs/scorecard/ctrend.md) — CTREND 스코어카드 (G5 선례)
- [docs/strategy/dashboard.md](../../../docs/strategy/dashboard.md) — 전략 상황판
- [docs/strategy/evaluation-standard.md](../../../docs/strategy/evaluation-standard.md) — 전략 평가 표준
