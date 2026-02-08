# 통합 어댑터 아키텍처 리팩토링 — 작업 컨텍스트

> **Last Updated**: 2026-02-08
> **Plan**: [unified-adapter-architecture-plan.md](unified-adapter-architecture-plan.md)
> **Research**: [unified-adapter-architecture.md](../research/unified-adapter-architecture.md)

---

## 전체 진행 현황

| Phase | 이름 | 상태 | 테스트 |
|-------|------|:----:|:------:|
| **6-A** | Port Protocol 정의 | **DONE** | 401 passed |
| **6-B** | Runner 팩토리 메서드 | **DONE** | 406 passed |
| **6-C** | 통합 Metrics 엔진 | **DONE** | 425 passed |
| 6-D | 통합 CLI | TODO | — |

---

## Phase 6-A: Port Protocol 정의 — DONE

### 완료일: 2026-02-08

### 변경 내역

| 파일 | 변경 유형 | 내용 |
|------|:--------:|------|
| `src/eda/ports.py` | **신규** | `DataFeedPort`, `ExecutorPort` Protocol 정의 (~55 lines) |
| `src/eda/oms.py` | 수정 | `Executor` Protocol 제거 → `ExecutorPort` import + `Executor = ExecutorPort` alias |
| `src/eda/runner.py` | 수정 | `feed` 변수 타입 어노테이션 `DataFeedPort`로 변경 |
| `tests/eda/test_ports.py` | **신규** | Protocol 준수 `isinstance` 테스트 4개 |

### 구현 세부사항

**1. `src/eda/ports.py`**
- `@runtime_checkable` Protocol 2개: `DataFeedPort`, `ExecutorPort`
- `from __future__ import annotations` 사용
- `EventBus`, `OrderRequestEvent`, `FillEvent`는 `TYPE_CHECKING` 블록에서 import
- structural subtyping — 기존 구현체에 변경 불필요

**2. `src/eda/oms.py`**
- 기존 `Executor` Protocol 정의 (34-50행) 제거
- `from src.eda.ports import ExecutorPort` 추가
- `Executor = ExecutorPort` 하위 호환 alias 유지
- `OMS.__init__` 파라미터: `executor: Executor` → `executor: ExecutorPort`
- ruff auto-fix: `FillEvent` import 제거 (`from __future__ import annotations` 하에서 runtime 불필요)

**3. `src/eda/runner.py`**
- `TYPE_CHECKING` 블록에 `from src.eda.ports import DataFeedPort` 추가
- `feed: HistoricalDataFeed | AggregatingDataFeed` → `feed: DataFeedPort`
- `HistoricalDataFeed`, `AggregatingDataFeed` import는 유지 (인스턴스 생성용)

**4. `tests/eda/test_ports.py`**
- `_make_single_data()` 헬퍼: 최소 MarketDataSet (5 bars)
- 4개 테스트:
  - `HistoricalDataFeed` → `DataFeedPort` ✅
  - `AggregatingDataFeed` → `DataFeedPort` ✅
  - `BacktestExecutor` → `ExecutorPort` ✅
  - `ShadowExecutor` → `ExecutorPort` ✅

### 검증 결과

```
ruff check:  0 errors
pyright:     0 errors, 0 warnings
pytest:      401 passed (기존 397 + 신규 4)
```

### 설계 메모

- `@runtime_checkable` Protocol의 `isinstance` 체크는 **메서드/프로퍼티 이름 존재 여부**만 검증 (시그니처 불일치는 감지 불가)
- `Executor = ExecutorPort` alias: 현재 외부에서 `from src.eda.oms import Executor` 사용하는 곳 없음. 방어적으로 유지
- `MetricsPort`는 Phase 6-C에서 정의 예정

---

## Phase 6-B: Runner 팩토리 메서드 — DONE

### 완료일: 2026-02-08

### 변경 내역

| 파일 | 변경 유형 | 내용 |
|------|:--------:|------|
| `src/eda/runner.py` | 수정 | `__init__`에서 feed/executor 생성, `_from_adapters` + 공개 팩토리 3개, `run()` 리팩토링 |
| `src/cli/eda.py` | 수정 | 3개 커맨드 (`run`, `run-agg`, `run-multi`) 팩토리 메서드 호출로 교체 |
| `tests/eda/test_runner_factory.py` | **신규** | 팩토리 메서드 테스트 5개 |

### 구현 세부사항

**1. `src/eda/runner.py` — 핵심 리팩토링**

- `__init__`: `self._data` 제거 → `self._data_timeframe` 으로 대체, feed/executor를 인스턴스에서 생성
- `_from_adapters()` private classmethod: `object.__new__(cls)`로 인스턴스 생성, 어댑터 직접 주입
- 공개 팩토리 3개:
  - `backtest()`: `HistoricalDataFeed` + `BacktestExecutor`
  - `backtest_agg()`: `AggregatingDataFeed` + `BacktestExecutor`
  - `shadow()`: `HistoricalDataFeed` + `ShadowExecutor`
- `run()`: feed/executor 생성 로직 제거 → `self._feed`/`self._executor` 사용
- `BacktestExecutor.on_bar()`: `isinstance(executor, BacktestExecutor)` 체크로 타입 안전하게 처리
- `self._data.timeframe` → `self._data_timeframe` 참조 변경
- import 추가: `ShadowExecutor` (runtime), `ExecutorPort` (TYPE_CHECKING)

**2. `src/cli/eda.py` — 팩토리 메서드 사용**

| 커맨드 | 변경 전 | 변경 후 |
|--------|---------|---------|
| `run` | `EDARunner(strategy, data, config, capital)` | `EDARunner.backtest(strategy, data, config, capital)` |
| `run-agg` | `EDARunner(..., target_timeframe=target_tf)` | `EDARunner.backtest_agg(strategy, data_1m, target_tf, config, capital)` |
| `run-multi` | `EDARunner(..., asset_weights=weights)` | `EDARunner.backtest(strategy, multi_data, config, capital, asset_weights=weights)` |

**3. `tests/eda/test_runner_factory.py` — 신규 테스트**

- `test_backtest_factory_creates_correct_types`: feed=HistoricalDataFeed, executor=BacktestExecutor
- `test_backtest_agg_factory_creates_correct_types`: feed=AggregatingDataFeed, executor=BacktestExecutor
- `test_shadow_factory_creates_correct_types`: feed=HistoricalDataFeed, executor=ShadowExecutor
- `test_original_init_backward_compat`: `EDARunner(strategy, data, config)` 정상 동작
- `test_backtest_factory_run_produces_metrics`: 실제 run() 실행하여 PerformanceMetrics 반환 확인

### 검증 결과

```
ruff check:  0 errors
pyright:     0 errors, 0 warnings
pytest:      406 passed (기존 401 + 신규 5)
```

### 설계 메모

- `__init__` 시그니처 동일 유지 → 기존 18개 테스트 하위 호환 통과
- `isinstance(executor, BacktestExecutor)` 체크: `hasattr` 대신 사용하여 pyright 타입 narrowing 활용
- `_from_adapters()`에 `data_timeframe` 파라미터 추가: `data` 객체 의존성 제거, timeframe 문자열만 전달
- 향후 `LiveDataFeed`, `PaperExecutor` 등 새 어댑터 추가 시 Runner 수정 없이 팩토리만 추가 가능

---

## Phase 6-C: 통합 Metrics 엔진 — DONE

### 완료일: 2026-02-08

### 변경 내역

| 파일 | 변경 유형 | 내용 |
|------|:--------:|------|
| `src/backtest/metrics.py` | 수정 | `TradeStatsResult`, `freq_to_periods_per_year()`, `compute_trade_stats()`, `build_performance_metrics()` 추가. `periods_per_year` 타입 `int → float` 변경 |
| `src/eda/analytics.py` | 수정 | `compute_metrics()` → `build_performance_metrics()` 호출로 교체. 헬퍼 함수 6개 제거 |
| `tests/eda/test_analytics.py` | 수정 | `_freq_to_hours` import → `freq_to_periods_per_year` 변경 |
| `tests/backtest/test_metrics_unified.py` | **신규** | `build_performance_metrics()` 단위 테스트 19개 |

### 구현 세부사항

**1. `src/backtest/metrics.py` — 통합 API 추가**

- `TradeStatsResult(NamedTuple)`: total_trades, winning/losing, win_rate, avg_win/loss, profit_factor
- `freq_to_periods_per_year(freq)`: timeframe 문자열 → 연간 기간 수 ("1D"→365, "4h"→2190, "1h"→8760, "15T"→35040)
- `compute_trade_stats(trades)`: `Sequence[TradeRecord]` → `TradeStatsResult`
- `build_performance_metrics(equity_curve, trades, ...)`: 기존 `calculate_*` 함수들을 조합하여 `PerformanceMetrics` 생성
- 기존 함수들의 `periods_per_year` 파라미터 타입을 `int → float`로 변경 (하위 호환)

**2. `src/eda/analytics.py` — 리팩토링**

- `compute_metrics()` 내부 ~60행 자체 계산 로직 → `build_performance_metrics()` 단일 호출로 교체
- 제거된 헬퍼 함수: `_avg_pnl_pct`, `_profit_factor`, `_annualized_sharpe`, `_max_drawdown`, `_compute_cagr`, `_freq_to_hours`
- 펀딩비 계산 로직은 유지 (`funding_drag_per_period` 파라미터로 전달)
- `numpy` import 제거

**3. `tests/backtest/test_metrics_unified.py` — 신규 테스트**

- `TestFreqToPeriodsPerYear`: 5개 (1D, 4h, 1h, 15T, unknown unit)
- `TestComputeTradeStats`: 5개 (empty, all winners, mixed, profit factor, NamedTuple)
- `TestBuildPerformanceMetrics`: 9개 (monotonic equity, drawdown, trades, empty, funding drag, sortino, skew/kurt, volatility, MDD positive)

### 설계 결정

- **Max drawdown 양수 규약**: `build_performance_metrics()`는 `abs()` 변환하여 양수 반환 (EDA 규약). `calculate_max_drawdown()`은 기존 음수 반환 유지 (하위 호환)
- **risk_free_rate=0.0 기본값**: EDA 기존 동작 유지 (VBT의 0.05와 다름)
- **VBT 미변경**: `PerformanceAnalyzer.analyze()`는 현 Phase에서 변경 없음. VBT는 `vbt_portfolio.stats()`에서 값 추출하는 구조
- **EDA 신규 메트릭**: sortino, calmar, skewness, kurtosis가 이제 채워짐 (기존에는 None)

### 검증 결과

```
ruff check:  0 errors
pyright:     0 errors, 0 warnings
pytest:      425 passed (기존 406 + 신규 19)
parity:      14/14 통과 (VBT vs EDA)
```

### 설계 메모

- `build_performance_metrics()`는 EDA와 (향후) VBT 양쪽에서 호출 가능한 단일 진입점
- `freq_to_periods_per_year()`는 기존 EDA `_freq_to_hours()`와 동일 파싱 로직, 반환값만 다름 (hours → periods/year)
- `compute_trade_stats()`는 `Sequence[TradeRecord]` 기반 — VBT `pd.Series` trade_returns와 독립

---

## Phase 6-D: 통합 CLI — TODO

### 핵심 작업

- `run` 커맨드에 `--mode` 옵션 추가 (backtest, backtest-agg, shadow)
- 기존 `run-agg`, `run-multi` 하위 호환 유지
- 내부적으로 Runner 팩토리 메서드 사용

### 의존성

- Phase 6-B 완료 필요 ✅
