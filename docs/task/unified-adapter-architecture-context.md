# 통합 어댑터 아키텍처 리팩토링 — 작업 컨텍스트

> **Last Updated**: 2026-02-08
> **Plan**: [unified-adapter-architecture-plan.md](unified-adapter-architecture-plan.md)
> **Research**: [unified-adapter-architecture.md](../research/unified-adapter-architecture.md)

---

## 전체 진행 현황

| Phase | 이름 | 상태 | 테스트 |
|-------|------|:----:|:------:|
| **6-A** | Port Protocol 정의 | **DONE** | 401 passed |
| 6-B | Runner 팩토리 메서드 | TODO | — |
| 6-C | 통합 Metrics 엔진 | TODO | — |
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

## Phase 6-B: Runner 팩토리 메서드 — TODO

### 핵심 작업

- `EDARunner`에 `@classmethod` 팩토리: `backtest()`, `backtest_agg()`, `shadow()`
- `_from_adapters()` private 팩토리로 어댑터 직접 주입
- `run()` 내부에서 `self._feed`, `self._executor` 사용
- `BacktestExecutor.on_bar()` 처리: `hasattr` 체크 패턴
- CLI에서 팩토리 메서드 사용
- 기존 `__init__` 하위 호환 유지

### 의존성

- Phase 6-A 완료 필요 ✅

---

## Phase 6-C: 통합 Metrics 엔진 — TODO

### 핵심 작업

- `PerformanceAnalyzer` (VBT)와 `AnalyticsEngine` (EDA) 메트릭 계산 로직 비교
- `src/backtest/metrics.py` 공통 순수 함수 추출
- 양쪽에서 `build_performance_metrics()` 호출하도록 리팩토링
- Parity 테스트

### 리스크

- 두 구현의 annualization factor, 거래일 수 가정, drawdown 기준점이 다를 수 있음
- VBT equity curve와 EDA equity curve의 형태(index, frequency) 차이

### 의존성

- Phase 6-A 완료 필요 ✅
- Phase 6-B와 병렬 진행 가능

---

## Phase 6-D: 통합 CLI — TODO

### 핵심 작업

- `run` 커맨드에 `--mode` 옵션 추가 (backtest, backtest-agg, shadow)
- 기존 `run-agg`, `run-multi` 하위 호환 유지
- 내부적으로 Runner 팩토리 메서드 사용

### 의존성

- Phase 6-B 완료 필요
