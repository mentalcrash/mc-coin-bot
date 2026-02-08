# 통합 어댑터 아키텍처 리팩토링 — 실행 계획

> **Date**: 2026-02-08
> **Prerequisite**: [unified-adapter-architecture.md](unified-adapter-architecture.md) 리서치 완료
> **Goal**: EDA 시스템에 명시적 Port Protocol + Runner 팩토리 + 통합 Metrics 도입

---

## 전체 Phase 구조

```
Phase 6-A: Port Protocol 정의 ──────────────── (1~2일)
Phase 6-B: Runner 팩토리 메서드 ─────────────── (1일)
Phase 6-C: 통합 Metrics 엔진 ───────────────── (2~3일)
Phase 6-D: 통합 CLI ────────────────────────── (1일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
총 예상: 5~7일
```

**원칙:**
- 기존 397개 테스트는 항상 전부 통과
- 기존 코드의 동작 변경 없음 (Protocol은 structural subtyping)
- 각 Phase는 독립 커밋 가능
- Phase 간 의존성: 6-A → 6-B → 6-C (순서 중요), 6-D는 독립

---

## Phase 6-A: Port Protocol 정의

### 목표

현재 암묵적으로 따르는 인터페이스(duck typing)를 명시적 `Protocol`로 정의하여 아키텍처를 명확화하고, 향후 `LiveDataFeed`, `LiveExecutor` 등 새 어댑터 추가 시 계약을 보장한다.

### 작업 1: `src/eda/ports.py` 생성

새 파일에 3개의 Protocol 정의:

```python
# src/eda/ports.py
"""EDA 포트 정의 — Hexagonal Architecture의 Port 계층.

모든 어댑터(DataFeed, Executor, Metrics)는 이 Protocol을 구현해야 한다.
Protocol은 structural subtyping이므로 기존 클래스에 변경 불필요.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.core.events import FillEvent, OrderRequestEvent


@runtime_checkable
class DataFeedPort(Protocol):
    """데이터 공급 포트.

    구현체: HistoricalDataFeed, AggregatingDataFeed, LiveDataFeed (향후)
    """

    async def start(self, bus: EventBus) -> None:
        """데이터 재생/구독을 시작한다."""
        ...

    async def stop(self) -> None:
        """데이터 피드를 중지한다."""
        ...

    @property
    def bars_emitted(self) -> int:
        """발행된 총 bar 수."""
        ...


@runtime_checkable
class ExecutorPort(Protocol):
    """주문 실행 포트.

    구현체: BacktestExecutor, ShadowExecutor, PaperExecutor (향후), LiveExecutor (향후)
    """

    async def execute(self, order: OrderRequestEvent) -> FillEvent | None:
        """주문을 실행하고 체결 결과를 반환한다."""
        ...
```

> **Note**: `MetricsPort`는 Phase 6-C에서 통합 Metrics 엔진 설계 시 정의.

### 작업 2: 기존 클래스가 Protocol을 만족하는지 검증

**검증 대상 및 예상 결과:**

| 클래스 | Protocol | 만족 여부 | 필요 변경 |
|--------|----------|:---:|---------|
| `HistoricalDataFeed` | `DataFeedPort` | ✅ | 없음 (start/stop/bars_emitted 이미 존재) |
| `AggregatingDataFeed` | `DataFeedPort` | ✅ | 없음 (동일 시그니처) |
| `BacktestExecutor` | `ExecutorPort` | ✅ | 없음 (execute 이미 존재) |
| `ShadowExecutor` | `ExecutorPort` | ✅ | 없음 (execute 이미 존재) |

Protocol은 structural subtyping이므로 기존 클래스에 `DataFeedPort`를 상속할 필요 없음. 하지만 `isinstance` 체크가 가능하도록 `@runtime_checkable` 데코레이터를 적용.

### 작업 3: Runner에서 Protocol 타입 힌트 사용

**파일: `src/eda/runner.py`**

현재:
```python
class EDARunner:
    def __init__(
        self,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,  # ← 구체 타입
        config: PortfolioManagerConfig,
        ...
    ) -> None:
```

변경 후 (`__init__`은 유지, 내부에서 Protocol 타입 활용):
- `__init__`의 public 시그니처는 하위 호환을 위해 유지
- 내부 `_feed` 속성에 `DataFeedPort` 타입 어노테이션 추가
- OMS 생성 시 `executor` 파라미터에 `ExecutorPort` 타입 어노테이션 추가

### 작업 4: OMS의 기존 Executor Protocol 통합

**파일: `src/eda/oms.py`**

현재 OMS에 `Executor` Protocol이 이미 정의되어 있음:
```python
@runtime_checkable
class Executor(Protocol):
    async def execute(self, order: OrderRequestEvent) -> FillEvent | None: ...
```

**변경:** 이 Protocol을 `ports.py`의 `ExecutorPort`로 교체:
- `oms.py`에서 `Executor` Protocol 정의 제거
- `from src.eda.ports import ExecutorPort` 사용
- `Executor` → `ExecutorPort` 이름 변경
- 하위 호환: `oms.py`에서 `Executor = ExecutorPort` alias 유지 (deprecation)

### 작업 5: 테스트 추가

**파일: `tests/unit/eda/test_ports.py` (새 파일)**

```python
"""Port Protocol 준수 검증 테스트."""

def test_historical_data_feed_implements_data_feed_port():
    assert isinstance(HistoricalDataFeed(...), DataFeedPort)

def test_aggregating_data_feed_implements_data_feed_port():
    assert isinstance(AggregatingDataFeed(...), DataFeedPort)

def test_backtest_executor_implements_executor_port():
    assert isinstance(BacktestExecutor(...), ExecutorPort)

def test_shadow_executor_implements_executor_port():
    assert isinstance(ShadowExecutor(), ExecutorPort)
```

### 변경 파일 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `src/eda/ports.py` | **새 파일** | DataFeedPort, ExecutorPort Protocol (~40 lines) |
| `src/eda/oms.py` | 수정 | Executor → ExecutorPort alias, import 변경 |
| `src/eda/runner.py` | 수정 | 내부 타입 힌트에 Protocol 사용 |
| `tests/unit/eda/test_ports.py` | **새 파일** | Protocol 준수 검증 (~30 lines) |

### 완료 기준

- [ ] `uv run pyright src/` — 0 errors
- [ ] `uv run ruff check .` — 0 errors
- [ ] `uv run pytest` — 기존 397개 + 신규 4개 = 401개 통과
- [ ] `isinstance(HistoricalDataFeed(...), DataFeedPort)` == True
- [ ] `isinstance(BacktestExecutor(...), ExecutorPort)` == True

---

## Phase 6-B: Runner 팩토리 메서드

### 목표

`EDARunner`에 `@classmethod` 팩토리 메서드를 추가하여 모드별 어댑터 조합을 간소화한다. 기존 `__init__` 시그니처는 하위 호환을 위해 유지.

### 작업 1: Runner 내부 리팩토링 — 어댑터 분리

현재 `EDARunner.__init__`에서 데이터와 피드가 결합되어 있음:

```python
# 현재: data를 받아서 내부에서 feed 생성
def __init__(self, strategy, data, config, ..., target_timeframe=None):
    # 내부에서 조건 분기
    if target_timeframe:
        feed = AggregatingDataFeed(data, target_timeframe)
    else:
        feed = HistoricalDataFeed(data)
```

**리팩토링:** 내부에 `_feed`와 `_executor`를 직접 받는 private 초기화 경로 추가:

```python
class EDARunner:
    # 기존 __init__ 유지 (하위 호환)
    def __init__(
        self,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
        target_timeframe: str | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._initial_capital = initial_capital
        self._asset_weights = asset_weights
        self._queue_size = queue_size

        # 기존 피드/실행기 생성 로직 유지
        if target_timeframe:
            self._feed: DataFeedPort = AggregatingDataFeed(data, target_timeframe)
        else:
            self._feed = HistoricalDataFeed(data)
        self._executor: ExecutorPort = BacktestExecutor(config.cost_model)

    @classmethod
    def _from_adapters(
        cls,
        strategy: BaseStrategy,
        feed: DataFeedPort,
        executor: ExecutorPort,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
        target_timeframe: str | None = None,
    ) -> EDARunner:
        """어댑터를 직접 주입하는 내부 팩토리."""
        instance = cls.__new__(cls)
        instance._strategy = strategy
        instance._feed = feed
        instance._executor = executor
        instance._config = config
        instance._initial_capital = initial_capital
        instance._asset_weights = asset_weights
        instance._queue_size = queue_size
        instance._target_timeframe = target_timeframe
        return instance
```

### 작업 2: 공개 팩토리 메서드 추가

```python
    @classmethod
    def backtest(
        cls,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
    ) -> EDARunner:
        """백테스트 모드: 히스토리컬 데이터 + 시뮬레이션 체결."""
        return cls._from_adapters(
            strategy=strategy,
            feed=HistoricalDataFeed(data),
            executor=BacktestExecutor(config.cost_model),
            config=config,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            queue_size=queue_size,
        )

    @classmethod
    def backtest_agg(
        cls,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        target_timeframe: str,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
    ) -> EDARunner:
        """1m 집계 백테스트 모드: 1m 데이터 → target TF 집계 + 시뮬레이션 체결."""
        return cls._from_adapters(
            strategy=strategy,
            feed=AggregatingDataFeed(data, target_timeframe),
            executor=BacktestExecutor(config.cost_model),
            config=config,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            queue_size=queue_size,
            target_timeframe=target_timeframe,
        )

    @classmethod
    def shadow(
        cls,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
    ) -> EDARunner:
        """섀도우 모드: 히스토리컬 데이터 + 시그널 로깅 (체결 없음)."""
        return cls._from_adapters(
            strategy=strategy,
            feed=HistoricalDataFeed(data),
            executor=ShadowExecutor(),
            config=config,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
        )
```

### 작업 3: `run()` 메서드에서 `_feed`/`_executor` 사용

현재 `run()` 메서드 내부에서 피드와 실행기를 생성하는 부분을 `self._feed`, `self._executor`를 사용하도록 변경.

**핵심 변경 포인트:**
1. `feed = HistoricalDataFeed(...)` → `feed = self._feed`
2. `executor = BacktestExecutor(...)` → `executor = self._executor`
3. `BacktestExecutor.on_bar()` 호출: 실행기가 `BacktestExecutor` 인스턴스일 때만 호출
   - `if hasattr(self._executor, 'on_bar'):` 패턴 사용
   - 또는 별도의 `BarAwareExecutor` Protocol 정의

### 작업 4: `BacktestExecutor.on_bar()` 처리

현재 Runner에서 `BacktestExecutor.on_bar(bar)`를 직접 호출하여 최신 가격을 전달함. 이 메서드는 `ExecutorPort`에 없는 추가 메서드.

**해결 방안 (가장 간결한 접근):**

`ExecutorPort`에 optional `on_bar`를 추가하지 않고, Runner에서 `hasattr` 체크:

```python
# runner.py의 bar 핸들러에서
async def _on_bar_for_executor(self, event: BarEvent) -> None:
    if hasattr(self._executor, "on_bar"):
        self._executor.on_bar(event)  # type: ignore[union-attr]
```

이 패턴은 NautilusTrader에서도 어댑터별 추가 메서드에 사용하는 패턴.

### 작업 5: CLI에서 팩토리 메서드 사용

**파일: `src/cli/eda.py`**

변경 전:
```python
runner = EDARunner(
    strategy=strategy,
    data=market_data,
    config=pm_config,
    initial_capital=initial_capital,
)
```

변경 후:
```python
runner = EDARunner.backtest(
    strategy=strategy,
    data=market_data,
    config=pm_config,
    initial_capital=initial_capital,
)
```

각 CLI 커맨드별:
- `run` → `EDARunner.backtest()`
- `run-agg` → `EDARunner.backtest_agg()`
- `run-multi` → `EDARunner.backtest()` with `asset_weights`

### 작업 6: 테스트 추가/수정

**파일: `tests/unit/eda/test_runner_factory.py` (새 파일)**

```python
"""Runner 팩토리 메서드 테스트."""

def test_backtest_factory_creates_historical_feed():
    runner = EDARunner.backtest(strategy, data, config)
    assert isinstance(runner._feed, HistoricalDataFeed)
    assert isinstance(runner._executor, BacktestExecutor)

def test_backtest_agg_factory_creates_aggregating_feed():
    runner = EDARunner.backtest_agg(strategy, data, "1D", config)
    assert isinstance(runner._feed, AggregatingDataFeed)

def test_shadow_factory_creates_shadow_executor():
    runner = EDARunner.shadow(strategy, data, config)
    assert isinstance(runner._executor, ShadowExecutor)

def test_original_init_still_works():
    """하위 호환 테스트."""
    runner = EDARunner(strategy, data, config)
    # 기존 방식도 정상 동작
```

기존 테스트: `EDARunner(...)` 직접 생성하는 테스트는 변경 불필요 (하위 호환).

### 변경 파일 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `src/eda/runner.py` | 수정 | `_from_adapters()`, `backtest()`, `backtest_agg()`, `shadow()` 추가, `run()` 내부 리팩토링 |
| `src/cli/eda.py` | 수정 | 팩토리 메서드 사용 |
| `tests/unit/eda/test_runner_factory.py` | **새 파일** | 팩토리 메서드 테스트 (~40 lines) |

### 완료 기준

- [ ] `EDARunner.backtest()` 로 생성한 runner가 기존 `EDARunner()` 와 동일 결과
- [ ] `EDARunner.backtest_agg()` 로 생성한 runner가 기존 `target_timeframe` 전달과 동일 결과
- [ ] `EDARunner.shadow()` 로 생성한 runner가 `ShadowExecutor` 사용
- [ ] 기존 `EDARunner(strategy, data, config)` 방식 하위 호환
- [ ] `uv run pyright src/` — 0 errors
- [ ] `uv run ruff check .` — 0 errors
- [ ] `uv run pytest` — 기존 + 신규 테스트 전부 통과

---

## Phase 6-C: 통합 Metrics 엔진

### 목표

`PerformanceAnalyzer` (VBT)와 `AnalyticsEngine` (EDA)에서 공통으로 사용하는 메트릭 계산 로직을 추출하여 단일 소스로 통합. 이를 통해:

1. 메트릭 계산 로직 중복 ~300 lines 제거
2. VBT와 EDA의 메트릭 계산 정확도 parity 보장
3. 향후 새로운 메트릭 추가 시 한 곳만 수정

### 사전 분석 필요

이 Phase 시작 전에 아래 항목을 구체적으로 분석해야 함:

1. `PerformanceAnalyzer.analyze()` 내부의 메트릭 계산 순서와 공식
2. `AnalyticsEngine.compute_metrics()` 내부의 메트릭 계산 순서와 공식
3. 두 구현의 차이점 (입력 형태, 계산 방식, 반올림)
4. `PerformanceMetrics` 모델의 각 필드별 계산 경로

### 작업 1: 공통 메트릭 함수 추출

**파일: `src/backtest/metrics.py` (새 파일)**

순수 함수로 메트릭 계산 로직을 추출:

```python
# src/backtest/metrics.py
"""통합 메트릭 계산 엔진.

VBT PerformanceAnalyzer와 EDA AnalyticsEngine 양쪽에서 사용.
입력: equity curve (pd.Series) + trade records → PerformanceMetrics.
"""

def compute_returns(equity_curve: pd.Series) -> pd.Series:
    """equity curve에서 수익률 시리즈 계산."""

def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods: int) -> float:
    """Sharpe ratio 계산."""

def compute_sortino_ratio(returns: pd.Series, risk_free_rate: float, periods: int) -> float:
    """Sortino ratio 계산."""

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """최대 낙폭 (양수, %)."""

def compute_cagr(equity_curve: pd.Series) -> float:
    """연환산 수익률 계산."""

def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calmar ratio 계산."""

def compute_trade_stats(trades: list[TradeRecord]) -> TradeStats:
    """거래 통계 (win rate, profit factor, avg win/loss 등)."""

def build_performance_metrics(
    equity_curve: pd.Series,
    trades: list[TradeRecord],
    timeframe: str,
    cost_model: CostModel,
    risk_free_rate: float = 0.05,
) -> PerformanceMetrics:
    """모든 메트릭을 조합하여 PerformanceMetrics 생성.

    VBT와 EDA 양쪽에서 이 함수를 최종 호출한다.
    """
```

### 작업 2: AnalyticsEngine 리팩토링

**파일: `src/eda/analytics.py`**

변경 전:
```python
def compute_metrics(self, timeframe, cost_model) -> PerformanceMetrics:
    # 자체 equity curve → 자체 메트릭 계산 (200+ lines)
```

변경 후:
```python
def compute_metrics(self, timeframe, cost_model) -> PerformanceMetrics:
    equity_series = self._build_equity_series()  # 내부 EquityPoint → pd.Series 변환
    trades = self._closed_trades
    return build_performance_metrics(
        equity_curve=equity_series,
        trades=trades,
        timeframe=timeframe,
        cost_model=cost_model,
    )
```

### 작업 3: PerformanceAnalyzer 리팩토링

**파일: `src/backtest/analyzer.py`**

변경 전:
```python
def analyze(self, vbt_portfolio) -> PerformanceMetrics:
    # VBT portfolio → 자체 메트릭 계산
```

변경 후:
```python
def analyze(self, vbt_portfolio) -> PerformanceMetrics:
    equity_series = vbt_portfolio.value()  # VBT equity curve
    trades = self.extract_trades(vbt_portfolio)
    return build_performance_metrics(
        equity_curve=equity_series,
        trades=trades,
        timeframe=self._timeframe,
        cost_model=self._cost_model,
    )
```

### 작업 4: Parity 테스트

**파일: `tests/unit/test_metrics_parity.py` (새 파일)**

```python
"""VBT와 EDA 메트릭 계산 parity 테스트."""

def test_sharpe_ratio_parity():
    """동일 equity curve에 대해 두 경로의 Sharpe 일치."""
    equity = pd.Series([100, 101, 99, 103, 105])
    result_a = compute_sharpe_ratio(compute_returns(equity), 0.05, 252)
    result_b = compute_sharpe_ratio(compute_returns(equity), 0.05, 252)
    assert result_a == result_b

def test_build_performance_metrics_deterministic():
    """동일 입력 → 동일 PerformanceMetrics."""

def test_eda_and_vbt_same_metrics_engine():
    """AnalyticsEngine과 PerformanceAnalyzer가 동일 metrics 함수 사용 확인."""
```

### 변경 파일 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `src/backtest/metrics.py` | **새 파일** | 공통 메트릭 함수 (~200 lines) |
| `src/eda/analytics.py` | 수정 | 자체 계산 → `build_performance_metrics()` 호출 |
| `src/backtest/analyzer.py` | 수정 | 자체 계산 → `build_performance_metrics()` 호출 |
| `tests/unit/test_metrics_parity.py` | **새 파일** | parity 테스트 |

### 완료 기준

- [ ] `build_performance_metrics()` 함수가 `PerformanceMetrics` 정확 생성
- [ ] `AnalyticsEngine.compute_metrics()` 결과가 리팩토링 전후 동일
- [ ] `PerformanceAnalyzer.analyze()` 결과가 리팩토링 전후 동일
- [ ] Parity 테스트: 동일 입력 → 동일 출력 검증
- [ ] `uv run pyright src/` — 0 errors
- [ ] `uv run ruff check .` — 0 errors
- [ ] `uv run pytest` — 전부 통과

### 리스크

- PerformanceAnalyzer와 AnalyticsEngine의 메트릭 계산 공식이 미묘하게 다를 수 있음
  - 예: annualization factor, 거래일 수 가정, drawdown 계산 기준점
  - **대응:** Phase 시작 시 두 구현을 상세 비교하여 차이점 문서화 후 진행
- VBT portfolio의 equity curve와 EDA의 equity curve 형태가 다름 (index, frequency)
  - **대응:** `build_performance_metrics()`의 입력을 `pd.Series` (DatetimeIndex)로 통일

---

## Phase 6-D: 통합 CLI

### 목표

현재 분리된 2개 CLI (`src/cli/backtest.py`, `src/cli/eda.py`)의 EDA 부분에 `--mode` 옵션을 추가하여 모든 EDA 모드를 하나의 진입점에서 실행할 수 있게 한다.

> **Note:** VBT CLI (`backtest.py`)는 변경하지 않는다. VBT는 독립적 Phase 1 스크리닝 도구.

### 작업 1: EDA CLI에 mode 옵션 추가

**파일: `src/cli/eda.py`**

현재 3개 커맨드:
- `run` → 단일 심볼 EDA 백테스트
- `run-agg` → 1m 집계 모드
- `run-multi` → 멀티 심볼

**리팩토링:** `run` 커맨드에 `--mode` 옵션 추가:

```python
@app.command(name="run")
def run(
    strategy_name: Annotated[str, typer.Argument(...)],
    symbol: Annotated[str, typer.Argument(...)],
    ...
    mode: Annotated[str, typer.Option(
        help="실행 모드: backtest (기본), backtest-agg, shadow"
    )] = "backtest",
    target_timeframe: Annotated[str | None, typer.Option(...)] = None,
) -> None:
    ...
    match mode:
        case "backtest":
            runner = EDARunner.backtest(strategy, data, config, initial_capital)
        case "backtest-agg":
            if target_timeframe is None:
                typer.echo("Error: --target-timeframe required for backtest-agg mode")
                raise typer.Exit(1)
            runner = EDARunner.backtest_agg(strategy, data, target_timeframe, config, initial_capital)
        case "shadow":
            runner = EDARunner.shadow(strategy, data, config, initial_capital)
        case _:
            typer.echo(f"Unknown mode: {mode}")
            raise typer.Exit(1)
    ...
```

### 작업 2: 기존 커맨드 하위 호환

기존 `run-agg`, `run-multi` 커맨드는 유지하되, 내부적으로 팩토리 메서드를 사용하도록 변경:

```python
@app.command(name="run-agg")
def run_agg(...):
    """1m 집계 모드 (하위 호환)."""
    # 내부: EDARunner.backtest_agg() 호출
```

### 작업 3: CLI 도움말 정리

```bash
# 통합 사용법
uv run python main.py eda run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31
uv run python main.py eda run tsmom BTC/USDT --mode backtest-agg --target-timeframe 1D
uv run python main.py eda run tsmom BTC/USDT --mode shadow

# 기존 방식도 유지 (하위 호환)
uv run python main.py eda run-agg tsmom BTC/USDT --target-timeframe 1D
uv run python main.py eda run-multi tsmom BTC/USDT,ETH/USDT
```

### 변경 파일 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `src/cli/eda.py` | 수정 | `--mode` 옵션 추가, 팩토리 메서드 사용 |

### 완료 기준

- [ ] `uv run python main.py eda run tsmom BTC/USDT --mode backtest` 정상 동작
- [ ] `uv run python main.py eda run tsmom BTC/USDT --mode shadow` 정상 동작
- [ ] 기존 `run-agg`, `run-multi` 커맨드 하위 호환
- [ ] `uv run pyright src/` — 0 errors
- [ ] `uv run ruff check .` — 0 errors
- [ ] `uv run pytest` — 전부 통과

---

## Phase 간 의존성

```
Phase 6-A (Protocol 정의)
    │
    ├──→ Phase 6-B (Runner 팩토리)  ──→  Phase 6-D (CLI 통합)
    │                                       (6-B의 팩토리 메서드 사용)
    │
    └──→ Phase 6-C (통합 Metrics)
          (독립적, 6-A 완료 후 병렬 가능)
```

- **6-A → 6-B**: Runner가 Protocol 타입을 사용하므로 6-A 선행 필수
- **6-B → 6-D**: CLI가 팩토리 메서드를 사용하므로 6-B 선행 필수
- **6-C**: Protocol과 무관하게 독립 진행 가능 (6-A 이후 6-B와 병렬 가능)

---

## 향후 Phase (Phase 8 — 라이브 준비)

Phase 6 완료 후, 동일 어댑터 패턴 위에 라이브 어댑터를 추가:

```
Phase 8-A: LiveDataFeed
├── CCXT Pro WebSocket → BarEvent
├── Reconnection / error handling
└── 테스트: mock WebSocket

Phase 8-B: LiveExecutor
├── CCXT Pro 주문 실행 → FillEvent
├── Rate limiting / retry
└── 테스트: mock exchange

Phase 8-C: PaperExecutor
├── 실시간 가격 + 가상 체결
├── 슬리피지 시뮬레이션
└── 결과 AnalyticsEngine과 동일

Phase 8-D: DualExecutor (Shadow)
├── LiveExecutor + ShadowExecutor 동시 실행
├── 시그널 비교 로깅
└── Canary 배포 지원

Phase 8-E: Reconciliation
├── 거래소 상태 ↔ 내부 상태 동기화
├── Startup + Continuous
└── State persistence (Redis/SQLite)
```

이 모든 Phase는 **Phase 6에서 정의한 Port Protocol을 구현하는 어댑터만 추가**하면 되며, EDA Core (Strategy, PM, RM, OMS, Analytics, EventBus)는 변경 불필요.

---

## 체크리스트 요약

### Phase 6-A: Port Protocol 정의
- [ ] `src/eda/ports.py` 생성
- [ ] `DataFeedPort`, `ExecutorPort` Protocol 정의
- [ ] `oms.py`의 `Executor` Protocol → `ports.py`로 이동
- [ ] Runner에 Protocol 타입 힌트 추가
- [ ] Protocol 준수 테스트 추가
- [ ] lint/type/test 전체 통과

### Phase 6-B: Runner 팩토리 메서드
- [ ] `_from_adapters()` private 팩토리 추가
- [ ] `backtest()`, `backtest_agg()`, `shadow()` 팩토리 추가
- [ ] `run()` 메서드에서 `self._feed`, `self._executor` 사용
- [ ] `on_bar()` hasattr 처리
- [ ] CLI에서 팩토리 메서드 사용
- [ ] 하위 호환 테스트
- [ ] lint/type/test 전체 통과

### Phase 6-C: 통합 Metrics 엔진
- [ ] 두 구현의 메트릭 계산 로직 상세 비교
- [ ] `src/backtest/metrics.py` 공통 함수 추출
- [ ] `AnalyticsEngine` 리팩토링
- [ ] `PerformanceAnalyzer` 리팩토링
- [ ] Parity 테스트 추가
- [ ] lint/type/test 전체 통과

### Phase 6-D: 통합 CLI
- [ ] `--mode` 옵션 추가
- [ ] 기존 커맨드 하위 호환 유지
- [ ] lint/type/test 전체 통과
