# Backtest Engine Architecture

이 문서는 MC Coin Bot의 백테스트 엔진 아키텍처를 설명합니다.
Clean Architecture 원칙에 따라 설계되었으며, EDA, Dry Run, Live 트레이딩으로의 확장을 고려합니다.

## 1. 아키텍처 개요

### 1.1 5-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Interface Layer                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │   CLI   │  │   EDA   │  │  REST   │  │  Cron   │             │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘             │
└───────┼────────────┼────────────┼────────────┼──────────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Application Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ BacktestRequest │  │  DryRunRequest  │  │   LiveRequest   │  │
│  │ BacktestEngine  │  │  DryRunEngine   │  │   LiveEngine    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Domain Layer                             │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐    │
│  │   Strategy   │  │   Portfolio   │  │ PerformanceAnalyzer│    │
│  │  (시그널생성)  │  │ (자금/포지션)  │  │   (성과 분석)       │    │
│  └──────────────┘  └───────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                        │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │ MarketDataService │  │   ExchangeAdapter │                   │
│  │  (데이터 제공)      │  │   (거래소 연동)    │                   │
│  └───────────────────┘  └───────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Stateless Engine** | BacktestEngine은 상태를 가지지 않음. 모든 정보는 Request로 주입 |
| **Separation of Concerns** | 데이터/전략/포트폴리오/분석 각각 독립적인 책임 |
| **Command Pattern** | BacktestRequest가 실행에 필요한 모든 정보를 캡슐화 |
| **Repository Pattern** | MarketDataService가 데이터 접근 추상화 |

---

## 2. 핵심 컴포넌트

### 2.1 MarketDataService (Infrastructure Layer)

데이터 접근을 추상화하는 서비스입니다.

```python
from src.data import MarketDataService, MarketDataRequest

# 요청 생성
request = MarketDataRequest(
    symbol="BTC/USDT",
    timeframe="1D",  # 1m, 1h, 1D 지원
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2025, 12, 31, tzinfo=UTC),
)

# 데이터 로드
service = MarketDataService()
data = service.get(request)

print(data.symbol)      # "BTC/USDT"
print(data.periods)     # 730 (daily candles)
print(data.freq)        # "1D" (VectorBT용)
print(data.ohlcv.head())  # DataFrame
```

**파일 위치:**
- `src/data/market_data.py`: MarketDataRequest, MarketDataSet DTOs
- `src/data/service.py`: MarketDataService

### 2.2 Portfolio (Domain Layer)

초기 자본과 집행 설정을 결합한 도메인 객체입니다.

```python
from src.portfolio import Portfolio

# 기본 포트폴리오
portfolio = Portfolio.create(initial_capital=10000)

# 프리셋 사용
portfolio = Portfolio.conservative(initial_capital=50000)
portfolio = Portfolio.aggressive(initial_capital=10000)
portfolio = Portfolio.paper_trading(initial_capital=100000)

# 커스텀 설정
from src.portfolio import PortfolioManagerConfig

portfolio = Portfolio.create(
    initial_capital=10000,
    config=PortfolioManagerConfig(
        max_leverage_cap=3.0,
        rebalance_threshold=0.03,
    ),
)
```

**파일 위치:**
- `src/portfolio/portfolio.py`: Portfolio 도메인 객체
- `src/portfolio/config.py`: PortfolioManagerConfig

### 2.3 PerformanceAnalyzer (Domain Layer)

성과 분석을 전담하는 컴포넌트입니다.

```python
from src.backtest import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# 성과 지표 추출
metrics = analyzer.analyze(vbt_portfolio)

# 벤치마크 비교
benchmark = analyzer.compare_benchmark(vbt_portfolio, data.ohlcv, "BTC/USDT")

# 거래 기록 추출
trades = analyzer.extract_trades(vbt_portfolio, "BTC/USDT")

# QuantStats용 수익률 시리즈
strat_ret, bench_ret = analyzer.get_returns_series(vbt_portfolio, data.ohlcv, "BTC/USDT")
```

**파일 위치:**
- `src/backtest/analyzer.py`: PerformanceAnalyzer

### 2.4 BacktestRequest (Application Layer)

실행에 필요한 모든 정보를 캡슐화하는 Command 객체입니다.

```python
from src.backtest import BacktestRequest

request = BacktestRequest(
    data=data,              # MarketDataSet
    strategy=strategy,      # BaseStrategy
    portfolio=portfolio,    # Portfolio
    analyzer=analyzer,      # PerformanceAnalyzer (optional)
)
```

**파일 위치:**
- `src/backtest/request.py`: BacktestRequest

### 2.5 BacktestEngine (Application Layer)

Stateless 백테스트 실행자입니다.

```python
from src.backtest import BacktestEngine

engine = BacktestEngine()

# 단순 실행
result = engine.run(request)

# 수익률 시리즈 포함 실행 (QuantStats 리포트용)
result, strat_ret, bench_ret = engine.run_with_returns(request)
```

**파일 위치:**
- `src/backtest/engine.py`: BacktestEngine, run_parameter_sweep

---

## 3. 사용 예시

### 3.1 기본 백테스트

```python
from datetime import UTC, datetime
from decimal import Decimal

from src.backtest import BacktestEngine, BacktestRequest
from src.data import MarketDataService, MarketDataRequest
from src.portfolio import Portfolio
from src.strategy.tsmom import TSMOMStrategy

# 1. 데이터 로드
data = MarketDataService().get(
    MarketDataRequest(
        symbol="BTC/USDT",
        timeframe="1D",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
    )
)

# 2. 요청 생성
request = BacktestRequest(
    data=data,
    strategy=TSMOMStrategy(),
    portfolio=Portfolio.create(initial_capital=Decimal("10000")),
)

# 3. 실행
result = BacktestEngine().run(request)

# 4. 결과 확인
print(f"Total Return: {result.metrics.total_return:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2f}%")
```

### 3.2 파라미터 최적화

```python
from src.backtest import run_parameter_sweep

results = run_parameter_sweep(
    strategy_class=TSMOMStrategy,
    data=data,
    param_grid={
        "lookback": [12, 24, 36, 48],
        "vol_target": [0.10, 0.15, 0.20],
    },
    portfolio=Portfolio.create(initial_capital=Decimal("10000")),
    top_n=10,
)

print(results[["lookback", "vol_target", "sharpe_ratio", "total_return"]])
```

### 3.3 QuantStats 리포트 생성

```python
from src.backtest import PerformanceAnalyzer
from src.backtest.reporter import generate_quantstats_report

request = BacktestRequest(
    data=data,
    strategy=TSMOMStrategy(),
    portfolio=Portfolio.create(initial_capital=Decimal("10000")),
    analyzer=PerformanceAnalyzer(),
)

result, strat_ret, bench_ret = BacktestEngine().run_with_returns(request)

report_path = generate_quantstats_report(
    returns=strat_ret,
    benchmark_returns=bench_ret,
    title="VW-TSMOM Backtest",
)
print(f"Report saved: {report_path}")
```

---

## 4. 확장 가이드

### 4.1 시나리오별 재사용성

| 시나리오 | MarketData | Strategy | Portfolio | Engine |
|----------|------------|----------|-----------|--------|
| **EDA** | MarketDataService | - | - | - |
| **Backtest** | MarketDataService | BaseStrategy | Portfolio | BacktestEngine |
| **Dry Run** | MarketDataService + Streaming | BaseStrategy | Portfolio | DryRunEngine (향후) |
| **Live** | ExchangeAdapter (실시간) | BaseStrategy | Portfolio | LiveEngine (향후) |

### 4.2 Dry Run 확장 예시

```python
# 향후 구현 예정
from src.dryrun import DryRunEngine, DryRunRequest

request = DryRunRequest(
    data_source=RealtimeDataSource(symbol="BTC/USDT"),
    strategy=TSMOMStrategy(),
    portfolio=Portfolio.create(initial_capital=10000),
)

engine = DryRunEngine()
await engine.run(request)
```

### 4.3 새 전략 추가

1. `src/strategy/` 디렉토리에 새 모듈 생성
2. `BaseStrategy` 상속
3. `preprocess()`, `generate_signals()` 구현
4. `BacktestRequest`에 전달

```python
class MyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "MyStrategy"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # 지표 계산
        ...

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        # 시그널 생성
        ...
```

---

## 5. 파일 구조

```
src/
├── backtest/
│   ├── __init__.py          # 모듈 exports
│   ├── analyzer.py          # PerformanceAnalyzer
│   ├── cost_model.py        # CostModel
│   ├── engine.py            # BacktestEngine, run_parameter_sweep
│   ├── metrics.py           # 순수 함수 기반 지표 계산
│   ├── reporter.py          # QuantStats 리포트 생성
│   └── request.py           # BacktestRequest DTO
├── data/
│   ├── __init__.py          # 모듈 exports
│   ├── bronze.py            # BronzeStorage
│   ├── fetcher.py           # DataFetcher
│   ├── market_data.py       # MarketDataRequest, MarketDataSet
│   ├── service.py           # MarketDataService
│   └── silver.py            # SilverProcessor
├── portfolio/
│   ├── __init__.py          # 모듈 exports
│   ├── config.py            # PortfolioManagerConfig
│   └── portfolio.py         # Portfolio 도메인 객체
└── strategy/
    ├── base.py              # BaseStrategy ABC
    ├── types.py             # StrategySignals, Direction
    └── tsmom/               # VW-TSMOM 전략
```

---

## 6. 설계 결정 근거

### 6.1 왜 Stateless Engine인가?

- **테스트 용이성**: 모든 의존성이 Request로 주입되어 Mock 가능
- **병렬 실행**: 상태가 없어 멀티프로세싱에 안전
- **재현성**: 동일한 Request는 동일한 결과 보장

### 6.2 왜 Portfolio와 Strategy를 분리하는가?

- **Strategy 책임**: 시그널 생성 (순수 함수적)
- **Portfolio 책임**: 자금 관리, 집행 규칙 (상태 관리)
- **재사용성**: 동일 Strategy를 다른 Portfolio 설정에서 사용 가능

### 6.3 왜 MarketDataService가 필요한가?

- **일관된 인터페이스**: CLI, EDA, 백테스트 모두 동일한 방식으로 데이터 접근
- **리샘플링 중앙화**: 타임프레임 변환 로직 중복 제거
- **캐싱 전략 적용 가능**: 향후 메모리/디스크 캐싱 추가 용이

---

## 7. CLI 사용법

```bash
# 기본 백테스트
uv run python -m src.cli.backtest run BTC/USDT --year 2024 --year 2025

# 리포트 생성
uv run python -m src.cli.backtest run BTC/USDT -y 2024 --report

# 초기 자본 변경
uv run python -m src.cli.backtest run BTC/USDT -c 50000 --report

# 파라미터 최적화
uv run python -m src.cli.backtest optimize BTC/USDT -y 2024 -y 2025

# 정보 출력
uv run python -m src.cli.backtest info
```
