# Backtest Engine Architecture

MC Coin Bot의 백테스트 엔진 아키텍처 문서.
Clean Architecture 원칙에 따라 설계되었으며, VBT(벡터화) + EDA(이벤트 기반) 이중 엔진 체계를 운용합니다.

---

## 1. 아키텍처 개요

### 1.1 6-Layer Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           Interface Layer                               │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   CLI   │  │ Pipeline │  │   EDA    │  │ Orchest. │  │   Live   │  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼────────────┼────────────┼────────────┼────────────┼────────────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Application Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │BacktestEngine│  │  EDARunner   │  │  OrchestratedRunner          │  │
│  │ (VBT 벡터화) │  │ (이벤트기반) │  │  (멀티전략 이벤트기반)       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬───────────────┘  │
│  ┌──────┴───────┐  ┌──────┴───────────────────────────────────────┐    │
│  │  Optimizer   │  │     Validation / IC / StressTest / Advisor   │    │
│  │  (Optuna)    │  │                                              │    │
│  └──────────────┘  └─────────────────────────────────────────────-┘    │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Domain Layer                                   │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐          │
│  │  BaseStrategy │  │   Portfolio   │  │ PerformanceAnalyzer│          │
│  │  (시그널생성) │  │ (자금/포지션) │  │   (성과 분석)      │          │
│  └──────────────┘  └───────────────┘  └────────────────────┘          │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐          │
│  │  CostModel   │  │  PM Config    │  │  SmartExecutor     │          │
│  │  (비용 모델) │  │ (리스크 규칙) │  │  (Limit 우선 실행) │          │
│  └──────────────┘  └───────────────┘  └────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Infrastructure Layer                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐  │
│  │ MarketDataService │  │   ExchangeAdapter │  │ CandleAggregator  │  │
│  │  (데이터 제공)    │  │   (거래소 연동)   │  │  (TF 집계)        │  │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 이중 엔진 체계

| 엔진 | 용도 | 특성 |
|------|------|------|
| **BacktestEngine (VBT)** | 전략 발굴, 파라미터 최적화, 빠른 검증 | 벡터화, 고속, Numba JIT |
| **EDARunner** | 라이브 시뮬레이션, EDA Parity 검증 | 이벤트 기반, 1m bar-by-bar, 실거래 동일 경로 |
| **OrchestratedRunner** | 멀티전략 포트폴리오 백테스트 | Pod 기반, 시그널 넷팅, Fill 귀속 |

### 1.3 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Stateless Engine** | BacktestEngine은 상태를 가지지 않음. 모든 정보는 Request로 주입 |
| **Separation of Concerns** | 데이터/전략/포트폴리오/분석 각각 독립적인 책임 |
| **Command Pattern** | BacktestRequest가 실행에 필요한 모든 정보를 캡슐화 |
| **Repository Pattern** | MarketDataService가 데이터 접근 추상화 |
| **Dual Engine Parity** | VBT↔EDA 결과 정합성 유지 (Sharpe 편차 < 2%) |

---

## 2. 핵심 컴포넌트

### 2.1 MarketDataService (Infrastructure Layer)

데이터 접근을 추상화하는 서비스입니다.

```python
from src.data import MarketDataService, MarketDataRequest

# 단일 심볼
data = MarketDataService().get(
    MarketDataRequest(
        symbol="BTC/USDT",
        timeframe="12H",  # 1m, 1h, 4h, 8h, 12H, 1D 지원
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2025, 12, 31, tzinfo=UTC),
    )
)

# 멀티 심볼
multi_data = MarketDataService().get_multi(
    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"],
    timeframe="12H",
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2025, 12, 31, tzinfo=UTC),
)
```

**데이터 모델:**

```python
@dataclass(frozen=True)
class MarketDataSet:
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    ohlcv: pd.DataFrame

    @property
    def periods(self) -> int       # 캔들 수
    @property
    def freq(self) -> str          # VBT 주파수 ("12h", "1d" 등)
    @property
    def duration_days(self) -> int

@dataclass
class MultiSymbolData:
    symbols: list[str]
    timeframe: str
    start: datetime
    end: datetime
    ohlcv: dict[str, pd.DataFrame]

    @property
    def n_assets(self) -> int
    @property
    def close_matrix(self) -> pd.DataFrame   # (periods × n_assets)
    @property
    def periods(self) -> int

    def get_single(self, symbol) -> MarketDataSet
    def slice_time(self, start, end) -> MultiSymbolData
    def slice_iloc(self, start_idx, end_idx) -> MultiSymbolData
```

**파일 위치:**

- `src/data/market_data.py`: MarketDataRequest, MarketDataSet, MultiSymbolData
- `src/data/service.py`: MarketDataService

### 2.2 BaseStrategy (Domain Layer)

모든 전략이 상속하는 추상 기반 클래스입니다.

```python
class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def required_columns(self) -> list[str]: ...

    @property
    def required_enrichments(self) -> list[str]:
        return []  # 대안데이터 의존 시 오버라이드

    @property
    def config(self) -> BaseModel | None:
        return None

    @property
    def params(self) -> dict[str, Any]: ...

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> StrategySignals: ...

    def run(self, df) -> tuple[pd.DataFrame, StrategySignals]:
        """preprocess + generate_signals 일괄 실행"""

    def run_incremental(self, df) -> tuple[pd.DataFrame, StrategySignals]:
        """증분 실행 (라이브용)"""

    @classmethod
    def from_params(cls, **params) -> BaseStrategy:
        """파라미터 딕셔너리로 전략 생성 (sweep/orchestrator용)"""

    @classmethod
    def recommended_config(cls) -> dict[str, Any]: ...

    def get_startup_info(self) -> dict[str, str]: ...
```

**전략 등록 패턴:**

```python
from src.strategy.registry import register

@register("tri-channel-trend")
class TriChannelTrendStrategy(BaseStrategy):
    @classmethod
    def from_params(cls, **params) -> TriChannelTrendStrategy:
        config = TriChannelConfig(**params)
        return cls(config)
```

**파일 위치:**

- `src/strategy/base.py`: BaseStrategy ABC
- `src/strategy/types.py`: StrategySignals, Direction
- `src/strategy/registry.py`: `@register` 데코레이터

### 2.3 Portfolio & CostModel (Domain Layer)

#### Portfolio

초기 자본과 집행 설정을 결합한 도메인 객체입니다.

```python
from src.portfolio import Portfolio, PortfolioManagerConfig

# 기본 포트폴리오
portfolio = Portfolio.create(initial_capital=Decimal("10000"))

# 프리셋
Portfolio.conservative(initial_capital=Decimal("50000"))
Portfolio.aggressive(initial_capital=Decimal("10000"))
Portfolio.paper_trading(initial_capital=Decimal("100000"))
Portfolio.binance_vip(initial_capital=Decimal("10000"), vip_level=1)

# 커스텀 설정
portfolio = Portfolio.create(
    initial_capital=Decimal("10000"),
    config=PortfolioManagerConfig(
        max_leverage_cap=3.0,
        system_stop_loss=0.10,
        use_trailing_stop=True,
        trailing_stop_atr_multiplier=3.0,
        use_intrabar_trailing_stop=False,  # VBT-EDA parity 필수
        rebalance_threshold=0.05,
    ),
)
```

#### PortfolioManagerConfig 주요 필드

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `execution_mode` | `"orders"` | `"orders"` (연속 리밸런싱) / `"signals"` (진입/청산) |
| `price_type` | `"next_open"` | Look-ahead bias 방지 |
| `rebalance_threshold` | `0.05` | 최소 비중 변화 5% |
| `max_leverage_cap` | `3.0` | 시스템 레버리지 상한 |
| `system_stop_loss` | `0.10` | 자기자본 대비 10% 손절 |
| `use_trailing_stop` | `False` | 트레일링 스톱 활성화 |
| `trailing_stop_atr_multiplier` | `2.0` | ATR 배수 (라이브: 3.0 권장) |
| `use_intrabar_stop` | `True` | High/Low 기반 SL (vs Close) |
| `use_intrabar_trailing_stop` | `True` | 1m bar TS 체크 (False=TF bar만) |
| `cash_sharing` | `False` | 멀티에셋 현금 공유 |
| `cost_model` | `CostModel()` | 비용 모델 |
| `smart_execution` | `SmartExecutorConfig()` | Limit order 설정 |

#### CostModel

```python
class CostModel(BaseModel):
    maker_fee: float = 0.0002       # 0.02% (Binance VIP0)
    taker_fee: float = 0.0004       # 0.04%
    slippage: float = 0.0005        # 0.05%
    funding_rate_8h: float = 0.0001 # 0.01%/8h
    market_impact: float = 0.0002   # 0.02%
    use_taker: bool = True

    @property
    def effective_fee(self) -> float      # taker or maker
    @property
    def total_fee_rate(self) -> float     # fee + slippage + impact
    @property
    def round_trip_cost(self) -> float    # 2 × total_fee_rate
    @property
    def daily_funding_cost(self) -> float # funding_rate_8h × 3

    # 프리셋
    CostModel.zero()             # 비용 0
    CostModel.conservative()     # 2배 비용
    CostModel.binance_futures()  # 바이낸스 선물 기본
```

#### SmartExecutorConfig

```python
class SmartExecutorConfig(BaseModel):
    enabled: bool = False                   # opt-in
    limit_timeout_seconds: float = 30.0     # Limit 대기
    price_offset_bps: float = 1.0           # mid 대비 1bp
    max_price_deviation_pct: float = 0.3    # 조기 취소 임계
    poll_interval_seconds: float = 2.0      # 상태 확인 주기
    max_concurrent_limit_orders: int = 4    # 동시 제한
    fallback_to_market: bool = True         # timeout → market
```

**파일 위치:**

- `src/portfolio/portfolio.py`: Portfolio
- `src/portfolio/config.py`: PortfolioManagerConfig
- `src/portfolio/cost_model.py`: CostModel
- `src/eda/smart_executor_config.py`: SmartExecutorConfig

### 2.4 PerformanceAnalyzer (Domain Layer)

성과 분석을 전담하는 컴포넌트입니다.

```python
from src.backtest import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# 성과 지표 추출
metrics = analyzer.analyze(vbt_portfolio)

# 벤치마크 비교
benchmark = analyzer.compare_benchmark(vbt_portfolio, data.ohlcv, "BTC/USDT")

# 거래 기록
trades = analyzer.extract_trades(vbt_portfolio, "BTC/USDT")

# QuantStats용 수익률 시리즈
strat_ret, bench_ret = analyzer.get_returns_series(
    vbt_portfolio, data.ohlcv, "BTC/USDT"
)

# 펀딩비 반영 수익률
funding_ret = analyzer.get_funding_adjusted_returns(
    vbt_portfolio, cost_model, freq="12h"
)

# Beta 분해
beta_result = analyzer.analyze_beta_attribution(
    diagnostics_df, benchmark_returns, window=60
)
```

**파일 위치:**

- `src/backtest/analyzer.py`: PerformanceAnalyzer

### 2.5 BacktestRequest (Application Layer)

실행에 필요한 모든 정보를 캡슐화하는 Command 객체입니다.

```python
from src.backtest import BacktestRequest
from src.backtest.request import MultiAssetBacktestRequest

# 단일 에셋
request = BacktestRequest(
    data=data,              # MarketDataSet
    strategy=strategy,      # BaseStrategy
    portfolio=portfolio,    # Portfolio
    analyzer=analyzer,      # PerformanceAnalyzer (optional)
)

# 멀티 에셋
request = MultiAssetBacktestRequest(
    data=multi_data,                    # MultiSymbolData
    strategy=strategy,                  # BaseStrategy
    portfolio=portfolio,                # Portfolio
    weights={"BTC/USDT": 0.5, ...},    # None → Equal Weight (1/N)
    asset_allocation=AssetAllocationConfig(
        method="inverse_volatility",
        rebalance_bars=14,
    ),                                  # 동적 배분 (optional)
    analyzer=analyzer,                  # PerformanceAnalyzer (optional)
)
```

**파일 위치:**

- `src/backtest/request.py`: BacktestRequest, MultiAssetBacktestRequest

### 2.6 BacktestEngine (Application Layer)

Stateless 백테스트 실행자입니다.

```python
from src.backtest import BacktestEngine

engine = BacktestEngine()

# === 단일 에셋 ===
result = engine.run(request)                          # BacktestResult
result, strat_ret, bench_ret = engine.run_with_returns(request)
result, validation = engine.run_validated(request, level="quick")

# === 멀티 에셋 ===
result = engine.run_multi(request)                    # MultiAssetBacktestResult
result, returns, benchmark = engine.run_multi_with_returns(request)
result, validation = engine.run_multi_validated(request, level="quick")
```

**내부 처리 흐름 (run_multi):**

```text
MultiAssetBacktestRequest
    ↓
1. 심볼별 독립 전략 실행 (strategy.run(df))
2. Enrichment 검증 (required_enrichments)
3. 동적 자산 배분 비중 계산 (IV/EW/RP/SW)
4. PM 규칙 적용 (SL/TS/Rebalance — Numba JIT)
5. VBT from_orders(cash_sharing=True, group_by=True) 실행
6. 펀딩비 반영 메트릭 조정 (365일 연환산)
7. 포트폴리오 + 심볼별 성과 분석
    ↓
MultiAssetBacktestResult
```

**Numba 최적화 함수 (모듈 레벨):**

| 함수 | 용도 |
|------|------|
| `apply_stop_loss_to_weights()` | SL 포지션 청산 (Long/Short) |
| `apply_trailing_stop_to_weights()` | ATR 기반 트레일링 스톱 |
| `apply_rebalance_threshold_numba()` | 리밸런싱 임계값 필터 |
| `_numba_inverse_vol_weights()` | 롤링 IV 비중 행렬 |
| `_numba_single_bar_iv()` | 단일 bar IV 비중 (재귀) |
| `_numba_clamp_normalize()` | Water-filling 비중 정규화 |

**파일 위치:**

- `src/backtest/engine.py`: BacktestEngine, `run_parameter_sweep()`, PM 규칙 함수

### 2.7 결과 모델 (Models Layer)

```python
class PerformanceMetrics(BaseModel):
    # 수익률
    total_return: float          # %
    cagr: float                  # %
    # 리스크 조정
    sharpe_ratio: float
    sortino_ratio: float | None
    calmar_ratio: float | None
    # 낙폭
    max_drawdown: float          # %
    avg_drawdown: float | None
    # 거래 통계
    win_rate: float              # %
    profit_factor: float | None
    total_trades: int
    winning_trades: int
    losing_trades: int
    # 분포
    volatility: float | None     # 연환산 %
    skewness: float | None
    kurtosis: float | None
    # 계산 필드
    risk_reward_ratio: float | None  # @computed_field

class BacktestResult(BaseModel):
    config: BacktestConfig
    metrics: PerformanceMetrics
    benchmark: BenchmarkComparison | None
    trades: tuple[TradeRecord, ...]

    def passed_minimum_criteria(
        self, min_sharpe=1.0, max_mdd=40.0, min_win_rate=40.0
    ) -> bool

class MultiAssetBacktestResult(BaseModel):
    config: MultiAssetConfig
    portfolio_metrics: PerformanceMetrics       # 전체 포트폴리오
    per_symbol_metrics: dict[str, PerformanceMetrics]
    correlation_matrix: dict[str, dict[str, float]]
    contribution: dict[str, float]              # 심볼별 기여도
```

**파일 위치:**

- `src/models/backtest.py`: BacktestResult, MultiAssetBacktestResult, PerformanceMetrics, TradeRecord, BenchmarkComparison

---

## 3. 검증 시스템 (Validation)

3단계 과적합 검증 시스템입니다.

### 3.1 검증 레벨

| 레벨 | 방법 | 기준 |
|------|------|------|
| **QUICK** | IS/OOS 분할 (70/30) | OOS Sharpe ≥ 0.0, Decay ≤ 50% |
| **MILESTONE** | Walk-Forward (5 fold, expanding) | OOS Sharpe ≥ 0.0, Decay ≤ 50% |
| **FINAL** | CPCV + Monte Carlo (1000회) | DSR ≥ 0.0, PBO 검증 |

### 3.2 사용법

```python
from src.backtest.validation import TieredValidator, ValidationLevel

validator = TieredValidator()

# 단일 에셋
result = validator.validate(
    level=ValidationLevel.QUICK,
    data=data,
    strategy=strategy,
    portfolio=portfolio,
)

# 멀티 에셋
result = validator.validate_multi(
    level=ValidationLevel.FINAL,
    data=multi_data,
    strategy=strategy,
    portfolio=portfolio,
    weights=weights,
)

print(result.passed)             # True/False
print(result.verdict)            # "PASS" | "WARN" | "FAIL"
print(result.avg_sharpe_decay)   # IS→OOS 감쇠율
print(result.consistency_ratio)  # 일관성 비율
print(result.overfit_probability)# 과적합 확률
```

### 3.3 ValidationResult 모델

```python
class ValidationResult(BaseModel):
    level: ValidationLevel
    fold_results: tuple[FoldResult, ...]
    monte_carlo: MonteCarloResult | None

    avg_train_sharpe: float
    avg_test_sharpe: float
    sharpe_stability: float      # Test Sharpe 표준편차
    passed: bool
    failure_reasons: tuple[str, ...]

    # 계산 필드
    avg_sharpe_decay: float      # (Train - Test) / Train
    consistency_ratio: float     # 일관된 fold 비율
    overfit_probability: float   # 0~1
    verdict: Literal["PASS", "WARN", "FAIL"]
```

**파일 위치:**

- `src/backtest/validation/validator.py`: TieredValidator
- `src/backtest/validation/models.py`: ValidationResult, FoldResult, MonteCarloResult
- `src/backtest/validation/levels.py`: ValidationLevel enum
- `src/backtest/validation/splitters.py`: IS/OOS, WFA, CPCV 분할기
- `src/backtest/validation/deflated_sharpe.py`: Deflated Sharpe Ratio, PSR
- `src/backtest/validation/pbo.py`: Probability of Backtest Overfitting
- `src/backtest/validation/report.py`: 검증 리포트 생성

---

## 4. 분석 & 최적화 도구

### 4.1 Optimizer (Optuna)

Optuna TPE 기반 파라미터 최적화입니다.

```python
from src.backtest.optimizer import optimize_strategy, generate_g3_sweeps

# Optuna 최적화
best_params = optimize_strategy(
    strategy_class=TriChannelTrendStrategy,
    data=multi_data,
    portfolio=portfolio,
    n_trials=100,
)

# G3 안정성 sweep 생성
sweeps = generate_g3_sweeps(config_class=TriChannelConfig)
```

주요 기능:

- `extract_search_space(config_class)`: Pydantic 필드 → Optuna 파라미터 자동 변환
- complement 필드 처리 (bb_weight ↔ rsi_weight)

**파일 위치:** `src/backtest/optimizer.py`

### 4.2 IC Analyzer

전략 시그널의 정보 계수(Information Coefficient)를 빠르게 점검합니다.

```python
from src.backtest.ic_analyzer import ICAnalyzer

result = ICAnalyzer.analyze(indicator_series, forward_returns)

print(result.ic)           # Pearson IC
print(result.rank_ic)      # Spearman Rank IC
print(result.ic_ir)        # IC / std(IC)
print(result.hit_rate)     # 방향 적중률 (%)
print(result.stable)       # 안정성 판정
```

**임계값:**

- `IC_ABS_THRESHOLD`: 0.02
- `IC_IR_ABS_THRESHOLD`: 0.1
- `HIT_RATE_THRESHOLD`: 52%

**파일 위치:** `src/backtest/ic_analyzer.py`

### 4.3 Stress Test

합성 충격을 주입하여 전략의 극단 상황 내성을 검증합니다.

```python
from src.backtest.stress_test import run_stress_test, BLACK_SWAN, FLASH_CRASH

result = run_stress_test(
    df=ohlcv_df,
    scenario=BLACK_SWAN,     # -30% 가격 충격, 5x 스프레드
    strategy_fn=strategy.run,
)

print(result.max_drawdown)
print(result.recovery_bars)
```

**사전 정의 시나리오:**

| 시나리오 | 가격 충격 | 스프레드 |
|----------|-----------|----------|
| `BLACK_SWAN` | -30% | 5x |
| `LIQUIDITY_CRISIS` | -10% | 10x |
| `FUNDING_SPIKE` | - | funding 0.3% |
| `FLASH_CRASH` | -15% | 3x |

**파일 위치:** `src/backtest/stress_test.py`

### 4.4 Beta Attribution

전략 수익의 Beta 분해 분석입니다.

```python
analyzer.analyze_beta_attribution(diagnostics_df, benchmark_returns, window=60)
analyzer.get_rolling_beta_attribution(...)  # 시계열 Beta
```

**파일 위치:** `src/backtest/beta_attribution.py`

### 4.5 Advisor

백테스트 결과를 분석하여 개선 제안을 생성합니다.

```text
src/backtest/advisor/
├── advisor.py          # 메인 어드바이저
├── suggestions.py      # 제안 생성
├── models.py           # 제안 모델
└── analyzers/          # Loss, Overfit, Regime, Signal 분석기
```

### 4.6 Parameter Sweep

Grid search 기반 파라미터 탐색입니다.

```python
from src.backtest import run_parameter_sweep

results = run_parameter_sweep(
    strategy_class=DonchMultiStrategy,
    data=data,
    param_grid={
        "lookback_short": [30, 40, 50],
        "lookback_mid": [60, 74, 90],
        "lookback_long": [140, 157, 180],
    },
    portfolio=portfolio,
    top_n=10,
)
```

### 4.7 Metrics (순수 함수)

```python
from src.backtest.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_cagr,
    calculate_rolling_sharpe,
    calculate_profit_factor,
)
```

**파일 위치:** `src/backtest/metrics.py`

---

## 5. EDA 엔진 통합

### 5.1 EDA와 VBT의 관계

```text
                    ┌─────────────────┐
                    │  BaseStrategy   │  ← 동일 전략 코드 공유
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  BacktestEngine │           │    EDARunner     │
    │  (VBT 벡터화)   │           │  (이벤트 기반)   │
    │                 │           │                  │
    │  전체 데이터를  │           │  1m bar 순차 처리│
    │  행렬 연산으로  │           │  → TF 집계       │
    │  일괄 처리      │           │  → 시그널 생성   │
    │                 │           │  → 다음 bar 체결 │
    └─────────────────┘           └─────────────────┘
          │                             │
          └──────────┐    ┌─────────────┘
                     ▼    ▼
              ┌───────────────┐
              │  Parity 검증  │  ← Sharpe 편차 < 2%
              └───────────────┘
```

### 5.2 EDA 이벤트 흐름

```text
[Backtest] 1m Parquet → CandleAggregator → BAR → StrategyEngine → SIGNAL → PM → RM → OMS → FILL
[Live]     WebSocket  → CandleAggregator → BAR → StrategyEngine → SIGNAL → PM → RM → OMS → FILL
[Multi]    1m → MultiTF Aggregator → BAR → Orchestrator → Pod SIGNAL → PM → RM → OMS → FILL
```

### 5.3 VBT-EDA Parity 핵심 포인트

| 항목 | VBT | EDA | Parity 해법 |
|------|-----|-----|-------------|
| 실행 가격 | Close[N] | Open[N+1] | Deferred fill = VBT shift(-1) |
| TS Peak 추적 | 전체 bar 일괄 | 1m bar 순차 | `use_intrabar_trailing_stop=False` |
| 비용 모델 | 벡터 브로드캐스트 | Smart: maker vs taker 분리 | BacktestExecutor smart_execution |
| SL/TS 즉시 체결 | 트리거 bar close | 명시적 가격 | 동일 |

### 5.4 EDA 핵심 컴포넌트

| 컴포넌트 | 파일 | 역할 |
|----------|------|------|
| EDARunner | `src/eda/runner.py` | 메인 실행기 |
| StrategyEngine | `src/eda/strategy_engine.py` | BaseStrategy → 이벤트 어댑터 |
| PortfolioManager | `src/eda/portfolio_manager.py` | 포지션/캐시/리밸런싱/SL/TS |
| RiskManager | `src/eda/risk_manager.py` | 사전 검증, 서킷 브레이커 |
| OMS | `src/eda/oms.py` | 주문 라우팅, 멱등성 |
| BacktestExecutor | `src/eda/executors.py` | Deferred fill 실행 |
| SmartExecutor | `src/eda/smart_executor.py` | Limit order 우선 (LiveExecutor 래핑) |
| CandleAggregator | `src/eda/candle_aggregator.py` | 1m → TF 집계 (순수 로직) |
| OrchestratedRunner | `src/eda/orchestrated_runner.py` | 멀티 Pod EDA 실행기 |

---

## 6. 사용 예시

### 6.1 기본 백테스트

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
        timeframe="12H",
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

### 6.2 멀티에셋 + 검증

```python
from src.backtest.request import MultiAssetBacktestRequest

request = MultiAssetBacktestRequest(
    data=multi_data,
    strategy=TriChannelTrendStrategy.from_params(
        scale_short=20, scale_mid=60, scale_long=150,
    ),
    portfolio=Portfolio.create(initial_capital=Decimal("10000")),
)

# 검증 결합 실행
result, validation = BacktestEngine().run_multi_validated(request, level="quick")

print(f"Sharpe: {result.portfolio_metrics.sharpe_ratio:.2f}")
print(f"Validation: {validation.verdict}")
print(f"OOS Decay: {validation.avg_sharpe_decay:.1%}")
```

### 6.3 QuantStats 리포트

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
    title="Tri-Channel 12H Backtest",
)
```

### 6.4 EDA 백테스트

```python
from src.eda.runner import EDARunner

metrics = await EDARunner.backtest(
    strategy=TriChannelTrendStrategy(),
    data=multi_data,           # 1m 데이터 필요
    target_timeframe="12H",
    config=pm_config,
    initial_capital=10000.0,
)
```

### 6.5 Orchestrated 백테스트

```python
from src.eda.orchestrated_runner import OrchestratedRunner

runner = OrchestratedRunner(
    orchestrator_config=orch_config,  # 4-Pod 설정
    data=multi_data_1m,               # 1m MultiSymbolData
    target_timeframe="12H",
    initial_capital=10000.0,
)

result = await runner.run()
```

---

## 7. CLI 사용법

```bash
# === 단일에셋 백테스트 ===
uv run mcbot backtest run BTC/USDT --strategy tsmom --year 2024 --year 2025
uv run mcbot backtest run BTC/USDT -y 2024 --report    # QuantStats 리포트
uv run mcbot backtest run BTC/USDT -c 50000 --report   # 자본 변경

# === 멀티에셋 백테스트 ===
uv run mcbot backtest run-multi -s tri-channel-trend -y 2024 -y 2025
uv run mcbot backtest run-multi --symbols BTC/USDT,ETH/USDT -c 50000
uv run mcbot backtest run-multi --validation quick

# === 파라미터 최적화 ===
uv run mcbot backtest optimize --strategy tri-channel-trend --symbol ETH/USDT

# === 과적합 검증 ===
uv run mcbot backtest validate -m quick       # IS/OOS
uv run mcbot backtest validate -m milestone   # Walk-Forward
uv run mcbot backtest validate -m final       # CPCV + DSR + PBO

# === 정보 ===
uv run mcbot backtest strategies              # 등록된 전략 목록
uv run mcbot backtest info --strategy tsmom   # 전략 메타데이터
uv run mcbot backtest diagnose --strategy tsmom  # 진단

# === EDA 실행 ===
uv run mcbot eda run config/default.yaml              # EDA 백테스트
uv run mcbot eda run-live config/paper.yaml --mode paper  # 페이퍼 트레이딩
```

---

## 8. 파일 구조

```text
src/
├── backtest/
│   ├── __init__.py              # 모듈 exports
│   ├── engine.py                # BacktestEngine, run_parameter_sweep, PM 규칙 (Numba)
│   ├── request.py               # BacktestRequest, MultiAssetBacktestRequest
│   ├── analyzer.py              # PerformanceAnalyzer, Beta Attribution
│   ├── metrics.py               # 순수 함수 기반 지표 (calculate_*)
│   ├── cost_model.py            # CostModel re-export
│   ├── reporter.py              # QuantStats 리포트
│   ├── optimizer.py             # Optuna 파라미터 최적화
│   ├── ic_analyzer.py           # Information Coefficient 분석
│   ├── stress_test.py           # 합성 충격 검증
│   ├── beta_attribution.py      # Beta 분해 분석
│   ├── advisor/                 # 전략 개선 제안 시스템
│   │   ├── advisor.py
│   │   ├── suggestions.py
│   │   ├── models.py
│   │   └── analyzers/           # Loss, Overfit, Regime, Signal
│   └── validation/
│       ├── __init__.py
│       ├── validator.py         # TieredValidator (validate, validate_multi)
│       ├── models.py            # ValidationResult, FoldResult, MonteCarloResult
│       ├── levels.py            # ValidationLevel enum
│       ├── splitters.py         # IS/OOS, WFA, CPCV 분할기 (multi-* 포함)
│       ├── deflated_sharpe.py   # DSR, PSR
│       ├── pbo.py               # Probability of Backtest Overfitting
│       ├── monte_carlo.py       # Monte Carlo 시뮬레이션
│       └── report.py            # 검증 리포트 생성
├── data/
│   ├── __init__.py
│   ├── market_data.py           # MarketDataRequest, MarketDataSet, MultiSymbolData
│   ├── service.py               # MarketDataService (get, get_multi)
│   ├── bronze.py                # BronzeStorage
│   ├── fetcher.py               # DataFetcher
│   └── silver.py                # SilverProcessor
├── eda/
│   ├── runner.py                # EDARunner (메인 실행기)
│   ├── strategy_engine.py       # BaseStrategy 이벤트 어댑터
│   ├── portfolio_manager.py     # PM (포지션/캐시/SL/TS)
│   ├── risk_manager.py          # RM (사전 검증/서킷 브레이커)
│   ├── oms.py                   # OMS (주문 라우팅/멱등성)
│   ├── executors.py             # BacktestExecutor, ShadowExecutor, LiveExecutor
│   ├── smart_executor.py        # SmartExecutor (Limit order 우선)
│   ├── smart_executor_config.py # SmartExecutorConfig
│   ├── candle_aggregator.py     # 1m → TF 집계 (순수 로직)
│   ├── data_feed.py             # HistoricalDataFeed
│   ├── live_data_feed.py        # WebSocket 실시간 피드
│   ├── orchestrated_runner.py   # 멀티 Pod EDA 실행기
│   ├── analytics.py             # 성과 계산
│   ├── reconciler.py            # 재시작 상태 복원
│   ├── exchange_stop_manager.py # STOP_MARKET 안전장치
│   └── persistence/             # 상태/거래 DB 영속화
├── models/
│   └── backtest.py              # BacktestResult, MultiAssetBacktestResult, PerformanceMetrics
├── portfolio/
│   ├── __init__.py
│   ├── portfolio.py             # Portfolio 도메인 객체
│   ├── config.py                # PortfolioManagerConfig
│   └── cost_model.py            # CostModel
├── strategy/
│   ├── base.py                  # BaseStrategy ABC
│   ├── types.py                 # StrategySignals, Direction
│   ├── registry.py              # @register 데코레이터
│   ├── tsmom/                   # VW-TSMOM
│   ├── donchian_ensemble/       # Donch-Multi (3-scale)
│   ├── tri_channel/             # Tri-Channel (3채널×3스케일)
│   ├── anchor_momentum/         # Anchor-Mom
│   └── ...                      # 30+ 전략 구현
└── orchestrator/
    ├── orchestrator.py          # 멀티 Pod 넷팅/귀속
    ├── pod.py                   # StrategyPod
    ├── config.py                # OrchestratorConfig
    ├── allocator.py             # 자본 배분 (Risk Parity, Kelly)
    ├── netting.py               # 포지션 넷팅 (one-way/hedge)
    ├── lifecycle.py             # Pod 졸업/퇴출 상태기계
    └── risk_aggregator.py       # 포트폴리오 레벨 리스크
```

---

## 9. 설계 결정 근거

### 9.1 왜 이중 엔진인가?

- **VBT (벡터화)**: 빠른 반복 → 전략 발굴/최적화에 적합 (수천 조합 탐색)
- **EDA (이벤트 기반)**: 실거래 동일 경로 → 라이브 배포 전 필수 검증
- **Parity**: 동일 전략이 두 엔진에서 유사한 결과 → 라이브 신뢰도 보장

### 9.2 왜 Stateless Engine인가?

- **테스트 용이성**: 모든 의존성이 Request로 주입되어 Mock 가능
- **병렬 실행**: 상태가 없어 멀티프로세싱에 안전
- **재현성**: 동일한 Request는 동일한 결과 보장

### 9.3 왜 Portfolio와 Strategy를 분리하는가?

- **Strategy 책임**: 시그널 생성 (순수 함수적)
- **Portfolio 책임**: 자금 관리, 집행 규칙 (상태 관리)
- **재사용성**: 동일 Strategy를 다른 Portfolio 설정에서 사용 가능

### 9.4 왜 Numba JIT인가?

- PM 규칙(SL/TS/Rebalance)은 bar 수 × 심볼 수 만큼 반복 → 순수 Python 병목
- Numba `@njit`로 50~100x 성능 향상
- VBT-EDA parity 유지하면서 벡터화 엔진 성능 확보

### 9.5 왜 SmartExecutor인가?

- Market order: taker fee + slippage → 비용 높음
- Limit order: maker fee + 0 slippage → 비용 절감
- 긴급성 분류: SL/TS → market (즉시), entry/rebal → limit (비용 절감)
- BacktestExecutor에서도 smart_execution=True → maker fee 적용

---

## 10. 현재 운용 현황

### 10.1 ACTIVE 전략 (2026-03-01)

| 전략 | TF | Sharpe (VBT/EDA) | 최적 에셋 | Short Mode |
|------|-----|-------------------|-----------|------------|
| Anchor-Mom | 12H | 1.36 | DOGE | HEDGE_ONLY |
| Donch-Multi | 12H | 1.61 / 1.45 | ETH, BTC | FULL |
| Tri-Channel | 12H | 2.17 / 1.99 | ETH, SOL | HEDGE_ONLY |

### 10.2 전략 탐색 통계

- **총 시도**: 182+ 전략
- **ACTIVE**: 3개 (12H TF)
- **RETIRED**: 179개
- **4H/8H 전멸**: 50+ 시도, 0 ACTIVE
- **1D OHLCV 고갈**: 92개 시도, 0 ACTIVE
- **ML 전략 전멸**: look-ahead bias로 4개 전부 폐기

### 10.3 Orchestrator 최종 설정

```text
Pod 1: Anchor-Mom  → DOGE/USDT        (HEDGE_ONLY, 25%)
Pod 2: Donch-Multi → BTC/USDT         (FULL, 25%)
Pod 3: Tri-Channel → ETH/USDT, SOL/USDT (HEDGE_ONLY, 25%)
Pod 4: Dual-Mom    → BTC,ETH,AVAX,LINK  (ACTIVE, 25%)
```

---

## 11. 개선 사항 & 알려진 제약

### 11.1 아키텍처 개선 제안

| 영역 | 현황 | 제안 |
|------|------|------|
| **engine.py 크기** | 1,715줄 단일 파일 | PM 규칙(Numba)과 동적 배분 로직을 별도 모듈로 분리 (`backtest/pm_rules.py`, `backtest/allocation.py`) |
| **VBT-EDA 중복** | SL/TS/Rebalance 로직이 `engine.py`와 `portfolio_manager.py`에 각각 구현 | 공통 인터페이스 추출하여 단일 출처(Single Source of Truth) 확보 |
| **Execution Mode** | `"orders"` / `"signals"` 2가지 | `"signals"` 모드는 사실상 미사용 → deprecated 후 제거 검토 |

### 11.2 Parity 관련 주의사항

| 항목 | 설명 | 대응 |
|------|------|------|
| `use_intrabar_trailing_stop` | `True`시 EDA에서 1m peak 추적 → 81% 편차 | Orchestrator에서 강제 `False` |
| HEDGE_ONLY + EDA | Drawdown 조건부 숏이 1m→12H 집계 차이에 민감 | 교훈 #086: DISABLED 전환으로 해소 |
| 펀딩비 연환산 | VBT 기본 252일 vs 크립토 365일 | `_adjust_metrics_for_funding()` 자동 보정 |

### 11.3 알려진 제약

- **1D OHLCV 검색공간 소진**: 새로운 1D 전략 발굴 가능성 매우 낮음
- **대안데이터 단독 alpha 부재**: on-chain/deriv/macro 데이터 단독으로는 edge 불확인 (181건 검증)
- **12H 단일 TF 의존**: TF 분산 불가 확정 (4H/8H 총 50+ 시도 전멸)
- **심볼 비중복 필수**: Pod 간 동일 심볼 → netting 상쇄 (One-way Mode 제약)

### 11.4 테스트 현황

- **총 테스트**: 7,600+
- **Coverage**: backtest/, eda/, strategy/, portfolio/ 전 모듈
- **Parity 테스트**: VBT vs EDA Sharpe 편차 < 2% 자동 검증
