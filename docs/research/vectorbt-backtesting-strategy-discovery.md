# VectorBT 백테스팅 아키텍처 & 전략 발굴 프로세스

> **Research Date**: 2026-02-07
> **Scope**: VectorBT 기반 백테스팅 최신 트렌드, 전략 발굴 프로세스, MC Coin Bot 적용 시사점

---

## 목차

1. [VectorBT Pro vs Open Source](#1-vectorbt-pro-vs-open-source)
2. [백테스팅 아키텍처 패턴](#2-백테스팅-아키텍처-패턴)
3. [전략 발굴 프로세스](#3-전략-발굴-프로세스)
4. [성능 최적화](#4-성능-최적화)
5. [리스크 관리 in 백테스팅](#5-리스크-관리-in-백테스팅)
6. [최신 트렌드 (2025-2026)](#6-최신-트렌드-2025-2026)
7. [Best Practices & Anti-patterns](#7-best-practices--anti-patterns)
8. [MC Coin Bot 적용 시사점](#8-mc-coin-bot-적용-시사점)

---

## 1. VectorBT Pro vs Open Source

### 1.1 Open Source (vectorbt)

GitHub에서 무료로 제공되는 오픈소스 버전:

- **Portfolio API**: `from_signals()`, `from_orders()`, `from_order_func()` 3가지 시뮬레이션 모드
- **IndicatorFactory**: 커스텀 인디케이터를 Numba JIT로 컴파일
- **기본 분석**: Sharpe, Sortino, MDD 등 표준 메트릭
- **시각화**: matplotlib/plotly 기반 차트
- **제약**: 단일 스레드, 메모리 관리 수동, WFA/CV 미지원

### 1.2 VectorBT Pro ($20/월)

Pro는 Open Source의 완전한 상위 집합으로, 핵심 추가 기능:

| 기능 | Open Source | Pro |
|------|:---------:|:---:|
| Portfolio Simulation | O | O (확장) |
| Chunked Execution | X | **O** |
| Disk Offloading | X | **O** |
| Dask/Ray 분산 처리 | X | **O** |
| Walk-Forward CV (Purging) | X | **O** |
| Random Search / Lazy Grid | X | **O** |
| Numba/NumPy/JAX 전환 | X | **O** |
| Stop Laddering | X | **O** |
| Limit Orders (TIF) | X | **O** |
| `from_order_func()` 확장 | 기본 | **상태 접근** |
| Live Trading Bridge | X | **O (Binance)** |

**Pro 핵심 혁신:**

1. **Chunked Execution**: 배열 자동 분할 → OOM 방지, 수십억 파라미터 조합 가능
2. **Chunk Caching**: 중간 결과 디스크 저장 → crash recovery
3. **Lazy Parameter Grids**: 전체 그리드 메모리 적재 없이 즉시 random 조합 생성
4. **`from_order_func()` 확장**: 시뮬레이션 중 open P&L, 포지션 정보 접근 가능 → path-dependent 전략 구현

### 1.3 성능 벤치마크

| 연산 | pandas | VBT (no parallel) | VBT (parallel) |
|------|--------|-------------------|----------------|
| Rolling Mean (1000x1000) | 45.6ms | 5.33ms | **1.82ms** |
| Rolling Sortino | 2.79s | - | **8.12ms** (340x) |

---

## 2. 백테스팅 아키텍처 패턴

### 2.1 Vectorized vs Event-Driven

**업계 컨센서스: Hybrid Architecture**

| 구분 | Vectorized | Event-Driven |
|------|-----------|-------------|
| **속도** | 500종목 10년 < 1초 | 500종목 분봉 15-30분 |
| **정확도** | 낮음 (fill 가정) | 높음 (실제 실행 모사) |
| **구현 복잡도** | 낮음 | 높음 |
| **적합 용도** | 일/주봉 리밸런싱, factor research | HFT, 복잡한 주문, 실시간 리스크 |

**현재 업계 표준 3단계 파이프라인:**

```
Stage 1 (Vectorized)     Stage 2 (Event-Driven)     Stage 3 (Production)
 빠른 시그널 스크리닝  →   고충실도 시뮬레이션    →   동일 코드 라이브 전환
 유니버스 축소            최종 후보 전략 검증          EDA 코드 그대로 사용
 factor research          리스크 관리 검증
```

### 2.2 VectorBT Portfolio Creation 모드

**Signal-based (`Portfolio.from_signals`)**:
```python
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,         # Boolean 시그널
    exits=exits,
    size=target_pct,
    size_type='targetpercent',
    group_by=True,           # 단일 그룹
    cash_sharing=True        # 자금 공유
)
```
- 추상화 수준 높음, 빠른 프로토타이핑
- 단순 진입/퇴출 전략에 적합

**Order-based (`Portfolio.from_orders`)**:
```python
pf = vbt.Portfolio.from_orders(
    close=price,
    size=order_sizes,        # 직접 사이즈 지정
    direction='both',
    group_by=True,
    cash_sharing=True
)
```
- 가장 직관적, 복잡한 주문 로직 직접 제어
- Vol-target, 연속 리밸런싱 전략에 최적

**Flex Order (`Portfolio.from_order_func`)** (Pro):
```python
pf = vbt.Portfolio.from_order_func(
    close=price,
    order_func_nb=my_order_func,  # Numba JIT
    # Pro: 시뮬레이션 중 포지션/PnL 접근 가능
)
```
- 최대 유연성, path-dependent 전략 가능
- Stop laddering, dynamic position sizing

### 2.3 Multi-Asset Cash Sharing 패턴

```python
pf = vbt.Portfolio.from_signals(
    close=multi_asset_df,      # DataFrame (columns = assets)
    entries=signals_df,
    group_by=True,             # 모든 자산 하나의 그룹
    cash_sharing=True,         # 자금 공유 → 진정한 포트폴리오
    init_cash=100_000
)
```

핵심 원칙:
- `cash_sharing=True`: cross-asset dependency → 같은 tick 내 모든 주문 순차 실행
- Lazy Broadcasting: 가격이 동일하면 Series → DataFrame 메모리 절약
- `group_by` index hierarchy → strategy/symbol/timeframe별 그룹 성과 분석

### 2.4 Custom Indicators (IndicatorFactory)

```python
MyIndicator = vbt.IndicatorFactory(
    class_name="MyIndicator",
    short_name="my",
    input_names=["close"],
    param_names=["fast_window", "slow_window"],
    output_names=["signal", "upper_band", "lower_band"]
).from_apply_func(
    my_numba_func,              # Numba JIT 함수
    fast_window=20,
    slow_window=60,
)

# 자동 파라미터 그리드 생성
result = MyIndicator.run(close, fast_window=[10, 20, 30], slow_window=[50, 60, 70])
```

자동 파라미터 broadcasting으로 9개 조합 동시 실행.

---

## 3. 전략 발굴 프로세스

### 3.1 전체 파이프라인

```
Strategy Hypothesis
    ↓
Parameter Sweep (Grid/Random Search)
    ↓
CPCV (Robustness 확인)
    ↓
DSR (Multiple Testing 보정)
    ↓
PBO (Overfitting 확률)
    ↓
WFA (실전 시뮬레이션)
    ↓
Monte Carlo (분포 추정)
    ↓
Final Holdout (1회 검증) — 반복 금지
```

### 3.2 Walk-Forward Analysis (WFA)

**Best Practices:**

1. **Rolling Window**: 6개월 training → 1개월 testing → sliding forward
2. **Walk-Forward Efficiency (WFE)**:
   ```
   WFE = Average Test Sharpe / Average Train Sharpe
   ```
   - WFE > 0.5: 건강한 전략
   - WFE < 0.3: overfitting 의심
3. **Anchored vs Rolling**:
   - Anchored (확장 윈도우): 데이터 축적 효과, 장기 패턴 포착
   - Rolling (고정 윈도우): regime 적응성, 최근 시장 반영
4. **Purging**: training/test 경계에서 label horizon만큼 데이터 제거 → 정보 누출 방지

VectorBT Pro 구현:
```python
@vbt.cv(
    splitter=vbt.RollingSplitter(window=252, step=21),
    purge_window=5,
    embargo_window=2
)
def optimize_strategy(data, params):
    ...
```

### 3.3 Combinatorial Purged Cross-Validation (CPCV)

CPCV가 WFA 대비 **false discovery 방지에서 현저히 우수**한 것으로 2024-2025 연구에서 확인됨.

**메커니즘:**
1. Time series를 N개 순차 그룹으로 분할
2. k개 그룹의 모든 조합 C(N,k)을 test set으로 선택
3. 나머지 N-k 그룹으로 training
4. **Purging**: training 관측값 중 label horizon이 test 기간과 겹치는 것 제거
5. **Embargo**: purging 이후 추가 buffer 기간 (autocorrelation 고려)

**최근 변형 (2024-2025):**
- **Bagged CPCV**: 앙상블 기법으로 robustness 강화
- **Adaptive CPCV**: 시장 조건에 따라 동적 조정
- CPCV는 stability와 efficiency에서 WFA 대비 우위
- 충분히 긴 시계열 필요 (최소 5-10년 일봉 데이터 권장)

### 3.4 Deflated Sharpe Ratio (DSR)

Bailey & Lopez de Prado의 DSR 프레임워크:

**핵심 문제**: 보고된 Sharpe ratio에서 누락된 정보 = **시도한 trial 수**

**보정 요소:**
- 독립 실험 수
- Sharpe ratio 분산
- 표본 길이
- Skewness, Kurtosis

**Haircut 가이드라인:**

| Sharpe Ratio | Haircut (할인율) |
|:---:|:---:|
| < 0.4 | 거의 항상 50% 초과 |
| 0.4 - 1.0 | 25% - 50% |
| > 1.0 | 최대 25% |

**Multiple Testing 조정 방법:**
- **Bonferroni**: 가장 보수적 (type I error 최소화)
- **Holm**: Bonferroni 개선 (단계적)
- **BHY (Benjamini-Hochberg-Yekutieli)**: FDR 제어 (가장 실용적)

### 3.5 Probability of Backtest Overfitting (PBO)

**CSCV (Combinatorially Symmetric Cross-Validation) 기반:**

1. 모든 trial의 성과를 N개 그룹으로 분할
2. C(N, N/2) 조합으로 IS/OOS 쌍 생성
3. IS에서 최적 전략이 OOS에서 중앙값 이하인 비율 = **PBO**
4. **PBO > 0.5**: 전략 선택 프로세스가 overfitting에 취약

### 3.6 Machine Learning Integration

**Triple Barrier Method + Meta-Labeling:**

```
Triple Barrier:
  ┌── Upper Barrier (Take Profit)  → Label: +1
  │
Price ──── Vertical Barrier (Time)  → Label:  0
  │
  └── Lower Barrier (Stop Loss)    → Label: -1
```

- 각 관측값의 **변동성 기반** barrier 설정 (고정 barrier 아닌 동적)
- Meta-labeling: 1차 모델이 방향(long/short) 결정 → 2차 ML 모델이 **bet 크기** 결정

**Regime Detection:**
- **HMM (Hidden Markov Model)**: bull/bear/neutral 상태 전이 탐지
- **Tree-based Ensemble + HMM Hybrid**: regime shift 탐지 후 HMM으로 시계열 모델링
- **GARCH + Mixture Models**: 이분산성 기반 확률적 regime 분류

**Feature Engineering 최신 트렌드:**
- 기술적 지표: SMA, EMA, RSI, Bollinger Bands
- 거시경제: 금리, VIX, 통화 강도
- 대체 데이터: 뉴스 sentiment(NLP), 소셜 미디어, on-chain 데이터
- **Multi-modal Fusion**: SentiStack 등 early/late fusion 기법 (IC 0.041, RIC 0.473, Sharpe 1.333)

---

## 4. 성능 최적화

### 4.1 Numba JIT Patterns

```python
from numba import njit, prange

@njit(parallel=True, cache=True)
def calculate_signals(close, fast_window, slow_window):
    n_cols = close.shape[1]
    result = np.empty_like(close)
    for col in prange(n_cols):        # 컬럼 단위 자동 병렬화
        for i in range(close.shape[0]):  # inner loop은 sequential
            ...
    return result
```

핵심 원칙:
- `@njit(parallel=True)`: outer loop 자동 병렬화
- `prange`로 명시적 병렬 범위 지정
- `cache=True`로 재컴파일 방지
- Pro: Numba/NumPy/JAX 구현 간 런타임 전환 가능

### 4.2 Memory-Efficient Parameter Sweeps

**VectorBT Pro 전략:**

1. **`@vbt.chunked`**: 배열 자동 분할 → OOM 방지
2. **Disk Offloading**: 중간 결과 디스크 저장 → crash recovery
3. **Lazy Parameter Grids**: 전체 그리드 메모리 적재 없이 즉시 조합 생성
4. **Lazy Broadcasting**: Series → DataFrame 변환 시 실제 메모리 복사 없이 처리

### 4.3 분산 처리 백엔드

| 백엔드 | 적합한 경우 | GIL |
|--------|-----------|-----|
| `ThreadPoolExecutor` | Numba 등 GIL 해제 함수 | 해제 |
| `ProcessPoolExecutor` | 무거운 Python 함수 | 유지 |
| `Dask` | 대규모 분산 처리, lazy eval | - |
| `Ray` | GPU 가속, ML 워크로드 | - |

```python
# Dask 또는 Ray backend 전환
vbt.settings.execution['engine'] = 'dask'  # or 'ray'
```

---

## 5. 리스크 관리 in 백테스팅

### 5.1 Realistic Fill Simulation

**VectorBT Pro 제공:**
- **Stop Laddering**: 다단계 stop (각 단계별 청산 비율 설정)
- **Limit Orders + TIF**: DAY/GTC/GTD time-in-force 지원
- **Order Delays**: 다음 바 실행 시뮬레이션
- **Leverage Modes**: Lazy(무제한) vs Eager(즉시 마진 검증)

### 5.2 Dynamic Slippage Model

**Square Root Market Impact:**
```
Impact = σ × √(Q / V) × π
```
- `σ`: 변동성 (시장 불안정 시 slippage 증가)
- `Q`: 주문 크기 (order book depth 대비)
- `V`: 거래량 (유동성 프록시)

**비용 수준별 가이드라인:**

| 수준 | Round-trip 비용 | 용도 |
|------|:---:|------|
| Conservative | 0.50% | 최종 검증 |
| Moderate | 0.30% | 일반적 백테스트 |
| Optimistic | 0.15% | 최적 조건 시뮬레이션 |

**Crypto-specific 비용:**
- Maker fee: 0.02% (Binance Futures VIP0)
- Taker fee: 0.04%
- Funding rate: 8시간마다 정산
- Liquidation risk premium

### 5.3 Position Sizing Methods

**Kelly Criterion:**
```python
kelly_fraction = (win_prob * avg_win - loss_prob * avg_loss) / avg_win
# 실무: Half-Kelly (50% 축소) → 파산 위험 감소
position_size = 0.5 * kelly_fraction * equity
```
- 장기 기대 로그 재산 최대화
- 대부분의 전략에서 2-4% risk per trade 시사

**Volatility Targeting:**
```python
target_vol = 0.15  # 연 15%
realized_vol = returns.rolling(60).std() * np.sqrt(252)
leverage = target_vol / realized_vol
position_size = leverage * equity
```
- Regime 변화에 자동 적응
- 최근 연구 (2025): **Kelly + VIX-based scaling hybrid** 접근

**Risk Parity:**
- 각 자산이 포트폴리오 총 위험에 **동일하게 기여**하도록 배분
- 수익률 기반이 아닌 **위험 기반** 배분
- 하락장 buffer 효과 우수

---

## 6. 최신 트렌드 (2025-2026)

### 6.1 Hybrid Architecture (Vectorized + Event-Driven)

**현재 업계 표준 패턴:**

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Vectorized     │     │  Event-Driven    │     │  Production   │
│  (VectorBT)     │ ──→ │  (EDA/Nautilus)  │ ──→ │  (Live)       │
│                 │     │                  │     │               │
│  - Factor scan  │     │  - Fill sim      │     │  - Same code  │
│  - Param sweep  │     │  - Risk mgmt     │     │  - Real feeds │
│  - Quick screen │     │  - Parity test   │     │  - Monitoring │
└─────────────────┘     └──────────────────┘     └───────────────┘
```

**NautilusTrader (2025 최신):**
- Rust/Cython 코어, Python 인터페이스
- **동일 코드로 backtest/sandbox/live 전환** (NautilusKernel 공유)
- OrderBookDepth10 지원
- 주간 릴리스 사이클

### 6.2 Real-time Strategy Deployment

핵심 과제: **Parity Problem** (backtest ↔ production 환경 불일치)

해결 접근:
- **Unified Codebase**: NautilusTrader, QuantConnect 채택
- **Docker 기반 배포**: QuantRocket의 self-hosted 패턴
- **Parity Test**: VBT(검증) → EDA(실행) 분리, 수렴 확인

### 6.3 Alternative Data (Crypto)

**On-chain 데이터:**
- SOPR (Spent Output Profit Ratio)
- Whale 이동 추적 → conviction shift vs 일시적 변동성 구분
- DeFi ecosystem 활동, AI/memecoin narrative 추적

**Funding Rate 신호:**
- 지속적 positive funding → 과도한 롱 → mean reversion 시그널
- Cross-exchange funding rate 차이 → arbitrage
- Open interest + funding rate + liquidation 결합 → 파생상품 시장 신호

**ADL (Autodeleveraging) 리스크:**
- 2025년 10월 대규모 ADL 발동 사례
- Insurance fund 소진 시나리오 모델링 중요
- Cascade liquidation 시뮬레이션 필수

### 6.4 QuantEvolve: Multi-Agent Strategy Discovery

2025년 10월 arxiv에 발표된 **QuantEvolve** 프레임워크:
- Multi-Agent Evolutionary Framework로 전략 자동 발굴
- LLM + genetic algorithm 결합
- 전략 코드 자동 생성 → 백테스트 → 진화 → 최적 전략 수렴

---

## 7. Best Practices & Anti-patterns

### 7.1 Common Backtesting Pitfalls

| Pitfall | 설명 | 방지 방법 |
|---------|------|----------|
| **Lookahead Bias** | 미래 정보 사용 | Point-in-time 데이터, `.shift(1)` |
| **Survivorship Bias** | 현존 종목만 테스트 | 상장폐지/인수 포함 전체 유니버스 |
| **Data Snooping** | 과도한 패턴 탐색 | DSR, PBO로 통계적 검증 |
| **Overfitting** | 파라미터 과적합 | WFA, CPCV, Monte Carlo |
| **Unrealistic Fills** | 종가 체결, 슬리피지 무시 | 동적 슬리피지, next-bar open 체결 |
| **Selection Bias** | 최고 성과만 보고 | Haircut SR, multiple testing 보정 |
| **Cherry-picking** | 유리한 기간만 선택 | Dataset scope 사전 정의 |

### 7.2 Proper Train/Test Splitting

**절대 원칙: 시간 순서 보존 (Random split 금지)**

```
|--- Training (60%) ---|-- Purge --|-- Embargo --|--- Test (20%) ---|-- Holdout (20%) --|
```

- **Purging**: training set 끝과 test set 시작 사이 label horizon만큼 제거
- **Embargo**: purging 이후 추가 안전 기간 (autocorrelation 고려)
- **Block Bootstrap**: 시계열 구조 유지 resampling → robust 통계 추정

### 7.3 Statistical Validation Checklist

| 단계 | 지표 | 목표 |
|------|------|------|
| 1. Minimum Track Record | 최소 필요 backtest 기간 | SR 기반 산출 |
| 2. DSR | 조정된 Sharpe Ratio | trial 수/skew/kurt 반영 |
| 3. PBO | Overfitting 확률 | **< 0.30** (목표) |
| 4. WFE | Walk-Forward Efficiency | **> 0.50** |
| 5. Monte Carlo | Bootstrap CI | 1,000+ 시뮬레이션 |
| 6. Haircut SR | 할인 Sharpe | Bonferroni/Holm/BHY |
| 7. Final Holdout | 최종 검증 | **1회 한정** (반복 금지) |

---

## 8. MC Coin Bot 적용 시사점

### 8.1 현재 아키텍처 평가

MC Coin Bot은 이미 업계 best practice에 부합하는 구조를 갖추고 있음:

| 영역 | 업계 Best Practice | MC Coin Bot 현황 | 평가 |
|------|-------------------|-----------------|------|
| Hybrid Architecture | Vectorized + Event-Driven | VBT(Phase 1-3) + EDA(Phase 4+) | **O** |
| Parity Test | Backtest ↔ Live 수렴 확인 | VBT 2.77% vs EDA 2.72% | **O** |
| Multi-Asset | Cash sharing, batch processing | `cash_sharing=True`, batch mode | **O** |
| Validation Framework | WFA, CPCV, DSR, PBO | 3-Tier (Quick/Milestone/Final) | **O** |
| Risk Management | SL, TS, Vol-target | PM 3-layer defense | **O** |
| Numba JIT | 리스크 룰 최적화 | `@njit(cache=True)` 3개 함수 | **O** |
| 비용 모델링 | Funding rate 포함 | CostModel + H-001 funding 조정 | **O** |
| 1m Aggregation | 고해상도 intrabar 체크 | CandleAggregator + AggregatingDataFeed | **O** |

### 8.2 강화 가능 영역

**A. 동적 슬리피지 모델 (우선순위: 높음)**

현재: 고정 fee 모델 (maker 0.02%, taker 0.04%, slippage 0.005%)
개선: Square Root Market Impact 모델 도입

```python
# 현재
class CostModel(BaseModel):
    slippage_pct: float = 0.005

# 개선안
class CostModel(BaseModel):
    slippage_model: Literal["fixed", "sqrt_impact"] = "fixed"
    slippage_pct: float = 0.005          # fixed mode
    volatility_window: int = 20           # sqrt_impact mode
    impact_coefficient: float = 0.1       # sqrt_impact mode
```

**B. Regime Detection 통합 (우선순위: 중간)**

HMM 기반 regime filter를 전략 레이어에 추가:
- Bull/Bear/Neutral 상태에 따라 포지션 사이징 조정
- TSMOM은 trending regime에서만 풀 사이즈
- BB-RSI는 mean-reverting regime에서만 활성화

**C. CPCV 고도화 (우선순위: 중간)**

- Bagged CPCV: 앙상블 기법으로 robustness 강화
- Adaptive CPCV: 시장 조건에 따라 split 수 동적 조정
- 현재 5-split × 2-test 설정은 적절하나 변형 추가 가능

**D. Triple Barrier + Meta-Labeling (우선순위: 낮음)**

전략 발굴의 다음 단계로:
1. 기존 TSMOM 시그널을 1차 모델로 사용
2. Triple Barrier로 라벨링
3. 2차 ML 모델이 bet 크기(position sizing) 결정
4. 기존 vol-target과 대체/보완 가능

**E. Parameter Sweep 고도화 (우선순위: 낮음)**

현재: Cartesian product 전수 탐색
개선 옵션:
- Random Search (대규모 파라미터 공간에서 효율적)
- Bayesian Optimization (순차적 최적화)
- ProcessPoolExecutor 병렬화 (독립 조합 동시 실행)

### 8.3 아키텍처 비교: MC Coin Bot vs NautilusTrader

| 영역 | MC Coin Bot | NautilusTrader |
|------|------------|---------------|
| **코어 언어** | Python 3.13 | Rust/Cython + Python |
| **백테스트 엔진** | VBT + EDA (dual) | NautilusKernel (unified) |
| **라이브 전환** | EDA Runner → (미구현) | 동일 코드 → adapter만 교체 |
| **데이터 처리** | Medallion (Bronze/Silver) | 내장 catalog |
| **전략 프레임워크** | BaseStrategy ABC + registry | Strategy ABC + 상태 머신 |
| **이벤트 시스템** | async Queue + EventBus | MessageBus (Rust native) |

**시사점**: MC Coin Bot의 EDA Runner 패턴은 NautilusTrader의 NautilusKernel과 유사한 방향. Phase 5-C(라이브 전환)에서 adapter 패턴으로 backtest → live 코드 공유가 가능.

### 8.4 다음 단계 로드맵 제안

```
Phase 6: Strategy Enhancement
├── 6.1 Dynamic Slippage Model
├── 6.2 HMM Regime Detection
├── 6.3 Bagged CPCV
└── 6.4 Parallel Parameter Sweep

Phase 7: ML Integration
├── 7.1 Triple Barrier Labeling
├── 7.2 Meta-Labeling (bet sizing)
├── 7.3 Feature Engineering Pipeline
└── 7.4 On-chain Data Integration

Phase 8: Production Readiness
├── 8.1 Live Trading Adapter
├── 8.2 Real-time Risk Monitoring
├── 8.3 Shadow Trading Mode
└── 8.4 Performance Attribution Dashboard
```

---

## References

### VectorBT
- [VectorBT Pro Overview](https://vectorbt.pro/features/overview/)
- [VectorBT Pro Performance](https://vectorbt.pro/features/performance/)
- [VectorBT Pro Optimization](https://vectorbt.pro/features/optimization/)
- [VectorBT Pro Portfolio](https://vectorbt.pro/features/portfolio/)
- [VectorBT Open Source API](https://vectorbt.dev/api/portfolio/base/)
- [From Backtest to Live with VectorBT (2025)](https://medium.com/@samuel.tinnerholm/from-backtest-to-live-going-live-with-vectorbt-in-2025-step-by-step-guide-681ff5e3376e)

### Strategy Discovery & Validation
- [CPCV Method - Towards AI](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)
- [DSR Paper - Bailey & Lopez de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [PBO Paper - SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [Backtest Overfitting Comparison (2024) - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- [Cross Validation with Embargo/Purging - QuantInsti](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
- [Walk-Forward Validation (2024)](https://arxiv.org/html/2512.12924v1)

### Machine Learning & Labeling
- [Triple Barrier & Meta-Labeling - mlfinlab](https://www.mlfinlab.com/en/latest/labeling/tb_meta_labeling.html)
- [Meta-Labeling Efficacy - Hudson & Thames](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [Crypto Triple Barrier + Deep Learning (2025) - Springer](https://link.springer.com/article/10.1186/s40854-025-00866-w)
- [Regime Detection with HMM - QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)

### Architecture & Deployment
- [NautilusTrader Architecture](https://nautilustrader.io/docs/latest/concepts/architecture/)
- [QuantEvolve: Multi-Agent Strategy Discovery (2025)](https://arxiv.org/html/2510.18569v1)
- [Vector vs Event-based Backtesting - IBKR](https://www.interactivebrokers.com/campus/ibkr-quant-news/a-practical-breakdown-of-vector-based-vs-event-based-backtesting/)
- [Best Backtesting Practices for Crypto CTA](https://medium.com/balaena-quant-insights/best-backtesting-practices-for-cta-trading-in-cryptocurrency-e79677cb6375)

### Risk & Cost Modeling
- [Realistic Backtesting Methodology](https://www.hyper-quant.tech/research/realistic-backtesting-methodology)
- [Kelly + VIX Hybrid (2025)](https://arxiv.org/html/2508.16598v1)
- [Crypto Derivatives Market Signals](https://web3.gate.com/crypto-wiki/article/what-are-crypto-derivatives-market-signals-how-funding-rates-open-interest-and-long-short-ratios-predict-price-movements-20260105)
- [ADL Research (2025)](https://arxiv.org/html/2512.01112v2)
- [Funding Rate Arbitrage - Amberdata](https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata)

### Anti-patterns & Statistics
- [Seven Sins of Quantitative Investing](https://bookdown.org/palomar/portfoliooptimizationbook/8.2-seven-sins.html)
- [Haircut Sharpe Ratio - quantstrat](https://rdrr.io/github/braverock/quantstrat/man/SharpeRatio.haircut.html)
- [Harvey & Liu: Backtesting](https://www.cmegroup.com/education/files/backtesting.pdf)
- [Backtesting Statistics - mlfinlab](https://www.mlfinlab.com/en/latest/backtest_overfitting/backtest_statistics.html)
