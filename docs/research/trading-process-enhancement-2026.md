# 자동매매 프로세스 개선 리서치

> **작성일:** 2026-02-13
> **목적:** 현행 시스템 Gap 분석 + 전략 지원 도구/프로세스 개선안 도출
> **범위:** 단일 에셋 Binance USDT-M Futures 기준

---

## Executive Summary

MC Coin Bot은 EDA 아키텍처, 50개 전략, G0~G7 검증 파이프라인, Live 인프라까지 완성도 높은 시스템이다.
그러나 전략이 아직 유의미한 성과를 내지 못하는 근본 원인은 **전략 자체**가 아니라 전략을 **둘러싼 인프라의 Gap**에 있다.

| Gap | 현재 | 문제 |
|-----|------|------|
| 데이터 | OHLCV만 사용 | 크립토 고유 alpha source (Funding Rate, OI) 미활용 |
| Regime | 전략 내부에 매립 | 전략 간 공유 불가, 추세장에서 MR 전략 손실 방치 |
| Position Sizing | 고정 vol-target | 약한 신호에도 풀 사이즈, 신뢰도 무관 |
| Risk 조절 | 10% CB (all-or-nothing) | 점진적 위험 축소 없음 |
| 전략 조합 | 단일 전략 단독 운용 | 38개 전략 자산을 활용하지 못함 |
| 실행 품질 | Market order 일괄 | Slippage 추정/분할 실행 없음 |

**핵심 제안 3가지:**
1. **Market Regime Service** — 모든 전략이 공유하는 시장 상태 판단 인프라
2. **Derivatives Data Pipeline** — Funding Rate / OI / Liquidation 데이터 수집
3. **Drawdown Throttle** — 점진적 리스크 축소 메커니즘

---

## 목차

- [A. 전략 지원 도구 (Feature / Signal Infrastructure)](#a-전략-지원-도구)
  - [A-1. Market Regime Service](#a-1-market-regime-service)
  - [A-2. Derivatives Data Pipeline](#a-2-derivatives-data-pipeline)
  - [A-3. Feature Store](#a-3-feature-store)
- [B. 매매 프로세스 개선 (Execution & Risk)](#b-매매-프로세스-개선)
  - [B-1. Adaptive Position Sizing](#b-1-adaptive-position-sizing)
  - [B-2. Drawdown Throttle](#b-2-drawdown-throttle)
  - [B-3. Slippage Estimator + Smart Execution](#b-3-slippage-estimator--smart-execution)
- [C. 전략 발굴 지원 (Ensemble & Optimization)](#c-전략-발굴-지원)
  - [C-1. Strategy Ensemble Framework](#c-1-strategy-ensemble-framework)
  - [C-2. Walk-Forward Optimizer](#c-2-walk-forward-optimizer)
- [우선순위 매트릭스](#우선순위-매트릭스)
- [권장 로드맵](#권장-로드맵)
- [참고 자료](#참고-자료)

---

## A. 전략 지원 도구

### A-1. Market Regime Service

#### 현재 상태

시스템에 `hmm-regime`, `ac-regime`, `vr-regime`, `hurst-regime` 등 regime 감지 전략이 있으나,
각 전략 **내부에 regime 로직이 매립**되어 있어 다른 전략에서 참조할 수 없다.

```
현재: Strategy A ──(내부 regime)──▶ Signal
      Strategy B ──(내부 regime)──▶ Signal   ← 동일 로직 중복

제안: RegimeService ──▶ regime state
      Strategy A ──(query)──▶ regime-aware Signal
      Strategy B ──(query)──▶ regime-aware Signal
```

#### 핵심 아이디어

Regime을 **전략이 아닌 인프라 레벨 서비스**로 분리하여, 모든 전략이 "지금 시장이 추세장인가, 횡보장인가"를 저비용으로 판단할 수 있게 한다.

#### 제안 인터페이스

```python
# src/market/regime_service.py

class MarketRegime(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"

class RegimeService:
    """전략들이 공유하는 시장 상태 판단 서비스.

    2-layer 구조:
    - Layer 1 (Volatility Regime): GARCH(1,1) 또는 rolling vol percentile
    - Layer 2 (Directional Regime): HMM 3-state (bull/bear/sideways)
    """

    def get_trend_regime(self, symbol: str) -> MarketRegime:
        """현재 추세 regime 반환 (AC + Hurst 기반)."""

    def get_vol_regime(self, symbol: str) -> VolRegime:
        """변동성 regime 반환 (GARCH 또는 rolling percentile)."""

    def get_trend_strength(self, symbol: str) -> float:
        """추세 강도 0~1 반환."""

    def get_composite_regime(self, symbol: str) -> CompositeRegime:
        """종합 판단 (trend + vol 결합)."""
```

#### 전략에서의 활용 예시

```python
# TSMOM 전략이 regime에 따라 행동 변경
class TSMOMStrategy(BaseStrategy):
    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        regime = self._regime_service.get_trend_regime(symbol)
        base_strength = compute_momentum_signal(df)

        # 추세장 → 풀 사이즈, 횡보장 → 사이즈 축소
        if regime == MarketRegime.MEAN_REVERTING:
            strength = base_strength * 0.3  # 70% 축소
        elif regime == MarketRegime.TRENDING_UP:
            strength = base_strength * 1.0  # 풀 사이즈
        ...
```

#### 방법론 비교

| Method | 장점 | 단점 | Crypto 적합도 |
|--------|------|------|---------------|
| **GaussianHMM 3-state** | 확률 기반, bull/bear/sideways 자연 분류 | expanding window 필요, 학습 느림 | **높음** (이미 구현) |
| **GARCH(1,1)** | 빠르고 안정적, vol clustering에 최적 | 방향성 없음 (vol만) | **높음** |
| **Rolling Vol Percentile** | 구현 가장 쉬움, 실시간 가능 | 학술적 근거 약함 | 중간 (baseline) |
| **Efficiency Ratio + Hurst** | 추세 vs MR 분류에 최적 | 두 지표 조합 필요 | 중간 (이미 구현) |
| **Markov Switching GARCH** | 가장 정교한 체제 전환 감지 | 계산 비용 높음, overfitting 위험 | 낮음 |

**권장:** 2-layer 구조

- **Layer 1 - Vol Regime**: GARCH(1,1) (`arch` 패키지). crypto alpha=0.09~0.37, beta=0.7+
- **Layer 2 - Direction Regime**: 기존 HMM 3-state 코드 추출 + expanding window 유지

#### EDA 통합 방안

```
BarEvent → RegimeService._on_bar() → regime 상태 업데이트
                                    → RegimeChangeEvent 발행 (옵션)

StrategyEngine._on_bar() → strategy.generate_signals(df, regime=current_regime)
```

- Backtest: vectorized로 전체 계산 (기존 방식)
- Live: BarEvent handler로 증분 업데이트

#### 기대 효과

- 모든 전략이 regime 정보를 저비용으로 활용
- **횡보장에서 추세 전략 손실 방지** (현재 가장 큰 PnL 드래그 원인)
- 기존 `ac-regime`/`vr-regime`/`hmm-regime` 코드를 추출하여 재사용 가능

#### 구현 난이도: **중** | 예상 기간: 3~5일

---

### A-2. Derivatives Data Pipeline

#### 현재 상태

데이터가 **OHLCV뿐**이다. 크립토 선물 고유 데이터가 빠져있어 전략 발굴 공간이 구조적으로 제한된다.

#### 가용 데이터 및 전략 활용

| 데이터 | Binance API | 주기 | 전략 활용 |
|--------|-------------|------|-----------|
| **Funding Rate** | `GET /fapi/v1/fundingRate` | 8시간 | 극단적 funding → mean-reversion |
| **Open Interest** | `GET /futures/data/openInterestHist` | 5m~1d | OI divergence → accumulation/distribution |
| **Long/Short Ratio** | `GET /futures/data/globalLongShortAccountRatio` | 5m~1d | 극단적 편향 → contrarian |
| **Taker Buy/Sell Volume** | `GET /futures/data/takerlongshortRatio` | 5m~1d | Aggressive flow 방향 |
| **Liquidation** | WebSocket `forceOrder` | 실시간 | 대량 청산 → 단기 반전 기회 |

#### Funding Rate — 크립토 검증된 Alpha Source

학술/실무 양쪽에서 검증된 크립토 고유 edge:

```python
# 핵심 로직
# funding_rate > +0.05% → 시장 과열 (롱 과다) → 숏 우위
# funding_rate < -0.03% → 과매도 (숏 과다) → 롱 우위 (short squeeze)

def funding_rate_signal(funding_rate: pd.Series) -> pd.Series:
    """Funding rate mean-reversion signal."""
    signal = pd.Series(0.0, index=funding_rate.index)
    signal[funding_rate > 0.0005] = -1.0   # 극단적 롱 → 숏
    signal[funding_rate < -0.0003] = 1.0    # 극단적 숏 → 롱
    return signal
```

- 2020~2024 기간 연 15~25% 리턴 보고 (비용 차감 후)
- 55~65% 방향 예측 정확도
- 8시간 주기 → 일봉 전략과 자연스럽게 호환

#### OI Divergence Signal

```python
# OI 증가 + 가격 하락 → hidden buying (accumulation) → 롱
# OI 감소 + 가격 상승 → short covering (약한 상승) → 경계
# OI 급등 + 가격 급등 → leverage buildup → liquidation cascade 위험

def oi_divergence_signal(
    close: pd.Series, oi: pd.Series, lookback: int = 24
) -> pd.Series:
    oi_change = oi.pct_change(periods=lookback)
    price_change = close.pct_change(periods=lookback)

    signal = pd.Series(0.0, index=close.index)
    # Hidden buying: OI↑ + Price↓
    signal[(oi_change > 0.05) & (price_change < -0.02)] = 1.0
    # Leverage flush risk: OI↑↑ + Price↑↑
    signal[(oi_change > 0.15) & (price_change > 0.10)] = -0.5
    return signal
```

#### Liquidation Cascade Detection

```python
# OI 급감 + 가격 급락 = forced liquidation → capitulation → reversal entry

def liquidation_cascade_signal(
    close: pd.Series, oi: pd.Series, window: int = 8
) -> pd.Series:
    oi_change = oi.pct_change(periods=window)
    price_change = close.pct_change(periods=window)

    # OI -10% 이상 감소 + 가격 -5% 이상 하락 → 강제 청산 추정
    cascade = (oi_change < -0.10) & (price_change < -0.05)
    return cascade.astype(float)  # 1.0 = reversal 기회
```

#### CCXT 지원 현황

```python
# Funding Rate (CCXT 네이티브 지원)
rates = await exchange.fetch_funding_rate_history('BTC/USDT:USDT')

# Open Interest (CCXT 네이티브 지원)
oi = await exchange.fetch_open_interest('BTC/USDT:USDT')

# Historical OI (Binance 전용 endpoint — CCXT에서 직접 미지원)
# → BinanceFuturesClient에 래퍼 추가 필요
oi_hist = await exchange._exchange.fapipublic_get_futures_data_openinteresthist({
    'symbol': 'BTCUSDT', 'period': '1h', 'limit': 500
})

# Long/Short Ratio (Binance 전용)
ls_ratio = await exchange._exchange.fapipublic_get_futures_data_globallongshortaccountratio({
    'symbol': 'BTCUSDT', 'period': '1h', 'limit': 500
})
```

#### 데이터 저장 구조

```
data/silver/{SYMBOL}/{YEAR}.parquet     ← 기존 OHLCV
data/silver/{SYMBOL}/{YEAR}_deriv.parquet ← 신규 Derivatives
  columns: [funding_rate, open_interest, long_short_ratio, taker_buy_sell_ratio]
```

또는 기존 Silver 파일에 추가 컬럼으로 병합:

```
columns: [open, high, low, close, volume, funding_rate, open_interest, ...]
```

#### 통합 방안

```
# 데이터 수집 (CLI)
uv run mcbot ingest derivatives BTC/USDT --year 2024 --year 2025

# 백테스트에서 자동 로드
MarketDataService.get(symbol, include_derivatives=True) → DataFrame with extra columns

# Live에서 주기적 fetch
LiveDataFeed에 fetch_derivatives_periodically() task 추가 (8시간 또는 1시간 주기)
```

#### 기대 효과

- OHLCV만으로 불가능한 **크립토 고유 alpha source** 확보
- Funding Rate 전략만으로 연 15~25% 기대 수익
- 기존 전략에 OI/Funding 조건을 **필터**로 추가하여 신호 품질 향상

#### 구현 난이도: **하** | 예상 기간: 2~3일

---

### A-3. Feature Store

#### 현재 상태

50개 전략이 `preprocess()`에서 **각각 독립적으로** ATR, RSI, EMA, Bollinger Band 등을 계산한다.
같은 데이터에 대해 동일 지표가 중복 계산된다.

#### 제안 아키텍처

```python
# src/market/feature_store.py

class FeatureStore:
    """지연 계산 + 캐시 기반 공통 지표 저장소.

    BAR 이벤트를 구독하여 공통 feature를 캐싱합니다.
    전략은 preprocess()에서 직접 계산 대신 store에서 조회합니다.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, tuple], pd.Series] = {}

    def atr(self, symbol: str, period: int = 14) -> pd.Series:
        """Average True Range (캐시됨)."""
        return self._get_or_compute(symbol, "atr", (period,))

    def rsi(self, symbol: str, period: int = 14) -> pd.Series:
        """Relative Strength Index (캐시됨)."""
        return self._get_or_compute(symbol, "rsi", (period,))

    def realized_vol(self, symbol: str, window: int = 30) -> pd.Series:
        """Annualized Realized Volatility (캐시됨)."""
        return self._get_or_compute(symbol, "realized_vol", (window,))

    def ema(self, symbol: str, span: int = 20) -> pd.Series:
        """Exponential Moving Average (캐시됨)."""
        return self._get_or_compute(symbol, "ema", (span,))

    def hurst_exponent(self, symbol: str, window: int = 100) -> pd.Series:
        """Hurst Exponent (캐시됨)."""
        return self._get_or_compute(symbol, "hurst", (window,))

    def autocorrelation(self, symbol: str, lag: int = 1) -> pd.Series:
        """Rolling Autocorrelation (캐시됨)."""
        return self._get_or_compute(symbol, "autocorr", (lag,))

    def _get_or_compute(
        self, symbol: str, name: str, params: tuple
    ) -> pd.Series:
        key = (symbol, name, params)
        if key not in self._cache:
            self._cache[key] = self._compute(symbol, name, params)
        return self._cache[key]
```

#### 주의사항 (MEMORY 교훈)

- **EWM + buffer truncation 금지**: EWM은 full history 의존. 버퍼 truncation 시 초기화 리셋되어 값 변질
- **Backtest vs Live 분리**: 백테스트는 전체 데이터 한번에 계산 (현재 방식 유지), Live에서만 증분 캐싱

#### 기대 효과

- 전략 코드 간소화 (`preprocess`에서 `store.rsi(14)` 한 줄)
- EDA Live 모드에서 중복 계산 제거 → 레이턴시 감소
- 새 전략 개발 시 빌딩블록 역할

#### 구현 난이도: **중** | 예상 기간: 3~5일 (점진적 마이그레이션)

---

## B. 매매 프로세스 개선

### B-1. Adaptive Position Sizing

#### 현재 상태

```python
# 현재 PM의 사이징 로직
target_weight = signal.strength  # vol_target / realized_vol
clamped = clamp(target_weight, -max_leverage_cap, max_leverage_cap)  # 3.0x
```

**문제:** 모든 신호에 동일 확신도 부여 → 약한 신호에도 풀 사이즈 진입

#### 방법 (a): Fractional Kelly Criterion

```python
def fractional_kelly(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.5,  # Half Kelly 권장
) -> float:
    """Fractional Kelly position size.

    Kelly 공식: f* = (p * b - q) / b
      p = 승률, b = 평균 승/패 비율, q = 1-p

    Half Kelly: 75% growth rate 유지, MDD 50% 감소.

    Args:
        win_rate: 최근 N 거래의 승률 (0~1)
        win_loss_ratio: 평균 이익 / 평균 손실
        fraction: Kelly 비율 (0.5 = Half Kelly)

    Returns:
        최적 포지션 비율 (0~1)
    """
    q = 1.0 - win_rate
    kelly_full = (win_rate * win_loss_ratio - q) / win_loss_ratio
    return max(0.0, kelly_full * fraction)
```

- Rolling window (최근 60~120 거래)로 `win_rate`, `win_loss_ratio` 업데이트
- Half Kelly가 Full Kelly 대비 **MDD 절반, 성장률 75%** (학술 검증)

#### 방법 (b): Drawdown-Adjusted Sizing

```python
def drawdown_adjusted_size(
    base_size: float,
    current_dd: float,
    max_dd: float,
) -> float:
    """Drawdown 비례 포지션 축소.

    Drawdown이 깊어질수록 포지션을 줄여 recovery 확률을 높인다.

    Args:
        base_size: 기본 포지션 크기
        current_dd: 현재 drawdown (0~1, 양수)
        max_dd: 최대 허용 drawdown (시스템 CB 수준)

    Returns:
        조정된 포지션 크기
    """
    if current_dd <= 0:
        return base_size
    scale = max(0.0, 1.0 - (current_dd / max_dd))
    return base_size * scale
```

기존 `EDARiskManager._peak_equity` 추적과 자연스럽게 연동.

#### 방법 (c): Signal Confidence Sizing

```python
# Regime + signal strength + 최근 적중률을 종합한 신뢰도
def compute_confidence(
    regime: MarketRegime,
    signal_strength: float,
    recent_hit_rate: float,
) -> float:
    """멀티 팩터 신뢰도 계산 (0.2 ~ 1.0).

    구성:
        - Regime 정합성 (40%): 추세장에서 추세 전략 = 높은 신뢰도
        - Signal 강도 (30%): 약한 신호 = 낮은 신뢰도
        - 최근 적중률 (30%): 실적 기반 피드백
    """
    regime_score = 1.0 if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN) else 0.5
    strength_score = min(abs(signal_strength) / 1.5, 1.0)
    hit_rate_score = max(recent_hit_rate - 0.4, 0) / 0.2

    raw = regime_score * 0.4 + strength_score * 0.3 + hit_rate_score * 0.3
    return 0.2 + 0.8 * min(raw, 1.0)  # 최소 20% 보장
```

#### PM 통합 포인트

```python
# EDAPortfolioManager._evaluate_rebalance() 내부
target_weight = strategy_signal.strength           # 이미 존재
kelly_scale = fractional_kelly(win_rate, wl_ratio) # 추가
dd_scale = drawdown_adjusted_size(1.0, dd, max_dd) # 추가
confidence = compute_confidence(regime, strength, hit_rate)  # 추가

final_weight = target_weight * kelly_scale * dd_scale * confidence
```

#### 기대 효과

- 약한 신호 → 작은 포지션 → 손실 시 피해 축소
- Regime과 불일치하는 신호 자동 축소
- **MDD 개선** (SOL sweep에서 SL보다 사이즈 조절이 MDD에 더 큰 영향)

#### 구현 난이도: **중** | 예상 기간: 3~4일

---

### B-2. Drawdown Throttle

#### 현재 상태

```
현재 RM: 고정 10% system stop-loss → circuit breaker (all-or-nothing)

Equity 추이: 100K → 97K → 95K → 93K → 91K → 90K (CB 발동, 전량 청산)
                                                     ↑ 갑작스러운 청산
```

**문제:** 5% 손실과 9% 손실에서 동일하게 풀 사이즈 운용 → 10%에서 갑자기 전체 청산

#### 제안: 점진적 리스크 감소

```python
class DrawdownThrottle:
    """Drawdown 깊이에 따라 포지션 사이즈를 점진적으로 축소.

    기존 circuit breaker(10% → 전량 청산)를 보완하여,
    그 이전 단계에서 노출을 줄여 CB 트리거 빈도를 감소시킨다.

    Equity 추이 예시:
        100K → 97K(풀) → 95K(70%) → 93K(40%) → 91K(청산)
                              ↑ 점진적 축소 시작
    """

    # (drawdown 임계값, 포지션 스케일 팩터)
    TIERS: list[tuple[float, float]] = [
        (0.03, 1.0),   # DD < 3%  → 풀 사이즈 (100%)
        (0.05, 0.7),   # DD 3~5%  → 70%
        (0.07, 0.4),   # DD 5~7%  → 40%
        (0.10, 0.0),   # DD > 10% → 전량 청산 (기존 CB)
    ]

    def scale_factor(self, current_drawdown: float) -> float:
        """현재 drawdown에 대한 포지션 스케일 팩터 반환.

        Args:
            current_drawdown: 현재 peak 대비 drawdown (0~1, 양수)

        Returns:
            0.0 ~ 1.0 범위의 스케일 팩터
        """
        for dd_threshold, factor in self.TIERS:
            if current_drawdown < dd_threshold:
                return factor
        return 0.0  # CB 수준 초과
```

#### RM 통합 포인트

```python
# EDARiskManager._on_order_request() 내부
throttle = DrawdownThrottle()
dd = (self._peak_equity - current_equity) / self._peak_equity
scale = throttle.scale_factor(dd)

if scale == 0.0:
    # 기존 circuit breaker 로직
    await self._trigger_circuit_breaker()
elif scale < 1.0:
    # 주문 크기 스케일링
    adjusted_order = order.with_scaled_quantity(scale)
    await self._bus.publish(adjusted_order)
```

#### 수치 비교

| 시나리오 | 현재 (CB only) | 제안 (Throttle + CB) |
|----------|---------------|---------------------|
| DD 4% | 풀 사이즈 운용 | 70%로 축소 |
| DD 6% | 풀 사이즈 운용 | 40%로 축소 |
| DD 8% | 풀 사이즈 운용 | 40%로 축소 (추가 손실 제한) |
| DD 10% | 전량 청산 | 전량 청산 (이미 40%만 노출) |
| **실효 MDD** | ~10% | **~6~7%** (40% 포지션에서의 추가 손실 제한) |

#### 기대 효과

- CB 전에 자동으로 익스포저 축소 → **CB 트리거 빈도 감소**
- 손실 회복이 빠름 (40% 사이즈에서의 -3%는 풀 사이즈의 -1.2%)
- 기존 CB 로직과 호환 (CB는 최후 방어선으로 유지)

#### 구현 난이도: **하** | 예상 기간: 1일 (RM에 ~20줄 추가)

---

### B-3. Slippage Estimator + Smart Execution

#### 현재 상태

- 백테스트: 고정 slippage (`CostModel.slippage_bps=5`)
- Live: Market order 즉시 체결, 실제 slippage 미추적

#### Slippage Estimator (사전 추정)

```python
class SlippageEstimator:
    """최근 체결 데이터 기반 실시간 slippage 추정.

    주문 크기와 시장 유동성을 기반으로 예상 slippage를 계산하고,
    주문 타입 추천을 제공한다.
    """

    # 최근 fill의 실제 slippage 추적 (rolling)
    _fill_history: list[FillSlippage]

    def estimate(self, symbol: str, order_size_usd: float) -> float:
        """예상 slippage (bps) 반환.

        단일 asset 기준:
        - $10K 이하:   ~1-2 bps (무시 가능)
        - $10K-$100K:  ~2-5 bps
        - $100K+:      order book depth 기반 계산 필요
        """

    def recommend_order_type(
        self, size_usd: float, urgency: Literal["high", "normal", "low"]
    ) -> OrderType:
        """주문 크기와 긴급도에 따라 MARKET/LIMIT/TWAP 추천."""
        if urgency == "high":
            return OrderType.MARKET
        if size_usd > 50_000:
            return OrderType.TWAP
        return OrderType.LIMIT
```

#### TWAP Executor (대형 주문 분할)

```python
class TWAPExecutor:
    """대형 주문을 N개 슬라이스로 시간 분할 실행.

    VWAP 대비 장점: volume curve 예측 불필요 (crypto 24/7).
    단일 거래소(Binance)만 사용하므로 smart order routing 불필요.
    """

    async def execute_twap(
        self,
        symbol: str,
        total_size: float,
        duration_minutes: int = 10,
        n_slices: int = 5,
    ) -> list[FillEvent]:
        slice_size = total_size / n_slices
        interval = duration_minutes * 60 / n_slices
        fills = []
        for _ in range(n_slices):
            fill = await self._execute_single(symbol, slice_size)
            fills.append(fill)
            await asyncio.sleep(interval)
        return fills
```

#### Post-trade TCA (Transaction Cost Analysis)

```python
@dataclass(frozen=True)
class TCAReport:
    """체결 품질 분석."""
    arrival_price: float       # 주문 시점 mid price
    execution_price: float     # 실제 체결 가격
    slippage_bps: float        # 슬리피지 (basis points)
    market_impact_bps: float   # 시장 충격
    execution_time_ms: float   # 체결 소요 시간
```

#### ExecutorPort 통합

```python
# 기존 ExecutorPort에 크기 기반 자동 선택 로직
order_size < $50K → market order (현재 방식)
order_size >= $50K → TWAP 분할 실행
```

#### 기대 효과

- BTC/USDT는 슬리피지 최소이나, SOL/DOGE 등 중소형은 유의미
- 큰 포지션 변동 시 (방향 전환) 시장 충격 최소화
- TCA 데이터 축적 → 백테스트 slippage 모델 정교화

#### 구현 난이도: **중** | 예상 기간: 3~5일
#### 참고: 단일 에셋 + 소규모 자본이면 급하지 않음. 자본 성장 시 중요도 증가

---

## C. 전략 발굴 지원

### C-1. Strategy Ensemble Framework

#### 현재 상태

50개 전략이 있지만 **개별 전략을 단독 운용**한다.
Tier 2 백테스트에서 이미 확인된 교훈: "단일지표 < 앙상블, 다양성이 알파"

#### 핵심 아이디어

개별 Sharpe 0.5인 전략 3개 → 앙상블 Sharpe 0.8~1.0 가능.
전략 발굴의 목표를 **"단독 우승"** → **"앙상블 기여도"**로 변경.

#### 가중치 방법론 비교

| 방법 | 복잡도 | Robust | 과적합 위험 | 권장 순서 |
|------|--------|--------|------------|-----------|
| **Equal Weight (EW)** | 최저 | 높음 | 없음 | Baseline |
| **Inverse-Volatility** | 낮음 | 높음 | 없음 | **1순위** |
| **Strategy Momentum** | 중간 | 중간 | 낮음 | 2순위 |
| **Meta-Labeling** | 높음 | — | 중간 | 3순위 (데이터 충분 시) |

#### 방법 (a): Inverse-Volatility Weighting (1순위)

```python
def inverse_vol_weights(
    strategy_returns: pd.DataFrame,
    lookback: int = 63,  # ~3개월
) -> pd.DataFrame:
    """최근 수익률 변동성의 역수로 가중치 계산.

    변동성이 높은 전략 → 비중 축소
    변동성이 낮은 전략 → 비중 증가

    가장 단순하고 robust한 방법. 과적합 위험 없음.

    Args:
        strategy_returns: 전략별 일간 수익률 (columns = 전략명)
        lookback: 변동성 계산 기간

    Returns:
        전략별 정규화 가중치 (합 = 1.0)
    """
    vol = strategy_returns.rolling(lookback).std()
    inv_vol = 1.0 / vol.clip(lower=1e-8)
    return inv_vol.div(inv_vol.sum(axis=1), axis=0)
```

#### 방법 (b): Strategy Momentum Weighting (2순위)

```python
def strategy_momentum_weights(
    strategy_returns: pd.DataFrame,
    lookback: int = 126,  # ~6개월
    top_n: int = 5,
) -> pd.DataFrame:
    """최근 N일 Sharpe가 높은 상위 K개 전략에 집중.

    "최근 잘한 전략이 계속 잘 한다"는 가설.
    전략 momentum은 약하지만 존재하는 것으로 보고됨.

    Args:
        strategy_returns: 전략별 일간 수익률
        lookback: Sharpe 계산 기간
        top_n: 선택할 상위 전략 수

    Returns:
        상위 전략에만 가중치 할당 (나머지 0)
    """
    rolling_sharpe = (
        strategy_returns.rolling(lookback).mean()
        / strategy_returns.rolling(lookback).std()
    )
    ranks = rolling_sharpe.rank(axis=1, ascending=False)
    selected = (ranks <= top_n).astype(float)
    return selected.div(selected.sum(axis=1), axis=0)
```

#### 방법 (c): Regime-Adaptive Weighting

```python
def regime_adaptive_weights(
    strategy_returns: pd.DataFrame,
    regime: pd.Series,  # RegimeService 출력
    strategy_types: dict[str, Literal["trend", "mr", "neutral"]],
) -> pd.DataFrame:
    """Regime에 따라 추세/역추세 전략 가중치 동적 변경.

    - TRENDING regime → 추세 전략 비중 증가, MR 전략 감소
    - MEAN_REVERTING regime → MR 전략 비중 증가, 추세 전략 감소

    RegimeService(A-1)에 의존.
    """
    weights = pd.DataFrame(0.0, index=regime.index, columns=strategy_returns.columns)
    for name, stype in strategy_types.items():
        if stype == "trend":
            weights[name] = np.where(
                regime.isin(["trending_up", "trending_down"]), 0.6, 0.2
            )
        elif stype == "mr":
            weights[name] = np.where(
                regime == "mean_reverting", 0.6, 0.2
            )
        else:
            weights[name] = 0.4  # neutral은 항상 중립
    # 정규화
    return weights.div(weights.sum(axis=1), axis=0)
```

#### 방법 (d): Meta-Labeling (3순위, 고급)

```python
class MetaLabeler:
    """Marcos Lopez de Prado의 Meta-Labeling 간소화 구현.

    1. Primary Model (M1): 기존 전략이 방향 시그널 생성 (이미 구현)
    2. Secondary Model (M2): Binary classifier가 각 시그널의 수익 여부 예측
    3. Position Sizing: M2의 확률을 포지션 크기로 변환

    필요: 충분한 거래 데이터 (최소 500+ 거래)
    """

    def __init__(self, features: list[str]) -> None:
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1
        )
        self.features = features

    def fit(self, X: pd.DataFrame, strategy_returns: pd.Series) -> None:
        """Triple Barrier labeling으로 학습."""
        labels = self._triple_barrier_label(strategy_returns)
        self.model.fit(X[self.features], labels)

    def predict_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """시그널 수익 확률 반환 → position size로 사용."""
        return self.model.predict_proba(X[self.features])[:, 1]
```

#### EDA 통합 구조

```
현재:
  StrategyEngine (1개) → SignalEvent → PM

제안:
  StrategyEngine_A → Signal_A ─┐
  StrategyEngine_B → Signal_B ──┼─▶ StrategyAllocator → CombinedSignal → PM
  StrategyEngine_C → Signal_C ─┘
```

`StrategyAllocator`는 EventBus에서 SignalEvent를 수집하고, 가중치를 적용하여 결합된 SignalEvent를 PM에 전달.

#### 기대 효과

- 50개 전략 자산을 최대한 활용
- 개별 Sharpe 0.5인 전략도 앙상블에서 가치 있음
- Regime-adaptive weighting으로 시장 상황에 맞는 전략 자동 선택

#### 구현 난이도: **중** | 예상 기간: 5~7일

---

### C-2. Walk-Forward Optimizer

#### 현재 상태

파라미터는 백테스트에서 한번 결정 → 라이브에서 **고정**.
시장 구조 변화 시 파라미터가 최적이 아닐 수 있다.

#### 제안

```python
class WalkForwardOptimizer:
    """Rolling window 기반 파라미터 자동 적응.

    기존 인프라 재사용:
    - BacktestEngine.sweep() → 파라미터 탐색
    - TieredValidator → G3 robustness 검증

    주기적으로 (월 1회 등) 최적 파라미터를 재탐색하여
    시장 구조 변화에 대응한다.
    """

    def __init__(
        self,
        strategy_name: str,
        lookback_days: int = 365,
        step_days: int = 30,
    ) -> None:
        self.strategy_name = strategy_name
        self.lookback_days = lookback_days
        self.step_days = step_days

    async def reoptimize(self) -> OptimizationResult:
        """최적 파라미터 재탐색 프로세스.

        1. 최근 lookback_days 데이터 로드
        2. Parameter sweep (기존 BacktestEngine.sweep 재사용)
        3. Robust plateau 확인 (기존 G3 검증 재사용)
        4. 변경이 유의미하면 전략 config 업데이트 제안
        """

    def should_reoptimize(self, last_optimized: datetime) -> bool:
        """재최적화 필요 여부 판단."""
        days_since = (datetime.now(UTC) - last_optimized).days
        return days_since >= self.step_days
```

#### 안전장치

- 자동 적용 없음 → **제안만** (사용자 승인 필수)
- G3 plateau 검증 통과 시에만 제안
- 변경 전후 Sharpe 비교 리포트 생성

#### 기대 효과

- 시장 구조 변화에 점진적 적응
- 기존 sweep/validation 인프라 재사용으로 구현 비용 낮음

#### 구현 난이도: **상** | 예상 기간: 1~2주

---

## 우선순위 매트릭스

| 순위 | 항목 | 임팩트 | 구현 난이도 | 기존 코드 재사용 | 예상 기간 |
|------|------|--------|-------------|----------------|-----------|
| **1** | A-1 Regime Service | ★★★★★ | 중 | HMM/AC/VR 추출 | 3~5일 |
| **2** | A-2 Derivatives Data | ★★★★★ | **하** | BinanceFuturesClient 확장 | 2~3일 |
| **3** | B-2 Drawdown Throttle | ★★★★ | **하** | RM에 ~20줄 추가 | 1일 |
| **4** | C-1 Ensemble Framework | ★★★★ | 중 | 50개 전략 재활용 | 5~7일 |
| **5** | B-1 Adaptive Sizing | ★★★★ | 중 | PM에 통합 | 3~4일 |
| **6** | A-3 Feature Store | ★★★ | 중 | 점진적 마이그레이션 | 3~5일 |
| **7** | B-3 Slippage/TWAP | ★★★ | 중 | ExecutorPort 확장 | 3~5일 |
| **8** | C-2 WF Optimizer | ★★ | 상 | sweep/validation 재사용 | 1~2주 |

---

## 권장 로드맵

### Phase 1: 즉시 (1~2주)

**목표:** 기존 시스템에 최소 변경으로 최대 효과

| # | 항목 | 작업 내용 | 기간 |
|---|------|----------|------|
| 1 | **Drawdown Throttle** | `EDARiskManager`에 tier 기반 스케일링 추가 | 1일 |
| 2 | **Derivatives Data** | `BinanceFuturesClient`에 funding rate/OI 래퍼 + Silver 저장 | 2~3일 |
| 3 | **Regime Service (기본)** | 기존 AC/Vol Regime 코드 추출 → 서비스화 | 3~5일 |

**Phase 1 완료 시 기대 효과:**
- MDD 10% → 6~7% (Throttle)
- 전략 발굴 공간 확장 (Funding Rate/OI 데이터)
- 기존 전략에 regime 필터 추가 가능

### Phase 2: 2~4주

**목표:** 전략 조합 및 사이징 고도화

| # | 항목 | 작업 내용 | 기간 |
|---|------|----------|------|
| 4 | **Ensemble Framework** | Inverse-Vol + Strategy Momentum allocator | 5~7일 |
| 5 | **Adaptive Sizing** | Half Kelly + Confidence scoring | 3~4일 |
| 6 | **Regime Service (고급)** | GARCH vol regime + HMM 통합 | 3~5일 |

**Phase 2 완료 시 기대 효과:**
- 50개 전략의 앙상블 활용 → Sharpe 개선
- 신뢰도 기반 사이징으로 약한 신호 손실 축소

### Phase 3: 1~2개월

**목표:** 실행 품질 및 자동 적응

| # | 항목 | 작업 내용 | 기간 |
|---|------|----------|------|
| 7 | **Feature Store** | 공통 지표 캐시 + 전략 점진적 마이그레이션 | 3~5일 |
| 8 | **Slippage/TWAP** | 크기 기반 주문 분할 + TCA | 3~5일 |
| 9 | **WF Optimizer** | 월간 파라미터 재탐색 자동화 | 1~2주 |

---

## 참고 자료

### Regime Detection
- [Step-by-Step Python Guide for Regime-Specific Trading Using HMM](https://blog.quantinsti.com/regime-adaptive-trading-python/)
- [Market Regime Detection using Hidden Markov Models | QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Bitcoin Price Regime Shifts: Bayesian MCMC and HMM Analysis](https://www.mdpi.com/2227-7390/13/10/1577)
- [GARCH Volatility Regime-Switch Detection (GitHub)](https://github.com/etatx0/Regime-Switch)

### Position Sizing
- [Risk-Constrained Kelly Criterion | QuantInsti](https://blog.quantinsti.com/risk-constrained-kelly-criterion/)
- [Position Sizing Strategies | QuantifiedStrategies](https://www.quantifiedstrategies.com/position-sizing-strategies/)
- [Sizing the Risk: Kelly, VIX, and Hybrid Approaches (2025)](https://arxiv.org/html/2508.16598v1)

### Market Microstructure (Funding Rate / OI)
- [How Funding Rates, Open Interest Predict Crypto Market Signals (2026)](https://dex.gate.com/crypto-wiki/article/how-do-futures-open-interest-funding-rates-and-liquidation-data-predict-crypto-derivatives-market-signals-in-2026-20260111)
- [Funding Rates: The Hidden Cost, Sentiment Signal, and Strategy Trigger](https://quantjourney.substack.com/p/funding-rates-in-crypto-the-hidden)
- [Crypto Derivatives Market Signals Predict Price Movements (2026)](https://www.gate.com/crypto-wiki/article/how-do-crypto-derivatives-market-signals-predict-price-movements-futures-open-interest-funding-rates-liquidation-data-long-short-ratio-and-options-explained-20260129)

### Execution Quality
- [Deep Learning for VWAP Execution in Crypto Markets (2025)](https://arxiv.org/html/2502.13722v1)
- [Comparing Global VWAP and TWAP for Better Trade Execution](https://blog.amberdata.io/comparing-global-vwap-and-twap-for-better-trade-execution)
- [Talos Market Impact Model for Crypto](https://www.talos.com/insights/understanding-market-impact-in-crypto-trading-the-talos-model-for-estimating-execution-costs)
- [TWAP and VWAP Strategies Minimize Market Impact in Crypto](https://www.ainvest.com/news/twap-vwap-strategies-minimize-market-impact-crypto-trading-2504-59/)

### Strategy Ensemble
- [Meta-Labeling: The Technique That Transformed Modern Quant Trading](https://whatworksintrading.substack.com/p/meta-labeling-the-technique-that)
- [Ensemble Methods for Stock & Crypto Trading (2025)](https://arxiv.org/html/2501.10709v1)
- [Meta-Labeling - Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling)

### Risk Management
- [Mapping Systemic Tail Risk in Crypto Markets](https://www.mdpi.com/1911-8074/18/6/329)
- [Correlation Risk Management Across Multiple Algorithms](https://breakingalpha.io/insights/correlation-risk-management-multiple-algorithms)

### Feature Store
- [Top 5 Feature Stores in 2025](https://www.gocodeo.com/post/top-5-feature-stores-in-2025-tecton-feast-and-beyond)

### Order Flow
- [Price Impact of Order Book Imbalance in Cryptocurrency Markets](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/)
- [VPIN Predicts Future Price Jumps in Bitcoin](https://www.sciencedirect.com/science/article/pii/S0275531925004192)
