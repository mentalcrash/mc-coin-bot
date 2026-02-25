# On-chain / Multi-Source 데이터 전략 구현 가이드

> 전략이 OHLCV 외 On-chain, Macro, Options 데이터를 필요로 하는 경우의 구현 패턴.
> 참조 구현: `src/strategy/onchain_accum/` (OnchainAccumStrategy)

---

## 가용 데이터 (60개 데이터셋, 57개 유휴)

### On-chain Global (전 에셋 공통)

| 데이터 | Silver 컬럼 | 소스 | 히스토리 | 해상도 |
|--------|------------|------|---------|:-----:|
| Stablecoin Total Supply | `oc_stablecoin_total_usd` | DeFiLlama | 2019+ | 1D |
| TVL Total | `oc_tvl_usd` | DeFiLlama | 2019+ | 1D |
| DEX Volume | `oc_dex_volume_usd` | DeFiLlama | 2019+ | 1D |
| DeFi Fees Total | `oc_fees_usd` | DeFiLlama | 2019+ | 1D |
| Fear & Greed Index | `oc_fear_greed` | Alternative.me | 2018+ | 1D |

### On-chain BTC/ETH 전용

| 데이터 | Silver 컬럼 | 소스 | 히스토리 |
|--------|------------|------|---------|
| MVRV Ratio | `oc_mvrv` | CoinMetrics | 2009/2015+ |
| Market Cap | `oc_mktcap_usd` | CoinMetrics | 2009/2015+ |
| Exchange Inflow | `oc_flow_in_ex_usd` | CoinMetrics | 2009/2015+ |
| Exchange Outflow | `oc_flow_out_ex_usd` | CoinMetrics | 2009/2015+ |
| Active Addresses | `oc_adractcnt` | CoinMetrics | 2009/2015+ |
| Transaction Count | `oc_txcnt` | CoinMetrics | 2009/2015+ |
| Hash Rate | `oc_hash_rate` | Blockchain.com | 2009+ |
| Supply | `oc_supply` | CoinMetrics | 2009/2015+ |

### Macro (글로벌)

| 데이터 | Silver 컬럼 | 소스 | 히스토리 |
|--------|------------|------|---------|
| DXY (Dollar Index) | `macro_dxy` | FRED | 2006+ |
| VIX | `macro_vix` | FRED | 1990+ |
| 10Y Treasury Yield | `macro_gs10` | FRED | 1962+ |
| M2 Money Supply | `macro_m2` | FRED | 1959+ |
| 10Y Breakeven Inflation | `macro_t10yie` | FRED | 2003+ |
| HY Credit Spread (OAS) | `macro_hy_spread` | FRED | 1996+ |
| Fed Total Assets | `macro_fed_assets` | FRED | 2002+ |
| Initial Jobless Claims | `macro_initial_claims` | FRED | 1967+ |
| Effective Fed Funds Rate | `macro_effr` | FRED | 1954+ |
| WTI Crude Oil | `macro_wti` | FRED | 1986+ |
| CME BTC Futures | `macro_btcf_close` | yfinance | 2017+ |
| iShares Bitcoin Trust (IBIT) | `macro_ibit_close` | yfinance | 2024+ |
| Emerging Markets ETF (EEM) | `macro_eem_close` | yfinance | 2003+ |

### Options (BTC/ETH)

| 데이터 | Silver 컬럼 | 소스 | 히스토리 |
|--------|------------|------|---------|
| DVOL (30D IV) | `opt_dvol` | Deribit | 2021+ |
| Put/Call Ratio | `opt_pc_ratio` | Deribit | 2026+ |

> **에셋 범위 주의**: BTC/ETH 전용 데이터를 사용하는 전략은 5-에셋 범용으로 강제하지 않는다.
> SOL/BNB/DOGE에서는 해당 컬럼이 NaN → Graceful Degradation으로 중립 처리.

---

## Multi-Source Context Architecture (12H + 1D)

### 핵심 개념

```
12H OHLCV  →  가격 시그널 (빠른 반응, 진입/청산 타이밍)
     ↕         merge_asof(direction="backward")로 자동 병합
1D On-chain →  컨텍스트/확신도 (느린 필터, 사이징 가중치)
```

- **가격 시그널(12H)**: 모멘텀, 추세, 변동성 기반 진입/청산 판단
- **On-chain 컨텍스트(1D)**: 시장 구조적 상태(저평가/과열, 자금흐름) → 확신도 가중
- 같은 해상도 불필요: On-chain은 본질적으로 1D 주기 (MVRV, Flow 등 수일~수주 변동)
- `merge_asof(direction="backward")`: 12H 봉에 가장 최근 1D 값 자동 매칭

### 데이터 병합 흐름 (자동)

```
MarketDataService.get(request)
  → _maybe_enrich_onchain(df, request)
    → OnchainDataService.precompute(symbol, ohlcv_index)
      → Silver Parquet 로드 → publication lag shift (T+1)
      → merge_asof(direction="backward")
    → df.join(oc_columns, how="left")
  → MarketDataSet (enriched)
```

- VBT Backtest: 자동 enrichment (추가 코드 불필요)
- EDA Backtest: `BacktestOnchainProvider`가 precomputed data 병합
- Live: `LiveOnchainFeed`가 polling + cache broadcast

---

## required_columns 패턴

**On-chain 컬럼은 `required_columns`에 포함하지 않는다** (Graceful Degradation).

```python
@property
def required_columns(self) -> list[str]:
    # OHLCV만 필수. On-chain은 optional (부재 시 중립 처리)
    return ["close", "high", "low", "volume"]
```

---

## preprocessor.py 패턴

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.{name}.config import {Name}Config

_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})

# On-chain 컬럼 (optional — Graceful Degradation)
_OC_MVRV = "oc_mvrv"
_OC_FLOW_IN = "oc_flow_in_ex_usd"
_OC_FLOW_OUT = "oc_flow_out_ex_usd"
_OC_STABLECOIN = "oc_stablecoin_total_usd"
_OC_FEES = "oc_fees_usd"
_OC_FEAR_GREED = "oc_fear_greed"

# Macro 컬럼 (optional — Graceful Degradation)
_MACRO_DXY = "macro_dxy"
_MACRO_VIX = "macro_vix"
_MACRO_HY_SPREAD = "macro_hy_spread"
_MACRO_FED_ASSETS = "macro_fed_assets"
_MACRO_WTI = "macro_wti"


def preprocess(df: pd.DataFrame, config: {Name}Config) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    close: pd.Series = df["close"]  # type: ignore[assignment]

    # === OHLCV 기반 feature ===
    returns = np.log(close / close.shift(1))
    realized_vol = returns.rolling(config.vol_window).std() * np.sqrt(
        config.annualization_factor
    )
    df["vol_scalar"] = config.vol_target / realized_vol.clip(
        lower=config.min_volatility
    )

    # === On-chain Context (Graceful Degradation) ===
    # 컬럼 부재 시 NaN 유지 → signal.py에서 NaN → 0 (중립) 처리

    # MVRV conviction (BTC/ETH 전용, 타 에셋 NaN)
    if _OC_MVRV in df.columns:
        mvrv: pd.Series = df[_OC_MVRV].ffill()  # type: ignore[assignment]
        # MVRV < 1.0 = 저평가 → 롱 확신↑, MVRV > 3.5 = 과열 → 롱 확신↓
        df["mvrv_conviction"] = np.where(
            mvrv < config.mvrv_undervalued, 1.2,
            np.where(mvrv > config.mvrv_overheated, 0.3, 1.0)
        )
    else:
        df["mvrv_conviction"] = 1.0  # 중립 (Graceful Degradation)

    # Exchange Netflow (BTC/ETH 전용)
    if _OC_FLOW_IN in df.columns and _OC_FLOW_OUT in df.columns:
        flow_in: pd.Series = df[_OC_FLOW_IN].ffill()  # type: ignore[assignment]
        flow_out: pd.Series = df[_OC_FLOW_OUT].ffill()  # type: ignore[assignment]
        net_flow = flow_in - flow_out
        df["netflow_zscore"] = (
            (net_flow - net_flow.rolling(config.flow_window).mean())
            / net_flow.rolling(config.flow_window).std().clip(lower=1e-10)
        )
    else:
        df["netflow_zscore"] = 0.0  # 중립

    # Fear & Greed (전 에셋 공통)
    if _OC_FEAR_GREED in df.columns:
        fg: pd.Series = df[_OC_FEAR_GREED].ffill()  # type: ignore[assignment]
        df["fg_normalized"] = fg / 100.0  # 0~1 정규화
    else:
        df["fg_normalized"] = 0.5  # 중립

    return df
```

---

## signal.py 패턴

```python
def generate_signals(df: pd.DataFrame, config: {Name}Config) -> StrategySignals:
    # Shift(1) Rule: 전봉 데이터 기반
    # On-chain 컬럼도 반드시 shift(1) 적용
    mvrv_conv = df["mvrv_conviction"].shift(1).fillna(1.0)
    netflow_z = df["netflow_zscore"].shift(1).fillna(0.0)
    vol_scalar = df["vol_scalar"].shift(1).fillna(0.0)

    # 가격 시그널 (12H OHLCV 기반)
    price_signal = ...  # 전략별 구현

    # On-chain 컨텍스트로 확신도 가중
    # 가격 시그널 방향은 변경하지 않음, 크기만 조절
    conviction = mvrv_conv  # 추가 컨텍스트 곱산 가능

    strength = price_signal * vol_scalar * conviction
    strength = strength.fillna(0.0)
    ...
```

---

## 테스트 fixture 패턴

```python
@pytest.fixture
def sample_ohlcv_with_onchain() -> pd.DataFrame:
    """OHLCV + On-chain 컬럼 포함 테스트 데이터."""
    np.random.seed(42)
    n = 300
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    # ... (OHLCV 생성) ...
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
        # On-chain columns (optional)
        "oc_mvrv": np.random.uniform(0.5, 4.0, n),
        "oc_flow_in_ex_usd": np.random.uniform(1e6, 1e9, n),
        "oc_flow_out_ex_usd": np.random.uniform(1e6, 1e9, n),
        "oc_fear_greed": np.random.uniform(10, 90, n),
    }, index=pd.date_range("2022-01-01", periods=n, freq="12h"))


@pytest.fixture
def sample_ohlcv_without_onchain() -> pd.DataFrame:
    """OHLCV만 포함 (On-chain 부재) — Graceful Degradation 테스트."""
    np.random.seed(42)
    n = 300
    close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n))
    # ... (OHLCV 생성) ...
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    }, index=pd.date_range("2022-01-01", periods=n, freq="12h"))
```

### 필수 테스트

```python
class TestGracefulDegradation:
    """On-chain 컬럼 부재 시 전략이 중립(0)으로 동작하는지 검증."""

    def test_without_onchain_columns(self, sample_ohlcv_without_onchain):
        """oc_* 컬럼 없이도 에러 없이 실행."""
        config = {Name}Config()
        result = preprocess(sample_ohlcv_without_onchain, config)
        signals = generate_signals(result, config)
        assert signals.strength is not None
        assert not signals.strength.isna().all()

    def test_with_partial_onchain(self, sample_ohlcv_with_onchain):
        """일부 oc_* 컬럼만 있어도 정상 동작."""
        df = sample_ohlcv_with_onchain.drop(columns=["oc_mvrv"])
        config = {Name}Config()
        result = preprocess(df, config)
        assert "mvrv_conviction" in result.columns
        assert (result["mvrv_conviction"] == 1.0).all()  # 중립

    def test_onchain_shift1(self, sample_ohlcv_with_onchain):
        """On-chain 파생 feature도 signal에서 shift(1) 적용 확인."""
        config = {Name}Config()
        result = preprocess(sample_ohlcv_with_onchain, config)
        signals = generate_signals(result, config)
        # 첫 봉 strength = 0 (shift(1) → NaN → 0)
        assert signals.strength.iloc[0] == 0.0
```

---

## 주의사항

1. **publication lag 자동 적용**: OnchainDataService가 T+1 shift 처리. preprocessor에서 추가 shift 불필요
2. **merge_asof 방향**: `direction="backward"` — 미래 데이터 참조 방지 (시스템 레벨)
3. **ffill() 필수**: On-chain 데이터는 주말/공휴일 누락 가능 → `ffill()` 처리
4. **NaN 비율 경고**: enrichment 후 oc_* 컬럼 NaN > 30%이면 시스템 경고 발생
5. **에셋 범위**: BTC/ETH 전용 데이터 → SOL/BNB/DOGE에서 NaN → 반드시 중립 처리
6. **required_columns에 미포함**: On-chain은 optional data. validate_input()에서 검증하지 않음
7. **CLI gap**: `mcbot backtest run`에 on-chain enrichment는 MarketDataService에서 자동 처리됨
