# Derivatives 데이터 전략 구현 가이드

> 전략이 OHLCV 외 Derivatives 데이터를 필요로 하는 경우의 구현 패턴.
> 참조 구현: `src/strategy/funding_carry/` (FundingCarryStrategy)

---

## 가용 데이터

| 데이터 | Silver 컬럼 | 간격 | 백테스팅 | 참고 |
|--------|------------|------|:-------:|------|
| Funding Rate | `funding_rate` | 8h | O | 전체 히스토리 |
| Open Interest | `open_interest` | 1h | X | 30일 제한 |
| LS Ratio | `ls_ratio` | 1h | X | 30일 제한 |
| Taker Ratio | `taker_ratio` | 1h | X | 30일 제한 |

---

## required_columns 패턴

strategy.py의 `required_columns`에 derivatives 컬럼을 추가:

```python
@property
def required_columns(self) -> list[str]:
    return ["close", "high", "low", "volume", "funding_rate"]
```

> BaseStrategy.validate_input()이 자동으로 컬럼 존재 여부 검증.
> MarketDataService.get(include_derivatives=True)로 데이터 로드 시 자동 병합.

---

## preprocessor.py 패턴

```python
_REQUIRED_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "funding_rate"})

def preprocess(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    funding_rate: pd.Series = df["funding_rate"]  # type: ignore[assignment]
    # NaN 처리: merge_asof 후 첫 구간 NaN 가능
    funding_rate = funding_rate.ffill()
    df["avg_funding_rate"] = funding_rate.rolling(config.lookback).mean()
    ...
```

---

## 테스트 fixture 패턴

```python
@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    # ... (기존 OHLCV 생성 코드) ...
    # Funding rate: -0.001 ~ +0.001 범위의 랜덤 값
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volume, "funding_rate": funding_rate},
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
```

---

## 주의사항

- `merge_asof(direction="backward")`: 각 OHLCV 타임스탬프에 가장 가까운 이전 derivatives 값
- 8h FR → 1D TF 사용 시: 하루 3개 FR 중 마지막 값이 매칭됨
- 첫 행 NaN 가능 → `ffill()` 또는 `dropna()` 처리 필요
- **CLI gap**: 현재 `mcbot backtest run`에 `--include-derivatives` 플래그 없음
  → `MarketDataService.get(include_derivatives=True)` 직접 호출 필요
