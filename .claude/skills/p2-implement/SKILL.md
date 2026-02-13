---
name: p2-implement
description: >
  p1-g0a-discover에서 발굴한 전략 후보를 무결점 코드로 구현하는 스킬.
  시니어 퀀트 개발자로서 4-file 구조(config/preprocessor/signal/strategy) + 테스트를
  Zero-Tolerance Lint Policy에 맞춰 작성하고, Registry 등록 + Dashboard 갱신까지 완수한다.
  사용 시점: (1) strategies/*.yaml에 CANDIDATE 전략이 있을 때,
  (2) "전략 구현", "implement", "코드 작성" 요청 시,
  (3) p1-g0a-discover 완료 후 다음 단계로,
  (4) G0A PASS 전략의 코드화가 필요할 때.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name>
---

# p2-implement: 전략 코드 구현

## 역할

**시니어 퀀트 개발자**로서 행동한다.
판단 기준: **"이 코드가 실제 돈을 운용하는 프로덕션 시스템에 배포되어도 안전한가?"**

핵심 원칙:

- **정확성 > 속도**: 한 줄의 shift(1) 누락이 수백만 원 차이를 만든다
- **벡터화 필수**: 모든 연산은 pandas/numpy 벡터 연산 (루프 절대 금지)
- **방어적 코딩**: NaN, 0 나눗셈, edge case를 사전 차단
- **일관된 패턴**: 기존 전략과 동일한 코드 구조 유지
- **테스트 우선**: 구현 완료 → 테스트 작성 → lint/typecheck/test 통과 → 등록

---

## 인수 파싱

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `accel-conv` |
| | | | |

> `strategy_name`은 kebab-case (e.g., `accel-conv`), 디렉토리명은 snake_case (e.g., `accel_conv`).

---

## Step 0: 후보 정보 수집

### 0-1. YAML에서 후보 로드

```bash
cat strategies/{strategy_name}.yaml
# meta.status가 CANDIDATE이고 gates.G0A.status가 PASS인지 확인
# YAML이 없으면: "p1-g0a-discover에서 pipeline create를 먼저 실행하세요"
```

> discovery에서 YAML을 이미 생성했으므로, YAML이 없으면 workflow 순서 오류.

다음 정보를 YAML에서 추출한다:

| 항목 | 필수 | 용도 |
|------|:----:|------|
| 전략명 (registry key) | O | `@register("name")` |
| 카테고리 | O | 코드 docstring |
| 타임프레임 | O | `annualization_factor` 결정 |
| ShortMode | O | signal.py 분기 로직 |
| Gate 0 점수 | O | YAML 메타데이터 기록 |
| 핵심 가설 | O | 코드 docstring + 주석 |
| 사용 지표 | O | preprocessor.py 구현 |
| 시그널 생성 로직 | O | signal.py 구현 |
| 차별화 포인트 | O | 폐기 전략과 구분 확인 |

### 0-2. annualization_factor 결정

| TF | annualization_factor | bars/year |
|-----|---------------------|-----------|
| 1D | 365.0 | 365 |
| 12H | 730.0 | 730 |
| 8H | 1095.0 | 1095 |
| 6H | 1460.0 | 1460 |
| 4H | 2190.0 | 2190 |
| 1H | 8760.0 | 8760 |

> **Critical**: annualization_factor 오류는 vol_scalar를 왜곡하여 포지션 사이징에 직접 영향.
> `realized_vol = returns.rolling(N).std() * sqrt(annualization_factor)`

### 0-3. 기존 전략과 중복 확인

```bash
# Registry에 이미 등록된 이름인지 확인
uv run python -c "from src.strategy import list_strategies; print(list_strategies())"
```

중복 시 중단하고 사용자에게 보고.

### 0-4. 폐기 전략 패턴 확인

[references/implementation-checklist.md](references/implementation-checklist.md)의 anti-pattern 목록 참조.
후보의 시그널 로직이 폐기된 전략의 실패 패턴과 유사하지 않은지 최종 확인.

---

## Step 1: 디렉토리 구조 생성

```bash
# 디렉토리 생성 (snake_case)
mkdir -p src/strategy/{name_snake}/
mkdir -p tests/strategy/{name_snake}/

# 빈 __init__.py 생성
touch tests/strategy/{name_snake}/__init__.py
```

**필수 파일 목록** (이후 단계에서 생성):

```
src/strategy/{name_snake}/
  __init__.py        # Step 6
  config.py          # Step 2
  preprocessor.py    # Step 3
  signal.py          # Step 4
  strategy.py        # Step 5

tests/strategy/{name_snake}/
  __init__.py        # (빈 파일)
  test_config.py     # Step 7
  test_preprocessor.py  # Step 7
  test_signal.py     # Step 7
  test_strategy.py   # Step 7
```

---

## Step 2: config.py 구현

### 템플릿

[references/code-templates.md](references/code-templates.md)의 Config Template 참조.

### 필수 요소

| 요소 | 설명 | 예시 |
|------|------|------|
| `from __future__ import annotations` | 상단 필수 | — |
| Pydantic `BaseModel` + `frozen=True` | 불변 설정 | `model_config = ConfigDict(frozen=True)` |
| `ShortMode(IntEnum)` | 숏 모드 열거형 | `DISABLED=0, HEDGE_ONLY=1, FULL=2` |
| 전략 파라미터 `Field()` | `ge`, `le`, `gt`, `lt` 검증 | `Field(default=30, ge=5, le=200)` |
| `vol_target: float` | Vol-target 파라미터 | `Field(default=0.35, gt=0.0, le=1.0)` |
| `min_volatility: float` | Vol 하한 | `Field(default=0.05, gt=0.0)` |
| `annualization_factor: float` | TF별 연환산 | Step 0-2 표 참조 |
| `short_mode: ShortMode` | 숏 모드 설정 | 후보 정보 기반 |
| HEDGE_ONLY 파라미터 | 헤지 진입 조건 | `hedge_threshold`, `hedge_strength_ratio` |
| `model_validator` | 교차 검증 | `vol_target >= min_volatility` |
| `warmup_periods()` | warmup bar 수 | 전략별 계산 |

### ShortMode 정의 규칙

```python
class ShortMode(IntEnum):
    """숏 포지션 모드."""
    DISABLED = 0    # 롱 전용
    HEDGE_ONLY = 1  # 하락장 헤지만 (drawdown 기반)
    FULL = 2        # 양방향 거래
```

### HEDGE_ONLY 필수 파라미터

HEDGE_ONLY 모드가 기본값인 전략은 반드시 다음을 포함:

```python
hedge_threshold: float = Field(default=-0.07, le=0.0)
hedge_strength_ratio: float = Field(default=0.8, gt=0.0, le=1.0)
```

### 검증 규칙

- `vol_target >= min_volatility` (model_validator)
- `lookback_long > lookback_short` (해당 시)
- 모든 window/period 파라미터에 `ge` 하한 설정
- `annualization_factor`는 TF에 맞는 정확한 값

### 주의사항

- `from __future__ import annotations` 사용 시 `TYPE_CHECKING` import 활용
- `ConfigDict` import는 `from pydantic import BaseModel, ConfigDict, Field, model_validator`
- IntEnum import: `from enum import IntEnum`

---

## Step 3: preprocessor.py 구현

### 템플릿

[references/code-templates.md](references/code-templates.md)의 Preprocessor Template 참조.

### 필수 요소

| 요소 | 설명 |
|------|------|
| `from __future__ import annotations` | 상단 필수 |
| `preprocess(df, config) -> pd.DataFrame` | 모듈 레벨 함수 |
| 원본 불변: `df = df.copy()` | 첫 줄에서 복사 |
| missing columns 검증 | `required = {"open", "high", "low", "close", "volume"}` |
| 벡터화 연산만 | for 루프 절대 금지 |
| `returns` 계산 | `np.log(close / close.shift(1))` |
| `realized_vol` 계산 | `returns.rolling(N).std() * np.sqrt(annualization_factor)` |
| `vol_scalar` 계산 | `(target / realized_vol.clip(lower=min_vol)).clip(upper=max_lev)` |
| 전략별 feature | 후보 문서의 "사용 지표" 구현 |
| `drawdown` 계산 | HEDGE_ONLY 모드용 (해당 시) |
| `atr` 계산 | 트레일링 스톱용 (해당 시) |

### 공통 유틸리티 재사용

기존 전략의 preprocessor에서 재사용 가능한 함수:

```python
# src/strategy/tsmom/preprocessor.py
from src.strategy.tsmom.preprocessor import (
    calculate_returns,
    calculate_realized_volatility,
    calculate_volatility_scalar,
)
```

또는 `src/strategy/vol_regime/preprocessor.py`에서:

```python
from src.strategy.vol_regime.preprocessor import (
    calculate_returns,
    calculate_drawdown,
    calculate_atr,
)
```

> 유틸리티가 있으면 반드시 재사용. 중복 구현 금지.

### vol_scalar 정확성 검증 공식

```python
realized_vol = returns.rolling(vol_window).std() * np.sqrt(annualization_factor)
realized_vol = realized_vol.clip(lower=min_volatility)
vol_scalar = config.vol_target / realized_vol
# Note: leverage cap은 PM에서 처리 (전략에서는 clip 안 함)
```

### drawdown 계산 (HEDGE_ONLY용)

```python
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max  # 항상 <= 0
```

### NaN 처리 원칙

| 상황 | 처리 |
|------|------|
| Rolling window warmup | NaN 유지 (fillna 금지) |
| 0 나눗셈 가능성 | `.clip(lower=epsilon)` 또는 `.replace(0, np.nan)` |
| 비율 계산 (range=0, doji) | `np.where(range > 0, ratio, 0.0)` |
| feature 간 연산 | NaN 전파 허용 (signal.py에서 처리) |

### 금지 패턴

```python
# WRONG: 전체 기간 통계
df["zscore"] = (df["close"] - df["close"].mean()) / df["close"].std()

# CORRECT: Rolling 통계
df["zscore"] = (df["close"] - df["close"].rolling(N).mean()) / df["close"].rolling(N).std()

# WRONG: inplace=True
df.fillna(0, inplace=True)

# CORRECT: 새 Series 할당
df["col"] = df["col"].fillna(0)

# WRONG: apply(axis=1)
df["signal"] = df.apply(lambda row: row["a"] + row["b"], axis=1)

# CORRECT: 벡터 연산
df["signal"] = df["a"] + df["b"]
```

---

## Step 4: signal.py 구현

### 템플릿

[references/code-templates.md](references/code-templates.md)의 Signal Template 참조.

### 필수 요소

| 요소 | 설명 |
|------|------|
| `from __future__ import annotations` | 상단 필수 |
| `generate_signals(df, config) -> StrategySignals` | 모듈 레벨 함수 |
| **Shift(1) Rule** | 시그널에 사용되는 **모든** feature에 `.shift(1)` 적용 |
| ShortMode 3-way 분기 | DISABLED / HEDGE_ONLY / FULL |
| `StrategySignals` 반환 | `(entries, exits, direction, strength)` |
| NaN → 0 처리 | strength의 NaN은 0으로, direction의 NaN은 0으로 |

### Shift(1) Rule — 가장 중요한 규칙

```python
# Step 1: 모든 feature를 shift(1)
indicator_a = df["indicator_a"].shift(1)
indicator_b = df["indicator_b"].shift(1)
vol_scalar = df["vol_scalar"].shift(1)

# Step 2: shift된 feature로만 시그널 생성
long_signal = (indicator_a > threshold_a) & (indicator_b > threshold_b)
short_signal = (indicator_a < -threshold_a) & (indicator_b < -threshold_b)
```

> **Critical**: preprocessor.py에서 계산한 feature를 signal.py에서 사용할 때,
> `shift(1)`은 signal.py에서 적용한다. preprocessor에서는 shift하지 않는다.
> 이렇게 해야 preprocessor 출력이 분석/디버깅에 직접 사용 가능하다.

### ShortMode 분기 패턴

```python
# --- Direction ---
if config.short_mode == ShortMode.DISABLED:
    direction = np.where(long_signal, 1, 0)

elif config.short_mode == ShortMode.HEDGE_ONLY:
    drawdown = df["drawdown"].shift(1)
    hedge_active = drawdown < config.hedge_threshold
    direction = np.where(
        long_signal, 1,
        np.where(short_signal & hedge_active, -1, 0)
    )

elif config.short_mode == ShortMode.FULL:
    direction = np.where(
        long_signal, 1,
        np.where(short_signal, -1, 0)
    )

direction_series = pd.Series(direction, index=df.index, dtype=int)
```

### Strength 계산 패턴

```python
# strength = direction * vol_scalar * conviction (해당 시)
strength = direction_series.astype(float) * vol_scalar.fillna(0)

# HEDGE_ONLY: 숏 포지션 강도 감쇄
if config.short_mode == ShortMode.HEDGE_ONLY:
    strength = np.where(
        direction_series == -1,
        strength * config.hedge_strength_ratio,
        strength,
    )
strength_series = pd.Series(strength, index=df.index).fillna(0.0)
```

### Entry/Exit 생성 패턴

```python
# entries: 포지션 변경 (0→1, 0→-1, 1→-1, -1→1)
prev_direction = direction_series.shift(1).fillna(0).astype(int)
entries = (direction_series != 0) & (direction_series != prev_direction)

# exits: 포지션 청산 (1→0, -1→0)
exits = (direction_series == 0) & (prev_direction != 0)

return StrategySignals(
    entries=entries.astype(bool),
    exits=exits.astype(bool),
    direction=direction_series,
    strength=strength_series,
)
```

### 금지 패턴

```python
# WRONG: shift 없이 당봉 데이터 사용
signal = df["close"] > df["sma_20"]

# WRONG: shift(-N) — 미래 참조
target = df["close"].shift(-5)

# WRONG: 동시 long + short
direction = np.where(long_signal, 1, np.where(short_signal, -1, 0))
# -> long_signal과 short_signal이 동시에 True일 수 있음!
# CORRECT: 우선순위 명시
direction = np.where(long_signal & ~short_signal, 1,
                     np.where(short_signal & ~long_signal, -1, 0))

# WRONG: strength가 NaN인데 entries=True
# -> fillna(0) 필수
```

---

## Step 5: strategy.py 구현

### 템플릿

[references/code-templates.md](references/code-templates.md)의 Strategy Template 참조.

### 필수 요소

| 요소 | 설명 |
|------|------|
| `@register("kebab-name")` | Registry 등록 데코레이터 |
| `BaseStrategy` 상속 | 추상 메서드 구현 |
| `name` property | registry key와 동일 |
| `required_columns` property | `["open", "high", "low", "close", "volume"]` |
| `config` property | Config 인스턴스 반환 |
| `preprocess()` | preprocessor.preprocess() 위임 |
| `generate_signals()` | signal.generate_signals() 위임 |
| `recommended_config()` | PM 설정 권장값 |
| `from_params()` | Config를 거쳐 인스턴스 생성 |
| `get_startup_info()` | CLI 표시용 핵심 파라미터 |

### recommended_config() 패턴

```python
@classmethod
def recommended_config(cls) -> dict[str, Any]:
    return {
        "stop_loss_pct": 0.10,
        "trailing_stop_enabled": True,
        "trailing_stop_atr_multiplier": 3.0,
        "rebalance_threshold": 0.10,
        "use_intrabar_stop": True,
    }
```

### from_params() 패턴

```python
@classmethod
def from_params(cls, **params: Any) -> BaseStrategy:
    config = {StrategyConfig}(**params)
    return cls(config=config)
```

---

## Step 6: __init__.py + Registry 등록

### 전략 모듈 __init__.py

```python
"""{StrategyName}: {한줄 설명}."""

from src.strategy.{name_snake}.config import {Config}, ShortMode
from src.strategy.{name_snake}.preprocessor import preprocess
from src.strategy.{name_snake}.signal import generate_signals
from src.strategy.{name_snake}.strategy import {Strategy}

__all__ = [
    "{Config}",
    "{Strategy}",
    "ShortMode",
    "generate_signals",
    "preprocess",
]
```

### src/strategy/__init__.py에 등록

기존 import 목록에 알파벳 순서로 추가:

```python
from src.strategy import {name_snake}  # noqa: F401 — registry side-effect
```

> 위치: 기존 import 목록의 알파벳 순서에 맞춰 삽입.
> `# pyright: reportUnusedImport=false` 주석이 파일 상단에 있으므로 추가 주석 불필요.

### 등록 확인

```bash
uv run python -c "from src.strategy import list_strategies; print('{name}' in list_strategies())"
# True 출력 확인
```

---

## Step 7: 테스트 작성

### 테스트 파일 구조

전략당 4개 테스트 파일, 총 **40~60개 테스트**.

### 7-1. test_config.py

**필수 테스트 클래스**:

```python
class TestShortMode:
    """ShortMode enum 테스트."""
    def test_values(self):           # 0, 1, 2 값
    def test_int_enum(self):         # isinstance(ShortMode.DISABLED, int)

class Test{Strategy}Config:
    """Config 기본값 + 검증 규칙."""
    def test_default_values(self):              # 기본 Config() 생성
    def test_frozen(self):                      # 수정 시 TypeError/ValidationError
    def test_{param}_range(self):               # 각 파라미터 경계값 검증
    def test_vol_target_gte_min_volatility(self): # 교차 검증
    def test_warmup_periods(self):              # warmup 계산 검증
    def test_annualization_factor_default(self): # TF에 맞는 값
```

### 7-2. test_preprocessor.py

**필수 테스트 클래스**:

```python
class TestPreprocess:
    """Preprocessor 출력 검증."""
    def test_output_columns(self):       # 필수 컬럼 존재
    def test_same_length(self):          # len(output) == len(input)
    def test_immutability(self):         # 원본 df 미수정
    def test_missing_columns(self):      # ValueError 발생
    def test_returns_log(self):          # log return 확인
    def test_realized_vol_positive(self): # > 0
    def test_vol_scalar_positive(self):  # > 0

class Test{FeatureName}:
    """전략 고유 feature 검증."""
    def test_{feature}_range(self):      # 값 범위 (0~1, >=0 등)
    def test_{feature}_edge_case(self):  # doji, 0 volume 등
```

### 7-3. test_signal.py

**필수 테스트 클래스**:

```python
class TestSignalStructure:
    """시그널 출력 구조."""
    def test_output_structure(self):     # entries, exits, direction, strength
    def test_entries_exits_bool(self):   # bool dtype
    def test_direction_values(self):     # {-1, 0, 1} subset
    def test_same_length(self):          # len == len(input)

class TestShift1Rule:
    """Look-ahead bias 방지."""
    def test_first_bar_neutral(self):    # iloc[0] direction == 0, strength == 0

class TestShortMode:
    """ShortMode 3-way 분기."""
    def test_disabled_no_shorts(self):        # direction != -1
    def test_full_allows_shorts(self):        # -1 가능
    def test_hedge_only_with_drawdown(self):  # 하락 시 숏 허용
    def test_hedge_only_no_drawdown(self):    # 상승 시 숏 억제

class Test{StrategySpecificLogic}:
    """전략 고유 시그널 로직."""
    # 후보 문서의 시그널 로직에 맞는 테스트
```

### 7-4. test_strategy.py

**필수 테스트 클래스**:

```python
class TestRegistry:
    """Registry 통합."""
    def test_registered(self):         # list_strategies() 포함
    def test_get_strategy(self):       # get_strategy() 조회

class Test{Strategy}Strategy:
    """전략 통합 테스트."""
    def test_name(self):               # name property
    def test_required_columns(self):   # required_columns
    def test_config(self):             # config property
    def test_preprocess(self):         # preprocess() 위임
    def test_generate_signals(self):   # generate_signals() 위임
    def test_run_pipeline(self):       # run() 전체 파이프라인
    def test_from_params(self):        # from_params() factory
    def test_recommended_config(self): # recommended_config()
    def test_get_startup_info(self):   # get_startup_info()
    def test_warmup_periods(self):     # warmup_periods() (해당 시)
    def test_custom_config(self):      # 커스텀 config 주입
    def test_params_property(self):    # params dict
    def test_repr(self):               # __repr__
```

### Fixture 패턴

```python
@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """전략 특성에 맞는 합성 데이터."""
    np.random.seed(42)
    n = 300  # warmup 충분히 고려
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)

    # high >= max(open, close), low <= min(open, close) 보장
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="{freq}"),
    )
```

> `{freq}`는 TF에 맞춰 설정: `"D"`, `"4h"`, `"6h"`, `"12h"` 등.

### Trending Data Fixture (ShortMode 테스트용)

```python
@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """순수 상승 추세 (HEDGE_ONLY 검증용)."""
    n = 200
    close = np.linspace(100, 200, n)
    high = close + 2.0
    low = close - 2.0
    open_ = close - 0.5
    volume = np.full(n, 5000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="{freq}"),
    )
```

---

## Step 8: 품질 검증

### 8-1. Ruff 린트 + 포맷

```bash
uv run ruff check src/strategy/{name_snake}/ tests/strategy/{name_snake}/ --fix
uv run ruff format src/strategy/{name_snake}/ tests/strategy/{name_snake}/
```

**0 error 필수**. 발견 시 즉시 수정.

### 8-2. Pyright 타입체크

```bash
uv run pyright src/strategy/{name_snake}/
```

**0 error 필수**. 주의사항:

- `from __future__ import annotations` 사용 시 runtime import와 TYPE_CHECKING 구분
- pandas Series 타입 → `pd.Series` 어노테이션 또는 적절한 `# type: ignore[assignment]`
- `100 * Series`는 `int`로 추론됨 → 변수에 `pd.Series` 타입 명시

### 8-3. 테스트 실행

```bash
# 전략 테스트만 실행
uv run pytest tests/strategy/{name_snake}/ -v

# 전체 테스트 (기존 테스트 깨짐 방지)
uv run pytest --tb=short -q
```

**0 failure 필수**. 기존 테스트가 깨지면 원인 분석 후 수정.

### 8-4. 최종 확인 체크리스트

| # | 항목 | 확인 방법 |
|---|------|----------|
| 1 | Registry 등록 | `list_strategies()`에 포함 |
| 2 | Ruff 0 error | `ruff check` 통과 |
| 3 | Pyright 0 error | `pyright src/strategy/{name_snake}/` 통과 |
| 4 | 테스트 전체 통과 | `pytest` 0 failure |
| 5 | Shift(1) 적용 | signal.py의 모든 feature에 shift(1) |
| 6 | ShortMode 분기 | 3가지 모드 모두 구현 |
| 7 | annualization_factor | TF에 맞는 값 |
| 8 | NaN 방어 | 0 나눗셈, edge case 처리 |
| 9 | 원본 불변 | `df.copy()` 사용 |
| 10 | docstring | 모듈/함수 docstring 작성 |

---

## Step 9: YAML 메타데이터 업데이트

### 9-1. YAML 메타데이터 업데이트 (필수)

`strategies/{strategy_name}.yaml`은 discovery 단계에서 이미 생성됨 (status: CANDIDATE).
구현 완료 후 status를 IMPLEMENTED로 업데이트한다:

```bash
# status 변경: CANDIDATE → IMPLEMENTED
uv run mcbot pipeline update-status {strategy_name} --status IMPLEMENTED
```

> parameters 추가가 필요하면 YAML 직접 편집 (config.py의 기본값과 동일하므로 보통 불필요).

---

## Step 10: Dashboard 갱신

### 10-1. Pipeline CLI로 Dashboard 자동 생성

```bash
uv run mcbot pipeline report
```

> YAML 데이터를 `pipeline report`로 콘솔 확인. `--output FILE`로 파일 저장 가능.
> 수동 편집 불필요.

### 10-2. YAML 상태 갱신

```bash
uv run mcbot pipeline update-status {strategy_name} --status IMPLEMENTED
```

---

## Step 11: 완료 리포트

```
============================================================
  STRATEGY IMPLEMENTATION REPORT
  전략: {DisplayName} ({registry-key})
  구현일: {YYYY-MM-DD}
  타임프레임: {TF}
  ShortMode: {mode}
============================================================

  파일 생성:
    src/strategy/{name_snake}/config.py
    src/strategy/{name_snake}/preprocessor.py
    src/strategy/{name_snake}/signal.py
    src/strategy/{name_snake}/strategy.py
    src/strategy/{name_snake}/__init__.py
    tests/strategy/{name_snake}/test_config.py
    tests/strategy/{name_snake}/test_preprocessor.py
    tests/strategy/{name_snake}/test_signal.py
    tests/strategy/{name_snake}/test_strategy.py

  품질 검증:
    Ruff:    PASS (0 errors)
    Pyright: PASS (0 errors)
    Tests:   PASS ({N} tests, 0 failures)

  Registry: {registry-key} ✓ ({total} strategies)

  다음 단계:
    1. /p3-g0b-verify {registry-key}  (Gate 0B 코드 검증)
    2. /p4-g1g4-gate {registry-key}    (G1~G4 검증)

============================================================
```

---

## Anti-Pattern 체크리스트

구현 완료 전 반드시 확인:

| # | Anti-Pattern | 확인 |
|---|-------------|------|
| 1 | `shift(-N)` 사용 (미래 참조) | signal.py에 없어야 함 |
| 2 | 전체 기간 `.mean()`, `.std()` (rolling 없이) | preprocessor.py에 없어야 함 |
| 3 | `for i in range(len(df))` | 어디에도 없어야 함 |
| 4 | `iterrows()`, `itertuples()` | 어디에도 없어야 함 |
| 5 | `apply(axis=1)` | 어디에도 없어야 함 |
| 6 | `inplace=True` | 어디에도 없어야 함 |
| 7 | `fillna(0)` 부적절 사용 | NaN이 의미 있는 경우 주의 |
| 8 | `except:` (광범위 예외) | 구체적 예외만 |
| 9 | `float` for prices | Decimal 미사용은 OK (전략 레벨) |
| 10 | 매직 넘버 | config 파라미터로 추출 |
| 11 | `# noqa`, `# type: ignore` 남용 | 정당한 사유만 |
| 12 | annualization_factor 365 하드코딩 | config에서 읽어야 함 |

---

## 참조 문서

| 문서 | 용도 |
|------|------|
| [references/code-templates.md](references/code-templates.md) | 4-file 코드 템플릿 |
| [references/implementation-checklist.md](references/implementation-checklist.md) | 구현 체크리스트 + 폐기 패턴 |
| [strategies/*.yaml](../../../strategies/) | YAML 파이프라인 (Single Source of Truth) |
| `pipeline report` | 전략 상황판 (CLI) |
| [.claude/rules/strategy.md](../../rules/strategy.md) | 전략 개발 규칙 |
| [src/strategy/base.py](../../../src/strategy/base.py) | BaseStrategy ABC |
