# 🚨 Common Mistakes & Workflow Guide

## 1. 피해야 할 흔한 실수

### 1. Lookahead Bias (미래 정보 유출)
**문제:** 같은 봉의 종가로 시그널 생성 후 같은 봉에서 실행

```python
# ❌ Bad (Lookahead Bias)
df['signal'] = (df['close'] > df['sma_20']).astype(int)
# 같은 봉의 close로 판단 → 미래 정보 사용

# ✅ Good (No Lookahead)
df['signal'] = (df['close'].shift(1) > df['sma_20'].shift(1)).astype(int)
# 이전 봉 데이터로 판단 → 의사결정 시점에 알 수 있는 정보만 사용
```

**영향:** 백테스트 성과가 실거래보다 과도하게 좋게 나옴

---

### 2. Float 정밀도 (CCXT API)
**문제:** Precision 함수를 거치지 않고 CCXT에 float 전송

```python
# ❌ Bad
await exchange.create_order("BTC/USDT", "limit", "buy", 0.001, 50000.5)
# Float 전송 → 부동소수점 오차 발생

# ✅ Good
amount = exchange.amount_to_precision("BTC/USDT", Decimal("0.001"))
price = exchange.price_to_precision("BTC/USDT", Decimal("50000"))
await exchange.create_order("BTC/USDT", "limit", "buy", amount, price)
# String 전송 → 정확한 정밀도 보장
```

**영향:** `INVALID_PRECISION` 에러, 주문 실패

---

### 3. 루프 성능 (벡터화 대신 iterrows)
**문제:** 벡터화 대신 `iterrows()` 사용

```python
# ❌ Bad (100x 느림)
for i, row in df.iterrows():
    if row['close'] > row['sma_20']:
        signals.loc[i] = 1

# ✅ Good (벡터화)
signals = np.where(df['close'] > df['sma_20'], 1, 0)
```

**영향:** 백테스트 시간 100배 이상 차이

---

### 4. 마켓 로딩 누락
**문제:** `load_markets()` 누락으로 정밀도 정보 없음

```python
# ❌ Bad
exchange = ccxt.binance(config)
# 정밀도 정보 없음 → 주문 실패 가능

# ✅ Good
async with ccxt.binance(config) as exchange:
    await exchange.load_markets()  # 필수!
    # 정밀도 정보 로드됨
```

**영향:** Precision 함수가 작동하지 않음

---

### 5. 연결 누수
**문제:** CCXT 거래소에 `async with` 컨텍스트 매니저 미사용

```python
# ❌ Bad
exchange = ccxt.binance(config)
# close() 호출 누락 → 연결 누수

# ✅ Good
async with ccxt.binance(config) as exchange:
    # 자동으로 close() 호출됨
    pass
```

**영향:** 소켓 고갈, 메모리 누수

---

### 6. 타임존 불일치
**문제:** 데이터 파이프라인에 비-UTC 타임스탬프 사용

```python
# ❌ Bad
df['timestamp'] = pd.to_datetime(df['timestamp'])
# 로컬 타임존 또는 타임존 없음

# ✅ Good
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.set_index('timestamp')
# UTC 명시적 지정
```

**영향:** 시간대 전환 시 데이터 불일치

---

### 7. 가변 연산 (inplace=True)
**문제:** pandas에서 `inplace=True` 사용 (Ruff PD002 위반)

```python
# ❌ Bad
df.fillna(0, inplace=True)
df.drop(columns=['col'], inplace=True)

# ✅ Good
df = df.fillna(0)
df = df.drop(columns=['col'])
```

**영향:** 불변 연산 원칙 위반, 디버깅 어려움

---

## 2. 코드베이스 작업 가이드

### 새 전략 추가

**1단계: 디렉터리 생성**
```bash
mkdir -p src/strategy/my_strategy
```

**2단계: 4개 파일 구현**

**`config.py`** - Pydantic 설정 모델
```python
from pydantic import BaseModel, Field

class MyStrategyConfig(BaseModel):
    lookback_period: int = Field(default=20, ge=1)
    threshold: float = Field(default=0.02, ge=0.0)
```

**`preprocessor.py`** - 지표 계산 (벡터화)
```python
import pandas as pd

def calculate_indicators(df: pd.DataFrame, config: MyStrategyConfig) -> pd.DataFrame:
    df['sma'] = df['close'].rolling(config.lookback_period).mean()
    return df
```

**`signal.py`** - 시그널 생성 로직
```python
import numpy as np
import pandas as pd

def generate_signals(df: pd.DataFrame, config: MyStrategyConfig) -> pd.Series:
    # shift(1) 사용하여 Lookahead Bias 방지
    condition = df['close'].shift(1) > df['sma'].shift(1)
    return np.where(condition, 1, 0)
```

**`strategy.py`** - 메인 전략 클래스
```python
from src.strategy.base import BaseStrategy
from src.strategy.registry import register_strategy

@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def __init__(self, config: MyStrategyConfig):
        self.config = config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return calculate_indicators(df, self.config)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return generate_signals(df, self.config)

    def get_config(self) -> MyStrategyConfig:
        return self.config
```

**3단계: 자동 등록 확인**
```bash
python -m src.cli.backtest strategies
# "my_strategy"가 목록에 나타나야 함
```

---

### 새 데이터 소스 추가

**1단계: `src/data/`에 fetcher 생성**
```python
# src/data/my_exchange_fetcher.py
import ccxt.pro as ccxt

class MyExchangeFetcher:
    async def fetch_ohlcv(self, symbol: str, timeframe: str) -> list:
        async with ccxt.myexchange() as exchange:
            await exchange.load_markets()
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe)
            return ohlcv
```

**2단계: Bronze 저장 로직 추가**
```python
# src/data/bronze.py에 추가
def save_to_bronze_myexchange(symbol: str, year: int, data: pd.DataFrame) -> None:
    path = BRONZE_DIR / "myexchange" / f"{symbol.replace('/', '_')}" / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path)
```

**3단계: Silver 처리 추가**
```python
# src/data/silver.py에 갭 채우기 로직 추가
def fill_gaps_myexchange(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df.fillna(method='ffill')  # Forward fill
    return df
```

**4단계: CLI 명령 업데이트**
```python
# src/cli/ingest.py에 새 exchange 옵션 추가
@app.command()
def fetch_myexchange(symbol: str):
    ...
```

---

### 포트폴리오 로직 수정

**Portfolio Manager 수정** (`src/portfolio/portfolio.py`)
```python
def calculate_position_size(
    self,
    signal: int,
    price: Decimal,
    volatility: Decimal
) -> Decimal:
    # Volatility Targeting 사이징
    target_vol = Decimal("0.02")  # 2% 일일 변동성 목표
    position_size = (self.balance * target_vol) / (price * volatility)
    return position_size
```

**Cost Model 수정** (`src/portfolio/cost_model.py`)
```python
def calculate_trading_cost(
    self,
    order_type: str,
    notional: Decimal
) -> Decimal:
    # Binance 현물 수수료: 0.1%
    if order_type == "spot":
        return notional * Decimal("0.001")
    # Binance 선물 수수료: 0.02% (Maker)
    elif order_type == "future":
        return notional * Decimal("0.0002")
```

**중요:** 모든 금액 계산에 `Decimal` 사용 (부동소수점 오류 방지)

---

## 3. 자주 쓰는 개발 명령어

### 환경 설정
```bash
# uv로 의존성 설치 (권장 패키지 매니저)
uv sync

# 개발 도구 포함 설치
uv sync --group dev

# 리서치/백테스트 도구 포함 설치
uv sync --group research

# 가상환경 활성화
source .venv/bin/activate
```

---

### 코드 품질 & 테스트
```bash
# 린터 실행 (코드 수정 없이 품질 검사)
uv run ruff check .

# 린트 이슈 자동 수정
uv run ruff check --fix .

# 코드 포맷팅 (Black 호환)
uv run ruff format .

# 타입 검사 (VSCode Pylance가 자동으로 실행)
# .vscode/settings.json: python.analysis.typeCheckingMode = "strict"
# pyproject.toml: [tool.pyright] 섹션 참조

# 전체 테스트 실행
uv run pytest

# 커버리지 리포트와 함께 테스트
uv run pytest --cov=src --cov-report=html

# 특정 테스트 파일 실행
uv run pytest tests/unit/test_portfolio.py

# 패턴에 맞는 테스트만 실행
uv run pytest -k "test_tsmom"
```

---

### 데이터 수집 파이프라인 (메달리온 아키텍처)
```bash
# Binance API에서 원시 데이터 수집 (Bronze 레이어)
python main.py ingest bronze BTC/USDT --year 2024 --year 2025

# 갭 채우기 처리 (Silver 레이어)
python main.py ingest silver BTC/USDT --year 2024 --year 2025

# Bronze → Silver 전체 파이프라인 실행
python main.py ingest pipeline BTC/USDT --year 2024 --year 2025

# 데이터 무결성 검증
python main.py ingest validate BTC/USDT --year 2025

# 수집 정보 조회
python main.py ingest info
```

---

### 백테스트
```bash
# 사용 가능한 전략 목록
python -m src.cli.backtest strategies

# 전략 상세 정보
python -m src.cli.backtest info tsmom

# 백테스트 실행
python -m src.cli.backtest run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31

# 파라미터 스윕/최적화 실행
python -m src.cli.backtest sweep tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31

# QuantStats 리포트 생성
python -m src.cli.backtest run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31 --report
```

---

## 4. Quick Tips

### Pandas 성능 최적화
```python
# ✅ 벡터화 연산
df['returns'] = df['close'].pct_change()

# ✅ NumPy 조건문
df['signal'] = np.where(df['close'] > df['sma'], 1, -1)

# ✅ PyArrow 백엔드 (메모리 효율)
df = pd.read_parquet(path, dtype_backend="pyarrow")
```

### CCXT 안전 패턴
```python
# ✅ Context Manager + Precision
async with ccxt.binance(config) as exchange:
    await exchange.load_markets()
    safe_amount = exchange.amount_to_precision(symbol, amount)
    safe_price = exchange.price_to_precision(symbol, price)
    await exchange.create_order(symbol, "limit", "buy", safe_amount, safe_price)
```

### Pydantic 불변 모델
```python
# ✅ Frozen 모델 (주문 데이터)
class Order(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    price: Decimal
    amount: Decimal
```
