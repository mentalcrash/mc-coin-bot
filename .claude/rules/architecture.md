# 🏗️ System Architecture

## 1. 메달리온 데이터 아키텍처 (Bronze → Silver → Gold)

### Bronze 레이어 (`data/bronze/`)
- **원시 데이터 저장소:** Binance API에서 수집한 변환 없는 OHLCV 데이터
- **파티셔닝 전략:** 심볼·연도별 분리 → `data/bronze/BTC_USDT/2024.parquet`
- **저장 형식:** Parquet (컬럼형 압축 포맷)
- **쓰기 정책:** Append-only (덮어쓰기 금지)
- **목적:** 데이터 원본 보존, 재처리 가능성 보장

### Silver 레이어 (`data/silver/`)
- **정제 데이터:** 검증·정제·갭 채우기 완료
- **데이터 품질 보장:**
  - 시간 갭 탐지 및 forward-fill로 채우기
  - 중복 제거, 타임스탬프 정렬
  - 이상치 검증 (가격 급등락 체크)
- **리샘플링:** 1분 기본 데이터를 상위 타임프레임(1h, 4h, 1d)으로 변환
- **타임스탬프:** DatetimeIndex 기준, UTC 타임존만 사용
- **목적:** 백테스트 및 실거래용 신뢰 가능한 데이터 제공

### Gold 레이어 (메모리 내 계산)
- **전략별 피처:** 기술적 지표, 파생 변수
- **생성 시점:** 백테스트 또는 실거래 시 on-the-fly 계산
- **저장 정책:** 디스크 저장 없음, 메모리에서만 사용
- **목적:** 전략별 커스터마이징, 빠른 반복 개발

---

## 2. 전략 엔진 설계

### BaseStrategy 인터페이스 (`src/strategy/base.py`)

모든 전략은 다음 메서드를 구현해야 합니다:

```python
class BaseStrategy(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산 (벡터화 연산만 사용)"""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """시그널 시리즈 반환 (-1: 매도, 0: 관망, 1: 매수)"""

    @abstractmethod
    def get_config(self) -> BaseConfig:
        """전략 설정 반환 (Pydantic 모델)"""
```

### 핵심 원칙

#### 1. 벡터화 연산 (Zero Loop Policy)
- **금지:** `for` 루프, `iterrows()`, `itertuples()`
- **필수:** pandas/numpy 벡터화 연산 사용
- **이유:** 백테스트 속도 향상 (100배 이상 차이)

```python
# ❌ Bad (Loop)
for i in range(len(df)):
    if df['close'].iloc[i] > df['sma_20'].iloc[i]:
        signals.iloc[i] = 1

# ✅ Good (Vectorized)
signals = np.where(df['close'] > df['sma_20'], 1, 0)
```

#### 2. Shift(1) 규칙 (Lookahead Bias 방지)
- **원칙:** 현재 봉 데이터로 시그널 생성 시 반드시 `.shift(1)` 사용
- **이유:** 같은 봉의 종가로 시그널을 생성하면 미래 정보 유출

```python
# ❌ Bad (Lookahead Bias)
signal = (df['close'] > df['sma_20']).astype(int)

# ✅ Good (No Lookahead)
signal = (df['close'].shift(1) > df['sma_20'].shift(1)).astype(int)
```

#### 3. 내부 로그 수익률 사용
- **계산:** 로그 수익률 `np.log(close / close.shift(1))` 사용
- **변환:** 리포트 생성 시에만 단순 수익률로 변환
- **이유:** 로그 수익률은 시간 가산성 보장, 복리 계산 정확도

### 전략 레지스트리 패턴

**자동 등록 시스템:**
```python
# 전략 클래스에 데코레이터 추가
@register_strategy("tsmom")
class TSMOMStrategy(BaseStrategy):
    ...

# CLI에서 자동 조회
strategy = get_strategy("tsmom", config)
all_strategies = list_strategies()
```

**디렉터리 구조:**
```
src/strategy/my_strategy/
├── config.py         # Pydantic 설정 모델
├── preprocessor.py   # 지표 계산 (벡터화)
├── signal.py         # 시그널 생성 로직
└── strategy.py       # @register_strategy 메인 클래스
```

---

## 3. 실행 시스템 (PM/RM/OMS 패턴)

헤지펀드 운영을 모델로 한 **3단계 방어 구조**로 치명적 손실을 방지합니다.

### Portfolio Manager (PM)
**위치:** `src/portfolio/portfolio.py`

**책임:**
- Signal 이벤트 수신 및 포지션 사이징 계산
- 포트폴리오 상태 관리 (포지션, 잔고)
- Fill 이벤트 수신하여 실제 체결 반영

**사이징 방식:**
- 고정 비율 (Fixed Fraction)
- Kelly Criterion
- 변동성 조정 (Volatility Targeting)
- 리스크 패리티 (Risk Parity)

**상태 관리:**
```python
class PortfolioManager:
    positions: dict[str, Position]  # 심볼별 포지션
    balance: Decimal                # 현재 잔고
    equity_curve: list[Decimal]     # 자산 곡선
```

### Risk Manager (RM)
**위치:** `src/execution/risk_manager.py` (예정)

**책임: 최종 관문 — 모든 주문 검증**
- 일일 손실 한도 (Daily Loss Limit)
- 포지션 한도 (Position Limit)
- 낙폭 한도 (Drawdown Limit)
- Fat-Finger 감지 (비정상적 주문 크기)

**Kill Switch:**
- 한도 초과 시 모든 주문 차단
- 관리자 승인 전까지 거래 중단

### Order Management System (OMS)
**위치:** `src/execution/oms.py` (예정)

**책임: 안전한 주문 실행**
- Client Order ID로 멱등한 주문 처리
- 네트워크 오류 시 지수 백오프 재시도
- WebSocket User Data Stream으로 실시간 동기화

**멱등성 패턴:**
```python
client_order_id = f"{strategy}_{symbol}_{timestamp}_{nonce}"
```
- 동일 ID로 재전송 시 중복 주문 방지
- 거래소가 멱등성 보장 (`ORDER_ALREADY_EXISTS` 응답)

---

## 4. 이벤트 기반 흐름

**모든 컴포넌트는 EventBus를 통해 이벤트로만 통신합니다:**

```
데이터 소스 (Binance WebSocket)
    ↓ [MarketData 이벤트]
전략 엔진 (무상태)
    ↓ [Signal 이벤트: BUY/SELL/HOLD]
Portfolio Manager
    ↓ [OrderRequest 이벤트: 수량/가격]
Risk Manager
    ↓ [Order 이벤트: 승인됨]
OMS
    ↓ [REST API: create_order]
거래소
    ↓ [Fill 이벤트]
Portfolio Manager (포지션 갱신)
```

### 이벤트 타입

| 이벤트 | 생성자 | 소비자 | 데이터 |
|--------|--------|--------|--------|
| `MarketData` | Data Fetcher | Strategy | OHLCV, Ticker |
| `Signal` | Strategy | Portfolio Manager | BUY/SELL/HOLD, Strength |
| `OrderRequest` | Portfolio Manager | Risk Manager | Symbol, Side, Amount, Price |
| `Order` | Risk Manager | OMS | 승인된 주문 상세 |
| `Fill` | OMS | Portfolio Manager | 체결 수량, 가격, 수수료 |
| `Error` | 모든 컴포넌트 | Logger, Discord | 에러 상세 |

### 이벤트 기반 설계 원칙

> [!IMPORTANT]
> **컴포넌트는 서로 직접 호출하지 않습니다.**
> - ❌ `portfolio_manager.update_position(order)` 직접 호출 금지
> - ✅ `event_bus.publish(FillEvent(order))` 이벤트 발행

**이유:**
- **디커플링:** 컴포넌트 간 의존성 최소화
- **테스트 용이성:** Mock EventBus로 격리 테스트
- **확장성:** 새 컴포넌트 추가 시 기존 코드 수정 불필요
- **재생 가능성:** 이벤트 로그로 백테스트 재현

---

## 5. 모듈 구조 & 책임

### 핵심 모듈 (`src/`)

#### `src/core/`
- **`logger.py`:** Loguru 기반 구조화 로깅 설정
- **`exceptions.py`:** 커스텀 예외 계층

#### `src/config/`
- **`settings.py`:** Pydantic Settings로 환경 설정
- 모든 설정은 `.env` 파일 지원 (`pydantic-settings`)

#### `src/models/`
엄격한 타입의 Pydantic v2 데이터 모델:
- **`ohlcv.py`:** OHLCV 캔들 데이터 모델
- **`signal.py`:** 트레이딩 시그널 모델
- **`backtest.py`:** 백테스트 결과 모델
- **불변성:** 트레이딩 데이터는 `frozen=True`로 생성

#### `src/data/`
- **`fetcher.py`:** CCXT 기반 비동기 데이터 페처
- **`bronze.py`:** Bronze 레이어 Parquet 저장
- **`silver.py`:** 갭 탐지/채우기 포함 Silver 레이어 처리
- **`market_data.py`:** 시장 데이터 요청/응답 모델
- **`service.py`:** 상위 수준 데이터 서비스 오케스트레이터

#### `src/exchange/`
- **`binance_client.py`:** CCXT Pro 기반 Binance API 래퍼
- **중요 규칙:**
  - 주문 전송 전 반드시 `amount_to_precision()`, `price_to_precision()` 호출
  - 가격/수량을 문자열로 전달 (float 금지)

#### `src/strategy/`
- **`base.py`:** BaseStrategy 추상 클래스
- **`registry.py`:** 전략 등록 및 탐색
- **`tsmom/`:** 거래량 가중 시계열 모멘텀 전략
- **`breakout/`:** 적응형 브레이크아웃 전략
- 각 전략: `config.py`, `preprocessor.py`, `signal.py`, `strategy.py`

#### `src/portfolio/`
- **`portfolio.py`:** 포트폴리오 상태 관리 및 포지션 사이징
- **`cost_model.py`:** 거래 비용 모델 (Binance 현물/선물)
- **`config.py`:** 포트폴리오 매니저 설정

#### `src/backtest/`
- **`engine.py`:** VectorBT 기반 백테스트 엔진
- **`analyzer.py`:** 성과 지표 계산
- **`reporter.py`:** QuantStats 리포트 생성
- **`beta_attribution.py`:** 베타 억제 분석
- **`metrics.py`:** 커스텀 성과 지표

#### `src/cli/`
- **`ingest.py`:** 데이터 수집용 Typer CLI
- **`backtest.py`:** 백테스트용 Typer CLI

#### `src/notification/`
- **`discord.py`:** 알림용 Discord webhook

#### `src/logging/`
- OpenTelemetry sink 지원 고급 로깅
- 전략별 로깅용 컨텍스트 매니저
- 치명적 오류용 Discord sink

---

## 6. 의존성 흐름 (단방향)

```
CLI/Main
  ↓
Strategy, Execution, Backtest
  ↓
Data, Exchange, Portfolio
  ↓
Models, Core
  ↓
Config
```

**금지 사항:**
- ❌ Models에서 Strategy import
- ❌ Data에서 Execution import
- ❌ Core에서 비즈니스 로직 import

**이유:** 순환 참조 방지, 테스트 용이성, 명확한 책임 분리

---

## 7. 파일 명명 규칙

| 타입 | 규칙 | 예시 |
|------|------|------|
| 모듈 | `snake_case.py` | `market_data.py` |
| 클래스 | `PascalCase` | `MarketDataService` |
| 함수/변수 | `snake_case` | `calculate_returns` |
| 상수 | `UPPER_SNAKE_CASE` | `MAX_POSITION_SIZE` |
| 테스트 | `test_*.py` | `test_portfolio.py` |

**테스트 미러링:** `tests/` 디렉터리는 `src/` 구조를 그대로 따릅니다.
