# CLAUDE.md

이 파일은 Claude Code(claude.ai/code)가 이 저장소의 코드를 작업할 때 참고하는 가이드입니다.

---

## 프로젝트 개요

**MC Coin Bot**은 이벤트 기반 아키텍처(EDA)로 구축된 전문가급 암호화폐 퀀트 트레이딩 시스템입니다.

**핵심 철학:**
- **이벤트 기반 아키텍처:** 모든 컴포넌트는 EventBus를 통해 이벤트로 통신 (MarketData, Signal, Order, Fill, Error)
- **무상태 전략 / 유상태 실행:** 전략 엔진은 포트폴리오 상태를 알지 못하고 시그널만 생성, 실행 시스템이 포지션·리스크·주문 관리
- **안전 우선 (PM/RM/OMS):** Portfolio Manager(자본 배분) → Risk Manager(최종 관문) → Order Management System(안전한 실행)
- **멱등성 & Fail-Safe:** 모든 주문 실행은 client order ID로 멱등하게 처리되며, 장애 시 시스템은 안전한 상태로 전환

---

## ⚠️ Agent Behavior Protocol (CRITICAL)

### Role Definition

**User Role:** Architect & Reviewer
- 시스템 아키텍처 결정권자
- 전략 로직 최종 승인자
- 코드 리뷰 및 품질 관리

**Claude Role:** Senior Implementation Engineer
- 사용자의 아키텍처 결정을 코드로 구현
- 테스트 작성 및 실행
- 코드 품질 표준 준수 (Ruff, basedpyright)

### Mandatory Workflow (MUST FOLLOW)

#### 1. Plan-Confirm-Implement Pattern
```
[사용자 요청]
  ↓
[Claude: 구현 계획 작성] ← 코드 작성 금지!
  ↓
[사용자: 계획 승인]
  ↓
[Claude: 코드 구현 + 테스트]
  ↓
[사용자: 리뷰 & 최종 승인]
```

**Implementation:**
- STEP 1: 작업 시작 시 반드시 `TodoWrite`로 계획 작성
- STEP 2: "계획을 확인해 주세요. 승인하시면 구현을 시작하겠습니다." 문구로 대기
- STEP 3: 승인 후에만 코드 작성 시작
- STEP 4: 각 단계 완료 후 "이 단계가 완료되었습니다. 리뷰 부탁드립니다." 보고

#### 2. Decision Gates (사용자 승인 필수)
다음 작업은 **반드시** 사용자 승인을 받아야 합니다:
- 전략 로직 변경 (Signal 생성 알고리즘)
- 리스크 파라미터 수정 (포지션 사이즈, 손절 레벨)
- 외부 API 호출 (Binance API, Discord Webhook)
- 데이터베이스 스키마 변경
- 의존성 추가/제거 (`pyproject.toml`)
- Git 작업 (commit, push, PR 생성)

#### 3. Autonomous Zone (자율 실행 가능)
다음 작업은 승인 없이 자율 실행 가능:
- 린트/포맷 수정 (`ruff check --fix`, `ruff format`)
- 타입 힌트 추가 (pyright 오류 해결)
- 단위 테스트 작성 및 실행
- 문서화 주석 추가 (docstring, type hints)
- 로그 메시지 개선

### Communication Style
- **간결성 우선:** 불필요한 존댓말이나 감탄사 제거
- **결과 중심:** "작업을 시작하겠습니다" → 바로 작업 실행
- **실패 투명성:** 에러 발생 시 즉시 보고, 재시도 전 사용자 확인
- **Markdown 링크 사용:** 파일 참조 시 반드시 `[파일명](경로)` 형식 사용

### Zero-Tolerance Lint Policy
**모든 코드는 다음 린트 도구의 에러가 0개여야 합니다:**
- **Ruff:** 코드 품질 및 스타일 검사
- **Pyright:** Strict 모드 타입 검사

**검사 실행:**
```bash
uv run ruff check .
uv run pyright src/
```

**코드 생성 후 필수 워크플로우:**
```bash
# 1. Ruff로 스타일 수정
uv run ruff check --fix . && uv run ruff format .

# 2. Pyright로 타입 검증
uv run pyright src/

# 3. 테스트 실행
uv run pytest --cov=src
```

> [!CAUTION]
> **`# noqa`, `# ruff: noqa`, `# type: ignore` 사용 절대 금지**
>
> 린트/타입 체커를 무력화하는 주석은 최후의 수단입니다. 코드를 수정하여 규칙을 준수하십시오.

---

## 핵심 개발 명령어

```bash
# 환경 설정
uv sync --group dev --group research

# 코드 품질 검사
uv run ruff check --fix .
uv run pyright src/

# 테스트 실행
uv run pytest --cov=src

# 데이터 수집 (메달리온 아키텍처)
python main.py ingest pipeline BTC/USDT --year 2024 --year 2025

# 백테스트 실행
python -m src.cli.backtest run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31
```

> [!NOTE]
> 전체 명령어 목록은 [.claude/rules/common-mistakes.md](.claude/rules/common-mistakes.md) 참조

---

## 아키텍처 핵심

### 메달리온 데이터 아키텍처
- **Bronze:** Binance API 원시 데이터 (Parquet 저장)
- **Silver:** 검증·정제·갭 채우기 완료
- **Gold:** 전략별 피처 (메모리 내 계산)

### 전략 엔진
- **BaseStrategy 인터페이스:** `preprocess()`, `generate_signals()`, `get_config()`
- **Zero Loop Policy:** 벡터화 연산만 사용 (iterrows 금지)
- **Shift(1) 규칙:** 현재 봉 데이터로 시그널 생성 시 `.shift(1)` 필수

### 실행 시스템 (3단계 방어)
- **Portfolio Manager:** Signal 수신, 포지션 사이징
- **Risk Manager:** 최종 관문, 한도 검증, Kill Switch
- **OMS:** 멱등한 주문 실행, Client Order ID 사용

### 이벤트 기반 흐름
```
WebSocket → MarketData → Strategy → Signal → PM → OrderRequest → RM → Order → OMS → Fill
```

> [!IMPORTANT]
> 컴포넌트는 서로 직접 호출하지 않고, EventBus를 통해서만 통신합니다.

> [!NOTE]
> 상세한 아키텍처 설명은 [.claude/rules/architecture.md](.claude/rules/architecture.md) 참조

---

## 필수 개발 표준

> [!NOTE]
> 상세한 개발 표준은 다음 모듈 파일들을 참조하십시오:
> - [.claude/rules/python-standards.md](.claude/rules/python-standards.md) - Python 3.13 & Pydantic V2
> - [.claude/rules/code-quality.md](.claude/rules/code-quality.md) - Ruff & Basedpyright
> - [.claude/rules/trading-standards.md](.claude/rules/trading-standards.md) - CCXT Integration
> - [.claude/rules/testing-standards.md](.claude/rules/testing-standards.md) - Pytest & Asyncio

### 핵심 요약

**타입 안전성 (basedpyright strict):**
- 모든 함수에 인자·반환값 타입 힌트 필수
- `Union[X, Y]` → `X | Y` 문법 사용
- `Decimal` 사용 (float 금지)

**CCXT 연동:**
- `ccxt.pro` (WebSocket) 사용
- `await exchange.load_markets()` 필수
- `amount_to_precision()`, `price_to_precision()` 호출 후 문자열 전달
- `async with` 컨텍스트 매니저 사용

**Pandas/NumPy:**
- DataFrame 행에 대한 `for` 루프 금지
- `iterrows()`, `itertuples()` 금지 → 벡터화 연산
- `inplace=True` 금지 → 불변 연산만
- UTC 타임스탬프, DatetimeIndex 사용

**Ruff 준수:**
- `# noqa` 주석 절대 금지
- import 순서: stdlib → third-party → `src.*`
- `os.path` 대신 `pathlib.Path`
- 구체적 예외 명시 (`except:` 금지)

**테스트:**
- 비동기 테스트에 `pytest-asyncio`
- 단위 테스트에서 실제 네트워크 호출 금지 (`AsyncMock` 사용)
- 핵심 경로 90% 이상 커버리지

**보안:**
- `.env` 파일 커밋 절대 금지
- API 키는 출금 비활성화
- 민감 데이터는 환경 변수로만 관리

---

## 프로젝트 구조 패턴

### 의존성 흐름 (단방향)
```
CLI/Main → Strategy, Execution, Backtest → Data, Exchange, Portfolio → Models, Core → Config
```

**금지 사항:**
- ❌ Models에서 Strategy import
- ❌ Data에서 Execution import
- ❌ Core에서 비즈니스 로직 import

### 파일 명명 규칙
- 모듈: `snake_case.py`
- 클래스: `PascalCase`
- 테스트: `test_*.py` (src/ 구조 미러링)

---

## 주요 아키텍처 제약

### Walk-Forward Analysis (WFA)
파라미터 최적화 구현 시:
- 롤링 train/test 윈도우 사용 (단일 분할 금지)
- In-sample: 파라미터 최적화
- Out-of-sample: 성과 검증
- 과적합 및 데이터 스누핑 방지

### 생존 편향 방지
여러 심볼 백테스트 시:
- 해당 시점에 존재했던 심볼만 사용
- 과거 테스트에 현재 상장 목록 사용 금지

### 미래 정보 유출 방지
- 현재 봉 지표로 시그널 생성 시 항상 `.shift(1)` 사용
- 예: `signal = (df['close'].shift(1) > df['sma_20'].shift(1))`

---

## 필요한 환경 변수

`.env.example`을 `.env`로 복사 후 설정:

```bash
# 필수
BINANCE_API_KEY=your_key_here
BINANCE_SECRET=your_secret_here

# 선택 (기본값 있음)
BRONZE_DIR=data/bronze
SILVER_DIR=data/silver
LOG_DIR=logs
RATE_LIMIT_PER_MINUTE=1200
```

---

## 추가 자료

### 프로젝트 문서
- 메인 README: 시스템 아키텍처 포함 한글 문서
- 전략 예시: [src/strategy/tsmom/](src/strategy/tsmom/), [src/strategy/breakout/](src/strategy/breakout/)
- 테스트 예시: [tests/](tests/) 디렉터리가 src/ 구조를 미러링

### Claude Code 설정
- **Agent 가이드:** 이 파일 (CLAUDE.md)
- **모듈화된 표준:**
  - [.claude/rules/python-standards.md](.claude/rules/python-standards.md) - Python 3.13 & Pydantic V2
  - [.claude/rules/code-quality.md](.claude/rules/code-quality.md) - Ruff & Basedpyright
  - [.claude/rules/trading-standards.md](.claude/rules/trading-standards.md) - CCXT Integration
  - [.claude/rules/testing-standards.md](.claude/rules/testing-standards.md) - Pytest & Asyncio
- **아키텍처 상세:**
  - [.claude/rules/architecture.md](.claude/rules/architecture.md) - 메달리온, PM/RM/OMS, 모듈 구조
- **작업 가이드:**
  - [.claude/rules/common-mistakes.md](.claude/rules/common-mistakes.md) - 피해야 할 실수, 워크플로우, 전체 명령어

### 레거시 (참고용)
- Cursor Rules (archived): `.cursor/rules/` - Claude Code로 마이그레이션 완료
