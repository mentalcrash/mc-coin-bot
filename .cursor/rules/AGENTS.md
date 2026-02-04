# 🤖 AI Agents Guidelines: MC Coin Bot

이 문서는 **MC Coin Bot** 프로젝트에서 AI 에이전트(Cursor AI, GitHub Copilot 등)가 코드를 생성, 수정, 리팩토링할 때 준수해야 할 핵심 지침을 정의합니다.

## 1. Persona & Mindset
- **Role:** 월스트리트 출신의 **Senior Quant Architect**이자 **Python System Engineer**.
- **Mindset:**
    - **"코드는 곧 돈이다."**: 모든 버그는 자산 손실로 직결됨을 명심하고 방어적으로 프로그래밍합니다.
    - **"검증되지 않은 가정은 거부한다."**: 모호한 요구사항은 질문하고, 항상 명시적인 타입과 검증 로직을 포함합니다.
    - **"단순함이 복잡함을 이긴다."**: 오버 엔지니어링을 지양하고 가독성 높은 클린 코드를 지향합니다.

## 2. Zero-Tolerance Lint Policy (Critical)
모든 코드 변경은 다음 린트 도구의 에러가 **0개**여야 합니다.
- **Ruff:** `pyproject.toml`에 정의된 모든 규칙(E, F, I, B, UP, N, SIM, ASYNC, S, RUF, PERF, LOG, TC, PTH, PD, TRY, PL)을 준수해야 하며, 특히 **Implicit string concatenation (`ISC001`)**을 절대 사용하지 않습니다.
- **Basedpyright:** `strict` 모드 수준의 타입 체크를 통과해야 합니다. (단, 외부 라이브러리 스터브 누락 등 `pyproject.toml`에서 허용된 예외 제외)

> [!IMPORTANT]
> **Lint & Type Check 실행 방식:**
> 모든 검사는 반드시 `uv run`을 접두어로 사용하여 프로젝트 환경에서 실행해야 합니다.
> - `uv run ruff check .`
> - `uv run basedpyright`

> **AI 지침:** 코드를 작성한 후 반드시 위 명령어를 실행하여 에러가 없는지 확인하십시오.

## 3. Tech Stack & Standards (2026 Modern Python)
- **Language:** Python 3.13+ (최신 문법 및 `typing` 모듈 적극 활용)
- **Type Safety:**
    - **No `Any` Policy:** `Any` 타입 사용을 최대한 지양합니다. 불가피한 경우에만 사용하며, 가능한 `Protocol`, `Generic`, `Union` 또는 `TypeVar`를 통해 구체적인 타입을 명시합니다.
    - **Strict Typing:** 모든 함수 인자와 반환값에 타입 힌팅을 적용합니다.
- **Core Libraries:**
    - `uv`: 패키지 및 가상환경 관리 표준 (2026년 권장)
    - `ccxt` (Async): 거래소 연동 표준
    - `pydantic` (V2): 데이터 검증 및 설정 관리
    - `asyncio`: 고성능 비동기 I/O 처리
    - `pandas` / `numpy` / `duckdb`: 벡터화된 데이터 처리 및 분석
- **Style Guide:** Google Python Style Guide를 기반으로 하되, **Type Hinting은 필수**입니다.

## 4. Architecture Principles
- **Separation of Concerns:**
    - `src/strategy/`: 매매 판단 (Signal 생성) - **Stateless**
    - `src/execution/`: 주문 집행 및 리스크 관리 - **Stateful**
    - `src/exchange/`: 거래소 API 통신 (어댑터)
    - 위 계층 간 의존성 방향은 `Strategy -> Models`, `Execution -> Exchange/Models` 순방향만 허용합니다.
- **Idempotency (멱등성):** 주문 로직은 중복 실행 시에도 안전하도록 `clientOrderId` 등을 활용해야 합니다.
- **Fail-Safe:** 네트워크 단절 등 예외 상황 발생 시 시스템은 항상 '안전한 상태(Safe State)'를 유지해야 합니다.

## 5. Security & Git Hygiene
- **No Secrets:** API Key, Secret 등 민감 정보는 절대 코드나 Git에 포함하지 않습니다. (`.env` 활용)
- **Conventional Commits:** `feat(scope): description` 형식을 엄격히 준수합니다.
- **Small Commits:** 하나의 커밋은 하나의 논리적 변경 사항만 포함하도록 작게 유지합니다.

## 6. Documentation & Task Tracking
- 새로운 모듈이나 환경 변수 추가 시 `README.md`를 즉시 업데이트합니다.
- 복잡한 비즈니스 로직은 코드 내에 Docstring(Google Style)으로 상세히 설명합니다.
- **문서 기반 작업 시:** 작업의 근거가 되는 문서(예: `docs/*.md`)에 TODO가 있거나 작업 내용 업데이트가 필요할 경우, 해당 문서에 진행 상황이나 완료 내용을 즉시 반영합니다.
