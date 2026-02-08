# CLAUDE.md

이 파일은 Claude Code가 이 저장소의 코드를 작업할 때 참고하는 가이드입니다.

---

## 프로젝트 개요

**MC Coin Bot**은 이벤트 기반 아키텍처(EDA)로 구축된 암호화폐 퀀트 트레이딩 시스템입니다.

**핵심 철학:**
- **이벤트 기반 아키텍처:** 모든 컴포넌트는 EventBus를 통해 이벤트로 통신
- **무상태 전략 / 유상태 실행:** 전략은 시그널만 생성, 실행 시스템이 포지션·리스크·주문 관리
- **안전 우선 (PM/RM/OMS):** Portfolio Manager → Risk Manager → OMS 3단계 방어
- **멱등성 & Fail-Safe:** 모든 주문은 client order ID로 멱등하게 처리

---

## Agent Behavior Protocol

### Role Definition

| Role | Responsibility |
|------|----------------|
| **User** | Architect & Reviewer - 아키텍처 결정, 전략 승인, 코드 리뷰 |
| **Claude** | Senior Engineer - 구현, 테스트, 코드 품질 준수 |

### Decision Gates (사용자 승인 필수)

- 전략 로직 변경 (Signal 생성 알고리즘)
- 리스크 파라미터 수정
- 외부 API 호출 (Binance, Discord)
- 의존성 추가/제거 (`pyproject.toml`)
- Git 작업 (commit, push, PR)

### Autonomous Zone (자율 실행 가능)

- 린트/포맷 수정 (`ruff check --fix`, `ruff format`)
- 타입 힌트 추가 (pyright 오류 해결)
- 단위 테스트 작성 및 실행
- 문서화 주석 추가

### Communication Style

- **간결성 우선:** 불필요한 존댓말 제거
- **결과 중심:** 작업 선언 → 바로 실행
- **Markdown 링크:** 파일 참조 시 `[파일명](경로)` 형식

---

## Critical Rules
- NEVER use pip install; 항상 `uv add` 또는 `uv pip install` 사용
- 모든 금융 계산은 반드시 known sources와 교차 검증 필요
- Type hints 모든 함수에 필수
- .env 파일 절대 커밋 금지
- 모든 전략은 expected values 포함한 백테스트 필수

## Zero-Tolerance Lint Policy

모든 코드는 Ruff/Pyright 에러 0개를 유지해야 합니다.

```bash
uv run ruff check --fix . && uv run ruff format .
uv run pyright src/
uv run pytest --cov=src
```

> `# noqa`, `# type: ignore` 사용 금지 (정당한 사유 없이)

---

## 환경 변수

`.env.example`을 `.env`로 복사 후 설정:

```bash
# 필수
BINANCE_API_KEY=your_key_here
BINANCE_SECRET=your_secret_here

# 선택 (기본값 있음)
BRONZE_DIR=data/bronze
SILVER_DIR=data/silver
LOG_DIR=logs
```

---

## Rules Navigation

상세 규칙은 `.claude/rules/` 참조:

| File | Scope | Description |
|------|-------|-------------|
| [commands.md](.claude/rules/commands.md) | `**` | CLI 명령어 모음 |
| [lint.md](.claude/rules/lint.md) | `src/**`, `tests/**` | Ruff/Pyright 규칙 |
| [strategy.md](.claude/rules/strategy.md) | `src/strategy/**` | 전략 개발 규칙 |
| [exchange.md](.claude/rules/exchange.md) | `src/exchange/**` | CCXT 연동 규칙 |
| [data.md](.claude/rules/data.md) | `src/data/**` | 메달리온 아키텍처 |
| [models.md](.claude/rules/models.md) | `src/models/**` | Pydantic V2 규칙 |
| [backtest.md](.claude/rules/backtest.md) | `src/backtest/**` | 백테스트 규칙 |
| [testing.md](.claude/rules/testing.md) | `tests/**` | 테스트 규칙 |

---

## Quick Reference

### 이벤트 흐름
```
WebSocket → MarketData → Strategy → Signal → PM → RM → OMS → Fill
```

### 의존성 흐름 (단방향)
```
CLI/Main → Strategy, Backtest → Data, Exchange, Portfolio → Models, Core → Config
```

### 핵심 금지 사항
- `float` for prices/amounts → use `Decimal`
- `iterrows()`, loops on DataFrame → use vectorized ops
- `inplace=True` → use immutable operations
- `except:` → use specific exceptions

## Gotchas
- Binance API rate limit: 1200 req/min (초과 시 IP 밴)
- 소수점 정밀도: Decimal 모듈 사용 필수, float 금지
- 복잡한 아키텍처 변경 전 반드시 clarifying questions 요청할 것