# Module Health Checklist

모듈별 건강도 평가 시 확인해야 할 항목.

## 공통 검사 항목

모든 모듈에 적용:

| # | 항목 | 검사 방법 | 심각도 |
|---|------|----------|--------|
| M1 | `assert` in production | `Grep "assert " src/{module}/` (test 제외) | HIGH (live path) / MEDIUM (기타) |
| M2 | Bare `except:` | `Grep "except:" src/{module}/` | MEDIUM |
| M3 | `# noqa` 남용 | `Grep "# noqa" src/{module}/` | LOW |
| M4 | `# type: ignore` 남용 | `Grep "type: ignore" src/{module}/` | LOW |
| M5 | TODO/FIXME 잔존 | `Grep "TODO\|FIXME" src/{module}/` | LOW |
| M6 | 순환 import | import 방향 분석 | HIGH |
| M7 | Magic number | 하드코딩 상수 검사 | LOW |
| M8 | 커버리지 | pytest --cov 결과 | 80% 미만: YELLOW, 70% 미만: RED |

## 모듈별 추가 검사

### src/eda (EDA 이벤트 시스템)

| # | 항목 | 기대 |
|---|------|------|
| E1 | EventBus flush 패턴 | DataFeed에서 `await bus.flush()` 호출 |
| E2 | BacktestExecutor deferred execution | price=None → pending, SL/TS → 즉시 |
| E3 | PM stop-loss guard | `_stopped_this_bar` set으로 재진입 방지 |
| E4 | OMS idempotency | `client_order_id` 기반 중복 방지 |
| E5 | RM circuit breaker | equity drawdown threshold → close-all |
| E6 | Batch mode | multi-asset에서 `flush_pending_signals()` |

### src/exchange (거래소 연동)

| # | 항목 | 기대 |
|---|------|------|
| X1 | Rate limit retry | `_retry_with_backoff()`, RateLimitExceeded 먼저 catch |
| X2 | Hedge mode setup | `setup_account()` idempotent |
| X3 | Position reconciliation | drift threshold 5% |
| X4 | Error hierarchy | InsufficientFundsError, OrderExecutionError |
| X5 | assert 사용 금지 | live path에서 assert → crash risk |

### src/strategy (전략 모듈)

| # | 항목 | 기대 |
|---|------|------|
| S1 | shift(1) rule | 모든 시그널 전봉 기반 |
| S2 | Vectorized ops | for loop 금지 |
| S3 | ShortMode 분기 | DISABLED/HEDGE_ONLY/FULL 3종 |
| S4 | StrategySignals 반환 | (entries, exits, direction, strength) |
| S5 | frozen Config | Pydantic frozen=True |

### src/data (데이터 파이프라인)

| # | 항목 | 기대 |
|---|------|------|
| D1 | Medallion architecture | Bronze → Silver 분리 |
| D2 | Gap detection | 결측 탐지 + 보간 |
| D3 | Timezone UTC | 모든 timestamp UTC |
| D4 | Client 재사용 | `self._client` 패턴 |

### src/pipeline (전략 관리)

| # | 항목 | 기대 |
|---|------|------|
| P1 | YAML serialization | `model_dump(mode="json")` 사용 |
| P2 | Store caching | lazy cache 패턴 |
| P3 | Gate 순서 gap 허용 | `continue` not `break` |

### src/notification (알림 시스템)

| # | 항목 | 기대 |
|---|------|------|
| N1 | SpamGuard | 5분 cooldown |
| N2 | Retry logic | 3회 exponential backoff |
| N3 | Queue bounded | asyncio.Queue(500) |
| N4 | TYPE_CHECKING imports | Discord.py 런타임 분리 |

### src/core (핵심 모듈)

| # | 항목 | 기대 |
|---|------|------|
| C1 | Flat events | 상속 없이 독립 모델 |
| C2 | AnyEvent union | 모든 이벤트 타입 포함 |
| C3 | EventBus isolation | handler error 격리 |
| C4 | Bounded backpressure | Queue maxsize |

### src/models (Pydantic 모델)

| # | 항목 | 기대 |
|---|------|------|
| MD1 | frozen=True | 불변 모델 |
| MD2 | Field validators | 적절한 검증 |
| MD3 | StrEnum 사용 | 문자열 직렬화 호환 |

## 건강도 종합 판정

```
if coverage >= 80% AND 코드리뷰 이슈 0건:
    GREEN
elif coverage >= 70% OR 경미한 이슈만:
    YELLOW
else:
    RED
```
