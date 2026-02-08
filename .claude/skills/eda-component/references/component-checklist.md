# EDA 컴포넌트 배포 전 체크리스트

새 EDA 컴포넌트 추가 시 아래 항목을 **모두** 확인한다.

---

## 이벤트 정의

- [ ] `EventType` enum에 새 타입 추가
- [ ] Flat frozen Pydantic model 정의 (`model_config = ConfigDict(frozen=True)`)
- [ ] `Literal[EventType.XXX]` 필드로 타입 식별
- [ ] `AnyEvent` union에 추가
- [ ] `EVENT_TYPE_MAP`에 등록
- [ ] `NEVER_DROP_EVENTS` 또는 `DROPPABLE_EVENTS` 분류 결정
- [ ] `timestamp` 필드 포함
- [ ] `correlation_id: str | None = None` 필드 포함

## 핸들러 구현

- [ ] `register(bus)` 메서드로 구독 등록
- [ ] `event.event_type` 방어적 검증
- [ ] `correlation_id` 전파 (파생 이벤트 발행 시)
- [ ] 핸들러 내 에러가 내부 상태를 오염시키지 않음
- [ ] 순환 이벤트 방지 (A → B → A 무한 루프)

## EventBus 통합

- [ ] `await bus.flush()` 호출 시점 확인
- [ ] DROPPABLE 이벤트는 `put_nowait` → 큐 포화 시 드롭 인지
- [ ] NEVER_DROP 이벤트는 `await put()` → 블로킹 가능 인지
- [ ] 큐 크기 충분성 (기본 10000)

## Runner 통합

- [ ] 생성 순서: Analytics → PM → RM → OMS → Strategy → 새 컴포넌트
- [ ] `await register()` 호출 추가
- [ ] 멀티에셋 batch mode와 호환 (해당 시)
- [ ] `flush_pending_signals()` 호출 영향 없음

## 테스트

- [ ] 단위 테스트: 핸들러 로직
- [ ] 통합 테스트: EventBus를 통한 이벤트 체인
- [ ] 에러 격리 테스트: 핸들러 에러 시 다른 핸들러 정상 동작
- [ ] async fixture 사용 (`@pytest.mark.asyncio`)
- [ ] 멀티에셋 시나리오 (해당 시)

## 코드 품질

- [ ] `from __future__ import annotations`
- [ ] TYPE_CHECKING 블록 활용
- [ ] ruff + pyright 0 error
- [ ] loguru 로깅 (중요 상태 변경)
- [ ] type hints 모든 public 메서드

## 문서

- [ ] event-chain.md 다이어그램 업데이트 (해당 시)
- [ ] MEMORY.md에 gotcha 기록 (발견 시)

---

## 자주 하는 실수 (Anti-patterns)

### 1. AnyEvent union 미등록

```python
# ❌ 새 이벤트 모델은 만들었지만 AnyEvent에 미추가
# → EventBus의 JSONL 감사 로그에서 직렬화 실패
```

### 2. DROPPABLE 오분류

```python
# ❌ 주문 관련 이벤트를 DROPPABLE로 분류
# → 큐 포화 시 주문 누락 → 포지션 불일치
```

### 3. flush 미호출

```python
# ❌ DataFeed에서 bar 연속 발행 후 flush 없음
# → 모든 파생 이벤트가 마지막 bar 가격으로 체결
```

### 4. Equity 이중 계산

```python
# ❌ total_equity = cash + notional + unrealized
# notional(=size*price)에 이미 unrealized 포함
# → equity가 실제의 2배로 계산됨 → 포지션 사이징 오류
```

### 5. ATR 계산 순서

```python
# ❌ pos.last_price = current_price 먼저 → ATR의 prev_close가 잘못됨
# → trailing stop 거리가 비정상
```

### 6. _stopped_this_bar 미체크

```python
# ❌ stop-loss 발동 후 같은 bar에서 새 시그널로 재진입
# → stop-loss가 무력화됨
```
