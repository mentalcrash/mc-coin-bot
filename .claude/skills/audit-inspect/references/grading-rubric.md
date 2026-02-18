# Grading Rubric

6카테고리별 A~F 등급 기준. Overall은 가중 평균 (risk-safety 2배).

## 등급 점수 변환

| Grade | Score | Grade | Score |
|-------|-------|-------|-------|
| A+    | 4.3   | C+    | 2.3   |
| A     | 4.0   | C     | 2.0   |
| A-    | 3.7   | C-    | 1.7   |
| B+    | 3.3   | D     | 1.0   |
| B     | 3.0   | F     | 0.0   |
| B-    | 2.7   |       |       |

## Overall 계산

```
weighted_sum = (arch + risk*2 + quality + data + test + perf) / 7
```

score -> grade 역변환: 가장 가까운 등급 선택.

---

## 1. Architecture

| Grade | 기준 |
|-------|------|
| A     | 의존성 단방향 (CLI -> Strategy/Backtest -> Data/Exchange/Portfolio -> Models/Core -> Config). 결합도 낮음. 모듈간 인터페이스 명확 (Port/Protocol). 순환 import 0건 |
| B     | 대부분 단방향. 경미한 coupling 1~2건. 인터페이스 일부 암시적 |
| C     | 순환 참조 존재. 모듈 경계 불명확. 직접 의존 3건+ |
| D     | 근본적 아키텍처 결함. 레이어 붕괴. 모듈 재설계 필요 |

**체크리스트**:

- [ ] 의존성 방향 위반 검사 (`from src.cli` import in `src/eda` 등)
- [ ] EventBus 이벤트 타입 결합도
- [ ] Port/Protocol 패턴 준수 (ExecutorPort, DataFeedPort)

---

## 2. Risk-Safety

| Grade | 기준 |
|-------|------|
| A     | 모든 live trading path에 방어적 코드. PM/RM/OMS 3단 방어 완비. assert 대신 if/raise. 자금 관련 edge case 모두 처리 |
| B     | 대부분 방어적. 경미한 결함 1~2건. 자금 손실 위험 없음 |
| C     | Critical 결함 존재 (assert crash, cash negative, persistence 미비). 자금 손실 간접 위험 |
| D     | 자금 손실 직접 위험. 방어 코드 부재. 즉시 수정 필수 |

**체크리스트**:

- [ ] `assert` 사용 in production code (src/eda, src/exchange)
- [ ] OMS idempotency 보장 (restart 시 pending order 유실?)
- [ ] Cash negative guard (음수 잔고 방지)
- [ ] Circuit Breaker 상태 persistence
- [ ] Rate limit 방어 (Binance 1200 req/min)
- [ ] Decimal precision (float 대신 Decimal)
- [ ] Position reconciliation drift threshold

---

## 3. Code-Quality

| Grade | 기준 |
|-------|------|
| A     | lint 0, type 0, coverage 90%+. `# noqa`/`# type: ignore` 0건 (정당 사유 제외). 명확한 상수. 단일 책임 |
| B     | lint 0, type 0, coverage 80%+. noqa 최소. 코드 스타일 일관 |
| C     | lint/type 에러 소수. coverage 70% 미만. noqa 남용 |
| D     | lint/type 에러 다수. coverage 50% 미만. 코드 가독성 낮음 |

**체크리스트**:

- [ ] `ruff check .` — 0 errors
- [ ] `pyright src/` — 0 errors
- [ ] Coverage >= 80%
- [ ] `# noqa` 사용 횟수 및 정당성
- [ ] `# type: ignore` 사용 횟수 및 정당성
- [ ] Magic number 잔존 여부
- [ ] Bare except 사용 여부

---

## 4. Data-Pipeline

| Grade | 기준 |
|-------|------|
| A     | Medallion (Bronze/Silver) 완비. Gap filling 검증. 타임프레임 변환 정확. Data validation 자동화 |
| B     | 기본 파이프라인 정상 작동. 경미한 이슈. 검증 대부분 자동화 |
| C     | 데이터 이슈 존재 (gap, timezone, NaN). 수동 검증 필요 |
| D     | 데이터 파이프라인 불안정. 잘못된 데이터로 백테스트 위험 |

**체크리스트**:

- [ ] Bronze -> Silver 파이프라인 동작
- [ ] Gap detection/filling
- [ ] Timezone 일관성 (UTC)
- [ ] NaN 처리 정책 명확
- [ ] `ingest validate` 통과

---

## 5. Testing-Ops

| Grade | 기준 |
|-------|------|
| A     | CI/CD 완비. Coverage 90%+. Integration test 존재. 모든 critical path 테스트. Fixture 재사용 |
| B     | Coverage 80%+. CI 존재. Unit test 대부분. Integration 일부 |
| C     | Coverage 70% 미만. CI 일부 누락. Critical path 테스트 미비 |
| D     | 테스트 부실. CI 없음 또는 비활성 |

**체크리스트**:

- [ ] pytest 전체 통과
- [ ] Coverage 수치
- [ ] CI/CD 설정 존재 (GitHub Actions 등)
- [ ] Live trading path integration test 존재 여부
- [ ] Fixture 품질 (conftest.py 구조)

---

## 6. Performance

| Grade | 기준 |
|-------|------|
| A     | Numba JIT 활용. 벡터화 완전. 대용량 데이터(2년+) 처리 가능. 메모리 효율적 |
| B     | 대부분 벡터화. Numba 일부. 성능 이슈 미미 |
| C     | for 루프 존재. 대용량 처리 시 지연. Numba 미활용 |
| D     | 성능 병목 다수. iterrows(). 대용량 처리 불가 |

**체크리스트**:

- [ ] `iterrows()` 사용 여부
- [ ] `for i in range(len(df))` 사용 여부
- [ ] Numba `@njit` 활용 여부
- [ ] 대용량 데이터 처리 시간 (2년 8종목)
- [ ] 메모리 사용량 합리성
