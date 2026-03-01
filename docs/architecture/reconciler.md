# PositionReconciler

거래소 실제 포지션과 PM(PortfolioManager) 내부 상태를 주기적으로 교차 검증합니다.
네트워크 장애, 부분 체결, 봇 재시작 등으로 인한 포지션 불일치를 감지하고,
선택적으로 자동 보정(auto-correction)할 수 있습니다.

---

## 1. 왜 필요한가

```text
PM 포지션 != 거래소 포지션 시나리오:
  1. 네트워크 장애로 Fill 이벤트 누락
  2. 부분 체결 후 잔량 처리 실패
  3. 봇 재시작 시 PM 초기 상태 불일치
  4. 수동 거래소 조작 (orphan 포지션)
  5. Cancel race — 취소와 체결 동시 발생
```

---

## 2. 아키텍처

```text
LiveRunner (60초 주기)
       │
       ▼
  PositionReconciler
       │
  ┌────┴────┐
  │         │
initial   periodic
check     check
  │         │
  ▼         ▼
_compare()
  ├─ 거래소 포지션 조회 (fetch_positions)
  ├─ PM 포지션 집계 (composite key → raw symbol)
  ├─ 심볼별 drift 계산
  ├─ DriftDetail 수집 → Discord 알림
  └─ auto_correct → PM 상태 보정 (optional)

check_balance()
  ├─ PM equity vs 거래소 잔고
  └─ drift > 5% → CRITICAL 로그
```

---

## 3. Drift 감지

### 3.1 Position Drift

| 임계값 | 동작 |
|--------|------|
| `< 2%` | 정상 — 무시 |
| `>= 2%` | CRITICAL — 경고 발행 |
| `>= 10%` | auto_correct 활성화 시 PM 보정 |

### 3.2 Balance Drift

| 임계값 | 동작 |
|--------|------|
| `< 2%` | 정상 |
| `2~5%` | WARNING 로그 |
| `> 5%` | CRITICAL 로그 |

### 3.3 Orphan 감지

PM과 거래소 중 한쪽만 포지션을 보유한 경우 orphan으로 판별합니다.

---

## 4. Hedge Mode Composite Key

Orchestrator 환경에서 동일 심볼이 여러 Pod에 분산되면 PM은
composite key (`pod_id|symbol`)로 포지션을 관리합니다.

```text
PM positions:
  "pod-anchor|DOGE/USDT" → LONG 100
  "pod-donch|BTC/USDT"   → LONG 0.5

Reconciler _aggregate_pm_positions():
  DOGE/USDT → long_size: 100
  BTC/USDT  → long_size: 0.5

거래소와 raw symbol 기준으로 비교
```

---

## 5. Auto-Correction

`PositionReconciler(auto_correct=True)` 설정 시:

- drift > 10% (`_AUTO_CORRECT_THRESHOLD`)인 포지션만 보정
- PM의 `size`를 거래소 값으로 덮어씀
- 거래소 포지션이 0이면 PM도 NEUTRAL + size=0
- `corrections_applied` 카운터 증가 + WARNING 로그

**주의**: 기본값은 `auto_correct=False` (safety-first, 경고만 발행).

---

## 6. 검증 흐름

### 6.1 시작 시 (initial_check)

```text
봇 시작 → initial_check()
  ├─ 불일치 없음 → "All positions match" 로그
  └─ 불일치 발견 → CRITICAL 로그 + 불일치 심볼 리스트 반환
```

### 6.2 주기적 (periodic_check)

LiveRunner에서 60초마다 호출. 실패 시 빈 리스트 반환 (봇 중단 방지).

### 6.3 잔고 검증 (check_balance)

PM `total_equity` vs 거래소 USDT total 비교. drift 비율을 `last_balance_drift_pct`에 저장.

---

## 7. Discord 알림

`src/notification/reconciler_formatters.py`의 `DriftDetail` dataclass로 알림 포맷팅:

- `symbol`, `pm_size`, `pm_side`, `exchange_size`, `exchange_side`
- `drift_pct`, `is_orphan`, `auto_corrected`

---

## 8. Code Map

```text
src/eda/
├── reconciler.py                  # PositionReconciler, ExchangePositionInfo
└── live_runner.py                 # 60초 주기 호출

src/notification/
└── reconciler_formatters.py       # DriftDetail, Discord 포맷팅
```

---

## 9. Parse 메서드 요약

| 메서드 | 용도 | 반환형 |
|--------|------|--------|
| `parse_exchange_positions()` | One-way mode 기본 | `{symbol: (size, Direction)}` |
| `parse_exchange_positions_full()` | One-way + entry_price | `{symbol: ExchangePositionInfo}` |
| `parse_exchange_positions_hedge()` | Hedge mode (long/short 분리) | `{symbol: {long_size, short_size}}` |
| `parse_exchange_positions_hedge_full()` | Hedge + entry_price | `{symbol: {long: Info, short: Info}}` |
