# Exchange Safety Stop (거래소 안전망)

소프트웨어 SL/TS가 봇 장애 시 무력화되는 문제를 거래소 STOP_MARKET 주문으로 보완합니다.

---

## 문제

현재 EDA 시스템의 Stop Loss / Trailing Stop은 **100% 소프트웨어 기반**입니다.
PM이 매 1분/TF bar마다 가격을 체크하고 Market order로 청산합니다.

| 시나리오 | SW SL/TS | Exchange Safety Stop |
|---------|----------|---------------------|
| 정상 운영 | 발동 | 발동 안 함 (더 넓음) |
| 봇 프로세스 크래시 | **무력화** | 거래소가 보호 |
| 네트워크 단절 | **무력화** | 거래소가 보호 |
| 서버 전원 차단 | **무력화** | 거래소가 보호 |

24/7 크립토 시장에서 봇 장애는 곧 **무방비 상태**를 의미합니다.

---

## 해결: Hybrid SL/TS

기존 소프트웨어 SL/TS를 유지하면서, 거래소 STOP_MARKET 주문을 **안전망**으로 추가합니다.
SW SL보다 약간 넓게 설정하여 정상 상황에서는 소프트웨어가 먼저 청산하고, 봇 장애 시에만 거래소 주문이 보호합니다.

```
정상:  1m bar → SW SL 발동 (10%) → Market close → Exchange stop (12%) 취소
장애:  봇 다운 → 가격 급락 → Exchange STOP_MARKET (12%) 발동 → 보호
```

### 동작 원리

1. **진입 Fill** 감지 → 거래소에 `STOP_MARKET(closePosition=true)` 주문 배치
2. **매 Bar** → Trailing Stop price 재계산 → 0.5%+ 변동 시에만 cancel+create 업데이트
3. **청산 Fill** 감지 → 거래소 stop 취소
4. **봇 Shutdown** → 기본적으로 거래소 stop **유지** (재시작까지 보호 지속)

### Ratchet 규칙

안전망이 좁아지는 방향으로는 절대 이동하지 않습니다.

| 포지션 | 허용 | 차단 |
|--------|------|------|
| LONG | stop price 올리기 (수익 보호 강화) | stop price 내리기 |
| SHORT | stop price 내리기 (수익 보호 강화) | stop price 올리기 |

---

## 설정

`portfolio` 섹션에 5개 필드를 추가합니다. **`use_exchange_safety_stop: true`만 설정하면 기본값으로 동작합니다.**

```yaml
# config/live.yaml
portfolio:
  system_stop_loss: 0.10              # SW SL 10%
  use_trailing_stop: true
  trailing_stop_atr_multiplier: 3.0   # SW TS 3x ATR

  # Exchange Safety Net (Live 모드 전용)
  use_exchange_safety_stop: true      # 활성화 (기본: false)
  exchange_safety_margin: 0.02        # SL 대비 추가 마진 2% (기본)
  exchange_trailing_safety_margin: 0.005  # TS 대비 추가 마진 0.5% (기본)
  exchange_stop_update_threshold: 0.005   # 업데이트 임계값 0.5% (기본)
  cancel_stops_on_shutdown: false     # Shutdown 시 stop 유지 (기본)
```

### 설정 필드 상세

| 필드 | 기본값 | 범위 | 설명 |
|------|--------|------|------|
| `use_exchange_safety_stop` | `false` | bool | 활성화 여부. Live 모드에서만 동작 |
| `exchange_safety_margin` | `0.02` | 0.0~0.10 | SL stop에 추가할 마진 |
| `exchange_trailing_safety_margin` | `0.005` | 0.0~0.05 | TS stop에 추가할 마진 |
| `exchange_stop_update_threshold` | `0.005` | 0.001~0.05 | 이 비율 이상 변동 시에만 거래소 stop 업데이트 |
| `cancel_stops_on_shutdown` | `false` | bool | `true`: shutdown 시 stop 취소. `false`: 유지 (권장) |

### Stop Price 계산

**SL Stop (System Stop Loss 기반):**

```
LONG:  entry_price * (1 - system_stop_loss - exchange_safety_margin)
SHORT: entry_price * (1 + system_stop_loss + exchange_safety_margin)
```

예시 (LONG, entry $50,000, SL 10%, margin 2%):
- SW SL: $50,000 * 0.90 = **$45,000**
- Exchange stop: $50,000 * 0.88 = **$44,000**
- Gap: $1,000 (2%) — 봇이 살아있으면 SW가 먼저 발동

**TS Stop (Trailing Stop 기반):**

```
LONG:  (peak_price - ATR * multiplier) * (1 - exchange_trailing_safety_margin)
SHORT: (trough_price + ATR * multiplier) * (1 + exchange_trailing_safety_margin)
```

**Combined (SL + TS 모두 활성):**
- LONG: `min(sl_stop, ts_stop)` — 더 넓은 쪽 (안전망이므로)
- SHORT: `max(sl_stop, ts_stop)` — 더 넓은 쪽

---

## 적용 범위

| 모드 | 동작 |
|------|------|
| `paper` | 비활성 (거래소 API 불필요) |
| `shadow` | 비활성 (실행 없음) |
| `live` | **활성** (`use_exchange_safety_stop: true` 시) |
| EDA backtest | 비활성 |

---

## 운영 시나리오

### 정상 운영

```
Entry fill → Exchange stop 배치 ($44,000)
  ↓
매 bar → SW SL/TS 체크 + Exchange stop 가격 업데이트 (0.5%+ 변동 시)
  ↓
SW SL 발동 → Market close → Exchange stop 취소
```

### 봇 장애

```
Entry fill → Exchange stop 배치 ($44,000)
  ↓
봇 크래시 / 네트워크 단절
  ↓
가격 $44,000 이하로 하락
  ↓
거래소 STOP_MARKET 자동 발동 → closePosition=true → 전량 청산
```

### 봇 재시작

```
재시작 → State 복구 (SQLite에서 PM/RM/OMS/stop 상태 로드)
  ↓
★ Reconciliation: 거래소 포지션 조회 → PM phantom positions 제거 → cash 재조정
  ↓
verify_exchange_stops: 거래소 stop 존재 확인 → stale 제거
  ↓
★ place_missing_stops: 포지션 있는데 stop 없으면 재배치
  ↓
initial_check: 최종 일치 검증 → 불일치 시 Discord 알림
```

### Direction Flip (LONG → SHORT)

```
Exit fill → 기존 LONG stop 취소
  ↓
Entry fill → 새 SHORT stop 배치
```

---

## Edge Cases

| 시나리오 | 처리 방식 |
|---------|----------|
| SW SL 발동 → exchange stop 취소 실패 | 무시 (exchange stop이 더 넓어서 안 걸림, 다음 reconcile에서 정리) |
| Exchange stop 체결 (봇 alive) | Reconciler가 position 소멸 감지 → PM auto_correct |
| ATR 미성숙 (14봉 미달) | SL stop만 사용 (TS stop = None) |
| `system_stop_loss = None` | Safety stop 비활성 (배치 안 함) |
| API 연속 5회 실패 | CRITICAL 로그, 다음 bar에서 재시도 (거래 차단 없음) |
| Multi-asset | 심볼별 독립 관리, rate limit 우려 없음 (드문 업데이트) |
| Preflight (기존 stop 존재) | `safety-stop-` prefix 주문은 stale order 경고에서 제외 |
| 수동 청산 후 재시작 | reconcile이 phantom 제거 → 정확한 cash 역산 → 정상 동작 |
| 거래소 API 실패 (재시작 시) | reconcile skip → 기존 동작 유지 (경고만, safety-first) |

---

## 아키텍처

```
ExchangeStopManager (독립 클래스)
  ├── EventBus 구독: FILL, BAR
  ├── 의존: BinanceFuturesClient, EDAPortfolioManager (읽기 전용)
  ├── 설정: PortfolioManagerConfig
  └── 상태: StateManager (SQLite 영속화)

등록 순서: PM → RM → OMS → Analytics → ExchangeStopManager (후순위)
```

- `ExchangeStopManager`는 PM/OMS 이후에 등록되어, fill 처리가 완료된 후 stop을 관리합니다
- PM 로직은 **전혀 변경 없음** — backtest-live parity 보존
- `closePosition=true` 사용으로 position size drift 무관하게 전량 청산

### 관련 파일

| 파일 | 역할 |
|------|------|
| `src/eda/exchange_stop_manager.py` | 핵심 모듈 — stop lifecycle 관리 |
| `src/portfolio/config.py` | 설정 필드 5개 |
| `src/exchange/binance_futures_client.py` | `create_stop_market_order()`, `cancel_all_symbol_orders()` |
| `src/eda/persistence/state_manager.py` | 상태 영속화 |
| `src/eda/live_runner.py` | 통합 (생성, 등록, shutdown, preflight) |
