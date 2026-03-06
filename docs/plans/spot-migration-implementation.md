# Spot SuperTrend Migration — Implementation Plan

> 상위 문서: [spot-supertrend-migration.md](spot-supertrend-migration.md)
>
> 이 문서는 Futures Multi-Strategy Orchestrator → Spot Single-Strategy 전환의
> 상세 구현 계획을 다룬다. 각 Phase별 작업 목록, 파일 변경 사항, 검증 기준을 포함한다.

---

## 목차

- [Phase 0: 코드베이스 정리 (Cleanup)](#phase-0-코드베이스-정리-cleanup)
- [Phase 1: Spot 핵심 구현](#phase-1-spot-핵심-구현)
- [Phase 2: Notification / Discord 최적화](#phase-2-notification--discord-최적화)
- [Phase 3: Monitoring (Prometheus/Grafana) 최적화](#phase-3-monitoring-prometheusgrafana-최적화)
- [Phase 4: 문서 / Rules / Skills 정리](#phase-4-문서--rules--skills-정리)
- [Phase 5: Paper Trading 검증](#phase-5-paper-trading-검증)
- [Phase 6: Live 배포](#phase-6-live-배포)
- [Appendix A: 파일 변경 총괄표](#appendix-a-파일-변경-총괄표)
- [Appendix B: 삭제 대상 전략 목록](#appendix-b-삭제-대상-전략-목록)

---

## Phase 0: 코드베이스 정리 (Cleanup)

> **목표**: 사용하지 않는 모듈/전략/설정/스크립트를 정리하여
> 코드베이스를 Spot Single-Strategy에 맞는 규모로 축소한다.
>
> **원칙**: 삭제가 아닌 **별도 브랜치 보관** 후 main에서 제거.
> `git tag archive/futures-multi-strategy` 로 현재 상태를 태깅한 뒤 진행.

### 0.1. 사전 작업

| # | 작업 | 명령/설명 |
|---|------|----------|
| 0.1.1 | 현재 상태 태깅 | `git tag archive/futures-multi-strategy` |
| 0.1.2 | 새 브랜치 생성 | `git checkout -b refactor/spot-migration` |
| 0.1.3 | 전체 테스트 통과 확인 | `uv run pytest` (현재 7,600+ tests) |

### 0.2. Orchestrator 제거

**대상**: `src/orchestrator/` (23 파일)

```
src/orchestrator/
  __init__.py, orchestrator.py, pod.py, lifecycle.py, config.py,
  allocator.py, asset_allocator.py, risk_aggregator.py, vol_targeting.py,
  netting.py, metrics.py, models.py, result.py, surveillance.py,
  backtest_surveillance.py, degradation.py, dashboard.py,
  state_persistence.py, allocation_comparator.py, asset_selector.py,
  regime_filter.py, dd_derisk.py, volume_matrix.py
```

**의존성 정리** (제거 전 수정 필요):

| # | 파일 | 변경 내용 |
|---|------|----------|
| 0.2.1 | `src/backtest/engine.py` | `AssetAllocationConfig` import 제거, 관련 multi-asset 로직 정리 |
| 0.2.2 | `src/backtest/request.py` | `AssetAllocationConfig` import 제거 |
| 0.2.3 | `src/eda/orchestrated_runner.py` | 파일 전체 삭제 |
| 0.2.4 | `src/eda/runner.py` | Orchestrator TYPE_CHECKING import 제거 |
| 0.2.5 | `src/cli/orchestrate.py` | 파일 전체 삭제 |
| 0.2.6 | `src/cli/__init__.py` | `orchestrate_app` 등록 제거 |
| 0.2.7 | `src/config/orchestrator_loader.py` | 파일 전체 삭제 |
| 0.2.8 | `src/notification/orchestrator_engine.py` | 파일 전체 삭제 |
| 0.2.9 | `src/notification/orchestrator_formatters.py` | 파일 전체 삭제 |
| 0.2.10 | `src/notification/bot.py` | Orchestrator slash commands 제거 (`/strategies`, `/strategy`) |
| 0.2.11 | `src/notification/report_scheduler.py` | Orchestrator report 로직 제거 |
| 0.2.12 | `src/portfolio/risk_monitor.py` | Orchestrator import 제거 |
| 0.2.13 | `src/eda/live_runner.py` | `orchestrated_paper()`, `orchestrated_live()` 팩토리 제거 |

**검증**: `uv run pytest` + `uv run ruff check .` + `uv run pyright src/`

### 0.3. Regime 제거

**대상**: `src/regime/` (9 파일)

```
src/regime/
  __init__.py, config.py, detector.py, ensemble.py, service.py,
  hmm_detector.py, msar_detector.py, vol_detector.py, derivatives_detector.py
```

**의존성 정리**:

| # | 파일 | 변경 내용 |
|---|------|----------|
| 0.3.1 | `src/eda/runner.py` | `RegimeService` TYPE_CHECKING import 제거, 관련 파라미터 제거 |
| 0.3.2 | `src/eda/strategy_engine.py` | `RegimeService` optional 참조 제거 |
| 0.3.3 | `src/eda/live_runner.py` | `RegimeService` 초기화/등록 코드 제거 |
| 0.3.4 | `src/core/events.py` | `RegimeContext` TYPE_CHECKING import 제거 |
| 0.3.5 | `src/cli/pipeline.py` | Regime 관련 type import 제거 |

### 0.4. FeatureStore 제거

**대상**: `src/market/feature_store.py` (1 파일)

**의존성 정리**:

| # | 파일 | 변경 내용 |
|---|------|----------|
| 0.4.1 | `src/eda/runner.py` | `FeatureStore` TYPE_CHECKING import 및 초기화 제거 |
| 0.4.2 | `src/eda/strategy_engine.py` | `FeatureStore` optional 참조 제거 |
| 0.4.3 | `src/eda/live_runner.py` | `FeatureStore` 초기화 코드 제거 |
| 0.4.4 | `src/eda/ports.py` | `FeatureStorePort` Protocol 제거 |
| 0.4.5 | `src/cli/backtest.py` | `FeatureStore` type import 제거 |

### 0.5. 불필요한 전략 제거

**대상**: `src/strategy/` 하위 190개 디렉토리 중 `supertrend/` 외 전부

**유지 파일**:
- `src/strategy/__init__.py` (수정: supertrend만 import)
- `src/strategy/registry.py` (유지)
- `src/strategy/base.py` (유지)
- `src/strategy/types.py` (유지)
- `src/strategy/supertrend/` (5 파일 유지)

**작업**:

| # | 작업 | 설명 |
|---|------|------|
| 0.5.1 | `__init__.py` 수정 | 189개 strategy import 제거, supertrend 1개만 유지 |
| 0.5.2 | 전략 디렉토리 삭제 | `src/strategy/*/` (supertrend 제외) 일괄 삭제 |

### 0.6. 불필요한 Data Feed 제거

**대상**: 대안 데이터 피드 (Spot 단순 전략에 불필요)

| # | 파일 | 설명 |
|---|------|------|
| 0.6.1 | `src/eda/derivatives_feed.py` | Funding Rate/OI — Spot 불필요 |
| 0.6.2 | `src/eda/onchain_feed.py` | On-chain — 미사용 |
| 0.6.3 | `src/eda/macro_feed.py` | Macro — 미사용 |
| 0.6.4 | `src/eda/options_feed.py` | Options — 미사용 |
| 0.6.5 | `src/eda/deriv_ext_feed.py` | Extended Derivatives — 미사용 |
| 0.6.6 | `src/eda/_feed_metrics.py` | Feed 진단 (필요 시 유지) |
| 0.6.7 | `src/eda/live_runner.py` | 위 피드들의 초기화/정지 코드 제거 |

### 0.7. SmartExecutor / Futures 관련 제거

| # | 파일 | 설명 |
|---|------|------|
| 0.7.1 | `src/eda/smart_executor.py` | Futures Limit Order Decorator — Spot 구조 다름 |
| 0.7.2 | `src/eda/smart_executor_config.py` | SmartExecutor 설정 |
| 0.7.3 | `src/exchange/binance_futures_client.py` | Futures 클라이언트 — Spot 전환 |

### 0.8. 불필요한 설정/스크립트 정리

**Config 정리** (`config/`):

| 유지 | 삭제/보관 |
|------|----------|
| `spot_supertrend.yaml` | `default.yaml` (TSMOM 1D) |
| `.env.example` | `paper.yaml` (Futures) |
| | `orchestrator-live.yaml` |
| | `ensemble-example.yaml` |
| | `supertrend_*.yaml` (11개, 개별 에셋 구버전) |
| | `donch-multi_*.yaml`, `tri-channel-trend_*.yaml` 등 |
| | `ma_cross_btc.yaml`, `ma_st_cross_btc.yaml`, `st_donch_btc.yaml` |
| | `sweep_vwap/` (9개) |

**Scripts 정리** (`scripts/`):

| 유지 | 삭제/보관 |
|------|----------|
| `healthcheck.py` (Docker) | `st_v11_*.py` (5개, 구 SuperTrend) |
| `st_adx_*.py` (4개, 분석 기록) | `sweep_*.py` (3개, 파라미터 탐색) |
| | `tri_channel_20assets.py` |
| | `evaluate_regime_detector.py`, `compare_regime_detectors.py` |
| | `stitch_matic_pol.py` |
| | `dune/` (전체) |

### 0.9. 테스트 정리

- Orchestrator/Regime/FeatureStore/삭제 전략 관련 테스트 파일 제거
- 잔여 테스트 전체 통과 확인: `uv run pytest`
- lint/typecheck 통과: `uv run ruff check . && uv run pyright src/`

### Phase 0 완료 기준

- [x] `uv run pytest` 전체 통과 (1963 passed, 4 skipped)
- [x] `uv run ruff check .` 0 errors
- [x] `uv run pyright src/` 0 errors
- [x] `git diff --stat` — 1999 files changed, 228,519 lines deleted
- [x] 남은 strategy: supertrend 1개만 등록 확인

---

## Phase 1: Spot 핵심 구현

> **목표**: BinanceSpotClient + SpotExecutor + SpotStopManager 구현.
> LiveRunner에 Spot 모드 팩토리 추가.

### 1.1. BinanceSpotClient 신규 (`src/exchange/binance_spot_client.py`)

**BinanceFuturesClient 패턴 차용, Spot 전용 축소판.**

```python
class BinanceSpotClient:
    """Binance Spot 주문 클라이언트.

    async with 패턴 + CCXT Pro.
    Futures 대비 차이: reduceOnly/positionSide/leverage 없음.
    """
```

| 메서드 | 설명 |
|--------|------|
| `__aenter__` / `__aexit__` | CCXT 라이프사이클 (load_markets 포함) |
| `create_market_buy(symbol, quote_amount)` | Spot Market Buy (USDT 금액 기준, `quoteOrderQty`) |
| `create_market_sell(symbol, base_amount)` | Spot Market Sell (보유 수량 기준) |
| `create_stop_limit_order(symbol, side, amount, stop_price, limit_price)` | Stop-Limit 주문 |
| `cancel_order(symbol, order_id)` | 주문 취소 |
| `fetch_balance()` | 에셋별 잔고 `dict[str, Decimal]` |
| `fetch_open_orders(symbol)` | 미체결 주문 목록 |
| `fetch_ticker(symbol)` | 현재가 조회 |

**구현 포인트**:
- String Protocol 준수: `Decimal` → `amount_to_precision()` → `str`
- RateLimitExceeded → NetworkError 순서 except
- `_retry_with_backoff()` 3회 재시도
- API health 모니터링 (`_consecutive_failures`)
- Client Order ID: `spot_{symbol}_{timestamp}_{nonce}`

**테스트**: Mock CCXT exchange, 주문/잔고/에러 시나리오 10+ cases

### 1.2. SpotExecutor 신규 (`src/eda/spot_executor.py`)

**ExecutorPort Protocol 준수. OMS에서 기존 LiveExecutor 자리 교체.**

```python
class SpotExecutor:
    """Spot 주문 실행기.

    LiveExecutor 대비 차이:
    - reduceOnly / positionSide 없음
    - Market Buy: quoteOrderQty (USDT 금액 기준)
    - Market Sell: 보유 수량 전량 매도
    - Long-Only: SHORT 방향 주문 거부
    """
```

| 메서드 | 설명 |
|--------|------|
| `async execute(order: OrderRequestEvent)` | ExecutorPort 준수 메인 진입점 |
| `_execute_buy(order)` | USDT 금액 → Market Buy |
| `_execute_sell(order)` | 보유 수량 → Market Sell |
| `_parse_fill(ccxt_result, order)` | CCXT 결과 → FillEvent 변환 |
| `set_pm(pm)` | PM 참조 (잔고/포지션 조회용) |

**핵심 로직**:
```
order.target_weight > 0 (진입):
  available_usdt = total_equity * target_weight
  → create_market_buy(symbol, available_usdt)

order.target_weight == 0 (청산):
  held_amount = pm.get_position(symbol).size
  → create_market_sell(symbol, held_amount)
```

**테스트**: 진입/청산/부분체결/잔고부족/API에러 시나리오

### 1.3. SpotStopManager 신규 (`src/eda/spot_stop_manager.py`)

**Stop-Limit Ratchet 방식. EventBus BAR/FILL 구독.**

```python
@dataclass
class SpotStopState:
    symbol: str
    order_id: str
    stop_price: float
    limit_price: float
    quantity: float
    high_watermark: float
    last_atr: float
    created_at: datetime

class SpotStopManager:
    """Stop-Limit TS 거래소 위임 + Ratchet."""
```

| 메서드 | 설명 |
|--------|------|
| `async register(bus)` | FILL, BAR 이벤트 구독 |
| `async _on_fill(fill)` | 진입 체결 → Stop-Limit 설정 |
| `async _on_bar(bar)` | 12H bar → ATR 갱신 → Ratchet 판단 |
| `_compute_stop(high_watermark, atr)` | `hwm - 3.0 * atr` |
| `_compute_limit(stop_price)` | `stop * 0.995` (0.5% 여유) |
| `async _place_stop(symbol, qty, stop, limit)` | Stop-Limit 주문 설정 |
| `async _cancel_and_replace(symbol, new_stop, new_limit)` | 기존 취소 → 재설정 |
| `async _on_signal_close(symbol)` | 반대 신호 → Stop-Limit 취소 |
| `async recover_from_exchange()` | 봇 재시작 시 `fetch_open_orders()` 복구 |

**Ratchet 로직**:
```
새 stop = high_watermark - 3.0 * current_atr
if 새 stop > 기존 stop:
    취소 → 재설정 (ratchet up)
else:
    유지 (절대 아래로 안 내림)
```

**엣지 케이스 처리**:

| 상황 | 처리 |
|------|------|
| Stop 발동 + limit 미체결 (가격 관통) | 미체결 감지 → Market Sell fallback |
| 부분 체결 | 잔량 확인 → 잔량에 대해 재설정 |
| 봇 재시작 | `fetch_open_orders()` → SpotStopState 복구 |
| API 실패 (stop 설정 실패) | 재시도 3회 → Discord CRITICAL 알림 |
| 12H bar 누락 | 마지막 stop 유지 (거래소에 살아있어 안전) |

**테스트**: Ratchet up/유지/봇재시작/관통/부분체결 시나리오

### 1.4. PM Equity 계산 변경 (`src/eda/portfolio_manager.py`)

**Futures vs Spot Equity 차이**:

```
Futures: cash (USDT margin) + unrealized PnL
Spot:    USDT잔고 + sum(에셋 보유량 * 현재가)
```

**변경 사항**:

| # | 변경 | 설명 |
|---|------|------|
| 1.4.1 | `total_equity` 프로퍼티 | Spot 모드: `cash + sum(pos.size * pos.last_price)` |
| 1.4.2 | Short 방향 차단 | `_on_signal()` 에서 target_weight < 0 무시 |
| 1.4.3 | `sync_capital()` | Live Spot: `fetch_balance()` → USDT + 에셋 시가 합산 |

### 1.5. LiveRunner Spot 모드 팩토리 (`src/eda/live_runner.py`)

**기존 `paper()` / `live()` 과 동일 패턴, Executor + StopManager만 교체.**

```python
@classmethod
def spot_paper(
    cls,
    strategy: BaseStrategy,
    symbols: list[str],
    target_timeframe: str,
    config: PortfolioManagerConfig,
    client: BinanceClient,
    initial_capital: float = 10000.0,
    asset_weights: dict[str, float] | None = None,
    ...
) -> LiveRunner:
    """Spot Paper 모드. BacktestExecutor + 실시간 데이터."""

@classmethod
def spot_live(
    cls,
    strategy: BaseStrategy,
    symbols: list[str],
    target_timeframe: str,
    config: PortfolioManagerConfig,
    client: BinanceClient,
    spot_client: BinanceSpotClient,
    asset_weights: dict[str, float] | None = None,
    ...
) -> LiveRunner:
    """Spot Live 모드. SpotExecutor + SpotStopManager."""
```

**초기화 흐름** (기존 `live()` 대비 차이점만):

```
spot_live() 초기화:
  1. LiveDataFeed (기존 BinanceClient — Spot WebSocket 그대로)
  2. SpotExecutor (NEW, BinanceFuturesClient 대신 BinanceSpotClient)
  3. SpotStopManager (NEW, ExchangeStopManager 대신)
  4. PM: hedge_mode=False 고정 (Long-Only)
  5. RM: max_leverage_cap=1.0 고정
  6. 제거: RegimeService, FeatureStore, DerivativesFeed, 기타 피드
  7. 제거: SmartExecutor wrapping
```

### 1.6. CLI Spot 명령 추가 (`src/cli/eda.py`)

```bash
# 기존 (유지)
uv run mcbot eda run config/spot_supertrend.yaml           # EDA Backtest
uv run mcbot eda run-live config/spot_supertrend.yaml       # EDA Paper

# 신규
uv run mcbot eda spot-paper config/spot_supertrend.yaml     # Spot Paper
uv run mcbot eda spot-live config/spot_supertrend.yaml      # Spot Live
```

### Phase 1 완료 기준

- [ ] `BinanceSpotClient` 구현 + 단위 테스트 (mock exchange)
- [ ] `SpotExecutor` 구현 + ExecutorPort 준수 테스트
- [ ] `SpotStopManager` 구현 + Ratchet 로직 테스트
- [ ] PM Spot equity 계산 테스트
- [ ] LiveRunner `spot_paper()` / `spot_live()` 팩토리 동작
- [ ] CLI `spot-paper` / `spot-live` 명령 동작
- [ ] 전체 테스트 통과 + lint + typecheck

---

## Phase 2: Notification / Discord 최적화

> **목표**: Orchestrator 전용 알림 제거, Spot 단일 전략에 맞게 간소화.
> 채널 구조 유지하되 불필요한 포매터/이벤트 정리.

### 2.1. 현재 Discord 구조 분석

**파일 목록** (`src/notification/`, 13 파일, ~100KB):

| 파일 | LOC | 역할 | 처리 |
|------|-----|------|------|
| `discord.py` | ~500 | Webhook 기반 전송 (legacy) | 유지 (fallback) |
| `bot.py` | ~900 | Discord Bot + Slash Commands | **수정** |
| `config.py` | ~80 | DiscordBotConfig | 유지 |
| `engine.py` | ~150 | NotificationEngine (EventBus 구독) | **수정** |
| `queue.py` | ~250 | NotificationQueue + SpamGuard | 유지 |
| `models.py` | ~40 | Severity, ChannelRoute | 유지 |
| `formatters.py` | ~900 | 트레이딩 이벤트 Embed | **수정** |
| `health_formatters.py` | ~350 | Health check Embed | **수정** |
| `health_collector.py` | ~500 | Health 데이터 수집 | **수정** |
| `health_models.py` | ~180 | Health 데이터 모델 | 유지 |
| `orchestrator_engine.py` | ~200 | Orchestrator 알림 | **삭제** |
| `orchestrator_formatters.py` | ~270 | Orchestrator Embed | **삭제** |
| `reconciler_formatters.py` | ~160 | Position Reconciliation Embed | 유지 |

### 2.2. 삭제 대상

| # | 파일 | 이유 |
|---|------|------|
| 2.2.1 | `orchestrator_engine.py` | Orchestrator Pod 알림 전용 |
| 2.2.2 | `orchestrator_formatters.py` | Pod 성과/배분 Embed 전용 |

### 2.3. 수정 대상

| # | 파일 | 변경 내용 |
|---|------|----------|
| 2.3.1 | `bot.py` | Orchestrator slash commands 제거 (`/strategies`, `/strategy <name>`, `/onchain`) |
| 2.3.2 | `engine.py` | Orchestrator 이벤트 구독 제거, Spot 관련 이벤트 추가 (Stop-Limit 발동 알림) |
| 2.3.3 | `formatters.py` | Futures 전용 필드 제거 (leverage, funding, positionSide), Spot 필드 추가 |
| 2.3.4 | `health_formatters.py` | Orchestrator/Pod 상태 제거, 단일 전략 6에셋 요약으로 간소화 |
| 2.3.5 | `health_collector.py` | Orchestrator/Regime 데이터 수집 제거 |

### 2.4. 신규 알림 이벤트

| 이벤트 | 채널 | Severity | 설명 |
|--------|------|----------|------|
| Stop-Limit 발동 | ALERTS | WARNING | 거래소 TS 자동 매도 감지 |
| Stop-Limit 설정 실패 | ALERTS | CRITICAL | API 3회 재시도 실패 |
| Stop-Limit Ratchet | TRADE_LOG | INFO | Stop price 상향 조정 |
| BNB 잔고 부족 | ALERTS | WARNING | 수수료 할인용 BNB < 최소량 |
| 입금/출금 감지 | TRADE_LOG | INFO | 잔고 변동 (다음 12H bar 반영) |

### 2.5. 채널 구조 변경

| 채널 | 현재 | 변경 후 |
|------|------|---------|
| **TRADE_LOG** | Fill + Balance + Position | 유지 (Spot 필드로 변경) |
| **ALERTS** | CircuitBreaker + RiskAlert | + Stop-Limit 발동/실패, BNB 부족 |
| **DAILY_REPORT** | Strategy Health (8h) + Daily/Weekly | 간소화: 6에셋 요약 |
| **HEARTBEAT** | System Health (1h) | 유지 (Orchestrator 상태 제거) |
| **MARKET_REGIME** | Regime Score + Derivatives (4h) | **삭제** (Regime/Derivatives 미사용) |

### Phase 2 완료 기준

- [ ] Orchestrator 알림 파일 삭제
- [ ] Bot slash commands 간소화
- [ ] Spot 전용 Embed 포매터 동작
- [ ] 신규 알림 이벤트 5종 동작
- [ ] MARKET_REGIME 채널 비활성화

---

## Phase 3: Monitoring (Prometheus/Grafana) 최적화

> **목표**: 100+ 메트릭을 Spot 단일 전략에 필요한 핵심만 남기고 정리.
> Grafana 대시보드 재구성.

### 3.1. 현재 Prometheus 메트릭 구조

**파일**: `src/monitoring/metrics.py` (43KB, 100+ 메트릭)

| Layer | 메트릭 수 | 용도 | 처리 |
|-------|----------|------|------|
| L1: Order Execution | ~6 | 주문 체결/지연/슬리피지 | **유지** |
| L2: Position & PnL | ~8 | Equity/DD/포지션/PnL | **유지** |
| L3: Exchange API | ~7 | API 호출/지연/WS 상태 | **유지** |
| L4: Bot Health | ~6 | Heartbeat/에러/EventBus | **유지** |
| L9: On-chain | ~6 | On-chain fetch/cache | **삭제** |
| L10: Surveillance | ~5 | 동적 에셋 탐색 | **삭제** |
| L11: Data Feeds | ~6 | 통합 피드 메트릭 | **삭제** |
| (Orchestrator) | ~10 | Pod/Allocation/Netting | **삭제** |
| (Regime) | ~5 | Regime 분류/전환 | **삭제** |

### 3.2. 유지 메트릭 (핵심 ~27개)

**L1: Order Execution**
```
mcbot_orders_total{status}              # 주문 수 (ack/filled/rejected)
mcbot_order_latency_seconds             # 주문→체결 지연
mcbot_slippage_bps                      # 슬리피지 (bps)
mcbot_fees_usdt_total                   # 누적 수수료
```

**L2: Position & PnL**
```
mcbot_equity_usdt                       # 현재 equity
mcbot_drawdown_pct                      # 현재 drawdown %
mcbot_cash_usdt                         # 가용 현금
mcbot_open_positions                    # 오픈 포지션 수
mcbot_position_notional_usdt{symbol}    # 에셋별 포지션 금액
mcbot_unrealized_pnl_usdt{symbol}       # 에셋별 미실현 PnL
mcbot_realized_profit_usdt_total        # 누적 실현 이익
mcbot_realized_loss_usdt_total          # 누적 실현 손실
```

**L3: Exchange API**
```
mcbot_exchange_api_calls_total{endpoint,status}
mcbot_exchange_api_latency_seconds{endpoint}
mcbot_exchange_ws_connected{symbol}
mcbot_ws_reconnects_total
mcbot_ws_last_message_age_seconds{symbol}
```

**L4: Bot Health**
```
mcbot_heartbeat_timestamp
mcbot_last_bar_age_seconds{symbol}
mcbot_errors_total{component,error_type}
mcbot_eventbus_queue_depth
```

**신규: Spot 전용**
```
mcbot_stop_limit_active{symbol}          # 활성 Stop-Limit 수
mcbot_stop_limit_ratchets_total{symbol}  # Ratchet 횟수
mcbot_stop_limit_triggers_total{symbol}  # Stop 발동 횟수
mcbot_bnb_balance                        # BNB 잔고 (수수료 할인용)
mcbot_spot_balance_usdt{asset}           # 에셋별 잔고 (USDT 환산)
```

### 3.3. 삭제 메트릭 (~70+ 개)

| 그룹 | 메트릭 예시 | 이유 |
|------|------------|------|
| On-chain (L9) | `mcbot_onchain_fetch_total` 등 6개 | On-chain 미사용 |
| Surveillance (L10) | `mcbot_surveillance_active_assets` 등 5개 | Surveillance 미사용 |
| Data Feeds (L11) | `mcbot_datafeed_fetch_total` 등 6개 | 대안 데이터 피드 미사용 |
| Orchestrator | Pod별 Sharpe/DD/Capital 등 ~10개 | Orchestrator 제거 |
| Regime | `mcbot_regime_*` 등 ~5개 | Regime 제거 |
| Futures 전용 | `mcbot_aggregate_leverage`, `mcbot_margin_used_usdt` | Spot 무관 |

### 3.4. Process Monitor 변경 (`src/monitoring/process_monitor.py`)

| 현재 | 변경 |
|------|------|
| Event loop lag 모니터 (10s) | 유지 |
| CPU/Memory/FD/GC 모니터 | 유지 |
| asyncio task count (warn > 200) | 임계값 하향 (warn > 50) — 구성 축소 |

### 3.5. Anomaly Detection 정리

| 파일 | 현재 | 처리 |
|------|------|------|
| `anomaly/gbm_drawdown.py` | GBM 기반 DD 이상 탐지 | **유지** |
| `anomaly/distribution.py` | KS-test 수익률 분포 | **유지** |
| `anomaly/execution_quality.py` | 체결 품질 이상 패턴 | **유지** |
| `anomaly/conformal_ransac.py` | RANSAC 구조 변화 | 유지 (optional) |

### 3.6. Grafana 대시보드 재구성

**현재**: Futures Multi-Strategy 대시보드 (20+ 패널)

**변경 후**: Spot Single-Strategy 대시보드 (12 패널)

```
Row 1: Overview
  [Equity Curve]  [Drawdown %]  [Open Positions (6 에셋)]

Row 2: Trading
  [Recent Trades]  [Win Rate / PF]  [Cumulative Fees]

Row 3: Stop-Limit
  [Active Stops per Symbol]  [Ratchet History]  [Stop Triggers]

Row 4: System Health
  [API Latency]  [WS Status]  [Event Loop Lag]
```

### 3.7. Health Check 간소화

**스케줄 변경**:

| 현재 | 변경 | 이유 |
|------|------|------|
| 1h Heartbeat | **유지** | 시스템 생존 확인 |
| 4h Market Regime | **삭제** | Regime 미사용 |
| 8h Strategy Health | **유지** (간소화) | 단일 전략 6에셋 요약 |
| Daily Report (00:00 UTC) | **유지** | Equity curve + monthly returns |
| Weekly Report (Mon 00:00) | **유지** | 주간 요약 |

### Phase 3 완료 기준

- [ ] 불필요한 메트릭 코드 제거 (70+ 개)
- [ ] Spot 전용 메트릭 5개 추가
- [ ] Process monitor 임계값 조정
- [ ] Grafana 대시보드 JSON 재구성 (12 패널)
- [ ] Health check 스케줄 간소화
- [ ] Prometheus `/metrics` 엔드포인트 정상 확인

---

## Phase 4: 문서 / Rules / Skills 정리

> **목표**: Futures/Orchestrator/Multi-Strategy 기준 문서를
> Spot Single-Strategy 현실에 맞게 갱신한다.

### 4.1. 아키텍처 문서 (`docs/architecture/`)

| 파일 | 현재 (LOC) | 변경 |
|------|-----------|------|
| `eda-orchestrator.md` (738) | EDA + Orchestrator 3-layer | **대폭 수정**: Orchestrator 섹션 제거, Spot 파이프라인으로 재작성 |
| `backtest-engine.md` (1,083) | VBT + EDA 백테스트 | **수정**: Orchestrator 관련 섹션 제거 |
| `regime-system.md` (842) | HMM + Ensemble Regime | **삭제** |
| `smart-executor.md` (143) | Limit Order Decorator | **삭제** (Spot에서 미사용) |
| `reconciler.md` (155) | Position Reconciliation | **유지** (Spot에서도 필요) |

**신규 문서**:

| 파일 | 내용 |
|------|------|
| `spot-system.md` | Spot 전체 아키텍처: 데이터→전략→실행→모니터링 |
| `stop-limit-ratchet.md` | Stop-Limit Ratchet 상세 설계 (migration 문서에서 분리) |

### 4.2. 운영 문서 (`docs/operations/`)

| 파일 | 현재 (LOC) | 변경 |
|------|-----------|------|
| `notification.md` (480) | Discord 채널/Embed/Slash | **수정**: Orchestrator 관련 제거, Spot 알림 추가 |
| `monitoring.md` (1,090) | 100+ 메트릭 + Anomaly | **수정**: 제거된 메트릭 삭제, Spot 메트릭 추가 |
| `exchange-safety-stop.md` (208) | Futures STOP_MARKET | **재작성**: Spot Stop-Limit Ratchet |

### 4.3. 가이드 문서 (`docs/guides/`)

| 파일 | 현재 (LOC) | 변경 |
|------|-----------|------|
| `strategy-pipeline.md` (305) | P1-P7 전략 라이프사이클 | 유지 (Pipeline은 향후 재사용 가능) |
| `lessons-system.md` (106) | 교훈 기록 시스템 | 유지 |
| `data-collection.md` (815) | Bronze/Silver/Gold 수집 | 유지 |
| `regime-classification-analysis.md` (249) | Regime 분석 | **삭제** |
| `supertrend-backtest.md` (558) | SuperTrend 예시 | **수정**: Spot 비용 모델 반영 |
| `ma-cross-backtest.md` (235) | MA Cross 예시 | **삭제** (전략 제거됨) |
| `multi-asset-backtest-comparison.md` (120) | 다중 에셋 비교 | 유지 |

### 4.4. 기획 문서 (`docs/planning/`)

| 파일 | 현재 (LOC) | 변경 |
|------|-----------|------|
| `roadmap.md` (287) | Tier 1-3 로드맵 | **재작성**: Spot 운영 로드맵 |
| `dynamic-asset-surveillance-backtest.md` (394) | Surveillance 기획 | **삭제** |

### 4.5. Claude Rules 정리 (`.claude/rules/`)

| 파일 | 현재 | 변경 |
|------|------|------|
| `commands.md` | CLI Quick Reference | **수정**: `mcbot orchestrate` 제거, `mcbot eda spot-*` 추가 |
| `lint.md` | Zero-tolerance lint | 유지 |
| `testing.md` | 테스트 패턴 | **수정**: Orchestrator/Regime 테스트 마커 제거 |
| `backtest.md` | 백테스트 규칙 | **수정**: Orchestrator 관련 제거, Spot 비용 모델 반영 |
| `data.md` | Medallion 아키텍처 | 유지 |
| `eda.md` | EDA 규칙 | **수정**: Orchestrator 이벤트 체인 제거, Spot 파이프라인 반영 |
| `strategy.md` | 전략 개발 | **수정**: 현재 supertrend만 활성, 향후 추가 시 동일 패턴 |
| `exchange.md` | CCXT 통합 | **수정**: Spot 주문 예시 추가, Futures 예시 제거 |
| `models.md` | Pydantic 규칙 | 유지 |

### 4.6. Claude Skills 정리 (`.claude/skills/`)

| Skill | 현재 | 변경 |
|-------|------|------|
| `p1-research` | 전략 발굴 | 유지 (향후 재사용) |
| `p2p3-build` | 전략 4-file 구현 | 유지 (향후 재사용) |
| `p4-backtest` | 백테스트 검증 | 유지 |
| `p5p6-validate` | 로버스트니스 검증 | 유지 |
| `p7-live` | EDA Parity | **수정**: Spot 파이프라인 반영 |
| `audit-inspect` / `audit-fix` | 아키텍처 감사 | **수정**: Orchestrator 감사 항목 제거 |

### 4.7. CLAUDE.md 갱신

| 섹션 | 변경 |
|------|------|
| Quick Reference 이벤트 흐름 | Orchestrator 흐름 제거, Spot 흐름 추가 |
| 의존성 흐름 | Orchestrator/Regime 제거 |
| Gotchas | Spot Stop-Limit 주의사항 추가 |

### 4.8. Memory 갱신 (`~/.claude/projects/.../memory/MEMORY.md`)

| 섹션 | 변경 |
|------|------|
| Strategic Direction | Spot 전환 완료 기록 |
| Implementation Progress | Phase 0-6 진행 상태 |
| Strategy Results Summary | ACTIVE 4개 → supertrend 1개 (Spot) |
| Lessons Learned | Orchestrator → Spot 교훈 추가 |

### Phase 4 완료 기준

- [ ] 아키텍처 문서 재작성 (Spot 중심)
- [ ] 운영 문서 갱신 (Discord/Monitoring)
- [ ] 삭제 대상 가이드/기획 문서 제거
- [ ] Claude Rules 9개 파일 검토/수정
- [ ] Skills 수정 (p7-live, audit)
- [ ] CLAUDE.md + MEMORY.md 갱신
- [ ] `markdownlint-cli2 --fix "docs/**/*.md"` 통과

---

## Phase 5: Paper Trading 검증

> **목표**: Spot Paper 모드로 실시간 데이터 검증. 1주일 이상 운영.

### 5.1. 사전 조건

- [ ] Phase 0-4 완료
- [ ] `config/spot_supertrend.yaml` 에 `mode: paper` 설정
- [ ] Docker 환경 준비 (또는 로컬)

### 5.2. 검증 항목

| # | 항목 | 확인 방법 | 기준 |
|---|------|----------|------|
| 5.2.1 | WebSocket 연결 | 6 심볼 1m bar 수신 확인 | ws_connected=1 × 6 |
| 5.2.2 | 12H 캔들 생성 | CandleAggregator 출력 로그 | 하루 2회 × 6 심볼 |
| 5.2.3 | 시그널 발생 | StrategyEngine 로그 | SuperTrend LONG/NEUTRAL |
| 5.2.4 | Stop price 계산 | SpotStopManager 로그 | `entry - 3.0 * ATR` |
| 5.2.5 | Ratchet 동작 | Stop price 상향 이력 | 상향만 확인, 하향 없음 |
| 5.2.6 | Equity 추적 | AnalyticsEngine 로그 | 일관된 equity 계산 |
| 5.2.7 | Discord 알림 | 채널별 메시지 수신 | 진입/청산/TS 알림 |
| 5.2.8 | Prometheus 메트릭 | `:8000/metrics` 확인 | 27개 핵심 메트릭 |
| 5.2.9 | 정상 종료 | SIGTERM 전송 | graceful shutdown 완료 |
| 5.2.10 | 재시작 복구 | 봇 재시작 | StateManager 복구 확인 |

### 5.3. 모니터링 체크리스트 (1주일)

| 일차 | 확인 |
|------|------|
| Day 1 | WebSocket 안정성, 12H bar 생성, 시그널 발생 |
| Day 2 | 첫 거래 시뮬레이션 (진입 시그널 기다림) |
| Day 3 | Stop-Limit 계산 정합성, Ratchet 동작 |
| Day 4-5 | Discord 알림 정상, Daily Report 생성 |
| Day 6-7 | 정상 종료/재시작 테스트, 메모리 누수 확인 |

### Phase 5 완료 기준

- [ ] 1주일 무중단 운영 (또는 계획된 재시작만)
- [ ] 12H bar 누락 0건
- [ ] 시그널 정합성 (EDA 백테스트와 동일 방향)
- [ ] Stop price 계산 정합성
- [ ] Discord/Prometheus 정상 동작

---

## Phase 6: Live 배포

> **목표**: 실제 자금으로 Binance Spot 거래.
> 소액 → 단일 에셋 → 전체 에셋 순차 투입.

### 6.1. 사전 조건

- [ ] Phase 5 Paper 검증 완료
- [ ] Binance Spot 계정 API Key 발급 (IP Whitelist)
- [ ] BNB 소량 보유 (수수료 25% 할인)
- [ ] `config/spot_supertrend.yaml` 에 `mode: live` 설정

### 6.2. 단계적 투입

| 단계 | 자금 | 에셋 | 기간 | 검증 |
|------|------|------|------|------|
| 6.2.1 | $100 | BTC 1개 | 3일 | 실주문 체결, Stop-Limit 설정, 수수료 확인 |
| 6.2.2 | $600 | 6개 전체 | 1주일 | EW 배분, 다중 심볼 동시 관리 |
| 6.2.3 | 목표 자금 | 6개 전체 | 지속 | 풀 운영, 입금 시 자연 반영 |

### 6.3. 실거래 체크리스트

| # | 항목 | 확인 |
|---|------|------|
| 6.3.1 | Market Buy 체결 | quoteOrderQty 정확, 수수료 BNB 차감 |
| 6.3.2 | Stop-Limit 설정 | 거래소 Open Orders에 표시 |
| 6.3.3 | Ratchet 갱신 | 12H bar마다 stop 상향 확인 |
| 6.3.4 | Market Sell (시그널 청산) | Stop-Limit 취소 → 전량 매도 |
| 6.3.5 | Stop-Limit 발동 (TS 청산) | 가격 하락 시 자동 매도 확인 |
| 6.3.6 | 봇 다운 시 보호 | Stop-Limit 거래소에 잔존 확인 |
| 6.3.7 | PositionReconciler | 거래소 잔고 vs PM 정합성 |
| 6.3.8 | 입금 반영 | USDT 추가 → 다음 진입 시 새 금액 사용 |

### Phase 6 완료 기준

- [ ] 단일 에셋 소액 거래 성공
- [ ] 6에셋 동시 운영 1주일 무사고
- [ ] Stop-Limit 발동 최소 1회 확인 (또는 시뮬레이션)
- [ ] 봇 재시작 시 상태 복구 + Stop-Limit 잔존 확인

---

## Appendix A: 파일 변경 총괄표

### 삭제 대상 (Phase 0)

| 범주 | 경로 | 파일 수 |
|------|------|:-------:|
| Orchestrator | `src/orchestrator/` | 23 |
| Orchestrated Runner | `src/eda/orchestrated_runner.py` | 1 |
| Regime | `src/regime/` | 9 |
| FeatureStore | `src/market/feature_store.py` | 1 |
| Strategies (supertrend 제외) | `src/strategy/*/` | ~190 dirs |
| Data Feeds | `src/eda/{derivatives,onchain,macro,options,deriv_ext}_feed.py` | 5 |
| SmartExecutor | `src/eda/smart_executor{,_config}.py` | 2 |
| Futures Client | `src/exchange/binance_futures_client.py` | 1 |
| CLI Orchestrate | `src/cli/orchestrate.py` | 1 |
| Config Loader | `src/config/orchestrator_loader.py` | 1 |
| Notification | `src/notification/orchestrator_{engine,formatters}.py` | 2 |
| Docs | `docs/architecture/{regime-system,smart-executor}.md` | 2 |
| Docs | `docs/guides/{regime-*,ma-cross-*}.md` | 2 |
| Docs | `docs/planning/dynamic-asset-surveillance-backtest.md` | 1 |
| **합계** | | **~240+** |

### 수정 대상 (Phase 0-4)

| 파일 | Phase | 변경 요약 |
|------|:-----:|----------|
| `src/backtest/engine.py` | 0 | Orchestrator import 제거 |
| `src/backtest/request.py` | 0 | AssetAllocationConfig 제거 |
| `src/eda/runner.py` | 0 | Regime/FeatureStore/Orchestrator 참조 제거 |
| `src/eda/strategy_engine.py` | 0 | Regime/FeatureStore 참조 제거 |
| `src/eda/live_runner.py` | 0+1 | 피드/Orchestrator 제거 + Spot 팩토리 추가 |
| `src/eda/ports.py` | 0 | FeatureStorePort 제거 |
| `src/eda/portfolio_manager.py` | 1 | Spot equity 계산 |
| `src/strategy/__init__.py` | 0 | 189 import 제거 |
| `src/cli/__init__.py` | 0 | orchestrate_app 제거 |
| `src/cli/eda.py` | 1 | spot-paper/spot-live 명령 추가 |
| `src/notification/bot.py` | 2 | Orchestrator commands 제거 |
| `src/notification/engine.py` | 2 | Spot 이벤트 추가 |
| `src/notification/formatters.py` | 2 | Spot Embed 필드 |
| `src/notification/health_*.py` | 2 | Orchestrator 데이터 제거 |
| `src/monitoring/metrics.py` | 3 | 70+ 메트릭 제거, 5 추가 |
| `CLAUDE.md` | 4 | Spot 파이프라인 반영 |
| `.claude/rules/*.md` | 4 | 5개 파일 수정 |

### 신규 생성 (Phase 1)

| 파일 | 설명 |
|------|------|
| `src/exchange/binance_spot_client.py` | Spot 주문 클라이언트 |
| `src/eda/spot_executor.py` | Spot Executor (ExecutorPort) |
| `src/eda/spot_stop_manager.py` | Stop-Limit Ratchet |
| `docs/architecture/spot-system.md` | Spot 아키텍처 문서 |
| `docs/architecture/stop-limit-ratchet.md` | TS Ratchet 설계 문서 |

---

## Appendix B: 삭제 대상 전략 목록

> `src/strategy/` 하위 디렉토리 중 `supertrend/` 를 제외한 전부.
> 태그 `archive/futures-multi-strategy` 에서 복원 가능.

**ACTIVE 전략 (4개, 보관)**:
- `anchor_mom/` — Sharpe 1.36
- `donch_multi/` — Sharpe 1.61
- `tri_channel_trend/` — Sharpe 2.17
- `mad_channel_trend/` — Sharpe 1.24

**기타 186개 RETIRED 전략**: Pipeline YAML에 기록 보관.

**미등록 전략 (git status untracked)**:
- `ma_cross/`, `ma_st_cross/`, `st_donch/`, `st_regime/`

---

## 작업 순서 요약

```
Phase 0 (Cleanup)     ← 가장 먼저. 코드베이스 축소
  ↓
Phase 1 (Spot Core)   ← 핵심 구현. 가장 중요
  ↓
Phase 2 (Discord)     ─┐
Phase 3 (Monitoring)  ─┤ 병렬 가능
Phase 4 (Docs/Rules)  ─┘
  ↓
Phase 5 (Paper)       ← 1주일 검증
  ↓
Phase 6 (Live)        ← 단계적 투입
```

**의존성**: Phase 0 → Phase 1 → Phase 2/3/4 (병렬) → Phase 5 → Phase 6
