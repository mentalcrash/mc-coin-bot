# MC Coin Bot

Event-Driven Architecture 기반 암호화폐 퀀트 트레이딩 시스템.

185개 전략을 7단계 Phase 파이프라인으로 평가한 결과,
**SuperTrend v1.1 (Spot Long-Only)** 1개 전략을 6에셋에서 운용 중.

---

## 아키텍처

### Spot Live 파이프라인

```
WebSocket → 1m Bar → CandleAggregator → 12H Bar
  → Strategy(SuperTrend+ADX) → Signal
  → PM → RM → OMS → SpotExecutor → Fill
                       ↓
              SpotStopManager (Stop-Limit Ratchet)
```

### Backtest 파이프라인

```
1m Parquet → CandleAggregator → 12H Bar → Strategy → Signal → PM → RM → OMS → Fill
```

> 상세: [`docs/architecture/backtest-engine.md`](docs/architecture/backtest-engine.md)

### 핵심 설계 원칙

- **Stateless Strategy / Stateful Execution** -- 전략은 시그널만 생성, PM/RM/OMS가 포지션 관리
- **Target Weights 기반** -- "사라/팔아라" 대신 "적정 비중은 X%"
- **Look-Ahead Bias 원천 차단** -- Signal at Close → Execute at Next Open
- **PM/RM 분리 모델** -- Portfolio Manager → Risk Manager → OMS 3단계 방어
- **Stop-Limit Ratchet** -- 거래소에 Stop-Limit 위임 + 12H bar마다 상향 조정 (봇 장애 시에도 안전)

### 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.13 |
| Package Manager | uv |
| Type Safety | Pydantic V2 + pyright |
| Exchange | CCXT Pro (WebSocket + REST) — Binance Spot |
| Backtesting | VectorBT + Numba |
| EDA Backtesting | EventBus + CandleAggregator |
| Data | Parquet (Medallion Architecture) |
| Monitoring | Prometheus + Grafana |
| Notification | Discord.py (Bot + Slash Commands) |
| Charts | matplotlib (Agg headless) |
| Logging | Loguru |
| CI/CD | GitHub Actions + Coolify |

---

## 전략 설정 (Config YAML)

모든 백테스트와 실행은 YAML 설정 파일 하나로 제어됩니다.
현재 활성 config는 `config/spot_supertrend.yaml` 1개입니다.

### 설정 구조

```yaml
# config/spot_supertrend.yaml
backtest:
  symbols:
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
    - AVAX/USDT
    - XRP/USDT
    - S/USDT
  timeframe: "12h"
  start: "2020-01-01"    # Backtest 전용. Live에서 무시.
  end: "2026-03-06"      # Backtest 전용. Live에서 무시.
  capital: 100000.0       # Paper/Backtest 전용. Live는 거래소 잔고 자동 조회.

strategy:
  name: supertrend
  params:
    atr_period: 7
    multiplier: 2.5
    adx_period: 14
    adx_threshold: 25
    short_mode: 0         # DISABLED (Long-Only)

portfolio:
  max_leverage_cap: 1.0   # Spot: 레버리지 없음
  rebalance_threshold: 0.05
  use_trailing_stop: true
  trailing_stop_atr_multiplier: 3.0
  use_intrabar_trailing_stop: false  # 12H bar에서만 TS 체크
  cost_model:
    maker_fee: 0.001      # Spot 0.1%
    taker_fee: 0.001
    slippage: 0.0005
    funding_rate_8h: 0.0  # Spot: Funding Rate 없음
    market_impact: 0.0
```

---

## 전략 파이프라인

전략은 **Alpha Research(P1)** → **Live Readiness(P7)** 까지 7단계 Phase를 순차 통과해야 합니다.
각 Phase에서 FAIL 시 즉시 RETIRED.

| 구분 | Phase | 검증 내용 |
|------|-------|----------|
| 발굴 · 구현 | P1 → P3 | 데이터 기반 알파 발굴, 4-file 구현, C1~C7 코드 검증 |
| 백테스트 검증 | P4 → P6 | 6에셋x6년, IS/OOS, 파라미터 최적화, WFA+CPCV+PBO+DSR |
| 라이브 전환 | P7 | VBT↔EDA Parity, 라이브 인프라 검증, 배포 설정 |

> Phase별 상세 기준: [`docs/guides/strategy-pipeline.md`](docs/guides/strategy-pipeline.md)

### 전략 현황

| 전략 | 에셋 | TF | Avg Sharpe | 상태 |
|------|------|-----|-----------|------|
| **SuperTrend v1.1** | BTC, ETH, SOL, XRP, AVAX, S(Sonic) | 12H | 1.104 | **ACTIVE (Spot)** |

> 185개 전략: 1 ACTIVE (Spot) + 184 RETIRED. 이전 Futures ACTIVE 4개는 Phase 0에서 Orchestrator와 함께 제거.
> 상세: `uv run mcbot pipeline report`

---

## 빠른 시작

### 환경 설정

```bash
uv sync --group dev --group research
cp .env.example .env  # API 키 설정
```

### 백테스트

> **VBT**: Vectorized 고속 백테스트 (탐색용) / **EDA**: Event-Driven (라이브 동일 코드, 최종 검증)

```bash
# VBT 백테스트
uv run mcbot backtest run config/spot_supertrend.yaml

# EDA 백테스트 (1m 데이터 → 12H 집계)
uv run mcbot eda run config/spot_supertrend.yaml

# 옵션: --report (QuantStats HTML), --advisor (Strategy Advisor), -V (Verbose)
```

### Live Trading

```bash
# Paper — 시뮬레이션 체결
uv run mcbot eda run-live config/spot_supertrend.yaml --mode paper

# Spot Live — Binance Spot 실주문
uv run mcbot eda spot-live config/spot_supertrend.yaml
```

Spot Live 모드는 Binance Spot에서 Long-Only(레버리지 없음)로 실행됩니다.
60초마다 거래소 잔고와 PM 상태를 교차 검증(PositionReconciler)하며, 불일치 시 경고를 발행합니다.
봇 장애 시 포지션 보호를 위한 Stop-Limit Ratchet은 [`docs/operations/exchange-safety-stop.md`](docs/operations/exchange-safety-stop.md) 참조.

---

## 데이터 수집

Medallion Architecture(Bronze→Silver)로 OHLCV 1분봉을 수집·정제합니다.

```bash
# OHLCV (1분봉)
uv run mcbot ingest pipeline BTC/USDT --year 2024 --year 2025

# 데이터 상태
uv run mcbot ingest info
```

> 상세: [`docs/guides/data-collection.md`](docs/guides/data-collection.md)

### 데이터 카탈로그

데이터셋 메타데이터를 [`catalogs/datasets.yaml`](catalogs/datasets.yaml)에서 YAML로 관리합니다.

```bash
uv run mcbot catalog list                      # 데이터셋 목록
uv run mcbot catalog show btc_metrics          # 상세 (컬럼, enrichment 설정)
```

---

## 배포 (Docker Compose + Coolify)

3개 서비스(트레이딩 봇, Prometheus, Grafana)를 `docker-compose.yaml`로 실행합니다.

```bash
docker compose up --build -d      # 전체 스택 실행
docker compose logs -f mc-bot     # 로그 확인
docker compose down               # 중지
```

### 환경 변수

Coolify로 배포합니다. `MC_*` 환경 변수로 실행 모드를 제어합니다.

| 환경 변수 | 기본값 | 설명 |
|----------|--------|------|
| `MC_EXECUTION_MODE` | `paper` | 실행 모드 (`paper` / `shadow` / `live` / `spot_live`) |
| `MC_CONFIG_PATH` | `config/spot_supertrend.yaml` | YAML 설정 파일 경로 |
| `MC_INITIAL_CAPITAL` | `10000` | 초기 자본 — `spot_live`에서는 거래소 잔고 자동 조회 |
| `MC_DB_PATH` | `data/trading.db` | SQLite DB 경로 (상태 영속화, 빈 문자열=비활성) |
| `MC_ENABLE_PERSISTENCE` | `true` | 상태 영속화 on/off |
| `MC_METRICS_PORT` | `8000` | Prometheus metrics 포트 (`0`이면 비활성) |

Discord 채널 ID 등 추가 환경 변수는 `.env.example` 참조.

### 모니터링 & 알림

- **Prometheus** (`localhost:9090`) + **Grafana** (`localhost:3000`) — 상세: [`docs/operations/monitoring.md`](docs/operations/monitoring.md)
- **Discord 알림**: 체결, Stop-Limit Ratchet, 리스크 알림 실시간 전송 + `/status`, `/kill`, `/balance` Slash Commands
- **System Heartbeat** (1시간): equity, drawdown, CB 상태
- **Strategy Health Report** (8시간): Rolling Sharpe, Win Rate, 6에셋 요약

---

## 전략 철학: 12H Timeframe 확정

### 왜 12H인가

185개 전략을 6년간 검증한 결과, **12H가 크립토 trend-following에 최적 TF**라는 결론에 도달했습니다.

| 근거 | 설명 |
|------|------|
| **1D 고갈** | 92개 1D OHLCV 전략 시도 → 0 ACTIVE. 검색공간 소진 |
| **4H/8H 사망** | 50개+ 4H/8H 전략 전멸 — 거래비용이 edge 잠식, CPCV 로버스트니스 미달 |
| **12H 최적** | SuperTrend P5 PASS (91.4% Sharpe ≥ 0.8) — 비용/노이즈/신호 균형점 |
| **TF 분산 불가** | 4H/8H에서 단 하나도 ACTIVE 없음. 12H 단일 TF 운용이 최적 |
| **대안데이터 한계** | On-chain/Deriv/Macro/TradeFlow 단독 alpha 0건 (181개+ 전략 검증) |

### "느리지 않은가?" — Stop-Limit Ratchet이 해결

시그널은 12H로 판단하지만, 리스크 관리는 거래소 Stop-Limit으로 상시 방어합니다.

```
시그널 생성:  12H bar close 시 (하루 2회, 노이즈 없는 판단)
리스크 방어:  Stop-Limit Ratchet (3.0x ATR, 거래소 위임 — 봇 장애 시에도 동작)
```

- **Stop-Limit Ratchet**: 12H bar마다 high watermark 갱신 → stop price 상향만 허용
- **System Stop-Loss**: 포트폴리오 레벨 circuit breaker
- **Deferred Execution**: 다음 bar open 체결 → look-ahead bias 차단

---

## 운영 도구

```bash
# 과적합 검증
uv run mcbot backtest validate -m quick       # IS/OOS Split
uv run mcbot backtest validate -m milestone   # Walk-Forward (5-fold)
uv run mcbot backtest validate -m final       # CPCV + DSR + PBO

# 시그널 진단
uv run mcbot backtest diagnose BTC/USDT -s supertrend

# 교훈 관리 (lessons/*.yaml)
uv run mcbot pipeline lessons-list            # 전체 교훈 목록

# 아키텍처 감사 (audits/)
uv run mcbot audit latest                     # 최신 스냅샷
```

---

## 문서

| 문서 | 설명 |
|------|------|
| **Architecture** | |
| [`docs/architecture/backtest-engine.md`](docs/architecture/backtest-engine.md) | 백테스트 엔진 아키텍처 (VBT + EDA 이중 엔진 + 검증) |
| [`docs/architecture/reconciler.md`](docs/architecture/reconciler.md) | PositionReconciler (거래소↔PM 잔고 교차 검증) |
| **Guides** | |
| [`docs/guides/strategy-pipeline.md`](docs/guides/strategy-pipeline.md) | **전략 파이프라인** (Phase 1~7, PASS 기준, YAML 스키마) |
| [`docs/guides/data-collection.md`](docs/guides/data-collection.md) | **데이터 수집 가이드** (OHLCV, 저장 구조, CLI) |
| [`docs/guides/lessons-system.md`](docs/guides/lessons-system.md) | Lessons 시스템 (102개 교훈, YAML 스키마, CLI) |
| **Operations** | |
| [`docs/operations/monitoring.md`](docs/operations/monitoring.md) | Prometheus + Grafana 모니터링 (메트릭, 대시보드, 알림 규칙) |
| [`docs/operations/notification.md`](docs/operations/notification.md) | Discord 알림 시스템 (체결, Stop-Limit, Slash Commands) |
| [`docs/operations/exchange-safety-stop.md`](docs/operations/exchange-safety-stop.md) | Stop-Limit Ratchet (봇 장애 시 거래소 안전망) |
| **Planning** | |
| [`docs/planning/roadmap.md`](docs/planning/roadmap.md) | Spot 운영 로드맵 |
| [`docs/plans/spot-migration-implementation.md`](docs/plans/spot-migration-implementation.md) | Spot Migration 구현 계획 (Phase 0~5) |
