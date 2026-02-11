# Paper Trading & Production Architecture Guide (2026)

> MC Coin Bot의 EDA 백테스트 → Paper Trading → Live 배포 전환을 위한 종합 가이드
>
> 작성일: 2026-02-09 | 갱신일: 2026-02-10 (v4: Phase 6 완료 반영, 축약) | Python 3.13 + asyncio + Pydantic V2 + CCXT Pro

---

## Table of Contents

1. [현재 구현 상태](#1-현재-구현-상태)
2. [아키텍처 요약](#2-아키텍처-요약)
3. [남은 로드맵](#3-남은-로드맵)
4. [Production 운영 레퍼런스](#4-production-운영-레퍼런스)
5. [Sources](#5-sources)

---

## 1. 현재 구현 상태

### 1.1 완료된 Phase

| Phase | 컴포넌트 | 핵심 파일 | 비고 |
|-------|----------|-----------|------|
| **1-5** | VBT/EDA 백테스트 | `src/backtest/`, `src/eda/` | 23개 전략, 1623 tests |
| **6-A** | LiveDataFeed | `src/eda/live_data_feed.py` | CCXT Pro `watch_ohlcv()`, CandleAggregator |
| **6-A** | LiveRunner | `src/eda/live_runner.py` | `.paper()` / `.shadow()` factory, graceful shutdown |
| **6-A** | CLI run-live | `src/cli/eda.py` | `launch_live()` 공유 함수, Docker entrypoint 재사용 |
| **6-B** | SQLite DB | `src/eda/persistence/database.py` | aiosqlite, WAL mode, schema auto-create |
| **6-B** | TradePersistence | `src/eda/persistence/trade_persistence.py` | FillEvent → trades 테이블 |
| **6-B** | StateManager | `src/eda/persistence/state_manager.py` | PM/RM 상태 직렬화, 5분 주기 자동 저장 |
| **6-C** | Dockerfile | `Dockerfile` | Multi-stage (uv builder), non-root, health check |
| **6-C** | DeploymentConfig | `src/config/settings.py` | `MC_*` 환경변수, `src/live/main.py` entrypoint |
| **6-C** | YAML 설정 | `config/*.yaml` | 27개 전략 설정 파일 |
| **6-D** | Discord Bot | `src/notification/bot.py` | discord.py 2.6+, /status /kill /balance |
| **6-D** | NotificationEngine | `src/notification/engine.py` | EventBus subscriber, 채널 라우팅 |
| **6-D** | NotificationQueue | `src/notification/queue.py` | asyncio.Queue(500), SpamGuard, exp backoff retry |
| **6-D** | Formatters | `src/notification/formatters.py` | Event → Discord Embed dict |

### 1.2 남은 작업

| 컴포넌트 | Phase | 우선순위 | 비고 |
|----------|-------|----------|------|
| docker-compose.yml | 6-C | P2 | 단일 컨테이너 배포에는 불필요, Prometheus/Grafana 시 필요 |
| **VPS 배포 + Paper 운영** | **7** | **P0** | **다음 단계** — Shadow warmup → Paper 전환 |
| ChartGenerator | 7.5 | P1 | matplotlib equity/drawdown PNG |
| ReportScheduler | 7.5 | P1 | daily/weekly/monthly 자동 리포트 |
| Prometheus/Grafana | 7.5 | P2 | 모니터링 대시보드 |
| CI/CD Pipeline | 7.5 | P2 | GitHub Actions |
| **LiveExecutor** | **8** | P1 | CCXT 실제 주문 (ExecutorPort 준비됨) |
| Reconciliation | 8 | P1 | 거래소 vs 로컬 교차 검증 |
| KillSwitch (파일 기반) | 8 | P1 | `/kill` + KILL_SWITCH 파일 |

### 1.3 ExecutionMode 현황

```python
class ExecutionMode(StrEnum):
    BACKTEST = "backtest"  # ✅ HistoricalDataFeed + BacktestExecutor
    SHADOW   = "shadow"    # ✅ LiveDataFeed + ShadowExecutor
    PAPER    = "paper"     # ✅ LiveDataFeed + BacktestExecutor (시뮬레이션 체결)
    CANARY   = "canary"    # ❌ LiveDataFeed + LiveExecutor (소액 실제 주문)
    LIVE     = "live"      # ❌ LiveDataFeed + LiveExecutor (전액 실제 주문)
```

---

## 2. 아키텍처 요약

### 2.1 이벤트 흐름 (구현 완료)

```
[Shadow/Paper Mode — ✅ 구현 완료]
LiveDataFeed (WebSocket) → BarEvent(1m) → CandleAggregator → BarEvent(target_tf)
  → StrategyEngine → SignalEvent
  → PM → OrderRequestEvent → RM → OMS → [Executor] → FillEvent
  → PM → PositionUpdateEvent + BalanceUpdateEvent
  → AnalyticsEngine → PerformanceMetrics
  → NotificationEngine → Discord Bot (채널 알림 + Slash Commands)
  → TradePersistence → SQLite (trades, equity, bot_state)
  → StateManager → 5분 주기 상태 저장 + 재시작 복구
```

### 2.2 Executor 교체 매트릭스

| Mode | DataFeed | Executor | 실제 자금 |
|------|----------|----------|----------|
| **Shadow** | LiveDataFeed ✅ | ShadowExecutor ✅ | ❌ |
| **Paper** | LiveDataFeed ✅ | BacktestExecutor ✅ | ❌ |
| **Canary/Live** | LiveDataFeed ✅ | LiveExecutor ❌ | ✅ |

### 2.3 데이터 아키텍처 (3-Tier Hybrid, 구현 완료)

```
HOT (In-Memory)     ← PM._positions, RM._peak_equity (매 bar 실시간)
  ↓ FillEvent
WARM (SQLite)        ← trades, equity_snapshots, bot_state (aiosqlite, WAL mode)
  ↓ 분석 시
COLD (Parquet+JSONL) ← data/bronze/, data/silver/, EventBus JSONL audit
```

---

## 3. 남은 로드맵

### Phase 7: Paper Trading 배포 (P0 — 다음 단계)

| 단계 | 기간 | 모드 | 검증 항목 |
|------|------|------|-----------|
| **7-A: VPS 배포** | 1일 | — | Docker 빌드 + 환경변수 설정 + 볼륨 마운트 |
| **7-B: Shadow Warmup** | 2-4시간 | `--mode shadow` | WebSocket 안정성, 1m bar 수신, 시그널 생성 |
| **7-C: Paper 전환** | 2-4주 | `--mode paper` | 시뮬 체결, PnL, equity curve, 알림, 재시작 복구 |
| **7-D: 결과 평가** | — | 분석 | 백테스트 대비 parity (수익 부호, 거래 수, Sharpe) |

**Paper 모드 성공 기준:**

- WebSocket 24시간 무중단 연결
- 시그널이 백테스트와 일관 (동일 bar에서 동일 방향)
- 시뮬 PnL 부호가 백테스트와 일치
- Discord 알림 정상 수신 + Slash Commands 응답
- 재시작 후 상태 정상 복구

**인프라 선택:**

| Provider | Spec | 비용 | 추천 |
|----------|------|------|------|
| **Oracle Cloud Always Free** | ARM 4 vCPU, 24GB RAM | $0/month | **Paper 최적** |
| **Hetzner CX22** | 2 vCPU, 4GB RAM | ~$4/month | Live (가성비) |
| **DigitalOcean Basic** | 1 vCPU, 1GB RAM | $6/month | 대안 |

> 일봉 TSMOM 전략은 ms 단위 지연 무의미. Spot Instance 사용 절대 금지.

### Phase 7.5: 알림 고도화 + 모니터링 (P1, Paper 운영 중 병행)

| Task | 설명 | 의존성 |
|------|------|--------|
| **7.5-A** | `ChartGenerator` — matplotlib equity/drawdown PNG (Agg backend) | matplotlib (이미 있음) |
| **7.5-B** | Discord Bot 차트 첨부 (`discord.File` + Embed) | ChartGenerator |
| **7.5-C** | `ReportScheduler` — daily/weekly/monthly (`discord.ext.tasks`) | ChartGenerator, DB |
| **7.5-D** | Prometheus metrics + Grafana 대시보드 | `uv add prometheus-client` |
| **7.5-E** | CI/CD Pipeline (GitHub Actions: lint → test → build → deploy) | Dockerfile |

**Prometheus 핵심 지표:**

| Category | Metric | Alert Threshold |
|----------|--------|-----------------|
| Uptime | Last heartbeat | > 5분 무응답 |
| Drawdown | Current DD % | > 15% (W), > 25% (C) |
| Queue | EventBus size | > 80% capacity |
| API Rate | Binance calls/min | > 1000 (limit: 1200) |

### Phase 8: Live Trading (P2)

| 단계 | Assets | Capital | 기간 | 성공 기준 |
|------|--------|---------|------|-----------|
| **8-A** | LiveExecutor 구현 | — | — | ExecutorPort → CCXT `create_order` |
| **8-B** | Reconciliation 구현 | — | — | 거래소 vs 로컬 교차 검증 |
| **8-C** | Testnet 검증 | 1 (BTC) | 1주 | API 주문 체결 정상 |
| **8-D** | Alpha | 1 (BTC) | $1,000 (10%) | Sharpe > 0, MDD < 10% |
| **8-E** | Beta | 3 (BTC,ETH,SOL) | $3,000 (30%) | 멀티에셋 정상 |
| **8-F** | Full | 8 (전체) | $10,000 (100%) | 백테스트 parity |

---

## 4. Production 운영 레퍼런스

### 4.1 Security Checklist

**Binance API Key:**

- [x] Enable Spot & Margin Trading
- [ ] Enable Futures (필요시만)
- [ ] Enable Withdrawals (**절대 비활성화**)
- [x] Restrict access to trusted IPs only
- [x] 데이터 조회용/거래용 API key 분리

**API Key 관리 수준:**

| Level | 방법 | 적합 상황 |
|-------|------|-----------|
| Level 1 | `.env` file (gitignored) | 로컬 개발 |
| Level 2 | **Docker Secrets** | VPS Docker — **권장** |
| Level 3 | SOPS (암호화된 git) | 소규모 팀 |

**VPS Hardening:**

```bash
# SSH key-only + Firewall + Fail2ban
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
ufw default deny incoming && ufw allow 22/tcp && ufw enable
apt install fail2ban unattended-upgrades
```

### 4.2 Binance Rate Limits (2026)

| Limit | Spot | Futures |
|-------|------|---------|
| Request Weight | 6,000/min per IP | 2,400/min per IP |
| Order Count | 50/10s, 160,000/24h | 1,200/min |
| WebSocket | 5 msg/s per connection | 10 connections per IP |

429 → exponential backoff. 반복 위반 → 418 (IP ban).

### 4.3 Circuit Breaker 계층

| Level | Trigger | Action |
|-------|---------|--------|
| Order-level | 단일 주문 > 자본의 30% | Reject |
| Daily Loss | 일일 손실 > 5% | 신규 주문 중지 |
| Drawdown | MDD > 15% | 전체 포지션 청산, 봇 중지 |
| System | API 연속 실패 > 5회 | 봇 중지, 긴급 알림 |
| Manual Kill | `/kill` 명령 or KILL_SWITCH 파일 | 즉시 전량 청산 |

### 4.4 Discord 알림 라우팅

| Severity | 채널 | 멘션 | 예시 |
|----------|------|------|------|
| INFO | #trade-log | 없음 | Trade entry/exit |
| WARNING | #alerts | `@here` | Drawdown 경고, 연결 불안정 |
| CRITICAL | #alerts | `@owner` | Circuit breaker, 시스템 정지 |
| EMERGENCY | #alerts | `@owner` + DM | 자금 이상, 보안 위협 |

### 4.5 State Recovery (재시작 복구)

복구 순서 (StateManager에 구현됨):

1. SQLite `bot_state` 테이블에서 PM/RM 상태 로드
2. Exchange position 조회 (`fetch_positions`) — **Live 전환 시**
3. Reconciliation (state vs exchange) — exchange를 truth로 사용
4. 놓친 bar 보충 (`fetch_ohlcv` since last_processed_ts)
5. 정상 운영 재개

### 4.6 Exchange Disconnection 대응

| 시나리오 | 대응 |
|---------|------|
| WebSocket 끊김 | 자동 재연결 (exponential backoff, max 5회) |
| REST API 타임아웃 | 재시도 (3회) |
| 장기 disconnection (> 5분) | 봇 일시 중지, 알림, 재연결 후 reconciliation |

### 4.7 남은 의존성

```
Phase 7.5에서 추가:
  uv add prometheus-client   # Prometheus metrics

분석 전용 (선택):
  uv add --group research duckdb  # Parquet+SQLite 크로스 쿼리
```

> Phase 6에서 이미 추가 완료: `aiosqlite`, `discord.py>=2.6.0`

---

## 5. Sources

### Database

- [DuckDB vs SQLite 2026](https://www.analyticsvidhya.com/blog/2026/01/duckdb-vs-sqlite/)
- [Freqtrade - SQLite-based Crypto Trading Bot](https://github.com/freqtrade/freqtrade)

### Discord Bot

- [discord.py Documentation](https://discordpy.readthedocs.io/en/stable/)
- [discord.py 2.0 Slash Commands](https://gist.github.com/AbstractUmbra/a9c188797ae194e592efe05fa129c57f)
- [Discord Rate Limits](https://discord.com/developers/docs/topics/rate-limits)

### Production Deployment

- [Binance API Rate Limits](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits)
- [Using uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/)
- [Graceful Shutdowns with asyncio](https://roguelynn.com/words/asyncio-graceful-shutdowns/)
- [CCXT Documentation](https://docs.ccxt.com/)
- [Pydantic Settings v2](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [FIA Automated Trading Risk Controls](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf)
