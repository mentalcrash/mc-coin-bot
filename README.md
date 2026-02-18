# MC Coin Bot

Event-Driven Architecture 기반 암호화폐 퀀트 트레이딩 시스템.

87개 전략을 8단계 Gate 파이프라인으로 평가하여 실전 운용 후보를 선별합니다.
현재 **2개 전략 ACTIVE** (CTREND, Anchor-Mom), **3개 CANDIDATE** 검증 중.

---

## 아키텍처

### 단일 전략 (EDA)

```
WebSocket → MarketData → Strategy → Signal → PM → RM → OMS → Fill
```

### 멀티 전략 (Orchestrator)

```
                    ┌─ Pod A (TSMOM)   ─┐
WebSocket → Data ──→├─ Pod B (Donchian) ─┼→ Netting → PM → RM → OMS → Fill
                    └─ Pod C (VolAdapt) ─┘
                         ▲                    ▲
                    Capital Allocator    Risk Aggregator
                    Lifecycle Manager    (5-check defense)
```

> 상세: [`docs/architecture/strategy-orchestrator.md`](docs/architecture/strategy-orchestrator.md)

### 핵심 설계 원칙

- **Stateless Strategy / Stateful Execution** -- 전략은 시그널만 생성, PM/RM/OMS가 포지션 관리
- **Target Weights 기반** -- "사라/팔아라" 대신 "적정 비중은 X%"
- **Look-Ahead Bias 원천 차단** -- Signal at Close → Execute at Next Open
- **PM/RM 분리 모델** -- Portfolio Manager → Risk Manager → OMS 3단계 방어
- **Pod 독립성** -- 각 전략은 독립 Pod으로 운영 (독립 P&L, 독립 리스크)
- **Net Execution** -- 심볼별 포지션 넷팅으로 마진 효율 극대화
- **자동 방어** -- Degradation 감지 → 자동 축소 → Probation → Retirement

### 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.13 |
| Package Manager | uv |
| Type Safety | Pydantic V2 + pyright |
| Exchange | CCXT Pro (WebSocket + REST) |
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
VBT 백테스트와 EDA 백테스트에서 동일한 설정을 공유합니다.

### 설정 구조

```yaml
# config/my_strategy.yaml
backtest:
  symbols: [BTC/USDT, ETH/USDT]   # 1개: 단일에셋, 2개+: 멀티에셋 (기본 Equal Weight, 동적 배분 가능)
  timeframe: "1D"                   # 1D, 4h, 1h 등
  start: "2020-01-01"
  end: "2025-12-31"
  capital: 100000.0

strategy:
  name: tsmom                       # 등록된 전략 이름 (strategies 명령으로 확인)
  params:                           # 전략별 파라미터 (각 전략의 config.py 참조)
    lookback: 30
    vol_window: 30
    vol_target: 0.35
    short_mode: 1                   # 0=DISABLED, 1=HEDGE_ONLY, 2=FULL
    hedge_threshold: -0.07
    hedge_strength_ratio: 0.3
  # sub_strategies:                 # (선택) 앙상블 전략 전용
  #   - name: tsmom
  #     params: { lookback: 30 }

portfolio:
  max_leverage_cap: 2.0
  rebalance_threshold: 0.10         # 비중 변화 10% 이상 시 리밸런싱
  system_stop_loss: 0.10            # 10% 시스템 손절 (안전망)
  use_trailing_stop: true
  trailing_stop_atr_multiplier: 3.0 # 3x ATR 트레일링 스톱
  cost_model:                       # 선택 (기본값 있음)
    maker_fee: 0.0002
    taker_fee: 0.0004
    slippage: 0.0005
    funding_rate_8h: 0.0001
    market_impact: 0.0002
```

### 새 전략 설정 작성법

1. `config/` 디렉토리에 YAML 파일 생성
2. `strategy.name`에 등록된 전략 이름 지정 (`uv run mcbot backtest strategies`로 확인)
3. `strategy.params`에 해당 전략의 파라미터 입력 (각 전략의 `src/strategy/<name>/config.py` 참조)
4. `backtest.symbols`에 테스트할 심볼 나열 (2개 이상이면 자동으로 Equal Weight 멀티에셋)

### 앙상블 전략 설정

여러 전략의 시그널을 하나로 결합하는 메타 전략입니다. `strategy.name: ensemble`로 지정하고 `sub_strategies`에 서브 전략을 나열합니다.

```yaml
# config/ensemble-example.yaml
strategy:
  name: ensemble
  params:
    aggregation: inverse_volatility  # equal_weight | inverse_volatility | majority_vote | strategy_momentum
    vol_lookback: 63
    vol_target: 0.35
  sub_strategies:
    - name: tsmom
      params: { lookback: 30, vol_target: 0.35 }
    - name: donchian-ensemble
      params: { lookbacks: [20, 60, 150] }
    - name: vol-adaptive
      params: {}
```

**Aggregation 방법 4가지:**

| 방법 | 설명 |
|------|------|
| `equal_weight` | 동일 가중 평균 (기본값) |
| `inverse_volatility` | 안정적인 전략에 높은 가중치 |
| `majority_vote` | 다수결 합의 (min_agreement 이상) |
| `strategy_momentum` | 최근 Sharpe 상위 top_n 선택 |

### Orchestrator 설정 (멀티 전략)

독립 전략들을 **사업부처럼** 병렬 운영합니다. 각 Pod은 독립 자본 슬롯과 P&L을 가지며, 성과에 따라 자본이 동적 배분됩니다.

```yaml
# config/orchestrator-example.yaml
orchestrator:
  allocation_method: risk_parity
  rebalance:
    trigger: hybrid
    calendar_days: 7

pods:
  - pod_id: pod-tsmom-major
    strategy: tsmom
    params: { lookback: 30, vol_target: 0.35 }
    symbols: [BTC/USDT, ETH/USDT, SOL/USDT]
    initial_fraction: 0.15
    max_fraction: 0.40

    # Pod 내 에셋 배분 (생략 시 equal_weight)
    asset_allocation:
      method: inverse_volatility
      vol_lookback: 60
      rebalance_bars: 5
      min_weight: 0.10
      max_weight: 0.50

  - pod_id: pod-donchian-alt
    strategy: donchian-ensemble
    symbols: [SOL/USDT, BNB/USDT]
    initial_fraction: 0.10
    # asset_allocation 생략 → equal_weight (1/N)
```

**Pod 내 에셋 배분 방법:**

| 방법 | 설명 | 적합 상황 |
|------|------|----------|
| `equal_weight` | 1/N 균등 (기본값) | 에셋 수 적거나 변동성 유사 |
| `inverse_volatility` | 저변동 에셋에 높은 비중 | BTC vs SOL 등 변동성 차이 큰 경우 |
| `risk_parity` | 리스크 기여 균등화 | 상관관계까지 반영한 정밀 배분 |
| `signal_weighted` | 시그널 강도 비례 | 전략의 확신도 차이 반영 |

> 상세 설정 및 아키텍처: [`docs/architecture/strategy-orchestrator.md`](docs/architecture/strategy-orchestrator.md)

---

## 전략 파이프라인

전략은 **아이디어 발굴(G0A)** → **실전 배포(G7)** 까지 8단계 Gate를 순차 통과해야 합니다.
각 Gate에서 FAIL 시 즉시 폐기. 87개 전략 중 **2개 ACTIVE + 3개 CANDIDATE + 83개 RETIRED**.

| Phase | Gates | 검증 내용 |
|-------|-------|----------|
| 발굴 & 구현 | G0A → G0B | 전략 후보 선정, Critical 7항목 코드 검증 |
| 백테스트 검증 | G1 → G4 | 5코인×6년, IS/OOS, 파라미터 Sweep, WFA+CPCV+PBO+DSR |
| 라이브 전환 | G5 → G7 | VBT↔EDA Parity, Paper Trading(2주+), Live 배포 |

Gate별 상세 기준과 전체 현황은 `uv run mcbot pipeline report`로 확인.

---

## 빠른 시작

### 환경 설정

```bash
uv sync --group dev --group research
cp .env.example .env  # API 키 설정
```

### 전략 목록 확인

```bash
uv run mcbot backtest strategies      # 등록된 전략 목록
uv run mcbot backtest info            # 전략 상세 정보
```

### 백테스트

> **VBT**: Vectorized 고속 백테스트 (탐색용) / **EDA**: Event-Driven (라이브 동일 코드, 최종 검증)

```bash
# VBT 백테스트 (단일에셋 / 멀티에셋은 config의 symbols 수로 자동 판별)
uv run mcbot backtest run config/default.yaml

# EDA 백테스트 (1m 데이터 → target TF 집계)
uv run mcbot eda run config/default.yaml

# 옵션: --report (QuantStats HTML), --advisor (Strategy Advisor), -V (Verbose)
```

### Live Trading

```bash
uv run mcbot eda run-live config/paper.yaml --mode paper    # Paper — 시뮬레이션 체결
uv run mcbot eda run-live config/paper.yaml --mode shadow   # Shadow — 시그널 로깅만
uv run mcbot eda run-live config/paper.yaml --mode live     # Live — Binance 실주문 ⚠️
```

Live 모드는 Binance USDT-M Futures에서 Hedge Mode(Cross Margin, 1x Leverage)로 실행됩니다.
60초마다 거래소 포지션과 PM 상태를 교차 검증(PositionReconciler)하며, 불일치 시 경고만 발행합니다(자동 수정 없음).
봇 장애 시 포지션 보호를 위한 거래소 STOP_MARKET 안전망은 [`docs/exchange-safety-stop.md`](docs/exchange-safety-stop.md) 참조.

### 일괄 백테스트

```bash
uv run python scripts/bulk_backtest.py   # 전 전략 일괄 백테스트
```

---

## 데이터 수집

OHLCV(1분봉), 파생상품(Funding/OI/LS/Taker), On-chain(6개 소스 22개 데이터셋) 데이터를 Medallion Architecture(Bronze→Silver)로 수집·정제합니다.

```bash
uv run mcbot ingest pipeline BTC/USDT --year 2024 --year 2025       # OHLCV
uv run mcbot ingest derivatives batch                                # 파생상품 (8 자산)
uv run mcbot ingest onchain batch --type all                         # On-chain (22 데이터셋)
uv run mcbot ingest info                                             # 데이터 상태
```

> 상세 (저장 구조, Rate Limit, Publication Lag, 데이터 품질 등): [`docs/data-collection.md`](docs/data-collection.md)

### 데이터 카탈로그

8개 소스, 28개 데이터셋의 메타데이터를 [`catalogs/datasets.yaml`](catalogs/datasets.yaml)에서 YAML로 관리합니다. 전략 발굴 시 어떤 데이터가 있고, 어떤 컬럼이 나오고, publication lag은 며칠인지 빠르게 탐색할 수 있습니다.

```bash
uv run mcbot catalog list                      # 전체 28개 데이터셋 목록
uv run mcbot catalog list --type onchain       # 유형 필터 (ohlcv, derivatives, onchain)
uv run mcbot catalog list --group stablecoin   # 그룹 필터 (stablecoin, tvl, coinmetrics, ...)
uv run mcbot catalog show btc_metrics          # 상세 (컬럼, enrichment 설정, 전략 힌트)
```

각 데이터셋에는 `strategy_hints`(전략 활용 아이디어)와 `enrichment`(OHLCV 병합 설정)가 포함되어 있어, 새 전략 발굴 시 데이터 탐색 → 아이디어 도출 흐름을 지원합니다.

---

## 배포 (Docker Compose + Coolify)

3개 서비스(트레이딩 봇, Prometheus, Grafana)를 `docker-compose.yml`로 실행합니다.

```bash
docker compose up --build -d      # 전체 스택 실행
docker compose logs -f mc-bot     # 로그 확인
docker compose down               # 중지
```

### 환경 변수

DigitalOcean Droplet + Coolify로 배포합니다. `MC_*` 환경 변수로 실행 모드를 제어합니다.

| 환경 변수 | 기본값 | 설명 |
|----------|--------|------|
| `MC_EXECUTION_MODE` | `paper` | 실행 모드 (`paper` / `shadow` / `live`) |
| `MC_CONFIG_PATH` | `config/paper.yaml` | YAML 설정 파일 경로 (orchestrator YAML 자동 감지) |
| `MC_INITIAL_CAPITAL` | `10000` | 초기 자본 (USD) |
| `MC_DB_PATH` | `data/trading.db` | SQLite DB 경로 (상태 영속화, 빈 문자열=비활성) |
| `MC_ENABLE_PERSISTENCE` | `true` | 상태 영속화 on/off |
| `MC_METRICS_PORT` | `8000` | Prometheus metrics 포트 (`0`이면 비활성) |

Discord 채널 ID 등 추가 환경 변수는 `.env.example` 참조.

### 모니터링 & 알림

- **Prometheus** (`localhost:9090`) + **Grafana** (`localhost:3000`) — 상세: [`docs/monitoring.md`](docs/monitoring.md)
- **Discord 알림**: 체결, Circuit Breaker, 리스크 알림 실시간 전송 + `/status`, `/kill`, `/balance` Slash Commands
- **System Heartbeat** (1시간): equity, drawdown, 레버리지, CB 상태
- **Market Regime Report** (4시간): Funding Rate, OI, LS Ratio → Regime Score
- **Strategy Health Report** (8시간): Rolling Sharpe, Win Rate, Alpha Decay 감지

---

## 전략 철학: 1D Timeframe 집중

### 왜 1D인가

87개 전략을 6년간 검증한 결과, **1D(일봉) 앙상블이 개인 퀀트에게 최적**이라는 결론에 도달했습니다.

| 근거 | 설명 |
|------|------|
| **실증 결과** | 4H 단일지표 전략은 전량 RETIRED. 1D 앙상블만 G5 도달 (CTREND Sharpe 2.05) |
| **거래비용** | 4H는 6배 잦은 리밸런싱 → 수수료·슬리피지가 edge 잠식 |
| **노이즈** | 4H intraday noise가 높아 단일지표로 신호 추출 곤란. 1D에서도 앙상블 필수 |
| **OOS 안정성** | 4H 파라미터는 IS→OOS 붕괴 빈번 (G2 실패 패턴). 1D 모멘텀은 구조적으로 강건 |
| **운영 효율** | 데이터 관리 간편, Rate Limit 여유, 본업 병행 가능 |

### "느리지 않은가?" — SL/TS가 해결

시그널은 1D로 판단하지만, 리스크 관리는 실시간입니다.

```
시그널 생성:  1D bar close 시 (하루 1회, 노이즈 없는 판단)
리스크 방어:  1m bar마다 SL/TS 체크 (급락 시 수 분 내 청산)
```

- Trailing Stop (3.0x ATR): 1분마다 체크 → MDD 방어의 핵심
- System Stop-Loss (5~10%): 포트폴리오 레벨 circuit breaker
- Deferred Execution: 다음 bar open 체결 → look-ahead bias 차단

### 성장 로드맵

현재 시스템은 이미 멀티 전략(Orchestrator) + 멀티 에셋(Pod) 구조를 갖추고 있습니다.
1D 프레임워크 안에서 다음 축으로 확장합니다.

1. **전략 풀 확대**: ACTIVE 2개 → 5~10개 (1D 앙상블 중심 발굴)
2. **자산 다각화**: 8종 → 15~20종 (Tier-2/3 altcoin 추가)
3. **에셋 배분 고도화**: ✅ Pod 내 동적 배분 구현 완료 — EW/IV/RP/SW 4가지 방법 + Numba 최적화 ([상세](docs/architecture/strategy-orchestrator.md#54-intra-pod-asset-allocation))
4. **Derivatives 활용**: Funding Rate(8h), OI, LS Ratio를 1D 전략의 보조 필터로 활용

---

## 전략 현황

| 전략 | Best Asset | TF | Sharpe | CAGR | 상태 |
|------|-----------|-----|--------|------|------|
| **CTREND** | SOL/USDT | 1D | 2.05 | +97.8% | ACTIVE |
| **Anchor-Mom** | DOGE/USDT | 12H | 1.36 | +49.8% | ACTIVE |

> 87개 전략: 2 ACTIVE + 3 CANDIDATE + 83 RETIRED.
> 상세 현황은 `uv run mcbot pipeline report`로 확인.

```bash
uv run mcbot pipeline status          # 현황 요약
uv run mcbot pipeline table           # 전체 Gate 진행도
uv run mcbot pipeline show ctrend     # 전략 상세
```

---

## 운영 도구

### 과적합 검증

```bash
uv run mcbot backtest validate -m quick       # IS/OOS Split
uv run mcbot backtest validate -m milestone   # Walk-Forward (5-fold)
uv run mcbot backtest validate -m final       # CPCV + DSR + PBO
```

### 시그널 진단

```bash
uv run mcbot backtest diagnose BTC/USDT -s tsmom
```

### 교훈 관리

74개 전략 평가 과정에서 축적된 30개 핵심 교훈을 `lessons/*.yaml`로 구조화 관리합니다.

```bash
uv run mcbot pipeline lessons-list                      # 전체 교훈 목록
uv run mcbot pipeline lessons-list -c strategy-design   # 카테고리/태그/전략/TF 필터
uv run mcbot pipeline lessons-show 1                    # 교훈 상세
```

### 아키텍처 감사

정기적인 아키텍처/보안/코드 품질 감사 결과를 `audits/`에 관리합니다.

```bash
uv run mcbot audit latest                               # 최신 스냅샷
uv run mcbot audit findings --status open               # 미해결 발견사항
uv run mcbot audit actions --priority P0                # 긴급 액션
uv run mcbot audit trend                                # 지표 추이
```

---

## 아키텍처 문서

| 문서 | 설명 |
|------|------|
| [`docs/data-collection.md`](docs/data-collection.md) | **데이터 수집 가이드** (OHLCV, Derivatives, On-chain, 저장 구조, CLI) |
| [`docs/architecture/eda-system.md`](docs/architecture/eda-system.md) | EDA 시스템 아키텍처 (이벤트 흐름, 컴포넌트) |
| [`docs/architecture/backtest-engine.md`](docs/architecture/backtest-engine.md) | 백테스트 엔진 설계 (VBT + 검증) |
| [`docs/architecture/strategy-orchestrator.md`](docs/architecture/strategy-orchestrator.md) | **멀티 전략 오케스트레이터** (Pod, 배분, 생애주기, 넷팅) |
| [`docs/exchange-safety-stop.md`](docs/exchange-safety-stop.md) | Exchange Safety Stop (봇 장애 시 거래소 STOP_MARKET 안전망) |
