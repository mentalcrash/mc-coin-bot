# MC Coin Bot

Event-Driven Architecture 기반 암호화폐 퀀트 트레이딩 시스템.

31개 전략을 체계적으로 평가하여 실전 운용 후보를 선별합니다.
VectorBT + EDA 이중 백테스트와 4단계 과적합 검증(IS/OOS, 파라미터 안정성, WFA, CPCV)을 거쳐 실거래로 전환합니다.
현재 **1개 전략 Gate 2 PASS** (CTREND), 2개 PENDING.

---

## 아키텍처

```
WebSocket → MarketData → Strategy → Signal → PM → RM → OMS → Fill
```

### 핵심 설계 원칙

- **Stateless Strategy / Stateful Execution** -- 전략은 시그널만 생성, PM/RM/OMS가 포지션 관리
- **Target Weights 기반** -- "사라/팔아라" 대신 "적정 비중은 X%"
- **Look-Ahead Bias 원천 차단** -- Signal at Close → Execute at Next Open
- **PM/RM 분리 모델** -- Portfolio Manager → Risk Manager → OMS 3단계 방어

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
| Logging | Loguru |

---

## 전략 설정 (Config YAML)

모든 백테스트와 실행은 YAML 설정 파일 하나로 제어됩니다.
VBT 백테스트와 EDA 백테스트에서 동일한 설정을 공유합니다.

### 설정 구조

```yaml
# config/my_strategy.yaml
backtest:
  symbols: [BTC/USDT, ETH/USDT]   # 1개: 단일에셋, 2개+: 멀티에셋 (Equal Weight)
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
2. `strategy.name`에 등록된 전략 이름 지정 (`uv run python -m src.cli.backtest strategies`로 확인)
3. `strategy.params`에 해당 전략의 파라미터 입력 (각 전략의 `src/strategy/<name>/config.py` 참조)
4. `backtest.symbols`에 테스트할 심볼 나열 (2개 이상이면 자동으로 Equal Weight 멀티에셋)

---

## 전략 평가 체계

전략은 Gate 0(아이디어) → Gate 7(실전 배포) 순서로 평가됩니다.
상세 기준은 [전략 평가 표준](docs/strategy/evaluation-standard.md), 전체 현황은 [전략 상황판](docs/strategy/dashboard.md)을 참조하세요.

| Gate | 검증 | 핵심 기준 | CLI |
|:----:|------|----------|-----|
| 0 | 아이디어 | >= 18/30점 | — |
| 1 | 백테스트 (5코인 x 6년) | Sharpe > 1.0, CAGR > 20%, MDD < 40% | `run {config}` |
| 2 | IS/OOS 70/30 | OOS Sharpe >= 0.3, Decay < 50% | `validate -m quick` |
| 3 | 파라미터 안정성 | 고원 존재, ±20% 안정 | `sweep {config}` |
| 4 | WFA + CPCV + PBO | WFA OOS >= 0.5, PBO < 40% | `validate -m milestone/final` |
| 5 | EDA Parity | VBT vs EDA 수익 부호 일치 | `eda run` |
| 6 | Paper Trading (2주+) | 시그널 일치 > 90% | `eda run-live` |

---

## 빠른 시작

### 환경 설정

```bash
uv sync --group dev --group research
cp .env.example .env  # API 키 설정
```

### 전략 목록 확인

```bash
# 등록된 전략 목록
uv run python -m src.cli.backtest strategies

# 전략 상세 정보
uv run python -m src.cli.backtest info
```

### VBT 백테스트

```bash
# 단일에셋 백테스트
uv run python -m src.cli.backtest run config/default.yaml

# 멀티에셋 포트폴리오 (config의 symbols 2개 이상)
uv run python -m src.cli.backtest run-multi config/default.yaml

# QuantStats HTML 리포트
uv run python -m src.cli.backtest run config/default.yaml --report

# Strategy Advisor 분석
uv run python -m src.cli.backtest run config/default.yaml --advisor

# Verbose 모드
uv run python -m src.cli.backtest run config/default.yaml -V
```

### EDA 백테스트

```bash
# EDA 백테스트 (1m 데이터 → target TF 집계, 단일/멀티 자동 판별)
uv run python main.py eda run config/default.yaml

# QuantStats 리포트 포함
uv run python main.py eda run config/default.yaml --report

# Shadow 모드 (시그널 로깅만, 체결 없음)
uv run python main.py eda run config/default.yaml --mode shadow
```

### 과적합 검증

```bash
# QUICK: IS/OOS Split
uv run python -m src.cli.backtest validate -m quick

# MILESTONE: Walk-Forward (5-fold)
uv run python -m src.cli.backtest validate -m milestone

# FINAL: CPCV + DSR + PBO
uv run python -m src.cli.backtest validate -m final

# 특정 심볼/전략 지정
uv run python -m src.cli.backtest validate -m quick -s tsmom --symbols BTC/USDT,ETH/USDT
```

### 시그널 진단

```bash
# TSMOM 시그널 파이프라인 분석
uv run python -m src.cli.backtest diagnose BTC/USDT -s tsmom

# Adaptive Breakout 진단
uv run python -m src.cli.backtest diagnose SOL/USDT -s adaptive-breakout -V
```

### 데이터 수집

```bash
# Bronze → Silver 파이프라인
uv run python main.py ingest pipeline BTC/USDT --year 2024 --year 2025

# 데이터 검증
uv run python main.py ingest validate data/silver/BTC_USDT_1m_2025.parquet

# 상위 N개 심볼 일괄 다운로드
uv run python main.py ingest bulk-download --top 100 --year 2024 --year 2025

# 데이터 상태 확인
uv run python main.py ingest info
```

### 일괄 백테스트 & 스코어카드

```bash
# 전 전략 일괄 백테스트 (23 전략 x 5 자산)
uv run python scripts/bulk_backtest.py

# 스코어카드 자동 생성
uv run python scripts/generate_scorecards.py
```

### 배포 (DigitalOcean + Coolify)

```bash
# Docker 빌드 (multi-stage, uv 기반)
docker build -t mc-coin-bot:latest .

# 로컬 실행 (환경 변수로 모드 제어)
docker run --env-file .env \
  -e MC_EXECUTION_MODE=paper \
  -e MC_CONFIG_PATH=config/paper.yaml \
  -e MC_INITIAL_CAPITAL=10000 \
  mc-coin-bot:latest
```

DigitalOcean Droplet + Coolify로 배포합니다. `MC_*` 환경 변수로 실행 모드를 제어합니다.

| 환경 변수 | 기본값 | 설명 |
|----------|--------|------|
| `MC_EXECUTION_MODE` | `paper` | 실행 모드 (`paper` / `shadow` / `live`) |
| `MC_CONFIG_PATH` | `config/paper.yaml` | YAML 설정 파일 경로 |
| `MC_INITIAL_CAPITAL` | `10000` | 초기 자본 (USD) |
| `MC_DB_PATH` | `data/trading.db` | SQLite 경로 |
| `MC_ENABLE_PERSISTENCE` | `true` | 상태 영속화 on/off |

---

## 전략 현황

31개 전략 평가 완료: 1개 활성 (CTREND) + 2개 PENDING + 28개 폐기.
상세 현황과 폐기 전략 목록은 **[전략 상황판](docs/strategy/dashboard.md)** 참조.

| 전략 | Best Asset | Sharpe | CAGR | G0 | G1 | G2 | G3 | G4 | 상태 |
|------|-----------|--------|------|:--:|:--:|:--:|:--:|:--:|------|
| **CTREND** | SOL/USDT | 2.05 | +97.8% | P | P | P | P | F | PBO 60%, EDA/Paper 검증 권고 |
