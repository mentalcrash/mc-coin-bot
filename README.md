# MC Coin Bot

Event-Driven Architecture 기반 암호화폐 퀀트 트레이딩 시스템.

24개 전략을 체계적으로 평가하여 4개 검증 전략을 선별, 멀티에셋 포트폴리오로 운용합니다.
VectorBT + EDA 이중 백테스트와 3단계 과적합 검증을 거쳐 실거래로 전환합니다.

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

전략은 Gate 0 → Gate 7 순서로 평가되며, 각 단계를 통과해야 다음 단계로 진행합니다.
상세 기준은 [전략 평가 표준](docs/strategy-evaluation-standard.md), 개별 결과는 [스코어카드](docs/scorecard/)를 참조하세요.

### Gate 0: Idea Viability (아이디어 검증)

구현 전 6가지 기준으로 아이디어를 평가합니다 (각 1~5점, 총 30점).

| 기준 | 설명 |
|------|------|
| Economic Rationale | 경제적 논거 강도 (왜 alpha가 존재하는가?) |
| Novelty | 기존 전략 대비 참신성 |
| Data Availability | 필요한 데이터 확보 용이성 |
| Complexity | 구현 복잡도 (낮을수록 좋음) |
| Capacity | 운용 수용량 (슬리피지, 유동성) |
| Regime Dependency | 레짐 의존성 (낮을수록 좋음) |

> **18/30점 이상**: PASS (구현 진행)

### Gate 1: Single-Asset Backtest (단일에셋 백테스트)

5개 자산(BTC, ETH, BNB, SOL, DOGE) x 6년(2020-2025) 백테스트 후 Best Asset 기준 판정.

| 판정 | 기준 |
|------|------|
| **PASS** | Sharpe > 1.0 + MDD < 40% + 거래 50건 이상 |
| **WATCH** | 0.5 <= Sharpe <= 1.0, 또는 25% <= MDD <= 40% |
| **FAIL** | 총수익 음수 + 거래 20건 미만, 또는 MDD > 50% |

> PASS 전략은 멀티에셋 포트폴리오 편입 후보. FAIL 전략은 코드 삭제.

### 이후 검증 단계

| Gate | 검증 내용 | CLI |
|------|----------|-----|
| Gate 2 | IS/OOS Split (70/30) | `validate -m quick` |
| Gate 3 | Walk-Forward Analysis (5-fold) | `validate -m milestone` |
| Gate 4 | CPCV + DSR + PBO | `validate -m final` |
| Gate 5 | EDA Parity (VBT vs EDA 수익 부호 일치) | `eda run` |
| Gate 6 | Paper Trading | - |

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

---

## 전략 스코어카드 (Gate 0-3 평가)

4개 활성 전략 + 20개 폐기 전략. 6년(2020-2025) 5-coin 단일에셋 백테스트 + IS/OOS 70/30 + 파라미터 안정성 검증 결과.

### 활성 전략 (G3 PASS)

| # | 전략 | Best Asset | Sharpe | CAGR | MDD | G0 | G1 | G2 | G3 | 스코어카드 |
|---|------|-----------|--------|------|-----|:--:|:--:|:--:|:--:|-----------|
| 1 | **KAMA** | DOGE/USDT | 1.14 | +35.8% | -13.2% | P | P | P | P | [scorecard](docs/scorecard/kama.md) |
| 2 | **Donchian Ensemble** | ETH/USDT | 0.99 | +10.8% | -9.7% | P | W | P | P | [scorecard](docs/scorecard/donchian-ensemble.md) |
| 3 | **MAX-MIN** | DOGE/USDT | 0.82 | +15.3% | -13.9% | P | W | P | P | [scorecard](docs/scorecard/max-min.md) |
| 4 | **BB-RSI** | SOL/USDT | 0.53 | +4.1% | -15.1% | P | W | P | P | [scorecard](docs/scorecard/bb-rsi.md) |

> `P` = PASS, `W` = WATCH, `F` = FAIL
>
> **Gate 3 기준**: 핵심 파라미터의 고원(plateau) 존재 + ±20% 범위에서 Sharpe 부호 유지.
> 4개 전략 모두 넓은 파라미터 범위에서 안정적 성과. Gate 4(WFA+CPCV) 진행 대상.

### 폐기된 전략 (Deprecated)

아래 전략은 Gate 평가에서 구조적 문제 또는 성과 부족으로 폐기 처리되었습니다.
동일 아이디어의 재구현을 방지하기 위해 실패 사유를 기록합니다.

#### Gate 3 실패 (파라미터 불안정)

| 전략 | Sharpe | 폐기 사유 | 스코어카드 |
|------|--------|----------|-----------|
| TTM Squeeze | 0.94 | bb_period, kc_mult 고원 부재 — bb_period ±2에서 Sharpe 급락, kc_mult 1.0에서 거래 0건 | [scorecard](docs/scorecard/fail/ttm-squeeze.md) |

#### Gate 2 실패 (IS/OOS 과적합)

| 전략 | Sharpe | OOS Sharpe | Decay | 폐기 사유 | 스코어카드 |
|------|--------|-----------|-------|----------|-----------|
| Vol Regime | 1.41 | 0.37 | 77.3% | IS 고성과가 OOS에서 유지 안됨 (Decay 77%) | [scorecard](docs/scorecard/fail/vol-regime.md) |
| TSMOM | 1.33 | 0.19 | 87.2% | OOS Sharpe 0.19, Decay 87% (WFA/CPCV는 PASS) | [scorecard](docs/scorecard/fail/tsmom.md) |
| Enhanced VW-TSMOM | 1.22 | 0.25 | 85.2% | OOS Sharpe 0.25, Decay 85% | [scorecard](docs/scorecard/fail/enhanced-tsmom.md) |
| Vol Structure | 1.18 | 0.59 | 57.2% | OOS Sharpe 양호하나 Decay 57% | [scorecard](docs/scorecard/fail/vol-structure.md) |
| Vol-Adaptive | 1.08 | -0.97 | 155.9% | OOS Sharpe 음수, 완전 과적합 | [scorecard](docs/scorecard/fail/vol-adaptive.md) |
| Donchian Channel | 1.01 | 0.12 | 91.1% | OOS Sharpe 0.12, OOS Return -0.6% | [scorecard](docs/scorecard/fail/donchian.md) |
| ADX Regime | 0.94 | -0.68 | 146.3% | OOS Sharpe 음수, OOS Return -26.8% | [scorecard](docs/scorecard/fail/adx-regime.md) |
| Stochastic Momentum | 0.94 | -0.34 | 124.9% | OOS Sharpe 음수, 포트폴리오 분산 효과 부재 | [scorecard](docs/scorecard/fail/stoch-mom.md) |
| GK Breakout | 0.77 | 0.39 | 59.0% | Decay 59%, OOS Sharpe borderline | [scorecard](docs/scorecard/fail/gk-breakout.md) |
| MTF-MACD | 0.76 | 0.21 | 78.1% | OOS Sharpe 0.21, 신호 빈도 부족 | [scorecard](docs/scorecard/fail/mtf-macd.md) |
| HMM Regime | 0.75 | -0.66 | 162.9% | OOS Sharpe 음수, HMM 수렴 불안정 | [scorecard](docs/scorecard/fail/hmm-regime.md) |
| Adaptive Breakout | 0.54 | -0.68 | 201.1% | OOS Sharpe 음수, Decay 201% (최악) | [scorecard](docs/scorecard/fail/adaptive-breakout.md) |
| Mom-MR Blend | 0.48 | -0.10 | 109.1% | OOS Sharpe 음수, Mom+MR alpha 상쇄 | [scorecard](docs/scorecard/fail/mom-mr-blend.md) |

#### Gate 1 실패 (코드 삭제)

| 전략 | Sharpe | 폐기 사유 | 스코어카드 |
|------|--------|----------|-----------|
| Larry VB | 0.15 | 1-bar hold 비용 구조적 문제 (연 125건 x 0.1% = 12.5% drag) | [scorecard](docs/scorecard/fail/larry-vb.md) |
| Overnight | 0.00 | 1H TF 데이터 부족 + 계절성 불안정 | [scorecard](docs/scorecard/fail/overnight.md) |
| Z-Score MR | -0.02 | 단일 z-score 평균회귀, 낮은 Sharpe | [scorecard](docs/scorecard/fail/zscore-mr.md) |
| RSI Crossover | -0.16 | RSI 단순 크로스오버, 통계적 무의미 | [scorecard](docs/scorecard/fail/rsi-crossover.md) |
| Hurst/ER Regime | 0.24 | Hurst exponent 추정 노이즈, 실용성 부족 | [scorecard](docs/scorecard/fail/hurst-regime.md) |
| Risk Momentum | 0.77 | TSMOM과 높은 상관, 차별화 부족 | [scorecard](docs/scorecard/fail/risk-mom.md) |

> **G3 교훈**: G2 통과 5개 중 **G3 통과는 4개**. TTM Squeeze는 G2 Decay 45.6%(borderline)이었으며, bb_period ±2에서 Sharpe가 0.94→0.53으로 급락하는 뾰족한 최적해.
> 반면 KAMA, Donchian Ensemble, MAX-MIN, BB-RSI는 넓은 파라미터 범위에서 안정적 고원 — **로버스트한 전략은 파라미터에 둔감**.
