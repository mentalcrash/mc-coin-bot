# MC Coin Bot

Event-Driven Architecture 기반 암호화폐 퀀트 트레이딩 시스템.

검증된 TSMOM 전략을 8-asset Equal-Weight 포트폴리오로 운용하며,
VectorBT 기반 백테스트와 3단계 과적합 검증을 거쳐 실거래로 전환합니다.

---

## 전략 성과 (6년 백테스트, 2020-2025)

| Sharpe | CAGR | MDD | AnnVol | Calmar | Sortino | Total Return |
|:------:|:----:|:---:|:------:|:------:|:-------:|:------------:|
| **2.41** | **+52.1%** | **-17.5%** | 21.7% | **2.98** | **3.33** | **+1140%** |

### 성과 변화 추이

| 지표 | BTC TSMOM (단일) | 8-asset EW | 헤지 최적화 | 리스크 최적화 | BTC B&H |
|------|:---:|:---:|:---:|:---:|:---:|
| Sharpe | 1.04 | 1.98 | 2.33 | **2.41** | 0.69 |
| CAGR | +31.3% | +40.7% | +49.1% | **+52.1%** | +26.5% |
| MDD | -35.4% | -20.2% | -20.7% | **-17.5%** | -81.2% |

---

## 확정 설정

```python
# 8-asset Equal-Weight TSMOM Portfolio
assets = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
          "DOGE/USDT", "LINK/USDT", "ADA/USDT", "AVAX/USDT"]
weight = 1/8  # Equal-Weight
timeframe = "1D"

TSMOMConfig(
    lookback=30,              # 30일 모멘텀
    vol_window=30,            # 30일 변동성
    vol_target=0.35,          # 균형 설정
    annualization_factor=365, # 일봉 연환산
    short_mode=ShortMode.HEDGE_ONLY,
    hedge_threshold=-0.07,    # -7% 드로다운 시 헤지
    hedge_strength_ratio=0.3, # 롱의 30%로 방어적 숏
)
PortfolioManagerConfig(
    max_leverage_cap=2.0,
    system_stop_loss=0.10,              # 10% 손절 (안전망)
    use_trailing_stop=True,
    trailing_stop_atr_multiplier=3.0,   # 3x ATR (MDD 핵심 방어)
    rebalance_threshold=0.10,           # 10% (거래비용 최적화)
)
```

---

## 검증 완료 항목

| 항목 | 최적값 | 방법 |
|------|--------|------|
| 에셋 구성 | 8-asset EW | 4 vs 8 비교 (Sharpe +5%) |
| vol_target | 0.35 | 8단계 스윕 (0.15~0.50), 320 백테스트 |
| 타임프레임 | 1D | 4개 TF x 6 lookback, 192 백테스트 |
| lookback | 30일 | 4개 TF 모두 30일 최적 |
| hedge_strength_ratio | 0.3 | 10단계 스윕, 656 백테스트 |
| trailing_stop | 3.0x ATR | 6단계 스윕, MDD 4.4pp 개선 |
| rebalance_threshold | 10% | 5단계 스윕, 거래비용 절감 |
| 리스크 파라미터 통합 | SL 10% + TS 3.0x + rebal 10% | 총 368 백테스트 |
| 과적합 검증 | PASS | IS/OOS, Walk-Forward, CPCV, DSR, PBO |

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
| Data | Parquet (Medallion Architecture) |
| Logging | Loguru |

---

## 프로젝트 구조

```
mc-coin-bot/
├── src/
│   ├── backtest/           # BacktestEngine, Analyzer, Validation
│   │   └── validation/     # IS/OOS, WFA, CPCV, DSR, PBO
│   ├── cli/                # Typer CLI (backtest, ingest)
│   ├── config/             # 환경변수, 설정
│   ├── core/               # 공통 모듈
│   ├── data/               # Medallion (Bronze/Silver), MarketDataService
│   ├── exchange/           # Binance 커넥터
│   ├── models/             # Pydantic 모델 (BacktestResult 등)
│   ├── portfolio/          # PortfolioManagerConfig, Portfolio
│   └── strategy/           # BaseStrategy, TSMOM, Breakout 등
├── tests/                  # pytest (191+ tests)
├── docs/                   # 상세 문서
│   ├── architecture/       # 백테스트 엔진 아키텍처
│   ├── strategy-evaluation/# 전략 평가 지식베이스 (핵심)
│   ├── backtesting-best-practices.md
│   └── portfolio-manager.md
└── data/                   # Bronze/Silver 데이터
    ├── bronze/
    └── silver/
```

---

## 빠른 시작

### 환경 설정

```bash
uv sync --group dev --group research
cp .env.example .env  # API 키 설정
```

### 백테스트 실행

```bash
# 단일에셋 백테스트
uv run python -m src.cli.backtest run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31

# 8-asset 멀티에셋 포트폴리오
uv run python -m src.cli.backtest run-multi -s tsmom -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025

# 파라미터 스윕
uv run python -m src.cli.backtest sweep tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31

# QuantStats HTML 리포트
uv run python -m src.cli.backtest run tsmom BTC/USDT --start 2024-01-01 --end 2025-12-31 --report
```

### 과적합 검증

```bash
# QUICK (IS/OOS)
uv run python -m src.cli.backtest validate -m quick

# MILESTONE (Walk-Forward)
uv run python -m src.cli.backtest validate -m milestone

# FINAL (CPCV + DSR + PBO)
uv run python -m src.cli.backtest validate -m final
```

### 데이터 수집

```bash
# Bronze → Silver 파이프라인
python main.py ingest pipeline BTC/USDT --year 2024 --year 2025

# 데이터 검증
python main.py ingest validate BTC/USDT --year 2025
```

---

## 개발

### 코드 품질 (Zero-Tolerance Lint Policy)

```bash
uv run ruff check --fix . && uv run ruff format .   # lint + format
uv run pyright src/                                   # type check
uv run pytest --cov=src                               # test + coverage
```

### 핵심 금지 사항

- `float` for prices/amounts → use `Decimal`
- `iterrows()`, loops on DataFrame → use vectorized ops
- `inplace=True` → use immutable operations
- `except:` → use specific exceptions
- `# noqa`, `# type: ignore` → 정당한 사유 없이 사용 금지

---

## 로드맵

```
Phase 1: 단일에셋 백테스트     ✅ 완료 (VectorBT, 4 strategies, Numba)
Phase 2: 멀티에셋 백테스트     ✅ 완료 (run_multi, cash_sharing, 8-asset EW)
Phase 3: 고급 검증             ✅ 완료 (IS/OOS, WFA, CPCV, DSR, PBO)
Phase 4: EDA 시스템            ← 현재 (EventBus + 이벤트 기반 백테스트)
Phase 5: Dry Run              예정 (Shadow → Paper → Canary)
Phase 6: Live Trading         예정 (점진적 자본 투입 5% → 100%)
Phase 7: 모니터링              예정 (Streamlit + Grafana + Discord)
```

상세 로드맵: [docs/strategy-evaluation/implementation-roadmap.md](docs/strategy-evaluation/implementation-roadmap.md)

---

## 문서

| 문서 | 설명 |
|------|------|
| [strategy-evaluation/](docs/strategy-evaluation/) | 전략 평가 지식베이스 (핵심) |
| [strategy-evaluation/README.md](docs/strategy-evaluation/README.md) | 전략 평가 요약 + 확정 설정 |
| [strategy-evaluation/implementation-roadmap.md](docs/strategy-evaluation/implementation-roadmap.md) | Phase 2~7 상세 로드맵 |
| [architecture/backtest-engine.md](docs/architecture/backtest-engine.md) | 백테스트 엔진 아키텍처 |
| [portfolio-manager.md](docs/portfolio-manager.md) | 포트폴리오 매니저 아키텍처 |
| [backtesting-best-practices.md](docs/backtesting-best-practices.md) | 백테스팅 모범사례 가이드 |

---

> **Disclaimer:** 이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 암호화폐 거래는 높은 위험을 수반하며, 실제 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
