# MC Coin Bot

Event-Driven Architecture 기반 암호화폐 퀀트 트레이딩 시스템.

검증된 TSMOM 전략을 8-asset Equal-Weight 포트폴리오로 운용하며,
VectorBT 기반 백테스트와 3단계 과적합 검증을 거쳐 실거래로 전환합니다.

---

## 전략 성과 (6년 백테스트, 2020-2025, 펀딩비 반영)

| Sharpe | CAGR | MDD | Calmar | Sortino | Profit Factor | Total Return |
|:------:|:----:|:---:|:------:|:-------:|:-------------:|:------------:|
| **1.48** | **+53.1%** | **-19.9%** | **2.66** | **2.42** | **2.00** | **+1188%** |

> 선물 펀딩비(0.01%/8h) 반영 후 검증된 수치. 코드 감사 6건 수정 완료.

### 성과 변화 추이

| 지표 | BTC TSMOM (단일) | 8-asset EW | 헤지 최적화 | 리스크 최적화 | 코드 감사 (최종) | BTC B&H |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Sharpe | 1.04 | 1.98 | 2.33 | 2.41 | **1.48** | 0.69 |
| CAGR | +31.3% | +40.7% | +49.1% | +52.1% | **+53.1%** | +26.5% |
| MDD | -35.4% | -20.2% | -20.7% | -17.5% | **-19.9%** | -81.2% |

> **코드 감사 (최종)**: 펀딩비 사후 보정(H-001), Aggregate Leverage 검증(M-001) 등 6건 수정 반영.
> Sharpe 하락은 펀딩 드래그(~4.85pp/yr) 반영에 의한 것으로, 실거래 환경에 가장 근접한 수치.

---

## 확정 설정 (`config/default.yaml`)

모든 트레이딩 설정은 YAML 파일 하나로 관리됩니다.
VBT 백테스트와 EDA 백테스트에서 동일한 설정을 공유합니다.

```yaml
# config/default.yaml — 8-Asset EW TSMOM (Best Configuration)
backtest:
  symbols: [BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT,
            DOGE/USDT, LINK/USDT, ADA/USDT, AVAX/USDT]
  timeframe: "1D"
  start: "2020-01-01"
  end: "2025-12-31"
  capital: 100000.0

strategy:
  name: tsmom
  params:
    lookback: 30              # 30일 모멘텀
    vol_window: 30            # 30일 변동성
    vol_target: 0.35          # 균형 설정
    short_mode: 1             # HEDGE_ONLY
    hedge_threshold: -0.07    # -7% 드로다운 시 헤지
    hedge_strength_ratio: 0.3 # 롱의 30%로 방어적 숏

portfolio:
  max_leverage_cap: 2.0
  rebalance_threshold: 0.10           # 10% (거래비용 최적화)
  system_stop_loss: 0.10              # 10% 손절 (안전망)
  use_trailing_stop: true
  trailing_stop_atr_multiplier: 3.0   # 3x ATR (MDD 핵심 방어)
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
| 코드 감사 (Grade B+ → A-) | 6건 수정, 펀딩비 반영 | 6-step audit, 0 CRITICAL |

### 코드 감사 상세 (EDA 승격 전 최종 검증)

6-step quant audit 수행 후 발견된 이슈 6건을 모두 수정하고, 펀딩비 반영 후 재검증 완료.

| ID | 등급 | 이슈 | 수정 내용 |
|---|---|---|---|
| H-001 | HIGH | 선물 펀딩비 미반영 (~10.95%/yr) | `_adjust_metrics_for_funding()` 사후 보정 |
| H-002 | HIGH | Signals 모드 Trailing Stop 누락 | `to_vbt_params()`에 `sl_trail=True` 전달 |
| M-001 | MEDIUM | Multi-Asset Aggregate Leverage 미검증 | `weights_df.abs().sum(axis=1)` 스케일링 |
| M-002 | MEDIUM | Stop-Loss Close-only (Intrabar 미체크) | `use_intrabar_stop` 옵션 추가 |
| M-003 | MEDIUM | `extract_trades()` iterrows 사용 | `to_dict("records")` 벡터화 |
| L-001 | LOW | CAGR 1년 미만 단순 연환산 | 복리 공식 통일 |

**감사 스코어카드**: Data Integrity 10/10, Signal Logic 9/10, Execution Realism 8/10 → **Overall 8.5/10**

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

## 빠른 시작

### 환경 설정

```bash
uv sync --group dev --group research
cp .env.example .env  # API 키 설정
```

### VBT 백테스트 실행

```bash
# 단일에셋 백테스트 (config 파일의 첫 번째 심볼 사용)
uv run python -m src.cli.backtest run config/default.yaml

# 8-asset 멀티에셋 포트폴리오
uv run python -m src.cli.backtest run-multi config/default.yaml

# QuantStats HTML 리포트
uv run python -m src.cli.backtest run config/default.yaml --report

# Strategy Advisor 분석
uv run python -m src.cli.backtest run config/default.yaml --advisor
```

### EDA 백테스트 실행

```bash
# EDA 백테스트 (config의 symbols 개수로 단일/멀티 자동 판별)
uv run python main.py eda run config/default.yaml

# Shadow 모드 (시그널 로깅만, 체결 없음)
uv run python main.py eda run config/default.yaml --mode shadow
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

## 전략 스코어카드 (Gate 1 일괄 평가)

23개 전략 대상 6년(2020-2025) 5-coin 백테스트 결과. Best Asset 기준 단일에셋 성과.

### 활성 전략 (17개)

| # | 전략 | Best Asset | Sharpe | CAGR | MDD | 판정 | 스코어카드 |
|---|------|-----------|--------|------|-----|------|-----------|
| 1 | **Vol-Regime** | ETH/USDT | 1.41 | +57.6% | -32.3% | **PASS** | [scorecard](docs/strategy/strategy-scorecard-vol-regime.md) |
| 2 | **TSMOM** | SOL/USDT | 1.33 | +50.0% | -26.0% | **PASS** | [scorecard](docs/strategy/strategy-scorecard-tsmom.md) |
| 3 | **Enhanced TSMOM** | BTC/USDT | 1.22 | +39.1% | -37.5% | **PASS** | [scorecard](docs/strategy/strategy-scorecard-enhanced-tsmom.md) |
| 4 | **Vol-Structure** | SOL/USDT | 1.18 | +31.8% | -28.3% | **PASS** | [scorecard](docs/strategy/strategy-scorecard-vol-structure.md) |
| 5 | **KAMA** | DOGE/USDT | 1.14 | +35.8% | -13.3% | **PASS** | [scorecard](docs/strategy/strategy-scorecard-kama.md) |
| 6 | **Vol-Adaptive** | SOL/USDT | 1.08 | +31.5% | -44.9% | **PASS** | [scorecard](docs/strategy/strategy-scorecard-vol-adaptive.md) |
| 7 | **Donchian** | SOL/USDT | 1.01 | +30.5% | -50.0% | **PASS** | [scorecard](docs/strategy/strategy-scorecard-donchian.md) |
| 8 | Donchian Ensemble | ETH/USDT | 0.99 | +10.8% | -9.7% | WATCH | [scorecard](docs/strategy/strategy-scorecard-donchian-ensemble.md) |
| 9 | TTM Squeeze | BTC/USDT | 0.94 | +17.7% | -37.7% | WATCH | [scorecard](docs/strategy/strategy-scorecard-ttm-squeeze.md) |
| 10 | ADX Regime | SOL/USDT | 0.94 | +23.1% | -33.9% | WATCH | [scorecard](docs/strategy/strategy-scorecard-adx-regime.md) |
| 11 | Stoch-Mom | SOL/USDT | 0.94 | +7.1% | -9.0% | WATCH | [scorecard](docs/strategy/strategy-scorecard-stoch-mom.md) |
| 12 | Max-Min | DOGE/USDT | 0.82 | +15.3% | -13.9% | WATCH | [scorecard](docs/strategy/strategy-scorecard-max-min.md) |
| 13 | GK Breakout | DOGE/USDT | 0.77 | +13.1% | -23.2% | WATCH | [scorecard](docs/strategy/strategy-scorecard-gk-breakout.md) |
| 14 | MTF-MACD | SOL/USDT | 0.76 | +6.6% | -8.4% | WATCH | [scorecard](docs/strategy/strategy-scorecard-mtf-macd.md) |
| 15 | HMM Regime | BTC/USDT | 0.75 | +18.7% | -31.7% | WATCH | [scorecard](docs/strategy/strategy-scorecard-hmm-regime.md) |
| 16 | Adaptive Breakout | SOL/USDT | 0.54 | +6.7% | -12.1% | WATCH | [scorecard](docs/strategy/strategy-scorecard-adaptive-breakout.md) |
| 17 | BB-RSI | SOL/USDT | 0.53 | +4.1% | -15.1% | WATCH | [scorecard](docs/strategy/strategy-scorecard-bb-rsi.md) |
| — | Mom-MR Blend | ETH/USDT | 0.48 | +6.7% | -29.2% | WATCH | [scorecard](docs/strategy/strategy-scorecard-mom-mr-blend.md) |

> **PASS**: Sharpe >= 1.0 + 구조적 결함 없음. **WATCH**: 0.5 <= Sharpe < 1.0, 멀티에셋/파라미터 최적화 후 재평가.

### 폐기된 전략 (Deprecated)

아래 전략은 Gate 1 평가에서 구조적 문제 또는 성과 부족으로 **코드 삭제** 처리되었습니다.
동일 아이디어의 재구현을 방지하기 위해 실패 사유를 기록합니다.

| 전략 | Sharpe | 폐기 사유 | 폐기일 |
|------|--------|----------|--------|
| Larry-VB | 0.15 | 1-bar hold 비용 구조적 문제 (연 125건 x 0.1% = 12.5% drag) | 2026-02-09 |
| Overnight | 0.00 | 1H 전용 전략, 1D 백테스트 거래 0건. 구조적 TF 불일치 | 2026-02-09 |
| Z-Score MR | -0.02 | 총수익 음수, MDD -52%. Mean-reversion 전략의 한계 | 2026-02-09 |
| RSI Crossover | -0.16 | 총수익 음수, 단순 RSI 전략의 한계 | 2026-02-09 |
| Hurst Regime | 0.24 | MDD -57%, 462건 거래에도 CAGR 1.9%. Hurst exponent 불안정 | 2026-02-09 |
| Risk-Momentum | 0.77 | MDD -95.5% — 사실상 파산. 리스크 조절 실패 | 2026-02-09 |

> **교훈**: 단일지표 < 앙상블, 단일코인 < 멀티에셋. 고빈도 거래는 비용 구조 확인 필수. MDD > 50%는 무조건 폐기.

---
