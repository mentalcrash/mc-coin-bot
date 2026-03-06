# MA Cross Backtest Report

MA Cross 전략의 백테스트 결과 및 개선 이력.

---

## Strategy Overview

| 항목 | 설명 |
|------|------|
| **전략명** | `ma-cross` |
| **로직** | SMA(50) > SMA(200) 골든크로스 + ADX 추세 강도 필터 |
| **포지션** | Long-only, 전체 진입 / 전체 청산 (레버리지 1x) |
| **타임프레임** | 1D |
| **대상 자산** | BTC/USDT |
| **파일** | `src/strategy/ma_cross/` (config, preprocessor, signal, strategy) |
| **설정** | `config/ma_cross_btc.yaml` |

### Signal Logic (현재 — v1.1)

```text
[전봉 기준 — Shift(1) Rule 적용]

Entry: SMA(50, t-1) > SMA(200, t-1) AND ADX(t-1) >= 25
Exit:  SMA(50, t-1) < SMA(200, t-1) OR  ADX(t-1) <  25

Entries/Exits = direction 상태 전환 감지 (0→1 = entry, 1→0 = exit)
```

---

## Backtest Config

```yaml
backtest:
  symbols: [BTC/USDT]
  timeframe: "1D"
  start: "2020-01-01"
  end: "2025-12-31"
  capital: 100000.0

strategy:
  name: ma-cross
  params:
    fast_period: 50
    slow_period: 200
    adx_period: 14
    adx_threshold: 25

portfolio:
  max_leverage_cap: 1.0
  rebalance_threshold: 0.05
  system_stop_loss: null       # SL 없음
  use_trailing_stop: false     # TS 없음
  cost_model:
    maker_fee: 0.0004
    taker_fee: 0.0004
    slippage: 0.0005
    funding_rate_8h: 0.0
    market_impact: 0.0
```

---

## Version History

### v1.1 — ADX Filter (2026-03-05)

ADX(14) >= 25 필터 추가. 횡보장 거짓 신호 필터링 목적.

#### Performance Metrics

| Metric | v1.1 (ADX) | v1.0 (Baseline) | Buy & Hold | v1.0 → v1.1 |
|--------|-----------|-----------------|------------|--------------|
| **Total Return** | +398.32% | +324.00% | +1,128.26% | +74.3%p |
| **CAGR** | +30.66% | +27.19% | — | +3.5%p |
| **Sharpe Ratio** | 0.86 | 0.74 | — | +0.12 |
| **Sortino Ratio** | 0.68 | 0.77 | — | -0.09 |
| **Max Drawdown** | -42.42% | -55.81% | — | **+13.4%p** |
| **Total Trades** | 34 | 7 | — | +27 |
| **Win Rate** | 50.0% (17W/17L) | 57.1% (4W/3L) | — | -7.1%p |
| **Profit Factor** | 1.92 | 3.55 | — | -1.63 |
| **Alpha vs B&H** | -729.94% | -804.26% | — | +74.3%p |
| **Beta** | 0.45 | 0.66 | — | -0.21 |

#### Key Changes from v1.0

| 항목 | 개선 | 악화 |
|------|------|------|
| **MDD** | -55.8% → -42.4% (**13.4%p 개선**) | — |
| **Return** | +324% → +398% | — |
| **Sharpe** | 0.74 → 0.86 | — |
| **Trades** | — | 7 → 34 (ADX 플리커링) |
| **Win Rate** | — | 57.1% → 50.0% |
| **Profit Factor** | — | 3.55 → 1.92 |

#### ADX Flickering 문제

ADX가 25 부근에서 빈번하게 진동 → 골든크로스 유지 중에도 잦은 진입/청산 발생.

대표 사례:

```text
2020-05-22 | ENTER | ADX=28.7  ← 골든크로스 + ADX 충족
2020-05-26 | EXIT  | ADX=22.8  ← 4일 만에 ADX 25 하회 → 청산
2020-06-07 | ENTER | ADX=25.2  ← 12일 후 ADX 회복 → 재진입
2020-06-12 | EXIT  | ADX=22.1  ← 5일 만에 다시 하회 → 청산

2024-07-30 | ENTER | ADX=24.7
2024-07-31 | EXIT  | ADX=24.0  ← 1일 만에 청산 (최단)
```

34건 중 데드크로스에 의한 **진짜 추세 전환 청산은 5건**뿐.
나머지 29건은 ADX 플리커링에 의한 불필요한 거래.

#### Analysis

- **MDD 대폭 개선**: 코로나 구간에서 ADX 하락 시 조기 청산 → -55.8% → -42.4%
- **Return/Sharpe 소폭 개선**: 하락 구간 회피 효과가 거래 비용보다 큼
- **Sortino 악화**: 잦은 거래로 소규모 손실 빈도 증가 → 하방 변동성 증가
- **Profit Factor 악화**: 대형 수익 거래가 여러 작은 거래로 분할됨
- **Beta 하락 (0.66→0.45)**: 시장 노출 감소 (ADX < 25 구간에서 현금 보유)

#### Known Issues

1. **ADX 플리커링** — threshold 25에서 ADX 진동 → 과도한 거래 (34건/6년)
   - 해결안: 히스테리시스 (진입 ADX≥25, 청산 ADX<20) 또는 ADX 평활화
1. **SL/TS 부재** — MDD -42%로 개선되었으나 여전히 높음
1. **단순 강도** — strength=1.0 고정. 변동성 기반 포지션 사이징 없음
1. **Long-only 한계** — 하락장 수익 기회 없음

---

### v1.0 — Baseline (2026-03-05)

초기 구현. SMA 50/200 골든/데드 크로스, Long-only, SL/TS 없음.

#### Performance Metrics

| Metric | MA Cross | Buy & Hold |
|--------|----------|------------|
| **Total Return** | +324.00% | +1,128.26% |
| **CAGR** | +27.19% | — |
| **Sharpe Ratio** | 0.74 | — |
| **Sortino Ratio** | 0.77 | — |
| **Max Drawdown** | -55.81% | — |
| **Total Trades** | 7 | — |
| **Win Rate** | 57.1% (4W / 3L) | — |
| **Profit Factor** | 3.55 | — |
| **Alpha vs B&H** | -804.26% | — |
| **Beta** | 0.66 | — |

#### Trade Log

| # | Entry Date | Entry Price | Exit Date | Exit Price | Days | Return | Note |
|---|------------|-------------|-----------|------------|------|--------|------|
| 1 | 2020-02-19 | $9,594 | 2020-03-26 | $6,737 | 36 | **-30%** | 코로나 급락 직격 |
| 2 | 2020-05-22 | $9,170 | 2021-06-20 | $35,600 | 394 | **+288%** | 2020-2021 불장 포착 |
| 3 | 2021-09-16 | $47,738 | 2022-01-15 | $43,084 | 121 | **-10%** | 2021 말 하락 전환기 |
| 4 | 2023-02-08 | $22,963 | 2023-09-13 | $26,222 | 217 | **+14%** | 2023 초 회복 구간 |
| 5 | 2023-10-31 | $34,640 | 2024-08-11 | $58,713 | 285 | **+70%** | ETF 승인 상승파 |
| 6 | 2024-10-29 | $72,736 | 2025-04-08 | $76,322 | 161 | **+5%** | 상승 모멘텀 약화 |
| 7 | 2025-05-23 | $107,318 | 2025-11-17 | $92,215 | 178 | **-14%** | 2025 하반기 조정 |

#### Analysis

- **수익 집중**: Trade #2 (+288%)가 전체 수익의 대부분. 나머지 6건은 미미하거나 손실.
- **느린 시그널**: SMA 200은 매우 후행. 연 1~2회 크로스 → 추세 전환 시 늦은 진입/청산.
- **MDD -55.81%**: Trade #1에서 코로나 급락을 정면으로 맞음 (골든크로스 직후 36일 만에 -30%).
- **B&H 대비 열위**: BTC가 6년간 11배 상승한 강한 상승장에서 Long-only 전략임에도 B&H의 1/3 수준.
- **SL/TS 미적용**: 손절매/트레일링스톱 없어 하락 시 방어 부재.

---

## Warmup Handling

### 문제

SMA(200)은 최소 200 bars의 선행 데이터가 필요. 기존에는 backtest start 시점부터
데이터를 로드해 처음 200일은 SMA=NaN → 강제 현금 보유 → 왜곡된 결과.

### 해결 (v1.0에 적용)

```text
전략 config.warmup_periods() = 210 bars
  → CLI: data_start = start_date - (210 x timeframe_seconds)
  → Engine: strategy.run(전체 데이터) → 앞 210 bars 트리밍
  → 결과: start_date부터 유효한 시그널로 백테스트
```

**변경 파일**:

- `src/backtest/request.py` — `warmup_bars: int = 0` 필드 추가
- `src/backtest/engine.py` — `_execute()`에서 warmup 트리밍 + benchmark 정렬
- `src/cli/backtest.py` — 전략의 `warmup_periods()`로 데이터 로드 시작일 자동 확장

### Warmup 유무 비교

| Metric | Warmup 없음 | Warmup 적용 |
|--------|------------|------------|
| Total Return | +502.35% | +324.00% |
| Trades | 6 | 7 |
| Alpha vs B&H | -625.91% | -804.26% |

Warmup 없이는 처음 200일 강제 현금 보유 → 2020 상반기 하락을 우연히 회피 → 과대 수익.
Warmup 적용 후 정직한 결과 (Trade #1 코로나 손실 포함).

---

## Related

- [SuperTrend Backtest Report](supertrend-backtest.md) --- SuperTrend 전략 백테스트 결과 (별도 문서)
- [Market Regime Classification Analysis](regime-classification-analysis.md) --- 레짐 분류 정확도 검증 (별도 문서)

---

## Improvement Roadmap

개선 시 아래 표에 행을 추가하고, 해당 버전 섹션을 위에 기록.

| Version | Date | Change | Sharpe | Return | MDD | Trades | Alpha |
|---------|------|--------|--------|--------|-----|--------|-------|
| **MA v1.0** | 2026-03-05 | Baseline (SMA 50/200, no SL/TS) | 0.74 | +324% | -55.8% | 7 | -804% |
| **MA v1.1** | 2026-03-05 | ADX(14) >= 25 필터 추가 | 0.86 | +398% | -42.4% | 34 | -730% |
| v2.0 | -- | (예정) Vol-target sizing | -- | -- | -- | -- | -- |
| v2.1 | -- | (예정) 숏 모드 추가 | -- | -- | -- | -- | -- |

---

## References

- MA Cross code: `src/strategy/ma_cross/`
- MA Cross config: `config/ma_cross_btc.yaml`
- QuantStats report: `reports/` (실행 시 자동 생성)
- Warmup implementation: `src/backtest/engine.py` `_execute()`, `src/backtest/request.py`
