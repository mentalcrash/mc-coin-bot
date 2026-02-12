# 전략 스코어카드: Flow Imbalance

> 자동 생성 | 평가 기준: [dashboard.md](../../strategy/dashboard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Flow Imbalance (`flow-imbalance`) |
| **유형** | Microstructure / Order Flow Proxy |
| **타임프레임** | 1H |
| **상태** | `폐기 (Gate 1 FAIL — 1H 재검증 확정)` |
| **Best Asset** | N/A (전 에셋 Sharpe 음수) |
| **경제적 논거** | BVC(Bulk Volume Classification)로 매수/매도 볼륨을 추정하고, OFI(Order Flow Imbalance)와 VPIN proxy로 주문 흐름 불균형을 감지하여 방향 예측 |

---

## 성과 요약 (6년, 2020-2025, 1H)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate | Alpha | Beta |
|------|------|--------|------|-----|--------|------|----------|-------|------|
| 1 | DOGE/USDT | -0.12 | -2.0% | -29.5% | 324 | 0.91 | 37.3% | -6045% | 0.03 |
| 2 | ETH/USDT | -0.47 | -3.3% | -27.0% | 290 | 0.82 | 41.0% | -2223% | 0.00 |
| 3 | SOL/USDT | -0.51 | -3.9% | -27.7% | 338 | 0.83 | 45.0% | -4299% | 0.00 |
| 4 | BTC/USDT | -0.63 | -4.5% | -37.4% | 317 | 0.78 | 36.6% | -1156% | 0.01 |
| 5 | BNB/USDT | -1.63 | -11.9% | -54.2% | 454 | 0.55 | 37.2% | -6248% | 0.01 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 23/30점
G0B 코드검증 [PASS] C1-C7 전항목 PASS (2026-02-12 재검증). Warning 2건 (W3 Regime, W4 Turnover)
G1 백테스트  [FAIL] 1H 재검증 확정. 전 에셋 Sharpe 음수 (-0.24 ~ -1.56)
G2 IS/OOS    [    ]
G3 파라미터  [    ]
G4 심층검증  [    ]
G5 EDA검증   [    ]
```

### Gate 1 상세

**판정**: **FAIL** (즉시 폐기 -- 전 에셋 Sharpe 음수)

**즉시 폐기 사유**:

- 즉시 폐기 조건 #2 해당: **전 5개 에셋에서 Sharpe 음수** (-0.12 ~ -1.63)
- BNB/USDT MDD -54.2% (조건 #1 해당: MDD > 50%)
- Profit Factor 전 에셋 < 1.0 (0.55 ~ 0.91)
- CAGR 전 에셋 음수 (-2.0% ~ -11.9%)

**근본 원인 분석**:

1. **OFI entry_threshold 0.6의 구조적 문제**: OFI = rolling_sum(buy_vol - sell_vol) / rolling_sum(volume)는 [-1, 1] 범위이지만, 6H rolling에서 0.6 초과는 매수 압력이 극도로 편향된 구간에서만 발생. 시그널이 극단적 편향 직후에 발생하여 **이미 움직임이 완료된 시점**에 진입하는 구조.

2. **BVC 근사의 정보 손실**: 1H bar에서 (close-low)/(high-low)로 매수/매도 볼륨을 추정하는 BVC 방식은, 1H 내 가격 경로를 단일 비율로 압축. 실제 order flow는 bar 내에서 반전될 수 있으나 이를 포착 불가. VPIN-Flow(1D)의 실패와 동일한 근본 원인 -- **해상도를 1H로 올려도 BVC 근사의 한계는 해결 불가**.

3. **낮은 Win Rate (36~45%)**: OFI 방향이 flow continuation을 포착하지 못함. 강한 매수 압력(OFI > 0.6) 후 mean reversion이 더 빈번하게 발생 -- 정보거래자가 flow 후 이익 실현하는 패턴.

4. **VPIN threshold의 무의미한 필터링**: `vpin_threshold=0.15` (buy_ratio의 rolling_std)가 대부분의 시간에서 초과되어 실질적으로 필터 역할을 하지 못함. 활동성 게이트가 없는 것과 동일.

5. **BNB 특이 현상**: BNB에서 Sharpe -1.63, 454 trades로 가장 나쁜 성과. BNB는 Binance exchange token으로 BTC/ETH 대비 독자적 flow 패턴 보유. OFI threshold가 BNB의 거래 구조에 부적합 -- 자전 거래(wash trading) 비율이 높아 BVC 근사가 더욱 부정확.

**CTREND 비교**:

| 지표 | CTREND Best (SOL) | Flow-Imbalance Best (DOGE) |
|------|-------------------|---------------------------|
| Sharpe | 2.05 | -0.12 |
| CAGR | +97.8% | -2.0% |
| MDD | -27.7% | -29.5% |
| Trades | 288 | 324 |

**VPIN-Flow(1D) 대비 개선점과 한계**:

- VPIN-Flow(1D)는 전 에셋 거래 0건 (threshold 도달 불가). Flow-Imbalance(1H)는 290~454건 거래 발생 -- 시그널 생성 문제는 해결.
- 그러나 시그널의 **방향성 예측력이 부재** -- 거래가 발생해도 랜덤 이하의 성과. BVC 근사 자체의 한계.

**교훈**: OHLCV 기반 BVC 근사는 timeframe을 1D→1H로 올려도 order flow 방향성 예측에 불충분. 실제 microstructure alpha를 포착하려면 L2 order book, tick data, 또는 real-time volume profile이 필요. OHLCV만으로 flow 방향을 추정하는 접근은 근본적으로 정보 손실이 크며, 이번 결과로 flow 기반 전략의 OHLCV 데이터 한계가 1D(VPIN-Flow)에 이어 1H에서도 재확인됨.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0 | PASS | 23/30점 |
| 2026-02-10 | G0B | PASS | shift(1) lookahead 방지, vectorized ops, Pydantic frozen config |
| 2026-02-10 | G1 | FAIL | 1D 왜곡: 전 에셋 Sharpe 음수 (-0.12~-1.63) |
| 2026-02-12 | G1 (1H 재검증) | **FAIL 확정** | 전 에셋 Sharpe 음수 (-0.24~-1.56). BNB MDD -51.1%. BVC 근사 한계 1H에서도 불변 확인 |
