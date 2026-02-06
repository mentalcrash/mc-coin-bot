# Risk Management Parameter Optimization

8-asset EW TSMOM 포트폴리오의 리스크 관리 파라미터 최적화 결과입니다.
system_stop_loss, trailing_stop (ATR), rebalance_threshold 3개 축을 순차 스윕했습니다.

## 고정 파라미터

```python
TSMOMConfig(
    lookback=30, vol_window=30, vol_target=0.35,
    short_mode=ShortMode.HEDGE_ONLY,
    hedge_threshold=-0.07, hedge_strength_ratio=0.3,
)
PortfolioManagerConfig(max_leverage_cap=2.0)
```

## 테스트 범위

| Phase | 파라미터 | 테스트 범위 | 백테스트 수 |
|-------|---------|------------|-----------|
| A | system_stop_loss | None, 5%, 7%, 10%, 15%, 20%, 30% | 56 |
| B | trailing_stop (ATR mult) | 1.5x, 2.0x, 2.5x, 3.0x, 4.0x, 5.0x | 48 |
| C | stop_loss x trailing_stop | 7 x 5 = 28 조합 | 224 |
| D | rebalance_threshold | 2%, 3%, 5%, 7%, 10% | 40 |
| **총계** | | | **368** |

기간: 2020-01-01 ~ 2025-12-31 (6년)

---

## Phase A: System Stop-Loss

> 고정: trailing_stop=OFF, rebalance_threshold=5%

| stop_loss | Sharpe | CAGR | MDD | AnnVol | Calmar | Sortino |
|:---------:|:------:|:----:|:---:|:------:|:------:|:-------:|
| **7%** | **2.33** | **+49.1%** | **-20.3%** | 21.1% | **2.42** | 3.21 |
| 10% | 2.33 | +49.1% | -20.7% | 21.1% | 2.37 | 3.21 |
| 5% | 2.32 | +48.6% | -20.4% | 21.0% | 2.38 | 3.20 |
| None | 2.28 | +48.2% | -21.9% | 21.1% | 2.20 | 3.09 |
| 20% | 2.28 | +48.4% | -22.2% | 21.2% | 2.18 | 3.10 |
| 30% | 2.27 | +47.9% | -21.8% | 21.1% | 2.20 | 3.08 |
| 15% | 2.23 | +47.4% | -23.1% | 21.2% | 2.05 | 3.04 |

### 분석

- **7% 손절이 최적** (Sharpe 2.33, Calmar 2.42)
- 5~10% 범위에서 성과 차이 미미 (Sharpe 2.32~2.33) — **둔감 영역**
- 15% 이상은 오히려 성과 하락 (MDD 방어 실패)
- None vs 7%: Sharpe +0.05, MDD 1.6pp 개선 — **손절 효과 확인**

---

## Phase B: Trailing Stop (ATR Multiplier)

> 고정: system_stop_loss=None, rebalance_threshold=5%

| ATR Mult | Sharpe | CAGR | MDD | AnnVol | Calmar | Sortino |
|:--------:|:------:|:----:|:---:|:------:|:------:|:-------:|
| **2.5x** | **2.34** | +48.1% | **-17.4%** | 20.5% | **2.76** | **3.27** |
| 3.0x | 2.33 | +47.9% | -17.7% | 20.6% | 2.70 | 3.22 |
| 4.0x | 2.26 | +47.4% | -24.5% | 21.0% | 1.94 | 3.06 |
| 5.0x | 2.25 | +47.5% | -24.0% | 21.1% | 1.98 | 3.05 |
| 2.0x | 2.20 | +43.3% | -17.0% | 19.7% | 2.55 | 3.05 |
| 1.5x | 1.99 | +37.7% | -18.1% | 19.0% | 2.09 | 2.77 |

### 분석

- **2.5x~3.0x ATR이 최적 영역** (Sharpe 2.33~2.34)
- MDD 획기적 개선: -21.9% → **-17.4%** (4.5pp, 2.5x ATR)
- 너무 타이트 (1.5x~2.0x): 조기 청산으로 CAGR 급감 (-10pp)
- 너무 루즈 (4.0x+): MDD 방어 실패, Stop-Loss 없는 것과 비슷
- **Trailing Stop만으로 Calmar 2.76 달성** (baseline 2.37 대비 +16%)

---

## Phase C: Stop-Loss x Trailing Stop 결합

> 고정: rebalance_threshold=5%

### Top 10 by Sharpe

| Stop-Loss | Trailing | Sharpe | CAGR | MDD | AnnVol | Calmar | Sortino |
|:---------:|:--------:|:------:|:----:|:---:|:------:|:------:|:-------:|
| 10% | 3.0x ATR | **2.36** | +48.8% | -18.2% | 20.7% | 2.68 | **3.28** |
| 20% | 3.0x ATR | 2.35 | +48.7% | -17.5% | 20.7% | **2.78** | 3.26 |
| 7% | 5.0x ATR | 2.34 | **+49.3%** | -20.6% | 21.1% | 2.39 | 3.22 |
| 7% | 3.0x ATR | 2.33 | +48.3% | -18.6% | 20.7% | 2.60 | 3.23 |
| None | 3.0x ATR | 2.33 | +47.9% | -17.7% | 20.6% | 2.70 | 3.22 |
| 30% | 3.0x ATR | 2.33 | +47.9% | -17.7% | 20.6% | 2.70 | 3.22 |
| 5% | 3.0x ATR | 2.32 | +47.8% | -18.8% | 20.6% | 2.55 | 3.21 |
| 5% | 5.0x ATR | 2.32 | +48.8% | -20.7% | 21.0% | 2.35 | 3.21 |
| 10% | 5.0x ATR | 2.31 | +48.9% | -22.1% | 21.1% | 2.21 | 3.20 |
| 7% | 4.0x ATR | 2.30 | +48.4% | -20.9% | 21.0% | 2.32 | 3.17 |

### 분석

- **SL=10% + TS=3.0x ATR이 Sharpe 최적** (2.36)
- **SL=20% + TS=3.0x ATR이 Calmar 최적** (2.78, MDD=-17.5%)
- 3.0x ATR trailing stop이 모든 상위 조합에 공통 — **핵심 파라미터**
- SL 5~20% 범위에서 둔감: trailing stop이 주 방어선 역할
- SL이 타이트할수록 (5~7%) trailing stop과의 중복 작동으로 소폭 CAGR 손실

---

## Phase D: Rebalance Threshold

> 고정: system_stop_loss=10%, trailing_stop=3.0x ATR (Phase C 최적)

| Threshold | Sharpe | CAGR | MDD | AnnVol | Calmar | Sortino |
|:---------:|:------:|:----:|:---:|:------:|:------:|:-------:|
| **10%** | **2.41** | **+52.1%** | **-17.5%** | 21.7% | **2.98** | **3.33** |
| 5% | 2.36 | +48.8% | -18.2% | 20.7% | 2.68 | 3.28 |
| 7% | 2.32 | +48.5% | -18.7% | 20.9% | 2.60 | 3.20 |
| 2% | 2.28 | +45.5% | -17.7% | 20.0% | 2.57 | 3.16 |
| 3% | 2.28 | +45.7% | -18.0% | 20.1% | 2.54 | 3.16 |

### 분석

- **10% 임계값이 모든 지표에서 최적** (Sharpe 2.41, Calmar 2.98)
- 높은 임계값 = 거래 빈도 감소 = 거래 비용 절감 → CAGR +3.3pp
- 2~3% 임계값: 과도한 거래로 비용 증가, CAGR -6pp
- TSMOM의 일봉 특성상 신호 변화가 점진적 → 높은 임계값이 유리

---

## 최종 결과 비교

| 설정 | Sharpe | CAGR | MDD | Calmar | Sortino | 총수익 |
|------|:------:|:----:|:---:|:------:|:-------:|:------:|
| **최적 (신규)** | **2.41** | **+52.1%** | **-17.5%** | **2.98** | **3.33** | **+1140%** |
| 이전 확정 | 2.33 | +49.1% | -20.7% | 2.37 | 3.21 | +997% |
| No Risk Mgmt | 2.28 | +48.2% | -21.9% | 2.20 | 3.09 | +959% |

### 개선 효과 (이전 확정 대비)

| 지표 | 이전 | 신규 | 변화 |
|------|:----:|:----:|:----:|
| Sharpe | 2.33 | **2.41** | **+3.4%** |
| CAGR | +49.1% | **+52.1%** | **+3.0pp** |
| MDD | -20.7% | **-17.5%** | **+3.2pp** |
| Calmar | 2.37 | **2.98** | **+25.7%** |
| Sortino | 3.21 | **3.33** | **+3.7%** |
| 총수익 | +997% | **+1140%** | **+143pp** |

---

## 권장 설정

```python
PortfolioManagerConfig(
    max_leverage_cap=2.0,
    system_stop_loss=0.10,              # 10% — 최후 방어선 (둔감 영역)
    use_trailing_stop=True,             # Trailing Stop 활성화
    trailing_stop_atr_multiplier=3.0,   # 3x ATR — MDD 핵심 방어
    rebalance_threshold=0.10,           # 10% — 거래비용 최적화
)
```

### 파라미터별 역할

| 파라미터 | 값 | 주 역할 | 영향도 |
|---------|:---:|--------|:------:|
| trailing_stop (3.0x ATR) | 핵심 | MDD 방어 (-21.9% → -17.5%) | **높음** |
| rebalance_threshold (10%) | 보조 | 거래비용 절감 (CAGR +3pp) | **높음** |
| system_stop_loss (10%) | 안전망 | 극단 상황 방어 (둔감) | 낮음 |

### 핵심 인사이트

1. **Trailing Stop이 가장 중요한 리스크 파라미터**: MDD를 4.4pp 개선하면서 Sharpe 유지
2. **Rebalance Threshold를 10%로 올리면 거래비용 절감**: 일봉 TSMOM 특성상 신호가 점진적이므로 높은 임계값 적합
3. **Stop-Loss는 5~10% 범위에서 둔감**: 정확한 값보다 존재 자체가 중요 (안전망)
4. **3중 방어 체계**: Trailing Stop (주 방어) → Stop-Loss (안전망) → Rebalance (비용 최적화)

---

## 검증 방법

```bash
uv run python scripts/multi_asset_risk_sweep.py
```

- 기간: 2020-2025 (6년)
- 에셋: BTC, ETH, BNB, SOL, DOGE, LINK, ADA, AVAX (8개)
- 총 368 백테스트 (Phase A~D)
- 결과 CSV: `data/multi_asset_risk_*.csv`
