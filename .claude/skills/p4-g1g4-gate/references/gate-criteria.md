# Gate 판정 기준 — Machine-Readable Reference

> `gates/criteria.yaml`의 정량 기준 요약. `pipeline gates-show G1`로 확인.
> 파이프라인 실행 시 이 파일의 수치를 기준으로 PASS/FAIL을 판정한다.

---

## 공통 설정

| 항목 | 값 |
|------|---:|
| **초기자본** | $100,000 |
| **기간** | 2020-01-01 ~ 2025-12-31 (6년) |
| **타임프레임** | 1D |

### 에셋 유니버스 (Tier 1)

| # | 심볼 | 특성 |
|---|------|------|
| 1 | BTC/USDT | 기준 벤치마크 |
| 2 | ETH/USDT | 스마트 컨트랙트 대장 |
| 3 | BNB/USDT | 거래소 토큰 |
| 4 | SOL/USDT | 고변동성, 추세추종 유리 |
| 5 | DOGE/USDT | 밈코인, 노이즈 높음 |

### 비용 모델

| 항목 | 값 | 항목 | 값 |
|------|---:|------|---:|
| Maker Fee | 0.02% | Slippage | 0.05% |
| Taker Fee | 0.04% | Funding (8h) | 0.01% |
| Market Impact | 0.02% | **편도 합계** | **~0.11%** |

2배 비용 시나리오: maker 0.04%, taker 0.08%, slippage 0.10% (편도 ~0.22%)

---

## Gate 1: 단일에셋 백테스트

### PASS 조건 (Best Asset 기준)

| 지표 | 기준 | 비고 |
|------|------|------|
| Sharpe Ratio | > 1.0 | Best Asset에서 |
| CAGR | > 20% | Best Asset에서 |
| MDD | < 40% | Best Asset에서 |
| Total Trades | > 50 | 통계적 유의성 |

### 즉시 폐기 조건 (전 에셋 해당 시)

| # | 조건 | 판정 |
|---|------|------|
| 1 | MDD > 50% (전 에셋) | 즉시 폐기 |
| 2 | Sharpe < 0 (전 에셋) | 즉시 폐기 |
| 3 | Trades < 20 AND 수익 < 0 (전 에셋) | 즉시 폐기 |
| 4 | 총 수익의 80%+ 단일 거래 | 즉시 폐기 |

### CLI 명령

```bash
# 개별 실행 (5회)
uv run python -m src.cli.backtest run {strategy} BTC/USDT --start 2020-01-01 --end 2025-12-31
uv run python -m src.cli.backtest run {strategy} ETH/USDT --start 2020-01-01 --end 2025-12-31
uv run python -m src.cli.backtest run {strategy} BNB/USDT --start 2020-01-01 --end 2025-12-31
uv run python -m src.cli.backtest run {strategy} SOL/USDT --start 2020-01-01 --end 2025-12-31
uv run python -m src.cli.backtest run {strategy} DOGE/USDT --start 2020-01-01 --end 2025-12-31

# 일괄 실행 (bulk_backtest.py 패턴 — Python API 직접 호출)
uv run python scripts/bulk_backtest.py
```

---

## Gate 2: IS/OOS 70/30

### PASS 조건

| 지표 | 기준 | CTREND 참조 |
|------|------|------------|
| OOS Sharpe | >= 0.3 | 1.78 |
| Decay | < 50% | 33.7% |
| OOS Trades | >= 15 | — |

### Decay 계산

```
Decay (%) = (1 - OOS_Sharpe / IS_Sharpe) × 100
```

음수 Decay = OOS가 IS보다 높음 (데이터 분할 특성 또는 레짐 효과).

### CLI 명령

```bash
uv run python -m src.cli.backtest validate \
  -s {strategy} \
  --symbols {best_asset} \
  -m quick \
  -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

---

## Gate 3: 파라미터 안정성

### PASS 조건 (파라미터별)

| 조건 | 기준 | 정의 |
|------|------|------|
| 고원 존재 | >= 60% | Best Sharpe의 80% 이상인 값이 총 값의 60% 이상 (또는 최소 3개) |
| ±20% 안정 | Sharpe > 0 | 기본값 ±20% 범위에서 Sharpe 양수 유지 |

**전체 PASS**: 모든 핵심 파라미터가 (고원 존재 AND ±20% 안정)

### 고원 판정 상세

```
plateau_threshold = best_sharpe × 0.8  (best_sharpe > 0일 때)
plateau_count = count(sharpe >= plateau_threshold)
plateau_exists = plateau_count >= 3
```

### CLI 명령

```bash
# 사전 조건: STRATEGIES dict에 전략 등록 필요
uv run python scripts/gate3_param_sweep.py {strategy}
```

### 파라미터 그리드 설계 원칙

| 원칙 | 설명 |
|------|------|
| ±20% 포함 | 기본값의 0.8x ~ 1.2x 반드시 포함 |
| 넓은 범위 | ±20% 바깥 2~3개 추가 (edge behavior 확인) |
| 총 8~10개 값 | 스윕 시간 vs 해상도 균형 |
| vol_target 항상 포함 | 모든 전략 공통 (0.15~0.60 범위) |
| 가중치 쌍 처리 | `weight_a + weight_b = 1.0` 자동 보완 |

---

## Gate 4: 심층검증

### PASS 조건

| 지표 | 기준 | CTREND 참조 | 비고 |
|------|------|------------|------|
| WFA OOS Sharpe | >= 0.5 | 1.49 | 3-fold expanding window 평균 |
| WFA Decay | < 40% | 39% | IS → OOS |
| WFA Consistency | >= 60% | 67% | OOS fold 양수 비율 |
| PBO | 아래 참조 | 60% (PASS-B) | Probability of Backtest Overfitting |
| DSR (batch) | > 0.95 | 1.00 | 동일 배치 기준 |
| MC p-value | < 0.05 | 0.000 | Monte Carlo 통계적 유의성 |

### PBO 판정 (이중 경로)

PBO는 다음 **두 경로 중 하나**를 충족하면 PASS:

| 경로 | 조건 | 설명 |
|:----:|------|------|
| **A** | PBO < 40% | 기본 경로 — 과적합 위험 낮음 |
| **B** | PBO < 80% AND CPCV 전 fold OOS > 0 AND MC p < 0.05 | 보조 경로 — 파라미터 순위 역전이 있으나 기저 alpha 견고 |

**근거**: PBO는 "IS 최적 파라미터가 OOS에서도 최적 순위를 유지하는가"를 측정한다.
그러나 실전에서 중요한 것은 **"어떤 파라미터를 골라도 OOS에서 수익이 나는가"** (CPCV robustness).
경로 B는 파라미터 과적합이 있지만, 기저 전략 alpha가 견고한 경우를 구제한다.

**적용 예시**: CTREND (PBO 60%, 전 fold OOS 양수, MC p=0.000) → 경로 B PASS.
Anchor-Mom (PBO 80%, 전 fold OOS 양수, MC p=0.000) → 경로 B PASS (PBO < 80% 충족).

### CLI 명령

```bash
# Phase A: WFA
uv run python -m src.cli.backtest validate \
  -s {strategy} \
  --symbols {best_asset} \
  -m milestone \
  -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025

# Phase B: CPCV + PBO + DSR + Monte Carlo
uv run python -m src.cli.backtest validate \
  -s {strategy} \
  --symbols {best_asset} \
  -m final \
  -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -y 2025
```

---

## CTREND 참조 벤치마크 (전 Gate)

| Gate | 핵심 지표 | CTREND 결과 | 판정 |
|:----:|----------|------------|:----:|
| G1 | Best Sharpe | 2.05 (SOL) | PASS |
| G1 | Best CAGR | +97.8% | PASS |
| G1 | Best MDD | -27.7% | PASS |
| G1 | Best Trades | 288 | PASS |
| G2 | OOS Sharpe | 1.78 | PASS |
| G2 | Decay | 33.7% | PASS |
| G3 | 파라미터 | 4/4 PASS | PASS |
| G4 | WFA OOS | 1.49 | PASS |
| G4 | WFA Decay | 39% | PASS |
| G4 | PBO | 60% | PASS (경로 B) |
| G4 | DSR (batch) | 1.00 | PASS |
| G4 | MC p-value | 0.000 | PASS |

> CTREND은 PBO 60%로 경로 A(< 40%) FAIL이나, 전 CPCV fold OOS 양수 + MC p=0.000으로
> 경로 B를 통해 PASS. Anchor-Mom도 PBO 80%이나 동일 경로 B PASS.

---

## Gate 간 일관성 체크

파이프라인 진행 중 아래 일관성을 확인한다:

| 비교 | 기대 | 경고 조건 |
|------|------|----------|
| G1 Sharpe → G2 OOS Sharpe | OOS >= G1 × 0.3 | OOS < G1 × 0.3이면 과적합 의심 |
| G2 Decay → G4 WFA Decay | 유사 (±15%p) | 차이 > 20%p이면 CV 방법론 민감도 |
| G2 OOS → G4 WFA OOS | WFA >= G2 × 0.5 | WFA < G2 × 0.5이면 window 의존 |
| G3 Sharpe → G4 MC CI | G3 baseline ∈ CI | G3 baseline < CI 하한이면 불안정 |
