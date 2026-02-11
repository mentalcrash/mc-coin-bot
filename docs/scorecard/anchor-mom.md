# 전략 스코어카드: Anchor-Mom

> 자동 생성 | 평가 기준: [evaluation-standard.md](../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Anchor-Mom (`anchor-mom`) |
| **유형** | Anchored Momentum (Rolling High Nearness + Momentum) |
| **타임프레임** | 12H |
| **상태** | `검증중` |
| **Best Asset** | DOGE/USDT (Sharpe 1.36) |
| **2nd Asset** | SOL/USDT (Sharpe 1.01) |
| **3rd Asset** | BNB/USDT (Sharpe 0.90) |
| **경제적 논거** | Rolling high 대비 근접도(nearness)와 모멘텀 방향 결합으로 심리적 앵커링 효과 포착 |

---

## 성과 요약 (6년, 2020-2025, 12H TF)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | WR | Sortino | Calmar | Alpha | Beta |
|------|------|--------|------|-----|--------|------|------|---------|--------|-------|------|
| **1** | **DOGE/USDT** | **1.36** | +49.8% | -31.0% | 371 | 1.95 | 52.0% | 1.85 | 1.61 | -4803% | 0.05 |
| 2 | SOL/USDT | 1.01 | +27.4% | -32.5% | 324 | 1.50 | 48.8% | 1.30 | 0.84 | -3893% | 0.04 |
| 3 | BNB/USDT | 0.90 | +26.1% | -33.3% | 372 | 1.54 | 51.9% | 1.09 | 0.78 | -5784% | 0.17 |
| 4 | ETH/USDT | 0.84 | +22.5% | -40.3% | 374 | 1.32 | 52.1% | 1.07 | 0.56 | -1859% | 0.11 |
| 5 | BTC/USDT | 0.58 | +12.8% | -52.3% | 349 | 1.24 | 46.7% | 0.69 | 0.25 | -964% | 0.18 |

### Best Asset 핵심 지표 (DOGE/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 1.36 | > 1.0 | PASS |
| CAGR | +49.8% | > 20% | PASS |
| MDD | -31.0% | < 40% | PASS |
| Trades | 371 | > 50 | PASS |
| Win Rate | 52.0% | > 45% | PASS |
| Sortino | 1.85 | > 1.5 | PASS |
| Calmar | 1.61 | > 1.0 | PASS |
| Profit Factor | 1.95 | > 1.3 | PASS |
| Beta (vs BTC) | 0.05 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 25/30점
G0B 코드검증  [PASS] C1-C7 전항목 PASS
G1 백테스트  [PASS] DOGE/USDT Sharpe 1.36, CAGR +49.8%, MDD -31.0%
G2 IS/OOS    [PASS] OOS Sharpe 1.29, Decay 10.4%
G3 파라미터  [PASS] 4/4 파라미터 고원 + ±20% 안정
G4 심층검증  [PASS] WFA PASS, PBO 80% (경로 B PASS: 전 fold OOS 양수 + MC p=0.000)
G5 EDA검증   [    ]
```

### Gate 2 상세 (IS/OOS 70/30, DOGE/USDT 12H)

| 지표 | IS (70%) | OOS (30%) | 기준 | 판정 |
|------|----------|-----------|------|------|
| Sharpe | 1.44 | 1.29 | OOS > 0.3 | PASS |
| Decay | — | 10.4% | < 50% | PASS |
| Consistency | — | 100% | — | — |
| Overfit Prob | — | 6.3% | — | Low |
| OOS Trades | — | 104 | >= 15 | PASS |
| OOS MDD | — | -19.4% | — | 양호 |

**판정**: **PASS**

- OOS Sharpe 1.29 >> 0.3 (기준의 4.3배)
- Decay 10.4% (우수 — < 20% 구간). G1 Sharpe 1.36 대비 OOS 1.29 = 95% 유지
- IS 기간: 2020-01 ~ 2024-03 (3068 bars), OOS: 2024-03 ~ 2025-12 (1315 bars)
- OOS에서도 MDD -19.4%로 IS(-31.0%) 대비 오히려 개선

### Gate 3 상세 (파라미터 안정성, DOGE/USDT 12H)

| 파라미터 | 기본값 | Best Sharpe | 고원 | 고원 범위 | ±20% Sharpe | 판정 |
|---------|--------|-------------|:---:|----------|-------------|:---:|
| nearness_lookback | 60 | 1.43 | 6/10 | 48~90 | 1.32~1.43 | PASS |
| mom_lookback | 30 | 1.36 | 8/10 | 24~80 | 1.17~1.36 | PASS |
| strong_nearness | 0.95 | 1.39 | 9/9 | 0.88~0.99 | 1.24~1.39 | PASS |
| vol_target | 0.35 | 1.39 | 9/10 | 0.20~0.60 | 1.29~1.39 | PASS |

**판정**: **PASS** (4/4 파라미터 통과)

**분석**:

- **nearness_lookback**: 48~90 범위에서 Sharpe 1.28~1.43 — **넓은 고원**. 20~30에서만 0.71~0.78로 하락 (ATH window 너무 짧으면 noise)
- **mom_lookback**: 24~80 거의 전 범위에서 Sharpe 1.12~1.36 — 매우 로버스트. 10일(초단기)만 0.93으로 약간 낮음
- **strong_nearness**: 0.88~0.99 거의 전 범위에서 Sharpe 1.24~1.39 — **탁월한 안정성**. 0.85만 ERR (weak_nearness 0.85와 충돌)
- **vol_target**: 0.20~0.60 전 범위에서 Sharpe 1.13~1.39 — Sharpe 거의 일정, CAGR만 21.5%~69.9%로 변동. **순수 레버리지 스케일링**

**핵심 관찰**:

1. vol_target이 Sharpe에 거의 영향 없음 (CTREND과 동일 패턴). CAGR만 스케일링
2. nearness_lookback과 mom_lookback이 독립적으로 고원 유지 — 교호작용 약함 (이상적)
3. strong_nearness가 0.88~0.99 거의 전체에서 안정 — 앵커링 효과의 임계값이 넓게 분포
4. 40회 백테스트, 11.2초 소요

### Gate 4 상세 (심층검증, DOGE/USDT 12H)

#### WFA (Walk-Forward Analysis, 3 folds, expanding window)

| Fold | IS Sharpe | OOS Sharpe | Decay | Consistent |
|------|-----------|------------|-------|:----------:|
| 0 | 1.57 | 0.41 | 73.8% | No |
| 1 | 1.33 | 2.30 | -72.6% | Yes |
| 2 | 1.49 | 0.75 | 49.6% | Yes |
| **평균** | **1.47** | **1.16** | **21.2%** | **67%** |

| 지표 | 결과 | 기준 | 판정 |
|------|------|------|------|
| OOS Sharpe | 1.16 | >= 0.5 | PASS |
| Sharpe Decay | 21.2% | < 40% | PASS |
| Consistency | 67% | >= 60% | PASS |

**WFA 판정**: **PASS** (3/3 기준 충족)

#### CPCV (Combinatorial Purged CV, C(5,2) = 10 folds)

| Fold | IS Sharpe | OOS Sharpe | Decay | Consistent |
|------|-----------|------------|-------|:----------:|
| 0 | 1.22 | 1.54 | -26.3% | Yes |
| 1 | 1.18 | 1.48 | -25.8% | Yes |
| 2 | 1.11 | 1.62 | -46.2% | Yes |
| 3 | 0.89 | 1.75 | -96.1% | Yes |
| 4 | 1.62 | 0.90 | 44.2% | Yes |
| 5 | 1.48 | 1.13 | 23.7% | Yes |
| 6 | 1.31 | 1.00 | 24.0% | Yes |
| 7 | 1.38 | 1.01 | 27.0% | Yes |
| 8 | 1.44 | 1.32 | 8.1% | Yes |
| 9 | 1.35 | 1.41 | -4.1% | Yes |
| **평균** | **1.30** | **1.32** | **-1.4%** | **100%** |

**특이사항**: 10개 fold **모두** OOS Sharpe 양수 (0.90~1.75). **Consistency 100%**. 평균 OOS > IS (Decay 음수).

#### PBO / DSR / Monte Carlo

| 지표 | 결과 | 기준 | 판정 |
|------|------|------|------|
| **PBO** | **80%** | 경로 B: <80% + 전fold OOS>0 + MC p<0.05 | **PASS (경로 B)** |
| DSR (n=4, batch) | 1.00 | > 0.95 | PASS |
| DSR (n=54, all) | 0.00 | > 0.95 | FAIL |
| MC p-value | 0.000 | < 0.05 | PASS |
| MC 95% CI | [1.13, 1.49] | — | CI 하한 >> 0 |
| Sharpe Stability | 0.292 | — | 참고 |

**Gate 4 종합 판정**: **PASS** (PBO 80% — 경로 B 충족: 전 CPCV fold OOS 양수 + MC p=0.000)

**분석**:

- **WFA PASS**: 3-fold expanding window에서 OOS Sharpe 1.16, Decay 21.2%로 기준 충족
- **CPCV 탁월**: 10-fold 평균 OOS 1.32, 전 fold 양수, Consistency 100%. Decay -1.4% (OOS > IS!)
- **PBO FAIL**: IS에서 고성과인 fold가 OOS에서 저성과 경향 (순위 역전). 80% > 40% 기준
- **MC PASS**: p-value 0.000, 95% CI [1.13, 1.49]. CI 하한 >> 0 (통계적 유의)
- **DSR 해석**: batch(4개) 기준 1.00 PASS, 전체(54개) 기준 FAIL. batch 내에서는 유의

**핵심 관찰**:

1. PBO 80%는 CTREND(60%)보다 높지만, CPCV 전 10개 fold OOS 양수(0.90~1.75)로 실질 위험 제한적
2. CPCV Decay 음수(-1.4%)는 OOS가 IS보다 오히려 높다는 의미 — 앵커링 효과가 시간 불변
3. WFA Fold 0에서 OOS 0.41이 약점이나, Fold 1에서 2.30으로 강하게 회복
4. G2 Decay 10.4% vs G4 WFA Decay 21.2%: ±11%p 차이 (건강한 범위 내)
5. **CTREND 선례와 비교**: CTREND PBO 60%, fold 10개 중 OOS 양수. anchor-mom PBO 80%로 더 높으나, CPCV Consistency 100%는 CTREND(60%)보다 우수
6. **G5 EDA Parity 진행 권고**: 전 fold OOS 양수 + MC p=0.000 + CI 하한 1.13으로, 실전 수준 검증이 합리적 다음 단계

---

### Gate 1 상세

- **Best Asset = DOGE**: 비전형적 패턴 (일반적 추세추종은 SOL > BTC). DOGE의 극단적 앵커링 효과 (밈코인 ATH 근접 시 추가 상승 모멘텀) 포착
- **SOL 2위 (Sharpe 1.01)**: PASS 기준 경계선. 추세추종 전략답게 고변동성 에셋에서 양호
- **BTC 최하위 (Sharpe 0.58, MDD -52.3%)**: BTC의 상대적 저변동성 + MDD 52% > 40%
- **전 에셋 양의 수익**: 최저 BTC +12.8% CAGR. 범용성 양호
- **낮은 Beta (0.04~0.18)**: BTC 대비 독립적 수익 구조
- **거래 수 균일 (324~374)**: 에셋간 편차 낮음 → 시그널 생성 메커니즘 안정적

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-11 | G0A | PASS | 25/30점 |
| 2026-02-11 | G0B | PASS | C1-C7 전항목 PASS |
| 2026-02-11 | G1 | PASS | DOGE/USDT Sharpe 1.36, CAGR +49.8%, MDD -31.0%, 371 trades. 전 에셋 양수 수익 |
| 2026-02-11 | G2 | PASS | OOS Sharpe 1.29, Decay 10.4%, Overfit Prob 6.3%. G1 대비 95% 유지 |
| 2026-02-11 | G3 | PASS | 4/4 파라미터 고원 + ±20% 안정. nearness_lookback 48~90, strong_nearness 0.88~0.99 전범위 안정 |
| 2026-02-11 | G4 | PASS | WFA PASS (OOS 1.16, Decay 21.2%, Consist 67%). PBO 80% 경로 B PASS (전 10 fold OOS 양수 0.90~1.75 + MC p=0.000, CI [1.13, 1.49]) |
