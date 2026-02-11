# 전략 스코어카드: Perm-Entropy-Mom

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Perm-Entropy-Mom (`perm-entropy-mom`) |
| **유형** | Momentum (PE conviction scaling) |
| **타임프레임** | 4H (annualization_factor=2190, 1D 데이터 백테스트) |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | SOL/USDT (Sharpe 0.67) |
| **경제적 논거** | Permutation Entropy로 시장 질서도를 측정, 낮은 PE(질서적 패턴)에서 모멘텀 conviction 증폭 |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate |
|------|------|--------|------|-----|--------|------|---------|
| **1** | **SOL/USDT** | **0.67** | +2398.0% | -86.5% | 4 | 143.25 | 66.7% |
| 2 | BTC/USDT | 0.32 | +121.9% | -8.0% | 1 | — | — |
| 3 | ETH/USDT | 0.16 | +95.0% | -15.4% | 1 | — | — |
| 4 | DOGE/USDT | 0.04 | +0.0% | -786.7% | 1 | — | — |
| 5 | BNB/USDT | -0.43 | -276.9% | -17.5% | 6 | 0.03 | 40.0% |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.67 | > 1.0 | **FAIL** |
| CAGR | +2398.0% | > 20% | PASS |
| MDD | -86.5% | < 40% | **FAIL** |
| Trades | 4 | > 50 | **FAIL** |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G0B 코드검증 [PASS] C1-C7 전항목 PASS (2026-02-12 재검증). Warning 1건 (W3 Regime)
G1 백테스트  [재검증 대기] 이전 결과는 1D 데이터 왜곡. 4H TF로 재실행 필요
```

### Gate 1 상세

- **극소 거래 문제**: 5개 에셋 합산 거래 13건 (BTC 1, ETH 1, BNB 6, SOL 4, DOGE 1). 6년간 1~6건은 통계적 유의성 전무
- **annualization_factor 불일치**: 4H TF 전략(2190)이 1D 데이터에서 실행됨. 1D bar를 4H로 해석하여 vol scaling이 왜곡 (실제 vol 과소추정 → 과대 레버리지 → MDD 급등)
- **SOL CAGR +2398%**: 4건 거래로 달성한 비현실적 수치. 단일 거래 집중 리스크 (즉시 폐기 조건 #4 근접)
- **DOGE MDD -786.7%**: annualization_factor 불일치로 인한 비정상 MDD. 실제 파산 수준
- **noise_threshold 0.95**: 1D 데이터에서 PE가 거의 항상 0.95 미만이지만, 4H 기준 pe_short_window=30(5일)이 1D에서 30일이 되어 PE가 과도하게 평활화 → 거의 항상 noise gate 통과하지만 conviction이 극도로 낮아 거래 미발생

**근본 원인**: 4H TF 전략 파라미터가 1D 데이터와 구조적으로 불일치. PE 윈도우(30/60 bars)가 4H에서는 5/10일이나 1D에서는 30/60일로 해석되어 전략 의도와 완전히 다른 동작.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 |
| 2026-02-10 | G0B | PASS | Critical 7항목 결함 0개 |
| 2026-02-11 | G1 | **FAIL** | 전 에셋 Trades < 50 (최대 6건), Best Sharpe 0.67 < 1.0, MDD 86.5% > 40%. 4H→1D TF 불일치로 전략 무효 |
