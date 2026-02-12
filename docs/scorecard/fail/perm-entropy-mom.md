# 전략 스코어카드: Perm-Entropy-Mom

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Perm-Entropy-Mom (`perm-entropy-mom`) |
| **유형** | Momentum (PE conviction scaling) |
| **타임프레임** | 4H (annualization_factor=2190) |
| **상태** | `폐기 (Gate 1 FAIL — 4H 재검증 확정)` |
| **Best Asset** | BNB/USDT (Sharpe 0.52, 4H 재검증) |
| **경제적 논거** | Permutation Entropy로 시장 질서도를 측정, 낮은 PE(질서적 패턴)에서 모멘텀 conviction 증폭 |

---

## 성과 요약 (6년, 2020-2025, 4H 재검증)

### 에셋별 비교 (4H TF, 2026-02-12 재검증)

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Win Rate |
|------|------|--------|------|-----|--------|------|---------|
| **1** | **BNB/USDT** | **0.52** | +5.6% | -23.6% | 141 | 1.99 | 40.0% |
| 2 | ETH/USDT | 0.49 | +7.6% | -31.5% | 114 | 3.09 | 33.6% |
| 3 | SOL/USDT | 0.17 | +1.5% | -47.5% | 42 | 1.44 | 39.0% |
| 4 | BTC/USDT | 0.15 | +0.5% | -11.9% | 187 | 1.26 | 37.6% |
| 5 | DOGE/USDT | -0.07 | -9.2% | -75.9% | 84 | 0.25 | 39.8% |

### Best Asset 핵심 지표 (BNB/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.52 | > 1.0 | **FAIL** |
| CAGR | +5.6% | > 20% | **FAIL** |
| MDD | -23.6% | < 40% | PASS |
| Trades | 141 | > 50 | PASS |

### 이전 결과 (1D 왜곡, 참고용)

| 순위 | 에셋 | Sharpe (1D) | Sharpe (4H) | 변화 |
|------|------|------------|------------|------|
| 1 | BNB/USDT | -0.43 | 0.52 | 개선 |
| 2 | ETH/USDT | 0.16 | 0.49 | 개선 |
| 3 | SOL/USDT | 0.67 | 0.17 | 악화 |
| 4 | BTC/USDT | 0.32 | 0.15 | 악화 |
| 5 | DOGE/USDT | 0.04 | -0.07 | 악화 |

---

## Gate 진행 현황

```
G0 아이디어  [PASS] 24/30점
G0B 코드검증 [PASS] C1-C7 전항목 PASS (2026-02-12 재검증). Warning 1건 (W3 Regime)
G1 백테스트  [FAIL] 4H 재검증 확정. Best BNB Sharpe 0.52 < 1.0, CAGR 5.6% < 20%
```

### Gate 1 상세 (4H 재검증, 2026-02-12)

- **Sharpe/CAGR 미달**: Best Asset BNB Sharpe 0.52 < 1.0, CAGR +5.6% < 20%
- **거래 수 개선**: 1D(13건) → 4H(568건 합산). 4H TF에서 시그널 생성이 정상화됨
- **DOGE MDD -75.9%**: DOGE에서 여전히 높은 MDD. HEDGE_ONLY 모드에서도 밈코인 변동에 취약
- **SOL 악화**: 1D Sharpe 0.67 → 4H Sharpe 0.17. PE 윈도우 정상화에도 SOL에서 momentum edge 부재
- **BNB/ETH만 양호**: BNB 0.52, ETH 0.49로 중하위 자산에서만 약한 양의 수익. 고변동 에셋(SOL/DOGE)에서 PE conviction이 무효

**4H 재검증 관찰**:
- 1D 왜곡 시 비정상적 수치(CAGR +2398%, MDD -786%)가 4H 정상 TF에서 해소
- PE conviction × momentum 조합이 4H에서도 alpha 생성에 실패
- CTREND Best SOL Sharpe 2.05 대비 perm-entropy-mom Best BNB 0.52 — 3.9배 열등

### Gate 1 상세 (이전 1D 왜곡, 참고)

- **극소 거래 문제**: 5개 에셋 합산 거래 13건. annualization_factor 불일치로 전략 무효
- **근본 원인**: 4H TF 전략 파라미터가 1D 데이터와 구조적으로 불일치

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 |
| 2026-02-10 | G0B | PASS | Critical 7항목 결함 0개 |
| 2026-02-11 | G1 | **FAIL** | 1D 왜곡: 전 에셋 Trades < 50, Best Sharpe 0.67 < 1.0 |
| 2026-02-12 | G1 (4H 재검증) | **FAIL 확정** | Best BNB Sharpe 0.52 < 1.0, CAGR +5.6% < 20%. 4H TF에서도 alpha 부재 확인 |
