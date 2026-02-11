# 전략 스코어카드: Entropy-Switch

> 자동 생성 | 평가 기준: [evaluation-standard.md](../../strategy/evaluation-standard.md)

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Entropy-Switch (`entropy-switch`) |
| **유형** | 정보이론 / 레짐필터 |
| **타임프레임** | 1D |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | BNB/USDT (Sharpe 0.52) |
| **2nd Asset** | ETH/USDT (Sharpe 0.44) |
| **경제적 논거** | Shannon Entropy로 시장 예측가능성을 측정. 낮은 엔트로피(규칙적 패턴)에서만 추세추종 진입, 높은 엔트로피(무작위)에서 거래 중단. Entropy+ADX 87% regime 정확도 학술 검증. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **BNB/USDT** | **0.52** | +11.1% | -47.9% | 158 | 1.40 | -6066.5% | 0.12 |
| 2 | ETH/USDT | 0.44 | +8.4% | -41.2% | 170 | 1.28 | -2079.7% | 0.06 |
| 3 | SOL/USDT | 0.37 | +5.5% | -38.4% | 89 | 1.26 | -4232.6% | 0.02 |
| 4 | BTC/USDT | 0.29 | +4.2% | -54.1% | 166 | 1.18 | -1074.4% | 0.07 |
| 5 | DOGE/USDT | 0.20 | -16.3% | -90.1% | 161 | 0.67 | -6044.4% | -0.10 |

### Best Asset 핵심 지표 (BNB/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.52 | > 1.0 | **FAIL** |
| CAGR | +11.1% | > 20% | **FAIL** |
| MDD | -47.9% | < 40% | **FAIL** |
| Trades | 158 | > 50 | PASS |
| Win Rate | 51.0% | — | — |
| Sortino | 0.58 | — | — |
| Calmar | 0.23 | — | — |
| Profit Factor | 1.40 | — | — |
| Alpha (vs BTC B&H) | -6066.5% | — | — |
| Beta (vs BTC) | 0.12 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 26/30점
G0B 코드감사  [PASS] 7/7 Critical 결함 0개
G1  백테스트  [FAIL] BNB/USDT Sharpe 0.52, CAGR +11.1%, MDD -47.9%
G2  IS/OOS   [    ] — (G1 FAIL로 미진행)
G3  파라미터  [    ] — (G1 FAIL로 미진행)
G4  심층검증  [    ] — (G1 FAIL로 미진행)
G5  EDA검증  [    ] — (G1 FAIL로 미진행)
G6  모의거래  [    ] — (G1 FAIL로 미진행)
G7  실전배포  [    ] — (G1 FAIL로 미진행)
```

### Gate 상세

#### G0A 아이디어 검증 (2026-02-10)

| 항목 | 점수 |
|------|:----:|
| 경제적 논거 | 4/5 |
| 참신성 | 5/5 |
| 데이터 확보 | 5/5 |
| 구현 복잡도 | 4/5 |
| 용량 수용 | 4/5 |
| 레짐 독립성 | 4/5 |
| **합계** | **26/30** |

**핵심 근거:**

- 완전히 새 카테고리 (Information Theory) — 기존 30개 전략과 겹치지 않음
- Entropy+ADX 조합 레짐 분류에서 87% 정확도 (Preprints 202502.1717)
- Permutation Entropy로 BTC 변동성의 예측가능성 8년 실증 (Physica A, 2024)
- OHLCV only — 외부 데이터 불필요
- 기존 레짐 감지 전략(ADX Regime FAIL)과 근본적 차이: "시장 상태 분류" vs "예측가능성 측정"

#### G0B 코드 감사 (2026-02-10)

- Critical C1-C7 결함 0개 PASS
- W2 (params=6, 4H 예상 ~200-500 trades)

#### Gate 1 상세 (5코인 x 6년 백테스트)

**판정**: **FAIL** — 전 에셋 Sharpe < 1.0, Best CAGR +11.1% < 20%, Best MDD -47.9% > 40%

**FAIL 사유**:

1. **Sharpe 전 에셋 미달**: Best 0.52 (BNB), 전 에셋 0.20~0.52 범위. CTREND Best 2.05 대비 1/4 수준
2. **CAGR 전 에셋 미달**: Best +11.1% (BNB) < 20% 기준. 6년간 BTC B&H 대비 열등
3. **MDD 심각**: BTC -54.1%, DOGE -90.1% (사실상 파산). BNB -47.9%도 40% 초과
4. **DOGE 음수 수익**: CAGR -16.3%, PF 0.67 — 엔트로피 필터가 DOGE 노이즈 환경에서 무효

**퀀트 해석**:

- Entropy 필터가 "거래 중단"을 통해 리스크는 부분적으로 줄이나, 알파 생성에 실패
- SOL 거래 수 89건 (다른 에셋 158-170건 대비 저조) — SOL의 높은 변동성이 entropy_low_threshold를 자주 초과
- ADX 필터 + Entropy 이중 필터링이 거래 기회를 과도하게 제한
- HEDGE_ONLY 모드로 Short 기회 제한 — FULL 모드였으면 2022 약세장 방어 가능했을 수 있으나, 근본적 alpha 부재는 해결 불가
- 교훈: 정보이론 기반 regime 분류가 학술적으로 유효해도, 신호→수익 변환 메커니즘 없으면 무의미

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 26/30점 — 경제적 논거 4, 참신성 5, 데이터 5, 구현 4, 용량 4, 레짐독립 4 |
| 2026-02-10 | G0B | PASS | 7/7 Critical 결함 0개, W2 (params=6) |
| 2026-02-10 | G1 | **FAIL** | 전 에셋 Sharpe < 1.0 (Best 0.52 BNB). CAGR +11.1% < 20%. MDD -47.9% > 40%. DOGE -16.3% 음수. Alpha 부재 |
