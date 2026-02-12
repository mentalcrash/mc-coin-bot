# 전략 스코어카드: VWAP-Disposition

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | VWAP-Disposition (`vwap-disposition`) |
| **유형** | 행동재무학 |
| **타임프레임** | 1D |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | SOL/USDT (Sharpe 0.96) |
| **2nd Asset** | ETH/USDT (Sharpe 0.92) |
| **경제적 논거** | Rolling VWAP를 시장 참여자의 평균 취득가(cost basis)로 사용. Capital Gains Overhang(CGO)에 따른 disposition effect로 매도/매수 압력 예측. BTC에서 2017년 이후 disposition effect 유의미 증가 확인. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| **1** | **SOL/USDT** | **0.96** | +30.4% | -38.2% | 228 | 1.42 | -3837.4% | 0.09 |
| 2 | ETH/USDT | 0.92 | +27.1% | -42.8% | 278 | 1.36 | -1767.4% | 0.06 |
| 3 | BNB/USDT | 0.45 | +10.2% | -55.5% | 317 | 1.13 | -6064.5% | 0.11 |
| 4 | BTC/USDT | 0.16 | +0.4% | -64.1% | 289 | 1.04 | -1097.8% | -0.03 |
| 5 | DOGE/USDT | -0.69 | -95.2% | -622.1% | 43 | 0.83 | -6358.9% | -0.77 |

### Best Asset 핵심 지표 (SOL/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.96 | > 1.0 | **FAIL** (0.04 미달) |
| CAGR | +30.4% | > 20% | PASS |
| MDD | -38.2% | < 40% | PASS |
| Trades | 228 | > 50 | PASS |
| Win Rate | 50.9% | — | — |
| Sortino | 1.04 | — | — |
| Calmar | 0.80 | — | — |
| Profit Factor | 1.42 | — | — |
| Alpha (vs BTC B&H) | -3837.4% | — | — |
| Beta (vs BTC) | 0.09 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 23/30점
G0B 코드감사  [PASS] 7/7 Critical 결함 0개
G1  백테스트  [FAIL] SOL/USDT Sharpe 0.96, CAGR +30.4%, MDD -38.2%
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
| 데이터 확보 | 4/5 |
| 구현 복잡도 | 4/5 |
| 용량 수용 | 3/5 |
| 레짐 독립성 | 3/5 |
| **합계** | **23/30** |

**핵심 근거:**

- 행동재무학 카테고리 완전 미탐색 — 기존 전략과 메커니즘 근본적 차이
- BTC에서 disposition effect 유의미 실증 (Schatzmann 2023, Digital Finance)
- On-chain MVRV > 3.5 = 고점, < 1.0 = 저점 패턴을 VWAP proxy로 OHLCV 구현
- CGO 극단값 + Volume 확인으로 capitulation/profit-taking 포착
- ShortMode FULL — 차익실현 압력 SHORT이 핵심 에지

#### G0B 코드 감사 (2026-02-10)

- Critical C1-C7 결함 0개 PASS
- W2 (params=6), W4 (720-bar VWAP window)

#### Gate 1 상세 (5코인 x 6년 백테스트)

**판정**: **FAIL** — Best Sharpe 0.96 < 1.0 (0.04 미달)

**FAIL 사유**:

1. **Sharpe 근소 미달**: Best 0.96 (SOL) — 기준 1.0에 0.04 부족. 경계 사례
2. **ETH도 양호**: Sharpe 0.92, CAGR +27.1% — 2개 에셋에서 준 PASS 수준
3. **DOGE 파산**: MDD -622.1%, CAGR -95.2% — FULL Short 모드에서 DOGE 급등에 무방비 (ShortMode.FULL + 밈코인 변동성 = 치명적)
4. **BTC 무수익**: CAGR +0.4%, MDD -64.1% — BTC에서 disposition effect 시그널 무효
5. **에셋간 극심한 편차**: SOL Sharpe 0.96 vs DOGE -0.69 — 편차 1.65 (Yellow Flag)

**퀀트 해석**:

- SOL/ETH에서 부분적 edge 존재: CGO 기반 mean-reversion + FULL short이 효과적
- BTC에서 실패하는 이유: BTC는 기관 주도 추세가 강해 disposition effect (개인 투자자 행동 편향) 약함
- DOGE MDD -622%는 short squeeze + 밈코인 급등에서 ShortMode.FULL의 구조적 위험 노출
  - DOGE 2021-01~05 급등(+12,000%) 시 Short 포지션 파산
- 720-bar VWAP window (1D 기준 약 2년)는 데이터 6년 대비 1/3 소비 — warmup 비효율
- 교훈: 행동재무학 시그널은 에셋별 투자자 구성(개인 vs 기관)에 크게 의존. 밈코인 FULL Short은 구조적 자살행위

**CTREND 비교**:

| 지표 | CTREND Best (SOL) | VWAP-Disposition Best (SOL) |
|------|-------------------|-----------------------------|
| Sharpe | 2.05 | 0.96 |
| CAGR | +97.8% | +30.4% |
| MDD | -27.7% | -38.2% |
| Trades | 288 | 228 |

**참고**: Sharpe 0.96은 포트폴리오 분산 용도로는 가치가 있으나 (CTREND와 상관 낮음, Beta 0.09), 단독 배포에는 불충분. 향후 앙상블 구성 시 sub-strategy 후보로 재고 가능.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 23/30점 — 경제적 논거 4, 참신성 5, 데이터 4, 구현 4, 용량 3, 레짐독립 3 |
| 2026-02-10 | G0B | PASS | 7/7 Critical 결함 0개, W2 (params=6), W4 (720-bar VWAP) |
| 2026-02-10 | G1 | **FAIL** | Best Sharpe 0.96 < 1.0 (SOL). DOGE MDD -622% (FULL short 파산). BTC 무수익. 에셋간 편차 극심 |
