# 전략 스코어카드: VR-Regime

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | VR-Regime (`vr-regime`) |
| **유형** | 레짐전환 |
| **타임프레임** | 1D |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | BNB/USDT (Sharpe 0.17) |
| **2nd Asset** | DOGE/USDT (Sharpe 0.05) |
| **경제적 논거** | Lo-MacKinlay Variance Ratio로 random walk hypothesis를 검정. VR > 1 → trending (momentum), VR < 1 → mean-reverting (contrarian). Non-parametric test 기반. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| 1 | BNB/USDT | 0.17 | +0.8% | -12.5% | 10 | 1.28 | -6180.3% | 0.006 |
| 2 | DOGE/USDT | 0.05 | -6.5% | -74.3% | 30 | 0.52 | -6014.7% | -0.053 |
| 3 | ETH/USDT | -0.11 | -0.4% | -5.0% | 4 | 0.75 | -2173.8% | -0.000 |
| 4 | SOL/USDT | -0.57 | -2.1% | -15.3% | 9 | 0.26 | -4289.8% | -0.003 |
| 5 | BTC/USDT | -0.62 | -3.1% | -17.2% | 2 | 0.00 | -1145.4% | -0.006 |

### Best Asset 핵심 지표 (BNB/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.17 | > 1.0 | **FAIL** |
| CAGR | +0.8% | > 20% | **FAIL** |
| MDD | -12.5% | < 40% | PASS |
| Trades | 10 | > 50 | **FAIL** |
| Win Rate | 60.0% | > 45% | PASS |
| Sortino | 0.04 | > 1.5 | FAIL |
| Calmar | 0.06 | > 1.0 | FAIL |
| Profit Factor | 1.28 | > 1.3 | FAIL |
| Alpha (vs BTC B&H) | -6180.3% | > 0% | FAIL |
| Beta (vs BTC) | 0.006 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 24/30점
G0B 코드감사  [PASS] Critical 0, High 0 (수정완료), Medium 4
G1  백테스트  [FAIL] 전 에셋 Sharpe < 1.0, Best BNB Sharpe 0.17, 극소 거래 (2~30건)
G2  IS/OOS   [ -- ] G1 FAIL로 미실행
G3  파라미터  [ -- ] G1 FAIL로 미실행
G4  심층검증  [ -- ] G1 FAIL로 미실행
G5  EDA검증  [ -- ] G1 FAIL로 미실행
G6  모의거래  [ -- ] G1 FAIL로 미실행
G7  실전배포  [ -- ] G1 FAIL로 미실행
```

### Gate 상세

#### G0B 코드 감사 (2026-02-10)

**종합 등급: A** (HIGH 이슈 전수 수정 완료)

| 항목 | 점수 |
|------|:----:|
| 데이터 무결성 | 10/10 |
| 시그널 로직 | 9/10 |
| 실행 현실성 | 9/10 |
| 리스크 관리 | 9/10 |
| 코드 품질 | 9/10 |

**수정 완료된 이슈:**

- [H-001] ~~`warmup_periods()` chained rolling 미반영~~ → `vr_window + vr_k` 반영 완료
- [H-002] ~~`use_heteroscedastic` 라벨 오해~~ → 주석 수정: "Lo-MacKinlay (1988) Eq.5 simplified form, NOT full Eq.10" 명시
- [추가] HEDGE_ONLY drawdown shift(1) 미적용 → `df["drawdown"].shift(1)` 적용 완료

**잘된 점:**

- shift(1) 규칙 정확 (vr, z_stat, mom_direction, vol_scalar, drawdown 모두 shift)
- theta_safe = `np.maximum(theta, 1e-10)` z-stat 0 나눗셈 방어
- VR 계산 시 `var_1.replace(0, np.nan)` 0 나눗셈 방어
- cross-field validation: `vr_k * 2 < vr_window`

> **최종 판정**: G1 FAIL — 폐기

#### Gate 1 상세 (5코인 × 6년 백테스트, 2020-2025)

- **전 에셋 Sharpe < 1.0**: Best Asset BNB/USDT조차 Sharpe 0.17에 불과
- **3/5 에셋 Sharpe 음수**: BTC (-0.62), ETH (-0.11), SOL (-0.57)
- **극소 거래 수**: BTC 2건, ETH 4건, SOL 9건, BNB 10건 — significance_z 1.96이 너무 보수적
- **DOGE MDD 74.3%**: 유일하게 30건 거래 발생했으나, 방향성 없는 과도한 MDD
- **전 에셋 Alpha 음수**: BTC B&H 대비 전면 열등 (-1145% ~ -6180%)
- **CAGR**: Best Asset BNB조차 +0.8%로 기준(>20%) 극히 미달

**FAIL 사유**: Lo-MacKinlay VR test의 significance_z=1.96 (95% CI)이 일봉 데이터에서 거의 시그널을 생성하지 못함.
6년간 BTC에서 단 2건 거래는 전략의 실용성 부재를 의미. VR > 1 (trending) 또는 VR < 1 (mean-reverting) 판별이
일봉 노이즈 대비 유의 수준을 넘기 어려움. 이론적 논거는 타당하나, 크립토 일봉 데이터에서의 통계적 검정력 부족.

**CTREND 대비**: SOL Sharpe 2.05 vs VR-Regime Best 0.17 — 12배 열등. 거래 수 288 vs 10 — 30배 부족.

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 — 경제적 논거 4, 참신성 4, 데이터 5, 구현 4, 용량 3, 레짐독립 4 |
| 2026-02-10 | G0B | PASS | Critical 0개. HIGH 2건 발견 → 전수 수정 완료 (warmup vr_k 반영, 주석 정정, drawdown shift) |
| 2026-02-10 | G1 | **FAIL** | 전 에셋 Sharpe < 1.0 (Best BNB 0.17), 극소 거래 (2~30건), 전 에셋 Alpha 음수. significance_z 1.96 과도 → 시그널 부재 |
