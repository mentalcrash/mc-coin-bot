# 전략 스코어카드: Range-Squeeze

> 자동 생성 | 평가 기준: `pipeline gates-list`

## 기본 정보

| 항목 | 값 |
|------|---|
| **전략명** | Range-Squeeze (`range-squeeze`) |
| **유형** | 변동성/돌파 |
| **타임프레임** | 1D |
| **상태** | `폐기 (Gate 1 FAIL)` |
| **Best Asset** | ETH/USDT (Sharpe 0.33) |
| **2nd Asset** | SOL/USDT (Sharpe 0.08) |
| **경제적 논거** | NR7 패턴 + range ratio squeeze 후 vol compression이 해소되며 발생하는 방향성 breakout을 포착. Crypto의 높은 vol로 squeeze-breakout 사이클이 뚜렷. |

---

## 성과 요약 (6년, 2020-2025)

### 에셋별 비교

| 순위 | 에셋 | Sharpe | CAGR | MDD | Trades | PF | Alpha | Beta |
|------|------|--------|------|-----|--------|------|-------|------|
| 1 | ETH/USDT | 0.33 | +3.8% | -27.5% | 331 | 1.12 | -2139.5% | 0.02 |
| 2 | SOL/USDT | 0.08 | +0.2% | -28.1% | 301 | 1.02 | -4273.1% | 0.00 |
| 3 | BTC/USDT | -0.42 | -6.5% | -41.4% | 310 | 0.87 | -1156.8% | 0.01 |
| 4 | BNB/USDT | -0.52 | -8.4% | -53.5% | 369 | 0.84 | -6222.1% | 0.01 |
| 5 | DOGE/USDT | -0.89 | -94.7% | -529.4% | 84 | 0.68 | -6359.3% | -0.77 |

### Best Asset 핵심 지표 (ETH/USDT)

| 지표 | 값 | 기준 | 판정 |
|------|---|------|------|
| Sharpe | 0.33 | > 1.0 | **FAIL** |
| CAGR | +3.8% | > 20% | **FAIL** |
| MDD | -27.5% | < 40% | PASS |
| Trades | 331 | > 50 | PASS |
| Win Rate | 45.0% | > 45% | FAIL |
| Sortino | 0.29 | > 1.5 | FAIL |
| Calmar | 0.14 | > 1.0 | FAIL |
| Profit Factor | 1.12 | > 1.3 | FAIL |
| Alpha (vs BTC B&H) | -2139.5% | > 0% | FAIL |
| Beta (vs BTC) | 0.02 | < 0.5 | PASS |

---

## Gate 진행 현황

```
G0A 아이디어  [PASS] 24/30점
G0B 코드감사  [PASS] Critical 0, High 0 (수정완료), Medium 4
G1  백테스트  [FAIL] ETH/USDT Sharpe 0.33, CAGR +3.8%, MDD -27.5%
G2  IS/OOS   [    ] (G1 FAIL로 미진행)
G3  파라미터  [    ] (G1 FAIL로 미진행)
G4  심층검증  [    ] (G1 FAIL로 미진행)
G5  EDA검증  [    ] (G1 FAIL로 미진행)
G6  모의거래  [    ] (G1 FAIL로 미진행)
G7  실전배포  [    ] (G1 FAIL로 미진행)
```

### Gate 상세

#### G0B 코드 감사 (2026-02-10)

**종합 등급: A** (HIGH 이슈 전수 수정 완료)

| 항목 | 점수 |
|------|:----:|
| 데이터 무결성 | 10/10 |
| 시그널 로직 | 9/10 |
| 실행 현실성 | 9/10 |
| 리스크 관리 | 8/10 |
| 코드 품질 | 9/10 |

**수정 완료된 이슈:**

- [H-001] ~~NR 패턴 floating-point `==` 비교~~ → `np.isclose(rtol=1e-9)` 적용 완료
- [H-002] ~~`daily_range` dirty data 가드 없음~~ → `.clip(lower=0)` 적용 완료
- [H-003] vol_scalar 무제한 — PM 클램핑에 의존 (전략 레벨 추가 방어 불요, 아키텍처 설계대로)
- [추가] HEDGE_ONLY drawdown shift(1) 미적용 → `df["drawdown"].shift(1)` 적용 완료

**잘된 점:**

- shift(1) 규칙 올바르게 적용 (is_nr, range_ratio, vol_scalar, drawdown 모두 shift)
- NR7 + range_ratio squeeze 이중 감지 로직
- avg_range `.replace(0, np.nan)` 0 나눗셈 방어
- 벡터화 연산, 깔끔한 4파일 분리 구조

#### Gate 1 상세 (단일에셋 백테스트, 5코인 x 6년)

**판정: FAIL** -- Best Asset(ETH) Sharpe 0.33 < 1.0, CAGR +3.8% < 20%

**실패 사유:**

1. **전 에셋 Sharpe < 1.0**: 최고 ETH 0.33으로 기준 1.0에 크게 미달 (기준의 33%)
2. **3개 에셋 음수 Sharpe**: BTC(-0.42), BNB(-0.52), DOGE(-0.89) 손실
3. **전 에셋 음수 Alpha**: BTC B&H 대비 전 에셋에서 크게 열등
4. **DOGE 파괴적 손실**: MDD -529.4%, CAGR -94.7%, 사실상 자본 전액 손실
5. **CAGR 최대 +3.8%**: 20% 기준의 19% 수준, 경제적 가치 없음

**퀀트 해석:**

- **에셋 순서 비정상**: ETH > SOL > BTC > BNB > DOGE. 추세추종이라면 SOL > BTC가 일반적이나, ETH가 1위이고 SOL이 거의 0 수익 -- squeeze-breakout 패턴이 crypto 일봉에서 유의미한 edge를 제공하지 못함
- **거래 수 과다**: 연 50~60건 거래 (일봉 대비 높음), Win Rate 45~49%로 동전 던지기 수준. Avg Win과 Avg Loss가 유사하여 edge 없음
- **DOGE 거래 84건으로 급감**: DOGE의 높은 변동성에서 squeeze 감지가 저조하여 거래 미발생 구간 다수
- **Beta 극히 낮음 (0.00~0.02)**: BTC와 무상관이나, 이는 전략 자체가 무작위에 가까움을 의미 (alpha도 대규모 음수)
- **비용 민감도**: 연 50+건 거래 × 편도 0.11% = 연 ~11% drag. CAGR +3.8%에서 비용 부담이 이미 수익을 잠식

**CTREND 비교:**

| 지표 | CTREND Best (SOL) | Range-Squeeze Best (ETH) | 비율 |
|------|-------------------|-------------------------|------|
| Sharpe | 2.05 | 0.33 | 16% |
| CAGR | +97.8% | +3.8% | 4% |
| MDD | -27.7% | -27.5% | 유사 |
| Trades | 288 | 331 | 유사 |

**근본 원인 분석:**
NR7 패턴과 range ratio squeeze는 전통 주식에서 유의미하나, crypto 일봉에서는:

1. 24/7 시장 특성상 squeeze 해소 후 방향 지속성이 약함
2. close-open 방향 (1 bar)으로 breakout 방향을 결정하는 것은 노이즈에 취약
3. vol-target sizing으로 squeeze 구간 (저변동성)에서 레버리지가 확대되어 false breakout 비용 증가

---

## 의사결정 기록

| 날짜 | Gate | 판정 | 근거 |
|------|------|------|------|
| 2026-02-10 | G0A | PASS | 24/30점 -- 경제적 논거 4, 참신성 3, 데이터 5, 구현 5, 용량 4, 레짐독립 3 |
| 2026-02-10 | G0B | PASS | Critical 0개. HIGH 3건 발견 -> 2건 수정 완료 (isclose, clip), 1건 아키텍처 의존 (vol_scalar PM 클램핑) |
| 2026-02-10 | G1 | **FAIL** | 전 에셋 Sharpe < 1.0 (Best ETH 0.33). CAGR +3.8% < 20%. 3/5 에셋 음수 Sharpe. 전 에셋 음수 Alpha. Squeeze-breakout edge 부재 |
