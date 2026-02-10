# Strategy Discovery — Temp Candidates

> Gate 0 PASS 아이디어의 임시 후보 목록. 구현 전 사용자 리뷰와 우선순위 결정을 위한 staging 문서.

---

## 2026-02-10 — Strategy Discovery Session (4H Timeframe)

### 후보 #1: Entropy Regime Switch (`entropy-switch`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Information Theory |
| **타임프레임** | 4H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 26/30 |
| **상태** | 🔵 후보 |

**핵심 가설**: Shannon Entropy로 시장 예측가능성을 측정하여, 낮은 엔트로피(규칙적 패턴)에서만 추세추종 진입하고 높은 엔트로피(무작위)에서는 거래를 중단한다.

**경제적 논거**: 엔트로피가 낮으면 가격 패턴이 반복적이므로 추세가 지속될 가능성이 높다. 높은 엔트로피는 무작위 변동을 의미하며 추세추종 전략이 손실을 보는 구간이다. Entropy+ADX 조합 레짐 분류에서 87% 정확도가 학술적으로 검증되었다. Permutation Entropy로 BTC 변동성의 예측가능성이 8년간 실증되었다.

**사용 지표**: Shannon Entropy (returns, window=120, bins=10), Momentum (close, 20), ADX (14) 보조

**시그널 생성 로직**:
```
1. entropy = scipy.stats.entropy(histogram(returns[-window:], bins))
2. IF entropy < low_threshold AND momentum > 0 → LONG
3. IF entropy < low_threshold AND momentum < 0 → SHORT (HEDGE_ONLY)
4. IF entropy > high_threshold → FLAT (no signal)
5. 중간 구간 → 기존 포지션 유지 (신규 진입 없음)
```

**CTREND 상관 예측**: 낮음 (정보이론 vs ML 기술적 앙상블)

**예상 거래 빈도**: ~100건/년

**차별화 포인트**: 레짐 감지 전략 5개 전멸과 근본적 차이. 기존 전략들은 "시장 상태(bull/bear/sideways)" 분류를 시도했으나, entropy-switch는 "예측가능성 수준"을 측정한다. 시장이 예측가능할 때만 거래하는 메타 전략. ADX Regime(FAIL)은 ADX가 주 시그널이었으나, 여기서는 Entropy가 주이고 ADX는 보조 확인용.

**출처**:
- Optimizing Trading with ML and Entropy (Preprints 202502.1717) — Entropy+ADX 87% 정확도
- Permutation Entropy Analysis of Bitcoin Volatility (Physica A, 2024) — 8년 BTC 검증
- Shannon Entropy Cryptocurrency Portfolios (Entropy, 2022) — 포트폴리오 최적화
- Trading with Less Surprise: Shannon Entropy (Medium/Codex) — 브레이크아웃 필터 실증

**Gate 0 상세 점수**:
- 경제적 논거: 4/5 (학술 검증 다수, 87% regime 정확도)
- 참신성: 5/5 (완전히 새 카테고리 — Information Theory)
- 데이터 확보: 5/5 (OHLCV only, scipy.stats.entropy)
- 구현 복잡도: 4/5 (entropy + momentum, 직관적)
- 용량 수용: 4/5 (4H에서 ~100건/년)
- 레짐 독립성: 4/5 (entropy 자체가 레짐 적응)

---

### 후보 #2: Adaptive Kalman Trend (`kalman-trend`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Statistical Filtering / Trend Following |
| **타임프레임** | 4H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 24/30 |
| **상태** | 🔵 후보 |

**핵심 가설**: 칼만 필터로 가격에서 노이즈를 베이지안 최적으로 분리하여, smoothed price와 velocity 시그널로 추세 방향을 감지한다. Realized volatility 기반 adaptive Q 파라미터로 변동성 레짐에 자동 적응한다.

**경제적 논거**: 칼만 필터는 베이지안 최적 추정기로, 고정 lookback MA와 달리 자동으로 노이즈 레벨에 적응한다. Velocity (1st derivative) > 0이면 상승 추세, < 0이면 하락 추세. MA 대비 lag 감소, false signal 60% 필터링, profit factor 개선이 학술적으로 확인되었다. arXiv 2601.06084에서 4H가 크립토의 "equilibrium zone"임을 실증했다.

**사용 지표**: Kalman state (smoothed price), Kalman velocity, Realized Volatility (20 bars) for adaptive Q

**시그널 생성 로직**:
```
1. state, velocity = kalman_update(price, Q_adaptive, R)
   where Q_adaptive = base_Q * (realized_vol / long_term_vol)
2. IF velocity > threshold → LONG
3. IF velocity < -threshold → SHORT (HEDGE_ONLY)
4. IF |velocity| < threshold → FLAT
5. Position sizing: ATR-based vol targeting
```

**CTREND 상관 예측**: 중간 (둘 다 추세추종이나 메커니즘이 다름)

**예상 거래 빈도**: ~60-100건/년

**차별화 포인트**: TSMOM/Enhanced TSMOM(FAIL, Decay 85-87%)은 고정 MA lookback에 의존. 칼만 필터는 lookback window 파라미터가 없어 과적합 여지가 적다. Q/R ratio 하나로 responsiveness가 결정되며, adaptive Q는 실시간 변동성에 자동 조절.

**출처**:
- Adaptive Kalman Filter vs EMA (PyQuantLab, 2025) — Sharpe/drawdown 우수
- Abstract Trend Without Hiccups (arXiv:1808.03297) — smooth trend extraction
- Who sets the range? (arXiv:2601.06084) — 4H equilibrium zone 실증
- Kalman beats MAs in Trading (Coding Nexus, Dec 2025) — lag/profit factor 비교

**Gate 0 상세 점수**:
- 경제적 논거: 4/5 (베이지안 최적 필터링, 학술/실전 검증)
- 참신성: 4/5 (프로젝트 내 칼만 필터 미사용)
- 데이터 확보: 5/5 (OHLCV only)
- 구현 복잡도: 3/5 (matrix operations, Q/R tuning)
- 용량 수용: 4/5 (~60-100건/년)
- 레짐 독립성: 4/5 (adaptive 설계)

---

### 후보 #3: VWAP Disposition Momentum (`vwap-disposition`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Behavioral Finance |
| **타임프레임** | 4H |
| **ShortMode** | FULL |
| **Gate 0 점수** | 23/30 |
| **상태** | 🔵 후보 |

**핵심 가설**: Rolling VWAP를 시장 참여자의 평균 취득가(cost basis)로 사용하여, 미실현 이익/손실 수준(Capital Gains Overhang)에 따른 매도/매수 압력을 예측한다.

**경제적 논거**: Disposition effect — 투자자는 이익은 빨리, 손실은 늦게 실현한다. Bitcoin에서 2017년 이후 disposition effect 유의미하게 증가 확인 (Schatzmann 2023). 미실현 이익 과다(CGO↑) → 차익실현 매도 압력 → 단기 약세. 미실현 손실 과다(CGO↓) → 항복 매도 후 반등. On-chain MVRV>3.5=고점, <1.0=저점으로 검증된 패턴을 VWAP proxy로 OHLCV 구현.

**사용 지표**: Rolling VWAP (720 bars = 120일), Price-to-VWAP ratio (CGO proxy), Volume ratio confirmation

**시그널 생성 로직**:
```
1. vwap_120d = rolling_vwap(price, volume, window=720)
2. cgo = (close - vwap_120d) / vwap_120d
3. IF cgo < -overhang_low AND volume_spike → LONG (항복 매도 후 반등)
4. IF cgo > +overhang_high AND volume_decline → SHORT (차익 실현 압력)
5. IF -overhang_low < cgo < +overhang_high → momentum direction follow
```

**CTREND 상관 예측**: 낮음 (행동재무학 vs ML 기술적)

**예상 거래 빈도**: ~60-80건/년

**차별화 포인트**: 행동재무학 카테고리 완전 미탐색. VW-TSMOM(FAIL, Decay 92%)은 volume-weighted momentum이 핵심이나, vwap-disposition은 VWAP를 cost basis proxy로 사용하여 투자자 심리를 측정. 방향이 근본적으로 다르다.

**출처**:
- Exploring investor behavior in Bitcoin (arXiv:2010.12415, Digital Finance 2023)
- On-Chain Cashflows and Cryptocurrency Returns (SSRN 4540433)
- Cryptocurrency Volume-Weighted TSMOM (SSRN 4825389) — Sharpe 2.17
- Behavioral biases of crypto investors (Emerald, 2024) — Prospect Theory

**Gate 0 상세 점수**:
- 경제적 논거: 4/5 (Bitcoin disposition effect 실증, 행동재무학 이론)
- 참신성: 5/5 (완전히 새 카테고리)
- 데이터 확보: 4/5 (OHLCV rolling VWAP, on-chain 없이 proxy)
- 구현 복잡도: 4/5 (rolling VWAP + deviation zones)
- 용량 수용: 3/5 (~60-80건/년, deviation threshold 제한적)
- 레짐 독립성: 3/5 (강세장에서 disposition effect 더 강함)

---
