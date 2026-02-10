# Strategy Candidates (Temp)

> Gate 0A PASS 아이디어 임시 후보 목록. 구현 전 사용자 리뷰 + 우선순위 결정용.

---

## 2026-02-10 — Strategy Discovery Session (1H Timeframe)

### 후보 #1: Session Breakout (`session-breakout`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Structural / Session Decomposition |
| **타임프레임** | 1H |
| **ShortMode** | FULL |
| **Gate 0 점수** | 27/30 |
| **상태** | :red_circle: G1 FAIL — 폐기 (전 에셋 Sharpe 음수, MDD 88~97%) |

**핵심 가설**: Asian session(00-08 UTC)의 low-vol range를 EU/US 세션 open 시 breakout하는 패턴을 포착.

**경제적 논거**: Asian 세션은 institutional 참여 부족으로 accumulation zone 형성. London/US open에서 fresh liquidity 유입 시 range breakout 발생. Stop-hunting: Asian H/L에 집중된 stop order sweep 후 방향 결정. FX 시장에서 수십 년간 검증된 구조적 edge.

**사용 지표**: Session High/Low (00-08 UTC), Range Width Percentile (30d rolling), ADX (regime filter)

**시그널 생성 로직**:
```
1. Asian range: 00:00-08:00 UTC 1H bar의 max(high), min(low)
2. Range width percentile: 30일 rolling (narrow < 50th → squeeze)
3. 08:00-20:00 UTC에서:
   - close > Asian_high → long (shift(1) 적용)
   - close < Asian_low → short (shift(1) 적용)
4. Stop-loss: Asian range 반대쪽
5. Exit: 22:00 UTC 또는 1.5x range width TP
6. Narrow range filter: range_pctl < 50 시에만 진입 (squeeze 효과)
```

**CTREND 상관 예측**: 낮음 (intraday session structure vs daily ML ensemble)

**예상 거래 빈도**: 100~200건/년

**차별화 포인트**: 기존 range-squeeze(NR7, daily)는 1D squeeze. 이 전략은 intraday session decomposition + time-of-day feature가 핵심. 프로젝트 내 session 기반 전략 없음.

**출처**: Shen/Urquhart/Wang(2022) Financial Review, FMZ Quant Asian Breakout, Herman Trading (17yr NQ backtest)

**Gate 0 상세 점수**:
- 경제적 논거: 4/5
- 참신성: 5/5
- 데이터 확보: 5/5
- 구현 복잡도: 5/5
- 용량 수용: 4/5
- 레짐 독립성: 4/5

---

### 후보 #2: Liquidity-Adjusted Momentum (`liq-momentum`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Trend-Following / Liquidity Regime |
| **타임프레임** | 1H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 25/30 |
| **상태** | :red_circle: G1 FAIL — 폐기 (전 에셋 Sharpe 음수, MDD ~100%) |

**핵심 가설**: Momentum 시그널의 유효성은 liquidity 상태에 따라 극적으로 변화. Low-liquidity 환경에서 price discovery 지연 → momentum 지속 시간 증가.

**경제적 논거**: Kyle(1985) model — liquidity가 낮으면 informed trader의 정보가 가격에 느리게 반영되어 momentum 지속. Amihud illiquidity measure와 momentum return 간 양의 상관 실증. 주말/야간 thin market에서 momentum amplification 확인.

**사용 지표**: Relative Volume (168H median), Amihud Illiquidity Ratio, 12H TSMOM, Realized Volatility

**시그널 생성 로직**:
```
1. Relative Volume = vol_1h / rolling_median(vol, 168H)
2. Amihud = |return_1h| / volume_1h (rolling 24H mean)
3. Liquidity state:
   - LOW: rel_vol < 0.5 OR Amihud > 75th percentile
   - HIGH: rel_vol > 1.5 AND Amihud < 25th percentile
4. TSMOM signal: sign(rolling_return_12H) * vol_target / realized_vol
5. Conviction scaling:
   - LOW liquidity: weight * 1.5 (momentum amplification)
   - HIGH liquidity: weight * 0.5 (MR risk)
6. Weekend flag: SAT/SUN → additional 1.2x multiplier
```

**CTREND 상관 예측**: 낮음 (1H liquidity regime vs 1D ML ensemble)

**예상 거래 빈도**: 50~120건/년

**차별화 포인트**: 기존 tsmom/enhanced-tsmom/vw-tsmom은 fixed lookback + vol-target. 이 전략은 liquidity regime에 따라 momentum conviction을 dynamic하게 조절. Amihud ratio + relative volume 조합은 프로젝트 미탐색 영역.

**출처**: Kyle(1985), Chu et al.(2020) RIBAF, Tzouvanas et al.(2020), Weekend Effect in Crypto(ACR 2023)

**Gate 0 상세 점수**:
- 경제적 논거: 5/5
- 참신성: 4/5
- 데이터 확보: 5/5
- 구현 복잡도: 4/5
- 용량 수용: 3/5
- 레짐 독립성: 4/5

---

### 후보 #3: Flow Imbalance (`flow-imbalance`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Microstructure / Order Flow Proxy |
| **타임프레임** | 1H |
| **ShortMode** | FULL |
| **Gate 0 점수** | 23/30 |
| **상태** | :red_circle: G1 FAIL — 폐기 (전 에셋 Sharpe 음수, BVC 방향 예측 불가) |

**핵심 가설**: 1H bar 내 close 위치(bar position)로 buying/selling pressure를 추정하고, 누적 OFI(Order Flow Imbalance) divergence로 방향을 예측.

**경제적 논거**: Informed trader 진입 시 order flow가 편향됨. Bar 내 close position이 buying/selling pressure의 proxy (BVC 이론). VPIN 상승은 informed trading 증가를 의미하며 큰 가격 변동 임박 신호. 1H 해상도는 1D 대비 24x 정밀한 flow 추정 가능.

**사용 지표**: Bar Position (close-low)/(high-low), OFI (6H rolling), VPIN proxy (24H rolling std of buy_ratio), Volume

**시그널 생성 로직**:
```
1. Buy ratio = (close - low) / (high - low)  → [0, 1]
2. Buy_vol = volume * buy_ratio
3. Sell_vol = volume * (1 - buy_ratio)
4. OFI = rolling_sum(buy_vol - sell_vol, 6H) / rolling_sum(volume, 6H)
5. VPIN proxy = rolling_std(buy_ratio, 24H)
6. Entry (shift(1) 적용):
   - OFI > 0.6 AND VPIN > threshold: long (strong buy pressure)
   - OFI < -0.6 AND VPIN > threshold: short (strong sell pressure)
7. Exit: |OFI| < 0.2 또는 24H timeout
```

**CTREND 상관 예측**: 낮음 (microstructure flow vs ML trend features)

**예상 거래 빈도**: 80~150건/년

**차별화 포인트**: vpin-flow(FAIL)는 1D OHLCV에서 BVC → VPIN threshold 0.7이 max 0.45로 도달 불가. 1H에서는 24x 데이터로 BVC 정밀도 대폭 향상. OFI 방향성 시그널 추가 (기존은 toxicity 감지만). Flow direction + activity gate 이중 필터.

**출처**: Al-Carrion(2020) BVC, Anastasopoulos(2024) Crypto Order Flow, ScienceDirect(2025) Bitcoin Order Flow Toxicity

**Gate 0 상세 점수**:
- 경제적 논거: 4/5
- 참신성: 4/5
- 데이터 확보: 5/5
- 구현 복잡도: 3/5
- 용량 수용: 3/5
- 레짐 독립성: 4/5

---

### 후보 #4: Hour Seasonality Overlay (`hour-season`)

| 항목 | 내용 |
|------|------|
| **카테고리** | Structural / Seasonality |
| **타임프레임** | 1H |
| **ShortMode** | HEDGE_ONLY |
| **Gate 0 점수** | 22/30 |
| **상태** | :red_circle: G1 FAIL — 폐기 (전 에셋 Sharpe 음수, 계절성 비정상) |

**핵심 가설**: 22:00-23:00 UTC에 통계적으로 유의한 positive return anomaly 존재. 시간대별 return 패턴을 기존 전략의 conviction overlay로 활용.

**경제적 논거**: 주요 시장 closed 시간대에 retail flow가 지배하며 systematic buying pressure 발생. EU-US overlap(16-17 UTC)에서 가장 효율적 가격 발견. NYSE 운영 여부가 crypto intraday return 구조에 영향 (coupling effect).

**사용 지표**: Hour-of-Day Return t-stat (30d rolling), Relative Volume, NYSE Open/Closed flag

**시그널 생성 로직**:
```
1. Rolling 30일 window로 hour-of-day별 평균 return 계산
2. Hour score = mean_return / stderr → t-stat
3. Entry (단독 모드):
   - Current hour score > +2.0: long bias
   - Current hour score < -2.0: short bias
4. Overlay 모드 (기존 전략과 결합):
   - favorable hour: position size * 1.2
   - unfavorable hour: position size * 0.8
5. NYSE open/closed binary feature로 regime 구분
6. Volume confirmation: high-volume hour의 signal만 신뢰
```

**CTREND 상관 예측**: 낮음 (time structure vs price features)

**예상 거래 빈도**: 단독 150~250건/년, overlay 시 추가 비용 없음

**차별화 포인트**: 프로젝트 내 time-of-day를 feature로 사용하는 전략이 전무. 단독 alpha보다 기존 전략의 overlay/filter로 사용 시 포트폴리오 수준 Sharpe 개선 기대. Vojtko(2023)의 simple 21-23 UTC strategy: 연 33%, MDD -22%.

**출처**: Vojtko/Javorska(2023 SSRN #4581124), Seo/Chai(2024 IRFE), QuantPedia Seasonal Anomalies, Mesicek/Vojtko(2025 SSRN #5748642)

**Gate 0 상세 점수**:
- 경제적 논거: 3/5
- 참신성: 5/5
- 데이터 확보: 5/5
- 구현 복잡도: 5/5
- 용량 수용: 2/5
- 레짐 독립성: 2/5

---
