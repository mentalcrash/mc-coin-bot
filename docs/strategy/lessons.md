## 핵심 교훈

| # | 교훈 |
|---|------|
| 1 | **앙상블 > 단일지표**: ML 앙상블(CTREND)의 낮은 Decay(33.7%)가 단일 팩터 전략 대비 일반화 우수 |
| 2 | **IS Sharpe ≠ 실전 성과**: Gate 1 PASS 전략 24개 중 Gate 4까지 도달한 전략은 4개뿐 |
| 3 | **SOL/USDT = Best Asset**: 높은 변동성 + 추세 지속성이 모멘텀/앙상블 전략에 유리 |
| 4 | **CAGR > 20% 필터의 위력**: 안정적이나 수익 낮은 전략 (Donchian +10.8%, BB-RSI +4.6%) 조기 제거 |
| 5 | **PBO 이중 경로**: 경로 A (PBO<40%) 또는 경로 B (PBO<80% + CPCV 전fold OOS>0 + MC p<0.05). 기저 alpha 견고하면 파라미터 순위 역전은 허용 |
| 6 | **다양성이 알파**: 단일코인 < 멀티에셋, 단일지표 < 앙상블 |
| 7 | **데이터 해상도 = 전략 해상도**: 마이크로스트럭처 전략(VPIN)은 tick/volume bar 데이터 필수. 1D OHLCV에서 BVC 근사는 정보 손실 심각 |
| 8 | **G0A 이론 ≠ G1 실전**: AC-Regime(27/30), VR-Regime(24/30) 등 G0A 고득점이 G1 백테스트 성과를 보장하지 않음. 39개 전략 중 G5 도달 1개(2.6%) |
| 9 | **통계적 검정 전략의 한계**: significance threshold (z=1.96)가 일봉 데이터에서 거래 빈도를 극단 제한 (BTC 6년간 2건). 학술적 엄밀성 ≠ 실용성 |
| 10 | **밈코인 FULL Short = 구조적 자살**: VWAP-Disposition DOGE MDD -622%. ShortMode.FULL + 밈코인 급등 = 치명적. DOGE 제외 시 SOL Sharpe 0.96 |
| 11 | **칼만 필터 ≠ 알파**: 학술적 최적 노이즈 분리가 크립토 1D에서 MA 대비 우위 없음. vel_threshold가 에셋 변동성에 미적응 (DOGE 거래 0건) |
| 12 | **레짐 필터의 양면성**: Entropy 필터가 거래 중단→리스크 감소에는 기여하나, alpha 생성 메커니즘 없으면 수익도 함께 감소. 43개 전략 중 G5 도달 여전히 1개(2.3%) |
| 13 | **FX Session Edge ≠ Crypto Edge** *(1H 재검증 확정)*: Asian session breakout은 FX 시장의 institutional flow 시간대 분리에 기반. 1H 재검증에서도 전 에셋 Sharpe 음수 (-0.40~-2.52). 크립토 24/7 시장에서 session 분리는 구조적으로 무효 |
| 14 | **Amihud Illiquidity** *(1H 재검증 확정)*: 1H에서도 Sharpe -0.93~-2.78, MDD 94~100%. Amihud 상태 전환이 너무 빈번 (6K+ trades/6yr), 모멘텀 conviction 확대가 역효과. Equity 미시구조 지표 → 크립토 전이 불가 |
| 15 | **BVC 근사의 TF 불변 한계** *(1H 재검증 확정)*: 1H에서도 전 에셋 Sharpe 음수 (-0.24~-1.56). OHLCV 기반 BVC 근사는 1D→1H 해상도 상승에도 order flow 방향성 예측력 부재. **L2 data 없이 flow 기반 전략은 근본적으로 불가** |
| 16 | **Intraday 계절성** *(1H 재검증 확정)*: 1H에서 Best ETH Sharpe 0.26, 3/5 에셋 음수. Trades 22~78건으로 극소. 30일 rolling t-stat은 크립토 1H에서도 과적합 (비정상 intraday 패턴) |
| 17 | **4H/1H TF 전략의 1D 백테스트 구조적 불일치 → 재검증 완료**: VBT CLI `timeframe="1D"` 하드코딩 수정 후 8종 재검증 (2026-02-12). **8개 전략 모두 올바른 TF에서도 G1 FAIL**. TF 불일치는 일부 전략의 수치를 왜곡했으나, 근본적 alpha 부재는 TF와 무관 |
| 18 | **FULL Short + OU MR = 최악의 조합**: OU z-score 기반 short이 크립토 밈코인(DOGE) 급등에 노출 시 MDD -19,669%. VWAP-Disposition(-622%)의 31.6배. MR 전략에서 FULL short는 구조적 자살, HEDGE_ONLY가 최소 방어선 |
| 19 | **6H TF Acceleration/Momentum = 구조적 실패**: Accel-Conv(-1.05~-2.42), QD-Mom(-0.51~-2.10) 전 에셋 Sharpe 음수. 6H bar에서 2차 미분(acceleration)은 노이즈 과민, quarter-day autocorrelation은 크립토 24/7에서 부재. 과다 거래(연 400~760건) → 비용 drag 45~77% |
| 20 | **PBO 이중 경로 도입 (2026-02-11)**: PBO < 40% 단일 기준 → 이중 경로 완화. 경로 B: PBO < 80% AND CPCV 전 fold OOS > 0 AND MC p < 0.05. CTREND(PBO 60%) + Anchor-Mom(PBO 80%) 모두 경로 B PASS. **CPCV 전 fold OOS 양수 여부가 PBO 수치보다 실전 위험의 더 좋은 지표** |
| 21 | **4H Reversal 전략의 크립토 부적합 (2026-02-12 확정)**: Candle-Reject(-0.71~-1.62), Vol-Climax(-0.05~-1.73), OU-MeanRev(-0.31~-1.18) 전 에셋 Sharpe 음수. 4H bar에서 rejection wick, volume climax, OU mean reversion 시그널 모두 크립토 추세 지속성에 패배. **반전(MR) 전략은 크립토에서 구조적 불리 — 추세추종/앙상블만 생존** |
| 22 | **50개 전략 중 G5 도달 2개 (4.0%)**: 재검증 8종 포함 48개 폐기 확정. **alpha 생성은 ML 앙상블(CTREND) + 멀티 모멘텀(Anchor-Mom)에 집중**. 단일 지표/반전/미시구조 전략은 크립토 OHLCV에서 체계적으로 실패 |
