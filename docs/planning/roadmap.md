# 개선 로드맵

Spot SuperTrend 단일 전략 운용 로드맵.

**갱신일**: 2026-03-06

---

## 현황 요약

| 지표 | 값 |
|------|-----|
| 전략 | SuperTrend v1.1 (ATR 10, mult 3.0, ADX 14, threshold 25) |
| 모드 | Long-only + Pyramid (40/35/25) |
| TF | 12H |
| Tier 1 에셋 | BTC, ETH, SOL, XRP, AVAX, FTM |
| Avg Sharpe (6에셋) | 1.104 |
| P5 Robustness | PASS (6/6 assets, 91.4% Sharpe >= 0.8) |
| 마이그레이션 | Phase 0-3 완료 (Futures Multi -> Spot Single) |

---

## Phase 5: Paper Trading (미착수)

Spot 실환경 검증 (실제 자금 투입 전).

| # | 항목 | 설명 | 난이도 |
|---|------|------|--------|
| 5-1 | Spot Paper 실행 | `spot-live` 커맨드로 paper 모드 운용 | 낮 |
| 5-2 | Stop-Limit Ratchet 검증 | SpotStopManager 안전망 동작 확인 | 중 |
| 5-3 | Reconciler 검증 | Spot One-Way 포지션 대조 | 낮 |
| 5-4 | Discord 알림 검증 | Fill/Risk/Health 알림 정상 수신 | 낮 |
| 5-5 | 24h+ 연속 운용 | 메모리 누수, WS 재연결 안정성 | 중 |

---

## Phase 6: Live Trading (미착수)

실제 자금 운용 시작.

| # | 항목 | 설명 | 난이도 |
|---|------|------|--------|
| 6-1 | 소액 라이브 ($100~500) | 단일 에셋 (BTC) 소액 검증 | 낮 |
| 6-2 | 에셋 확장 | Tier 1 에셋 순차 추가 | 낮 |
| 6-3 | 자본 확대 | Paper 대비 수익률 일치 확인 후 | 중 |

---

## 향후 개선 (우선순위)

### A. 전략 고도화

| # | 항목 | 핵심 효과 | 난이도 |
|---|------|----------|--------|
| A1 | P6 WFA + CPCV 10-fold | OOS Sharpe decay 30% 이내 확인 | 중 |
| A2 | Tier 1 포트폴리오 합산 | 6에셋 EW 합산 성과 검증 | 낮 |
| A3 | P7 EDA Parity | VBT<->EDA 편차 10% 이내 확인 | 중 |

### B. 실행 품질

| # | 항목 | 핵심 효과 | 난이도 |
|---|------|----------|--------|
| B1 | 동적 슬리피지 모델 | 백테스트 정확도 향상 | 낮 |
| B2 | Alpha Decay 모니터링 | 전략 수명 관리 | 낮 |

### C. 추가 전략 탐색

| # | 항목 | 핵심 효과 | 난이도 |
|---|------|----------|--------|
| C1 | 6H TF 탐색 | 12H/4H 사이 sweet spot | 낮 |
| C2 | On-chain Context 결합 | 12H OHLCV + 1D On-chain | 중 |

---

## 고갈 확인 영역 (비추천)

| 영역 | 시도 수 | 결론 |
|------|--------|------|
| 1D OHLCV | 92개 | 검색공간 완전 고갈 |
| 4H/8H TF | 50+개 | 구조적 비용 벽 |
| ML 전략 | 4개 | Look-ahead bias 극복 불가 |
| 대안데이터 단독 | 20+개 | On-chain/Deriv alpha 0 |
| FULL Short | 20에셋 검증 | 19/20 Sharpe 악화 |
| PM 방어 (SL+TS) | 20에셋 검증 | SuperTrend 내장 손절과 간섭 |
