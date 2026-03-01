# 개선 로드맵

미착수 개선 항목 요약. 완료 항목은 정식 문서로 이동 완료.

**갱신일**: 2026-03-01

---

## 실행 품질 + 리스크 개선

> 원본: execution-and-risk-improvements.md (삭제됨)
> 완료 항목: SmartExecutor → [`docs/architecture/smart-executor.md`](../architecture/smart-executor.md)

| # | 항목 | 핵심 효과 | 난이도 | 상태 |
|---|------|----------|--------|------|
| ~~1~~ | ~~Limit Order~~ | ~~비용 40-50% 절감~~ | ~~중간~~ | ✅ 완료 |
| 2 | 동적 슬리피지 모델 | 백테스트 정확도 향상 | 쉬움 | 미착수 |
| 3 | Multi-TF Fusion | 4H/8H를 보조 TF로 활용 | 중간-높음 | 미착수 |
| 4 | Funding Rate 위험 감지 | 과열/폭락 사전 방어 | 쉬움 | 미착수 |
| 5 | OI(미결제약정) 경고 | 연쇄 청산 사전 감지 | 쉬움 | 미착수 |
| 6 | Alpha Decay 모니터링 | 전략 수명 관리 | 쉬움 | 미착수 |

### 권장 순서

```text
Phase 1 (1주):  #4 FR 위험감지 + #5 OI 경고 + #6 Alpha Decay
Phase 2 (1주):  #2 동적 슬리피지
Phase 3 (2주):  #3 Multi-TF Fusion (실험적)
```

### 핵심 설계 요약

**#2 동적 슬리피지**: `dynamic_slippage = base × asset_factor × vol_factor`

- BTC 0.7x, DOGE 1.5x, vol 0.5x~3.0x 범위

**#4+#5 파생상품 리스크 모니터**: `DerivativesRiskMonitor` 클래스

- FR 등급: 정상(<0.03%) / 경고(0.03~0.05%) / 위험(0.05~0.10%) / 극단(>0.10%)
- OI 등급: 90일 percentile 80/90/95th + 24h 변화율
- FR+OI+저변동성 3중 경고 시 포지션 즉시 축소

**#6 Alpha Decay**: `AlphaDecayMonitor` — 30/60/90일 Rolling Sharpe 기반 건강 상태

- 건강(>0.5) / 경고(0~0.5) / 위험(<0, 30d) / 사망(<0, 60d)

---

## Taker Buy Base Volume 인프라

> 원본: taker-buy-volume-infra.md (삭제됨)

| Phase | 항목 | 상태 |
|-------|------|------|
| ~~1~~ | ~~데이터 모델 + 지표 (`taker_cvd`, `taker_buy_ratio`)~~ | ✅ 완료 |
| 2 | API 캡처 (`publicGetKlines` column 9) | 미착수 |
| 3 | Storage 파이프라인 (Bronze/Silver) | 미착수 |
| 4 | EDA 통합 (BarEvent, CandleAggregator) | 미착수 |
| 5 | 데이터 재수집 (5 에셋) | 미착수 |
| 6 | CVD 전략 통합 | 미착수 |

### 핵심 포인트

- Binance klines column 9 (`taker_buy_base_volume`) — 추가 API 비용 없음
- `OHLCVCandle.taker_buy_base_volume: Decimal | None = None` (역호환)
- `BinanceClient.fetch_ohlcv_extended()` 신규 메서드 필요
- LiveDataFeed: MVP에서는 None (CCXT `watch_ohlcv()` 미지원)
