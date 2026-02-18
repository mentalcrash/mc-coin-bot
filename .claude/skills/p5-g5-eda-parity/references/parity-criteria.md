# Parity 정량 기준 + 괴리 원인 카탈로그

> Gate 5 EDA Parity 검증의 상세 판정 기준과 VBT-EDA 괴리 원인을 체계적으로 분류한다.

---

## 1. 정량 판정 기준

### 핵심 기준 (PASS/FAIL)

| # | 조건 | 기준 | 설명 |
|---|------|------|------|
| P1 | **수익 부호 일치** | VBT 양수 → EDA 양수 | 가장 중요한 기준. 부호 불일치 = 실행 로직 결함 |
| P2 | **수익률 편차** | < 20% | `|eda_return - vbt_return| / max(|vbt_return|, 1.0) × 100` |
| P3 | **거래 수 비율** | 0.5x ~ 2.0x | `eda_trades / vbt_trades`. 구조적 사유 시 예외 허용 |

### 보조 기준 (참고, PASS/FAIL에 직접 영향 없음)

| # | 지표 | 정상 범위 | 주의 범위 | 경고 범위 |
|---|------|----------|----------|----------|
| A1 | Sharpe 편차 | < 30% | 30-50% | > 50% |
| A2 | MDD 편차 | < 10%p | 10-20%p | > 20%p |
| A3 | Win Rate 편차 | < 10%p | 10-15%p | > 15%p |
| A4 | Profit Factor 편차 | < 30% | 30-50% | > 50% |

### 편차 계산 공식

```python
# 수익률 편차 (%)
return_deviation = abs(eda_return - vbt_return) / max(abs(vbt_return), 1.0) * 100

# 거래 수 비율
trade_ratio = eda_trades / max(vbt_trades, 1)

# Sharpe 편차 (%)
sharpe_deviation = (eda_sharpe - vbt_sharpe) / max(abs(vbt_sharpe), 0.01) * 100

# MDD 편차 (%p, 절대)
mdd_deviation = eda_mdd - vbt_mdd

# Win Rate 편차 (%p, 절대)
winrate_deviation = eda_winrate - vbt_winrate
```

---

## 2. VBT vs EDA 구조적 차이 (불가피한 괴리 원인)

### 2-1. 실행 타이밍 차이

| 구분 | VBT | EDA | 영향 |
|------|-----|-----|------|
| **시그널 생성** | 전체 벡터 연산 | bar-by-bar incremental | 동일 (fast mode) |
| **체결 시점** | `shift(1)` → 다음 bar open 가정 | deferred execution → 다음 TF bar open | 미세 차이 |
| **SL/TS 발동** | 벡터화 price 비교 | 1m intrabar 체크 또는 TF bar 단위 | MDD 차이의 주 원인 |

**결론**: VBT는 이상적 체결, EDA는 현실적 체결. EDA가 더 보수적.

### 2-2. PM/RM 방어 메커니즘 (EDA only)

| 메커니즘 | VBT 적용 | EDA 적용 | 효과 |
|----------|:--------:|:--------:|------|
| **rebalance_threshold** | X | O | 미세 거래 필터링 → 비용 절감 → 수익 향상 |
| **max_order_size** | X | O | 과대 주문 거부 → 거래 수 감소 |
| **system_stop_loss** | X | O | 자본 방어 → MDD 개선 가능 |
| **trailing_stop (ATR)** | △ (벡터화) | O (이벤트) | MDD 방어 → drawdown 감소 |
| **circuit_breaker** | X | O | 극단 DD 시 전량 청산 |

**결론**: EDA의 PM/RM은 VBT에 없는 추가 방어. 대부분 수익 향상 또는 리스크 감소 방향.

### 2-3. 비용 처리 차이

| 항목 | VBT | EDA |
|------|-----|-----|
| **거래 수수료** | 정적 적용 (CostModel) | 동일 CostModel 사용 |
| **Funding Rate** | 일괄 차감 (IS/OOS 분할 시) | bar별 funding drag 적용 |
| **Slippage** | 정적 % 가정 | 동일 % 가정 |

**결론**: 비용 처리는 거의 동일. funding rate bar별 적용이 미세 차이 유발 가능.

---

## 3. 괴리 원인 카탈로그

### 카탈로그 A: EDA > VBT (EDA가 더 높은 수익)

| # | 원인 | 빈도 | 심각도 | 설명 |
|---|------|:----:|:------:|------|
| A1 | PM rebalance threshold | 높음 | 낮음 | 불필요한 거래 제거 → 비용 절감. **정상** |
| A2 | Trailing stop 효과 | 중간 | 낮음 | MDD 방어로 복구 시간 단축 → CAGR 향상. **정상** |
| A3 | RM 주문 거부 | 낮음 | 낮음 | 과도한 레버리지 주문 필터 → 리스크 감소. **정상** |
| A4 | Signal pre-computation | 낮음 | 중간 | fast mode에서 전체 데이터 시그널 → VBT와 동일해야 하지만, PM/RM 차이로 최종 수익 다름 |

### 카탈로그 B: EDA < VBT (EDA가 더 낮은 수익)

| # | 원인 | 빈도 | 심각도 | 설명 |
|---|------|:----:|:------:|------|
| B1 | Funding drag bar별 적용 | 중간 | 낮음 | 복리 효과로 VBT보다 약간 낮은 수익. **정상** |
| B2 | SL/TS 조기 발동 | 낮음 | 중간 | intrabar 체크로 VBT보다 먼저 stop → 재진입 시 기회비용 |
| B3 | 시그널 buffering 차이 | 낮음 | 중간 | incremental buffer의 EWM 초기화 → fast mode로 해결 |
| B4 | Batch mode equity snapshot | 낮음 | 낮음 | 멀티에셋에서 동시 주문 → 개별 체결 대비 보수적 사이징 |

### 카탈로그 C: 부호 불일치 (Critical)

| # | 원인 | 빈도 | 심각도 | 설명 |
|---|------|:----:|:------:|------|
| C1 | **시그널 계산 버그** | 드묾 | 치명적 | incremental 버퍼에서 edge effect. fast mode로 재실행 |
| C2 | **deferred fill 미스매치** | 드묾 | 치명적 | executor의 fill 시점이 VBT와 다른 bar 참조 |
| C3 | **PM 상태 오류** | 드묾 | 치명적 | 포지션 추적 drift → 잘못된 방향으로 리밸런싱 |
| C4 | **데이터 불일치** | 드묾 | 치명적 | 1m → TF aggregation이 silver 데이터와 불일치 |
| C5 | **Config 불일치** | 중간 | 높음 | VBT와 EDA에서 다른 파라미터 사용 (short_mode 등) |

---

## 4. 괴리 분석 디시전 트리

```
수익 부호 일치?
├─ No → FAIL (C1~C5 원인 조사)
│   ├─ Config 동일? → No → C5 수정 후 재실행
│   ├─ fast_mode 사용? → No → fast_mode로 재실행 (C1 해결)
│   ├─ 1m 데이터 동일? → No → C4 데이터 재수집
│   └─ 위 전부 Yes → C2/C3 코드 레벨 디버깅 필요
│
└─ Yes → 편차 분석
    ├─ 편차 < 20% → PASS
    │   └─ 원인 기록 (카탈로그 A/B 참조)
    │
    └─ 편차 >= 20% → 원인 분석
        ├─ PM threshold 필터링으로 거래 감소 → 구조적. PASS 가능
        ├─ EWM/forward_return edge effect → fast_mode로 재실행
        ├─ SL/TS 차이 → MDD 비교하여 합리성 확인
        └─ 설명 불가 → FAIL (코드 디버깅 필요)
```

---

## 5. CTREND 선례 분석

### 편차 패턴

| 지표 | 편차 | 카탈로그 | 설명 |
|------|------|---------|------|
| Sharpe +37.6% | EDA > VBT | A1 | PM threshold 필터링으로 비용 절감 |
| CAGR +77.7% | EDA > VBT | A1+A2 | 거래 감소 + trailing stop 효과 |
| MDD -28.5% | EDA < VBT | A2 | Trailing stop ATR 3.0x 방어 |
| Trades -75.0% | 0.25x | A1 | rebalance_threshold 10% 필터링 |

### 판정 근거

1. **수익 부호 일치**: 양쪽 모두 강한 양수 (97.8% vs 173.8%)
1. **편차 > 20%**: 있으나, PM/RM 구조적 차이로 설명 가능
1. **거래 수 0.25x**: 기준(0.5x) 미달이나, PM threshold의 구조적 필터링으로 예외 허용

### 시사점

- PM rebalance_threshold는 거래 수를 크게 감소시키지만, 성과 향상으로 이어짐
- 거래 수 기준(0.5x~2.0x)은 "구조적 사유 면책" 조항이 필수
- EDA > VBT는 PM/RM 효과이며, 라이브에서도 동일하게 적용됨

---

## 6. TF별 검증 주의사항

### 1D TF

- 1년 ≈ 365 TF bar, ~525,600 1m bar
- fast mode 속도: ~2-5분
- standard mode 속도: ~10-30분

### 4H TF

- 1년 ≈ 2,190 TF bar, ~525,600 1m bar
- annualization_factor: 2190
- SL/TS intrabar 체크 빈도 높음 → standard mode에서 MDD 차이 클 수 있음

### 1H TF

- 1년 ≈ 8,760 TF bar, ~525,600 1m bar
- annualization_factor: 8760
- 시그널 빈도 매우 높음 → PM threshold 효과 극대화 → 거래 수 비율 변동 큼

### 6H/12H TF

- 1년 ≈ 1,460/730 TF bar
- annualization_factor: 1460/730
- VBT CLI에서 TF 반영 확인 필수 (이전 1D 하드코딩 버그 주의)

---

## 7. 멀티에셋 검증 (참고)

Gate 5는 **단일에셋** 검증이 기본이지만, 멀티에셋 EDA 검증도 가능:

### 멀티에셋 추가 확인 사항

| 항목 | 확인 |
|------|------|
| asset_weights 동일 | VBT와 EDA에서 동일 EW 비율 사용 |
| batch mode 활성화 | len(asset_weights) > 1 →_batch_mode=True 자동 |
| flush_pending_signals | Runner에서 호출 확인 |
| common_index | 심볼 간 데이터 교집합 처리 |

멀티에셋은 G5 필수 아님. 단일에셋 PASS 후 선택적 실행.
