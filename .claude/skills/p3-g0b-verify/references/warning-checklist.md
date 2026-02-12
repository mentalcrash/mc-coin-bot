# Warning Checklist (W1-W5) — 기록용 검증 가이드

> FAIL 사유는 아니지만 **반드시 기록**하여 후속 Gate에서 참고.
> 복수 WARNING이 누적되면 전략의 로버스트성에 의문을 제기해야 함.

---

## W1: Warmup Period

### 정의

지표 계산에 필요한 최소 bar 수가 `warmup_periods()` 반환값에 정확히 반영되는가.
부족하면 NaN 구간에서 잘못된 시그널이 발생할 수 있음.

### 검증 방법

1. `config.py`의 `warmup_periods()` 메서드 확인
2. `preprocessor.py`에서 사용하는 **모든** rolling/expanding/ewm 윈도우 나열
3. 최대 윈도우 + 1 <= `warmup_periods()` 반환값인지 확인

### 예시

```python
# config.py
class MyConfig(BaseModel):
    fast_ema: int = 12
    slow_ema: int = 26
    signal_ema: int = 9
    atr_period: int = 14
    vol_lookback: int = 30

    def warmup_periods(self) -> int:
        return max(self.slow_ema, self.atr_period, self.vol_lookback) + 1
        # WARNING: MACD signal line = slow_ema + signal_ema
        # 실제 필요: slow_ema + signal_ema = 35 > vol_lookback = 30
        # 올바른 값: max(slow_ema + signal_ema, vol_lookback) + 1 = 36
```

### 체크리스트

| 확인 | 내용 |
|------|------|
| rolling(N) | warmup >= N |
| ewm(span=N) | warmup >= N (실질적 수렴에 2N 권장) |
| 중첩 rolling | rolling(A) of rolling(B) -> warmup >= A + B |
| numba 함수 | 내부 루프 초기 구간 확인 |
| shift(1) | warmup에 +1 추가 |

---

## W2: Parameter Count

### 정의

자유 파라미터 수가 거래 수 대비 과다하면 과적합 위험 증가.
Rule of thumb: **Trades / Params > 10**

### 검증 방법

1. `config.py`에서 전략 로직에 영향을 주는 파라미터 수 세기
2. 공통 파라미터(vol_target, min_volatility 등)는 **제외** (PM에서 관리)
3. 전략 고유 파라미터만 카운트

### 파라미터 분류

| 분류 | 예시 | 카운트 |
|------|------|--------|
| **전략 고유** | fast_period, slow_period, threshold | 포함 |
| **공통 변동성** | vol_target, min_volatility, annualization_factor | 제외 |
| **공통 숏모드** | short_mode, hedge_threshold, hedge_strength_ratio | 제외 |
| **파생 파라미터** | warmup_periods() (다른 파라미터에서 계산) | 제외 |
| **이산 선택** | use_log_returns (bool) | 0.5로 카운트 |

### 판정

| Trades/Params | 판정 |
|--------------|------|
| > 20 | OK — 충분한 자유도 |
| 10 ~ 20 | OK — 합리적 |
| 5 ~ 10 | WARNING — 과적합 주의 |
| < 5 | WARNING (강) — Gate 3에서 파라미터 안정성 면밀 검증 필요 |

> 거래 수는 Gate 1에서 확인되므로, Gate 0B 시점에서는 **파라미터 수만 기록**.
> 예상 거래 빈도(일봉: ~50-200/6년, 시간봉: ~500-2000/6년)로 대략 추정.

---

## W3: Regime Concentration

### 정의

수익의 대부분이 특정 시장 레짐(2020-2021 상승장 등)에 집중되지 않는가.
코드 수준에서는 **구조적 편향**을 탐지.

### 검증 방법 (코드 수준)

1. **Long-only 편향**: ShortMode.DISABLED + 추세추종 = 하락장에서 손실 집중
2. **변동성 의존**: 급변동 시장에서만 시그널 발생하는 구조 (NR7 squeeze 등)
3. **레짐 필터 없음**: 모든 시장 환경에서 동일 로직 적용

### 체크리스트

| 확인 | WARNING 조건 |
|------|-------------|
| ShortMode | DISABLED이면서 추세추종 전략 -> 하락장 무방비 |
| 방향 편향 | direction이 항상 양수(Long)인 구조 |
| 시그널 빈도 | 특정 조건(극단 변동성)에서만 시그널 -> 편향 |
| 레짐 적응 | 레짐 인식 로직 없으면 기록 (FAIL 아님) |

---

## W4: Turnover

### 정의

연간 회전율이 비용 대비 합리적인가. 고빈도 전략은 비용 민감도 높음.

### 검증 방법 (코드 수준)

1. `signal.py`에서 entry 조건의 빈도 추정
2. `rebalance_threshold` 사용 여부 확인 (PM에서 필터링)
3. 매 bar entry 가능한 구조인지 확인

### 판정

| 예상 연간 거래 | 판정 |
|---------------|------|
| < 50 | WARNING — 통계적 유의성 부족 가능 |
| 50 ~ 200 | OK — 일봉 전략 적정 |
| 200 ~ 500 | OK — 시간봉 전략 적정 |
| > 500 | WARNING — 비용 민감도 높음, 비용 2배 시나리오 필수 |

### 비용 영향 추정

```
연간 거래 수 × 왕복 비용(0.22%) = 연간 비용 drag
예: 300회 × 0.22% = 66% → CAGR에서 차감
예: 50회 × 0.22% = 11% → 합리적
```

---

## W5: Correlation with Existing Strategies

### 정의

기존 활성 전략과의 수익률 상관계수가 0.7 이하인가.
높으면 포트폴리오 분산 효과 부족.

### 검증 방법 (코드 수준)

Gate 0B에서는 **코드 구조적 유사성**으로 사전 판단:

| 확인 | WARNING 조건 |
|------|-------------|
| 동일 지표 | 기존 전략과 핵심 지표가 동일 (예: 둘 다 EMA cross) |
| 동일 로직 | 시그널 생성 로직이 기존 전략의 변형 |
| 동일 시그널 소스 | 가격, 변동성, 모멘텀 등 동일 카테고리 |

### 이 프로젝트 기존 전략 카테고리 (30종)

| 카테고리 | 전략 예시 | 핵심 시그널 소스 |
|---------|----------|----------------|
| 추세추종 | TSMOM, Donchian, Breakout | 모멘텀, 채널 돌파 |
| 평균회귀 | BB-RSI, Stoch-Mom | RSI, 볼린저 밴드 |
| 변동성 | Vol-Adaptive, Vol-Structure, TTM Squeeze | ATR, 변동성 비율 |
| 레짐 | Hurst, HMM, AC-Regime, VR-Regime | 자기상관, 분산비 |
| 플로우 | VPIN-Flow | 거래량, 정보비대칭 |
| 스퀴즈 | Range-Squeeze | NR7, 레인지 비율 |
| 멀티TF | MTF-MACD | 다중 시간프레임 |

> **새 전략이 기존 카테고리와 동일 시그널 소스를 사용하면 WARNING 기록.**
> 실제 상관 계수는 Gate 1 이후 백테스트 결과로 정량 검증.

---

## 크립토 시장 특화 추가 경고

Gate 0B 필수 항목은 아니지만, 리포트에 해당 사항 기록:

### 펀딩비 민감도

- 전략이 장기 Long 편향이면 연간 펀딩비 추정 기록
- `0.01% × 3회/일 × 365일 = 10.95%/년` (평균)
- 장기 Short 편향이면 펀딩비 수취 가능 (양의 효과)

### 유동성 제약

- 소형 알트코인 대상 전략이면 유동성 경고
- BTC/ETH 외 알트코인의 대규모 주문 체결 현실성

### 24/7 시장 특성

- 일봉 기준 시간(UTC 00:00) 명확한지 확인
- 주말/공휴일 데이터 갭 처리 (크립토는 없어야 정상)
