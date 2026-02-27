# 실행 품질 + 리스크 개선 기획서

**상태**: 항목 #1 구현 완료 / 나머지 미착수
**작성일**: 2026-02-27
**배경**: 173개 전략 중 4H/8H 전멸 원인 분석 → 전략 아이디어 부재가 아니라 실행 비용 + 리스크 감지 부재가 근본 원인

## 배경 요약

### 현재 상황

- 12H 3개 ACTIVE (Anchor-Mom, Donch-Multi, Tri-Channel) — 이것은 올바른 결정
- 4H 42개+ 전멸, 8H 5개+ 전멸 — 비용 구조적 문제
- 업계 리서치 결과: 기관 수익의 75-90%가 execution alpha (마켓메이킹, 재정거래, 캐리)
- 리테일이 개선할 수 있는 영역: 실행 비용 절감 + 리스크 감지 강화

### 개선 6가지 요약

| # | 개선사항 | 핵심 효과 | 난이도 | 예상 기간 |
|---|---------|----------|--------|----------|
| 1 | ~~Limit Order (지정가 주문)~~ | 거래 비용 40-50% 절감 | 중간 | ✅ 완료 |
| 2 | 동적 슬리피지 모델 | 백테스트 정확도 향상 | 쉬움 | 3-5일 |
| 3 | Multi-TF Fusion | 4H/8H를 보조 TF로 활용 | 중간-높음 | 2-3주 |
| 4 | Funding Rate 위험 감지 | 과열/폭락 사전 방어 | 쉬움 | 3-5일 |
| 5 | OI(미결제약정) 경고 | 연쇄 청산 사전 감지 | 쉬움 | 3-5일 |
| 6 | Alpha Decay 모니터링 | 전략 수명 관리 | 쉬움 | 3-5일 |

### 구현 권장 순서

```
Phase 1 (1주):  #4 FR 위험감지 + #5 OI 경고 + #6 Alpha Decay
Phase 2 (1주):  #2 동적 슬리피지 + #1 Limit Order 기본
Phase 3 (2주):  #1 Limit Order 고도화 + #3 Multi-TF Fusion
```

---

## 1. Limit Order (지정가 주문)

### 1.1 개요

| 항목 | 내용 |
|------|------|
| 목적 | Market order(시장가) 대신 Limit order(지정가)를 사용하여 수수료 절감 |
| 핵심 효과 | 거래 비용 40-50% 절감 → 연 5-7%p 순수익 증가 |
| 영향 범위 | `src/eda/executors.py`, `src/eda/oms.py`, `src/portfolio/` |
| 난이도 | 중간 (1-2주) |

### 1.2 현재 상태

현재 시스템의 주문 흐름:

```
Signal → PM → OrderRequest(price=None) → OMS → LiveExecutor
  → exchange.create_market_order()  ← 항상 시장가!
  → taker fee 0.04% 지불
```

- `src/eda/executors.py`의 `LiveExecutor`는 시장가만 사용
- SL/TS 주문에서는 `price` 필드를 사용하지만, 일반 진입/청산은 항상 `price=None`
- ccxt의 `create_limit_order()`는 이미 사용 가능하지만 호출하지 않음

### 1.3 설계

#### 주문 유형 분류

```
주문 종류에 따라 실행 방식 결정:

  [긴급 주문] → 시장가 (Market Order)
    - Stop-Loss 발동
    - Trailing Stop 발동
    - Circuit Breaker (시스템 스탑로스)
    - 포지션 방향 전환 (롱→숏 / 숏→롱)

  [일반 주문] → 지정가 (Limit Order) 우선 시도
    - 신규 진입 (새 포지션 열기)
    - 리밸런싱 (포지션 크기 조정)
    - 일반 청산 (시그널 기반 청산)
```

#### SmartExecutor 로직

```python
class SmartExecutor:
    """시장가/지정가를 자동 판단하는 실행기"""

    async def execute(self, order: OrderRequestEvent) -> FillEvent:
        if self._is_urgent(order):
            # 긴급 → 즉시 시장가
            return await self._market_execute(order)

        # 일반 → 지정가 시도
        limit_price = self._calc_limit_price(order)
        fill = await self._limit_execute(order, limit_price, timeout_sec=15)

        if fill is None:
            # 15초 내 미체결 → 시장가로 전환
            await self._cancel_order(order)
            return await self._market_execute(order)

        return fill

    def _calc_limit_price(self, order):
        """현재가 대비 약간 유리한 가격으로 지정가 설정"""
        ticker = self._fetch_ticker(order.symbol)
        if order.side == "BUY":
            # 매수: 현재 매도호가(ask)에서 살짝 아래
            return ticker.ask * 0.9999  # 0.01% 유리한 가격
        else:
            # 매도: 현재 매수호가(bid)에서 살짝 위
            return ticker.bid * 1.0001
```

#### 비용 절감 계산

```
Binance Futures 수수료:
  Taker fee:  0.04%  (시장가)
  Maker fee:  0.02%  (지정가)

가정: 70% 주문이 지정가로 체결됨 (15초 타임아웃)

12H 전략 (연 146회 왕복 거래):
  현재:  146 × 2 × 0.04% = 11.68% 연간 수수료
  개선:  146 × 2 × (0.02%×0.7 + 0.04%×0.3) = 6.47%
  절감:  연 5.21%p

8H 전략 (연 219회 왕복 거래):
  현재:  219 × 2 × 0.04% = 17.52%
  개선:  219 × 2 × 0.024% = 10.51%
  절감:  연 7.01%p → 8H 전략 가능성 확대
```

### 1.4 구현 계획

| 단계 | 작업 | 파일 | 비고 |
|------|------|------|------|
| 1 | `SmartExecutor` 클래스 생성 | `src/eda/executors.py` | `LiveExecutor` 래핑 |
| 2 | 긴급/일반 주문 분류 로직 | `SmartExecutor._is_urgent()` | OrderRequest에 `urgency` 필드 추가 |
| 3 | Limit price 계산 | `SmartExecutor._calc_limit_price()` | bid/ask 기반 |
| 4 | 체결 대기 + 타임아웃 | `SmartExecutor._limit_execute()` | 폴링 또는 WebSocket |
| 5 | 미체결 시 시장가 전환 | `SmartExecutor._fallback_market()` | 주문 취소 → 재주문 |
| 6 | 부분 체결 처리 | `SmartExecutor._handle_partial()` | 잔량 시장가 전환 |
| 7 | Paper 모드 시뮬레이션 | `BacktestExecutor` 확장 | Fill probability 모델 |
| 8 | 테스트 | `tests/eda/test_smart_executor.py` | - |

### 1.5 위험 요소

| 위험 | 대응 |
|------|------|
| 15초 내 미체결 → 가격 악화 | 타임아웃 후 시장가 전환 시 슬리피지 증가 가능. 타임아웃을 5-15초 범위로 조정 가능 |
| 급변 시장에서 limit order 적체 | 변동성 급등 시 자동으로 시장가 모드 전환 (ATR 기반) |
| 부분 체결 관리 복잡도 | 99% 이상 체결 시 완료 처리, 미만 시 잔량 시장가 |

---

## 2. 동적 슬리피지 모델

### 2.1 개요

| 항목 | 내용 |
|------|------|
| 목적 | 고정 슬리피지(0.05%) → 시간/자산/변동성에 따라 변동하는 슬리피지 |
| 핵심 효과 | 백테스트 정확도 향상 → 전략 발굴/폐기 판단 개선 |
| 영향 범위 | `src/portfolio/cost_model.py`, `src/backtest/engine.py` |
| 난이도 | 쉬움 (3-5일) |

### 2.2 현재 상태

`src/portfolio/cost_model.py`:

```python
CostModel:
    slippage: float = 0.0005      # 항상 0.05%
    market_impact: float = 0.0002  # 항상 0.02%
```

문제점:
- BTC(유동성 풍부)와 DOGE(유동성 부족)에 같은 슬리피지 적용
- 새벽 3시(거래량 적음)와 오후 9시(거래량 많음)에 같은 슬리피지 적용
- 2025년 10월 폭락(슬리피지 5배 증가) 같은 극단 상황 반영 불가

### 2.3 설계

#### 슬리피지 계산 공식

```
dynamic_slippage = base_slippage × asset_factor × volatility_factor

  base_slippage:     0.03% (기본값, 현재 0.05%보다 낮음)
  asset_factor:      자산별 유동성 반영
  volatility_factor: 시장 변동성 반영
```

#### 자산별 계수 (asset_factor)

```python
ASSET_LIQUIDITY_FACTOR = {
    "BTC/USDT":  0.7,   # 가장 유동적 → 슬리피지 작음
    "ETH/USDT":  0.8,
    "BNB/USDT":  1.0,   # 기준
    "SOL/USDT":  1.2,
    "DOGE/USDT": 1.5,   # 유동성 부족 → 슬리피지 큼
}
# 미등록 자산: 1.5 (보수적)
```

#### 변동성 계수 (volatility_factor)

```python
def volatility_factor(realized_vol: float, vol_baseline: float = 0.6) -> float:
    """변동성이 높을수록 슬리피지 증가"""
    ratio = realized_vol / vol_baseline
    return max(0.5, min(3.0, ratio))  # 0.5x ~ 3.0x 범위 제한

# 예시:
#   평상시 (RV=0.6):  factor = 1.0 → 슬리피지 0.03%
#   저변동 (RV=0.3):  factor = 0.5 → 슬리피지 0.015% (과대평가 방지)
#   고변동 (RV=1.8):  factor = 3.0 → 슬리피지 0.09% (과소평가 방지)
```

#### 적용 예시

```
[BTC, 평상시]
  0.03% × 0.7 × 1.0 = 0.021%  (현재 0.05%보다 낮음 → 백테스트 수익 과소평가 해결)

[DOGE, 고변동]
  0.03% × 1.5 × 2.5 = 0.113%  (현재 0.05%보다 높음 → 위험 과소평가 방지)

[ETH, 폭락장]
  0.03% × 0.8 × 3.0 = 0.072%  (극단 상황 반영)
```

### 2.4 구현 계획

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | `DynamicCostModel` 클래스 추가 | `src/portfolio/cost_model.py` |
| 2 | `asset_factor` 매핑 테이블 | `src/portfolio/cost_model.py` |
| 3 | `volatility_factor()` 함수 | `src/portfolio/cost_model.py` |
| 4 | BacktestEngine에서 봉마다 동적 슬리피지 적용 | `src/backtest/engine.py` |
| 5 | 기존 `CostModel`과 하위 호환 유지 | `CostModel.dynamic()` 팩토리 메서드 |
| 6 | 테스트 | `tests/portfolio/test_dynamic_cost.py` |

### 2.5 하위 호환성

```python
# 기존 코드 (변경 없음)
cost = CostModel.binance_futures()  # 고정 슬리피지 (기존 방식)

# 새 코드
cost = CostModel.dynamic()          # 동적 슬리피지
cost.effective_slippage("BTC/USDT", realized_vol=0.8)  # → 봉마다 계산
```

---

## 3. Multi-TF Fusion (다중 시간대 결합)

### 3.1 개요

| 항목 | 내용 |
|------|------|
| 목적 | 12H(방향 결정) + 4H/8H(타이밍 결정) 계층적 시그널 결합 |
| 핵심 효과 | 진입 가격 개선, 4H/8H를 독립 전략이 아닌 "보조"로 활용 |
| 영향 범위 | `src/orchestrator/`, `src/eda/strategy_engine.py`, 신규 컴포넌트 |
| 난이도 | 중간-높음 (2-3주) |

### 3.2 현재 상태

현재 Orchestrator 구조:

```
Pod 1 (Anchor-Mom, 12H, DOGE) ─→ 독립 시그널 → PM → OMS
Pod 2 (Donch-Multi, 12H, BTC) ─→ 독립 시그널 → PM → OMS
Pod 3 (Tri-Channel, 12H, ETH) ─→ 독립 시그널 → PM → OMS

각 Pod가 독립적으로 판단. Pod 간 정보 교환 없음.
```

Multi-TF 인프라:

- `MultiTimeframeCandleAggregator`: 1m → 여러 TF 동시 집계 가능 (이미 구현됨)
- `StrategyEngine`: TF별 독립 전략 실행 (이미 구현됨)
- 부재: **서로 다른 TF의 시그널을 계층적으로 결합하는 컴포넌트**

### 3.3 설계

#### 핵심 개념: Direction Overlay (방향 오버레이)

```
                  ┌─────────────────────────────┐
                  │ 12H Strategy (방향 결정자)     │
                  │ "ETH는 상승 추세" (dir = +1)   │
                  └──────────┬──────────────────┘
                             │ direction 전달
                             ↓
                  ┌─────────────────────────────┐
                  │ 4H Strategy (타이밍 결정자)     │
                  │ 12H가 +1 → 매수 시그널만 허용   │
                  │ 12H가 -1 → 매도 시그널만 허용   │
                  │ 12H가  0 → 거래 안 함          │
                  └─────────────────────────────┘
```

#### 구현 방식 비교

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **A. Direction Filter** | 12H 방향과 일치하는 4H 시그널만 통과 | 단순, 안전 | 4H 독자 판단 제한 |
| **B. Entry Timing** | 12H가 "매수"면 4H 눌림목에서 진입 | 더 좋은 가격 | 기회 놓칠 수 있음 |
| **C. Confidence Weight** | 12H 강도로 4H 포지션 크기 조절 | 유연함 | 복잡 |

권장: **방식 A (Direction Filter)** — 가장 단순하고 검증하기 쉬움

#### Direction Filter 상세 로직

```python
class DirectionOverlay:
    """12H 방향을 4H 전략에 전달하는 오버레이"""

    def filter_signal(
        self,
        parent_direction: int,   # 12H 전략의 방향 (+1/0/-1)
        child_signal: SignalEvent,  # 4H 전략의 시그널
    ) -> SignalEvent | None:
        """12H 방향과 일치하는 4H 시그널만 통과"""

        if parent_direction == 0:
            # 12H가 중립 → 4H 거래 금지
            return None

        if child_signal.direction == parent_direction:
            # 방향 일치 → 통과
            return child_signal

        if child_signal.direction == 0:
            # 4H가 청산 시그널 → 항상 통과 (포지션 정리는 허용)
            return child_signal

        # 방향 불일치 → 차단
        return None
```

#### Orchestrator 확장

```
현재:
  Pod 1 (독립) ──→ Netting ──→ PM
  Pod 2 (독립) ──→ Netting ──→ PM
  Pod 3 (독립) ──→ Netting ──→ PM

개선:
  [Parent Pod] 12H Tri-Channel (ETH) ──→ direction = +1
                                              │
                                              ↓ Direction Filter
  [Child Pod]  4H Timing (ETH)  ──→ 매수 시그널만 통과 ──→ PM
```

### 3.4 구현 계획

| 단계 | 작업 | 파일 | 비고 |
|------|------|------|------|
| 1 | `DirectionOverlay` 컴포넌트 | `src/orchestrator/direction_overlay.py` | 신규 |
| 2 | Pod간 parent-child 관계 설정 | `src/orchestrator/pod.py` 확장 | config에 `parent_pod_id` 추가 |
| 3 | Parent direction 구독 | `src/orchestrator/orchestrator.py` | Parent SignalEvent 캐시 |
| 4 | 4H Timing 전략 구현 | `src/strategy/timing_entry_4h/` | 신규 — 눌림목 감지 |
| 5 | CandleAggregator 멀티 TF 설정 | config YAML | 12H + 4H 동시 집계 |
| 6 | 백테스트 지원 | `src/backtest/engine.py` 확장 | 멀티 TF 시뮬레이션 |
| 7 | 테스트 | `tests/orchestrator/test_direction_overlay.py` | - |

### 3.5 기대 효과

```
12H만 사용 (현재):
  시그널 발생 → 12H 봉 시가에 진입 (50,000달러)
  → 12시간 동안의 최적 가격을 놓침

12H + 4H 결합 (개선):
  12H 시그널 "매수" → 4H에서 눌림목 대기
  → 4H 봉에서 EMA 아래로 내려올 때 매수 (49,500달러)
  → 진입 가격 1% 개선

  연간 146회 거래 × 평균 0.5% 가격 개선 = 연 3.6%p 추가 수익
```

### 3.6 위험 요소

| 위험 | 대응 |
|------|------|
| 눌림목이 안 와서 진입 못 함 | 타임아웃: 12H 시그널 후 N봉(예: 6봉=24시간) 내 미진입 시 시장가 진입 |
| 4H 노이즈로 잘못된 타이밍 | Direction Filter로 방향은 12H가 결정하므로 방향 실수는 없음 |
| 복잡도 증가 | Phase 1에서는 Direction Filter만 구현 (가장 단순한 방식) |

---

## 4. Funding Rate 위험 감지

### 4.1 개요

| 항목 | 내용 |
|------|------|
| 목적 | 비정상적 펀딩레이트 감지 → 포지션 크기 자동 축소로 폭락 방어 |
| 핵심 효과 | MDD 방어 (2025.10 같은 폭락에서 손실 최소화) |
| 영향 범위 | `src/eda/risk_manager.py`, `src/eda/derivatives_feed.py` |
| 난이도 | 쉬움 (3-5일) |

### 4.2 현재 상태

데이터 인프라:

```
✅ DerivativesDataService: funding_rate 데이터 수집 가능 (Silver _deriv 파일)
✅ LiveDerivativesFeed: 실시간 funding_rate polling (8h 주기)
✅ merge_asof: OHLCV와 자동 병합 가능

❌ RiskManager: funding_rate를 전혀 참조하지 않음
❌ PM: funding_rate 기반 포지션 크기 조절 없음
❌ 경고/알림: funding_rate 이상 시 알림 없음
```

### 4.3 설계

#### 위험 등급 체계

```
펀딩레이트(FR) 절대값 기준 (8h당):

  [정상]  |FR| < 0.03%
    → 전략 정상 운영
    → 포지션 크기 100% 유지

  [경고]  0.03% ≤ |FR| < 0.05%
    → 포지션 크기를 원래의 50%로 축소
    → Discord 알림: "⚠️ {symbol} FR {value}% — 포지션 50% 축소"

  [위험]  0.05% ≤ |FR| < 0.10%
    → 포지션 크기를 원래의 25%로 축소
    → 신규 진입 금지 (기존 포지션만 유지)
    → Discord 알림: "🚨 {symbol} FR {value}% — 신규 진입 차단"

  [극단]  |FR| ≥ 0.10%
    → 기존 포지션도 청산 검토 (사용자 확인 후)
    → Discord 알림: "🔴 {symbol} FR {value}% — 전 포지션 검토 필요"
```

#### 방향 경고 (추가)

```
FR > 0 (롱이 숏에게 지불) + 내 포지션이 롱:
  → "나도 롱 군중의 일부. 과열 경고!"
  → 포지션 축소 강화 (위 등급의 한 단계 위 적용)

FR < 0 (숏이 롱에게 지불) + 내 포지션이 숏:
  → "나도 숏 군중의 일부. 숏 스퀴즈 경고!"
  → 포지션 축소 강화
```

#### RiskManager 통합

```python
# src/eda/risk_manager.py 확장

class EDARiskManager:

    def _check_funding_rate_risk(
        self, symbol: str, direction: int
    ) -> float:
        """펀딩레이트 기반 포지션 크기 배율 반환 (0.0 ~ 1.0)"""
        fr = self._latest_funding_rate.get(symbol, 0.0)
        abs_fr = abs(fr)

        # 기본 등급
        if abs_fr < 0.0003:    # 0.03%
            scale = 1.0
        elif abs_fr < 0.0005:  # 0.05%
            scale = 0.5
        elif abs_fr < 0.001:   # 0.10%
            scale = 0.25
        else:
            scale = 0.1

        # 방향 일치 시 추가 축소
        if (fr > 0 and direction == 1) or (fr < 0 and direction == -1):
            scale *= 0.5  # "군중과 같은 방향" → 추가 50% 축소

        return scale
```

### 4.4 구현 계획

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | `_latest_funding_rate` 캐시 | `src/eda/risk_manager.py` |
| 2 | `LiveDerivativesFeed` → RiskManager 연결 | `src/eda/live_runner.py` |
| 3 | `_check_funding_rate_risk()` 로직 | `src/eda/risk_manager.py` |
| 4 | PM vol_scalar에 FR scale 적용 | `src/eda/portfolio_manager.py` |
| 5 | Discord 알림 | `src/notification/` |
| 6 | 백테스트 지원 (Silver _deriv 데이터 활용) | `src/backtest/engine.py` |
| 7 | 테스트 | `tests/eda/test_fr_risk.py` |

### 4.5 실제 사례 시뮬레이션

```
2025년 10월 대폭락 시나리오:

  10/1: FR = 0.06% → [위험] → 포지션 25% 축소
  10/2: FR = 0.08% → [위험] → 포지션 25% 유지, 신규 진입 차단
  10/3: $3.21B 청산 발생 → 가격 급락
        → 우리 손실: 원래의 25%만 (포지션이 축소되어 있으므로)

  FR 경고 없었을 때: MDD -40% (예시)
  FR 경고 사용 시:   MDD -10% (75% 방어)
```

---

## 5. OI(미결제약정) 경고

### 5.1 개요

| 항목 | 내용 |
|------|------|
| 목적 | OI 과열/급감 감지 → 연쇄 청산 사전 대비 |
| 핵심 효과 | Liquidation cascade(연쇄 청산) 사전 감지 및 포지션 방어 |
| 영향 범위 | `src/eda/risk_manager.py`, `src/eda/derivatives_feed.py` |
| 난이도 | 쉬움 (3-5일, FR 경고와 동시 구현) |

### 5.2 현재 상태

```
✅ DerivativesDataService: open_interest 데이터 수집 가능
✅ LiveDerivativesFeed: 실시간 OI polling (1h 주기)

❌ OI 데이터를 리스크 판단에 사용하지 않음
❌ OI 이상 감지 없음
```

### 5.3 설계

#### OI 위험 지표

```
지표 1: OI Percentile (장기 과열 감지)

  OI의 90일 rolling percentile 계산
  → 현재 OI가 최근 90일 중 몇 % 위치인지

  [정상]    < 80th percentile  → "빚이 평균 수준"
  [주의]    80-90th percentile → "빚이 많아지고 있음"
  [경고]    90-95th percentile → "역대급 빚! 포지션 50% 축소"
  [극단]    > 95th percentile  → "역대 최고 수준! 포지션 25% 축소"


지표 2: OI 급변 (단기 위험 감지)

  OI의 24시간 변화율 계산

  [급증]   24h OI 변화 > +15% → "급격한 레버리지 증가, 경계"
  [급감]   24h OI 변화 < -20% → "연쇄 청산 진행 중! 신규 진입 금지"
```

#### FR + OI 복합 경고

```
3중 경고 조건 (가장 위험):
  ① OI > 90th percentile (빚이 역대급)
  ② FR > 0.05% (과열)
  ③ Realized Volatility < 30th percentile (변동성 극저 = 폭풍 전 고요)

  → 3개 모두 충족 시: "🔴 3중 경고! 포지션 즉시 축소"
  → 2025년 10월 대폭락 직전에 정확히 이 패턴이 나타남
```

#### 구현 코드 구조

```python
class DerivativesRiskMonitor:
    """FR + OI 복합 리스크 모니터"""

    def compute_risk_scale(
        self, symbol: str, direction: int
    ) -> float:
        """0.0 ~ 1.0 범위의 포지션 크기 배율"""
        fr_scale = self._fr_risk_scale(symbol, direction)
        oi_scale = self._oi_risk_scale(symbol)

        # 두 지표 중 더 보수적인 값 사용
        return min(fr_scale, oi_scale)

    def _oi_risk_scale(self, symbol: str) -> float:
        oi_pctl = self._oi_percentile(symbol, window=90)
        oi_change_24h = self._oi_change_rate(symbol, hours=24)

        # 급감 = 연쇄 청산 진행 중
        if oi_change_24h < -0.20:
            return 0.0  # 신규 진입 금지

        # 장기 과열
        if oi_pctl > 0.95:
            return 0.25
        if oi_pctl > 0.90:
            return 0.50
        if oi_pctl > 0.80:
            return 0.75

        return 1.0
```

### 5.4 구현 계획

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | `DerivativesRiskMonitor` 클래스 | `src/eda/deriv_risk_monitor.py` (신규) |
| 2 | OI 90일 percentile 계산 | 위 클래스 내부 |
| 3 | OI 24h 변화율 계산 | 위 클래스 내부 |
| 4 | FR 경고 (#4)와 통합 | 위 클래스에서 FR + OI 복합 |
| 5 | RiskManager 연동 | `src/eda/risk_manager.py` |
| 6 | Discord 3중 경고 알림 | `src/notification/` |
| 7 | 백테스트 지원 | Silver _deriv OI 데이터 활용 |
| 8 | 테스트 | `tests/eda/test_deriv_risk_monitor.py` |

---

## 6. Alpha Decay 모니터링

### 6.1 개요

| 항목 | 내용 |
|------|------|
| 목적 | 전략 성과가 점점 악화되는 것을 조기 감지 → 전략 교체 타이밍 결정 |
| 핵심 효과 | 전략 수명 관리, 손실 누적 방지 |
| 영향 범위 | 신규 모니터링 컴포넌트, `src/notification/` |
| 난이도 | 쉬움 (3-5일) |

### 6.2 현재 상태

```
❌ 전략 성과를 추적하는 시스템 없음
❌ Rolling Sharpe 계산 없음
❌ 전략 건강 상태 대시보드 없음
❌ 성과 저하 시 알림 없음

현재 흐름:
  전략 구현 → 백테스트 PASS → 라이브 투입 → (아무 모니터링 없음)
  → 큰 손실 발생 후 수동으로 "이 전략 별로네" 판단
```

### 6.3 설계

#### 건강 상태 체계

```
Rolling Sharpe 기반 상태 판정:

  [🟢 건강]  30일 Sharpe > 0.5
    → "전략이 잘 작동 중"
    → 조치 없음

  [🟡 경고]  0.0 ≤ 30일 Sharpe ≤ 0.5
    → "수익이 줄어들고 있음"
    → Discord 알림
    → 원인 분석 권장 (시장 레짐 변화? alpha decay?)

  [🟠 위험]  30일 Sharpe < 0.0
    → "전략이 손실 중!"
    → Discord 알림
    → 포지션 크기 자동 50% 축소 (선택적)

  [🔴 사망]  60일 Sharpe < 0.0
    → "2개월째 손실, 전략 교체 필요"
    → Discord 긴급 알림
    → 전략 비활성화 검토 (사용자 확인)
```

#### 추가 지표

```
1. Sharpe Trend (추세)
   30일 Sharpe의 30일 이동평균 기울기
   → 양수: 성과 개선 중
   → 음수: 성과 악화 중 (decay 진행)

2. Hit Rate (승률)
   최근 30일 거래 중 수익 거래 비율
   → 50% 이상: 건강
   → 40% 이하: 경고

3. Profit Factor
   총 이익 / 총 손실
   → 1.5 이상: 건강
   → 1.0 이하: 위험 (손실 > 이익)
```

#### AlphaDecayMonitor 구조

```python
class AlphaDecayMonitor:
    """전략별 성과 추적 및 decay 감지"""

    def __init__(self, strategies: list[str]):
        self._daily_returns: dict[str, list[float]] = {s: [] for s in strategies}

    def record_daily_return(self, strategy: str, daily_return: float) -> None:
        """매일 장 마감 후 호출"""
        self._daily_returns[strategy].append(daily_return)

    def health_check(self, strategy: str) -> StrategyHealth:
        """전략 건강 상태 반환"""
        returns = self._daily_returns[strategy]

        sharpe_30d = self._rolling_sharpe(returns, 30)
        sharpe_60d = self._rolling_sharpe(returns, 60)
        sharpe_trend = self._sharpe_slope(returns, 30)

        if sharpe_60d < 0:
            return StrategyHealth.DEAD       # 🔴
        if sharpe_30d < 0:
            return StrategyHealth.DANGER     # 🟠
        if sharpe_30d < 0.5:
            return StrategyHealth.WARNING    # 🟡
        return StrategyHealth.HEALTHY        # 🟢

    def dashboard(self) -> str:
        """전략 건강 대시보드 텍스트"""
        lines = ["전략 건강 대시보드", "=" * 50]
        for name in self._daily_returns:
            s30 = self._rolling_sharpe(self._daily_returns[name], 30)
            s60 = self._rolling_sharpe(self._daily_returns[name], 60)
            s90 = self._rolling_sharpe(self._daily_returns[name], 90)
            health = self.health_check(name)
            lines.append(f"  {name:20s}  30D={s30:.2f}  60D={s60:.2f}  90D={s90:.2f}  {health.emoji}")
        return "\n".join(lines)
```

#### 대시보드 예시

```
전략 건강 대시보드 (2026-02-27)
==================================================
  전략              30D    60D    90D    상태
  ────────────────  ─────  ─────  ─────  ────
  anchor-mom        1.2    1.1    1.0    🟢 건강
  donch-multi       0.8    0.9    1.1    🟢 건강
  tri-channel       0.3    0.5    0.8    🟡 경고  ← 주시!
```

### 6.4 구현 계획

| 단계 | 작업 | 파일 |
|------|------|------|
| 1 | `AlphaDecayMonitor` 클래스 | `src/monitoring/alpha_decay.py` (신규) |
| 2 | `StrategyHealth` enum | 위 파일 |
| 3 | Pod별 daily return 기록 연동 | `src/orchestrator/orchestrator.py` |
| 4 | `dashboard()` 텍스트 생성 | `AlphaDecayMonitor` |
| 5 | Discord 주간/일간 리포트 | `src/notification/` |
| 6 | 상태 전이 알림 (건강→경고, 경고→위험) | `AlphaDecayMonitor.check_transitions()` |
| 7 | 선택: 위험 시 자동 포지션 축소 | `src/eda/risk_manager.py` 연동 |
| 8 | 테스트 | `tests/monitoring/test_alpha_decay.py` |

### 6.5 데이터 저장

```
일간 성과 기록 위치:

  라이브: Orchestrator의 Pod별 equity 변화 → daily_return 계산
  저장: data/monitoring/alpha_decay/{strategy_name}.csv

  CSV 형식:
    date,daily_return,cumulative_return,sharpe_30d,health
    2026-02-27,0.003,0.15,1.2,HEALTHY
    2026-02-28,-0.001,0.149,1.1,HEALTHY
```

---

## 부록: 통합 아키텍처

### 개선 후 전체 흐름

```
[데이터]
  1m WebSocket ──→ CandleAggregator ──→ 4H/8H/12H Bar
                                              │
[파생 데이터]                                    │
  DerivativesFeed ──→ FR, OI, LS Ratio         │
                          │                     │
[리스크 모니터링]            ↓                     │
  DerivativesRiskMonitor ─→ risk_scale (0~1)    │
  AlphaDecayMonitor ──────→ health_status       │
                                                │
[전략]                                           ↓
  12H Strategy ──→ direction (+1/0/-1)
        │              │
        │    DirectionOverlay (Multi-TF Fusion)
        │              │
        ↓              ↓
  4H Strategy ──→ filtered signal
                       │
[실행]                  ↓
  PM ──→ OrderRequest ──→ RiskManager
                              │
                      FR/OI risk_scale 적용
                              │
                              ↓
                         SmartExecutor
                         ├─ 긴급 → Market Order
                         └─ 일반 → Limit Order (15s timeout)
                              │
                              ↓
                         FillEvent ──→ AlphaDecayMonitor 기록
```

### 설정 예시 (YAML)

```yaml
# config/improved.yaml
execution:
  smart_executor:
    enabled: true
    limit_timeout_sec: 15
    urgent_types: [stop_loss, trailing_stop, circuit_breaker]

cost_model:
  type: dynamic
  base_slippage: 0.0003
  asset_factors:
    BTC/USDT: 0.7
    ETH/USDT: 0.8
    DOGE/USDT: 1.5

risk:
  derivatives_monitor:
    enabled: true
    fr_thresholds: [0.0003, 0.0005, 0.001]
    fr_scales: [1.0, 0.5, 0.25, 0.1]
    oi_percentile_window: 90
    oi_alert_percentile: 0.90
    triple_warning: true  # FR + OI + low vol

  alpha_decay:
    enabled: true
    sharpe_windows: [30, 60, 90]
    healthy_threshold: 0.5
    danger_threshold: 0.0
    auto_scale_on_danger: true
    auto_scale_factor: 0.5

multi_tf:
  enabled: false  # Phase 3에서 활성화
  parent_tf: 12H
  child_tf: 4H
  filter_mode: direction  # direction / timing / confidence
```

---

## 참고 자료

### 업계 리서치 소스

- [Amberdata: 2025년 6대 Market Regime](https://blog.amberdata.io/the-six-market-regimes-of-2025-a-forensic-analysis)
- [1Token: Crypto Quant Strategy Index](https://blog.1token.tech/crypto-quant-strategy-index-viii-nov-2025/)
- [Amphibian Capital: Crypto Alpha](https://thehedgefundjournal.com/amphibian-quant-crypto-alpha-volatility-inefficiency/)
- [Maven Securities: Alpha Decay](https://www.mavensecurities.com/alpha-decay-what-does-it-look-like-and-what-does-it-mean-for-systematic-traders/)
- [PwC/AIMA 7th Annual Crypto Hedge Fund Report](https://caymanfinance.ky/wp-content/uploads/2025/11/7th-Annual-Global-Crypto-Hedge-Fund-Report.pdf)

### 내부 교훈

- #087: 4H TF 사망 재확인 (42+ 시도, 0 ACTIVE)
- #088: 대안데이터 graceful degradation ≠ alpha
- #089: Wavelet denoising으로 TF 한계 불극복
