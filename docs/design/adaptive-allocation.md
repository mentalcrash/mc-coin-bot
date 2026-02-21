# Adaptive Multi-Level Allocation System

## Context

현재 4개 ACTIVE 전략(CTREND, CTREND-X, GBTrend, Anchor-Mom)이 5개 에셋에서 운영 중이다.
143개 전략 탐색 결과 단일에셋 1D 전략 공간은 사실상 소진되었으며,
포트폴리오 Sharpe 향상의 현실적 경로는 **에셋 다각화 + 배분 고도화**이다.

**목표**:

1. 에셋 유니버스 확장 (5 → 12 → 16 → 20)
2. Pod 내 성과 미달 에셋 자동 제외/재진입 (AssetSelector)
3. Pod 간 성과 기반 리밸런싱 + 턴오버 제약
4. 전략 제거 시 포지션 안전 청산
5. 모든 상태 재시작 후에도 유지

---

## 1. 현재 아키텍처 (이미 구현)

| 레이어 | 모듈 | 역할 | 상태 |
|--------|------|------|------|
| Pod 간 배분 | `CapitalAllocator` | EW/IV/RP/Kelly → pod_id → fraction | :white_check_mark: |
| Pod 내 에셋 배분 | `IntraPodAllocator` | EW/IV/RP/Signal → symbol → weight | :white_check_mark: |
| Pod 라이프사이클 | `LifecycleManager` | INCUBATION→PRODUCTION→WARNING→PROBATION→RETIRED | :white_check_mark: |
| 상태 영속화 | `OrchestratorStatePersistence` | SQLite bot_state, 5분 주기 | :white_check_mark: |
| 리밸런싱 | `Orchestrator._execute_rebalance()` | Calendar/Threshold/Hybrid | :white_check_mark: |

**핵심 파일**:

- `src/orchestrator/allocator.py` &mdash; `compute_weights(pod_returns, pod_states, lookback=90)`
- `src/orchestrator/asset_allocator.py` &mdash; `on_bar(returns, strengths) → {symbol: weight}`
- `src/orchestrator/lifecycle.py` &mdash; `evaluate(pod) → LifecycleState`
- `src/orchestrator/pod.py` &mdash; `compute_signal()`, `to_dict()`, `restore_from_dict()`
- `src/orchestrator/orchestrator.py` &mdash; `_execute_rebalance()`, `_flush_net_signals()`
- `src/orchestrator/state_persistence.py` &mdash; `save()`, `restore()`
- `src/orchestrator/config.py` &mdash; `PodConfig`, `OrchestratorConfig`, `AssetAllocationConfig`

---

## 2. 신규 개발 범위

| # | 기능 | 신규/변경 | 우선순위 |
|---|------|----------|---------|
| A | Turnover Constraint | `orchestrator.py` + `config.py` 변경 | P0 |
| B | AssetSelector (에셋 선별 FSM) | **신규 모듈** `asset_selector.py` | P0 |
| C | Pod 통합 (Selector + Allocator) | `pod.py` 변경 | P1 |
| D | All-Excluded → Pod WARNING | `orchestrator.py` 변경 | P1 |
| E | 전략 제거 시 포지션 청산 | `live_runner.py` 변경 | P1 |
| F | 에셋 다각화 백테스트 | Config + 데이터만 | P2 |
| G | Pod 배분 방식 전환 | Config 변경만 | P3 |

---

## 3. AssetSelector 설계

### 3.1 핵심 원칙: WHO vs HOW MUCH 분리

```
AssetSelector (WHO participates)    IntraPodAllocator (HOW MUCH weight)
        │                                    │
        ▼                                    ▼
  multiplier: 0.0~1.0              weight: normalized allocation
        │                                    │
        └──── strength × multiplier × weight × n_assets ────┘
```

- `AssetSelector`는 에셋별 participation multiplier (0.0~1.0)를 관리
- `IntraPodAllocator`는 참여 중인(active) 에셋의 비중만 계산
- 두 모듈은 독립적으로 동작, 결과를 곱셈으로 결합

### 3.2 에셋 라이프사이클 FSM

```
               score < exclude (N bars 연속)
   ┌─────────┐ ─────────────────────────▶ ┌──────────────────┐
   │  ACTIVE │                            │ UNDERPERFORMING  │
   │ mult=1.0│ ◀────recovery during───── │ mult: 1.0→0.66→  │
   └────▲────┘     ramp-down              │      0.33→0.0    │
        │                                 └────────┬─────────┘
        │ ramp-up                                  │ ramp 완료
        │ complete                                 ▼
   ┌────┴────┐     cooldown 경과          ┌──────────────────┐
   │RE_ENTRY │ ◀── + score > include ──── │    COOLDOWN      │
   │mult: ↑  │     (N bars 연속)          │ mult=0, min 30봉 │
   └─────────┘                            └──────────────────┘
                                                   ▲
                  Hard Exclude (ANY → COOLDOWN):    │
                  Sharpe < -1.0 AND DD > 15% ───────┘
```

| From | To | 조건 |
|------|----|------|
| ACTIVE | UNDERPERFORMING | score < exclude_threshold, N bars 연속 |
| UNDERPERFORMING | COOLDOWN | ramp-down 완료 (multiplier = 0) |
| UNDERPERFORMING | RE_ENTRY | score > include_threshold, N bars 연속 (ramp 도중 회복) |
| COOLDOWN | RE_ENTRY | cooldown 경과 + score > include_threshold, N bars 연속 |
| RE_ENTRY | ACTIVE | ramp-up 완료 (multiplier = 1.0) |
| RE_ENTRY | UNDERPERFORMING | score 재하락 (ramp 도중 악화) |
| ANY | COOLDOWN | Hard exclude (Sharpe < -1.0 AND DD > 15%) |

### 3.3 Whipsaw 방지 메커니즘

| 메커니즘 | 설계 |
|---------|------|
| **Hysteresis Band** | exclude_threshold(0.20) != include_threshold(0.35) &mdash; 갭 존재 |
| **Confirmation Period** | exclude: 5봉 연속 / include: 3봉 연속 기준 미달/충족 필요 |
| **Minimum Cooldown** | 제외 후 최소 30봉(1개월) 대기, 충분한 신규 데이터 수집 |
| **Gradual Ramp** | 즉시 0/1 전환 금지. 3단계(1.0→0.66→0.33→0.0, 역순 동일) |
| **Min Active Assets** | 최소 N개 에셋 활성 유지 (전 에셋 탈락 방지) |

### 3.4 복합 점수 산출

```
score = 0.4 × sharpe_rank(60D) + 0.3 × return_rank(30D) + 0.3 × (1 - drawdown_rank)
```

- **Cross-sectional ranking**: Pod 내 에셋 간 상대 순위 (0~1 백분위)
- 단일 메트릭 위험 방지: Sharpe만 사용 시 비활성 에셋 제거 안 됨
- 데이터 부족(< 10봉) 시 neutral score (0.5) 부여

---

## 4. Config 구조

### 4.1 AssetSelectorConfig (신규, `config.py`)

```python
class AssetSelectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False

    # Scoring weights (합 = 1.0)
    sharpe_weight: float = 0.4       # ge=0, le=1
    return_weight: float = 0.3
    drawdown_weight: float = 0.3

    # Lookback
    sharpe_lookback: int = 60        # bars, ge=10
    return_lookback: int = 30        # bars, ge=5

    # Hysteresis thresholds
    exclude_score_threshold: float = 0.20   # 이하 시 제외 프로세스 시작
    include_score_threshold: float = 0.35   # 이상 시 재진입 프로세스 시작

    # Hard exclusion
    hard_exclude_sharpe: float = -1.0
    hard_exclude_drawdown: float = 0.15

    # Confirmation
    exclude_confirmation_bars: int = 5
    include_confirmation_bars: int = 3

    # Cooldown
    min_exclusion_bars: int = 30

    # Ramp
    ramp_steps: int = 3              # 1.0 → 0.66 → 0.33 → 0.0

    # Safety
    min_active_assets: int = 2
```

**Validation**: `include_score_threshold > exclude_score_threshold` (hysteresis 보장)

### 4.2 TurnoverConstraintConfig (신규, `config.py`)

```python
class TurnoverConstraintConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_pod_turnover_per_rebalance: float = 0.10   # Pod당 최대 +-10%p
    max_total_turnover_per_rebalance: float = 0.30  # 전체 최대 30%
```

### 4.3 기존 모델 확장

`PodConfig`에 추가:

```python
asset_selector: AssetSelectorConfig | None = None
```

`OrchestratorConfig`에 추가:

```python
turnover_constraint: TurnoverConstraintConfig = Field(default_factory=TurnoverConstraintConfig)
```

### 4.4 YAML 예시

```yaml
orchestrator:
  allocation_method: equal_weight
  turnover_constraint:
    max_pod_turnover_per_rebalance: 0.10
    max_total_turnover_per_rebalance: 0.30

  pods:
    - pod_id: pod-ctrend
      strategy_name: ctrend
      symbols: [SOL/USDT, BTC/USDT, BNB/USDT, ETH/USDT,
                ADA/USDT, AVAX/USDT, LINK/USDT, ATOM/USDT]
      timeframe: "1D"
      initial_fraction: 0.25
      asset_allocation:
        method: inverse_volatility
        vol_lookback: 60
        rebalance_bars: 5
        min_weight: 0.05
        max_weight: 0.40
      asset_selector:
        enabled: true
        exclude_score_threshold: 0.20
        include_score_threshold: 0.35
        min_exclusion_bars: 30
        ramp_steps: 3
        min_active_assets: 3
```

**하위 호환성**: `asset_selector` 미지정 또는 `enabled: false` → 기존과 동일 동작

---

## 5. 통합 포인트

### 5.1 StrategyPod (`pod.py`)

**변경 사항**:

1. `__init__`에 `_asset_selector: AssetSelector | None` 추가
2. `compute_signal()` 내 `_apply_asset_weight()` 수정:

```python
def _apply_asset_weight(self, symbol: str, strength: float) -> float:
    # (1) AssetSelector multiplier (WHO)
    selector_mult = 1.0
    if self._asset_selector is not None:
        selector_mult = self._asset_selector.multipliers.get(symbol, 1.0)
    if selector_mult < 1e-8:
        return 0.0  # 제외된 에셋

    # (2) IntraPodAllocator weight (HOW MUCH)
    if self._asset_allocator is None:
        return strength * selector_mult
    n = len(self._config.symbols)
    asset_w = self._asset_allocator.weights.get(symbol, 1.0 / n)
    return strength * asset_w * n * selector_mult
```

3. `_maybe_rebalance_assets()`에서 selector 호출 (allocator보다 먼저):

```python
# AssetSelector 평가 (매 bar)
if self._asset_selector is not None:
    self._asset_selector.on_bar(returns=self._asset_returns,
                                close_prices=self._get_close_prices())

# IntraPodAllocator (active symbols만)
if self._asset_allocator is not None:
    active = self._asset_selector.active_symbols if self._asset_selector else self._config.symbols
    filtered_returns = {s: r for s, r in self._asset_returns.items() if s in active}
    self._asset_allocator.on_bar(filtered_returns)
```

4. `to_dict()` / `restore_from_dict()`에 selector 상태 추가

### 5.2 Orchestrator (`orchestrator.py`)

**변경 1**: Turnover Constraint &mdash; `_execute_rebalance()` 내:

```python
new_weights = self._allocator.compute_weights(...)
constrained = self._apply_turnover_constraints(new_weights)
# constrained 적용
```

`_apply_turnover_constraints()` 로직:

- Pod별 `|delta| <= max_pod_turnover_per_rebalance` 클램프
- RETIRED Pod는 bypass (즉시 0)
- 전체 합 `sum(|delta|) <= max_total_turnover_per_rebalance` 초과 시 비례 축소

**변경 2**: All-Excluded 감지 &mdash; lifecycle 평가 후:

```python
for pod in self._pods:
    if pod.is_active and pod._asset_selector and pod._asset_selector.all_excluded:
        if pod.state == LifecycleState.PRODUCTION:
            pod.state = LifecycleState.WARNING
```

### 5.3 전략 제거 시 포지션 청산 (`live_runner.py`)

현재: 제거된 Pod의 저장 상태 무시, 포지션 방치

**변경**: `_restore_orchestrator_state()` 이후 orphan 감지 + 청산:

```python
async def _cleanup_orphaned_positions(self, state_mgr):
    # 1. 저장된 pod_id 집합 vs 현재 config pod_id 집합 비교
    # 2. 제거된 Pod에서 quantity != 0인 symbol 추출
    # 3. 현재 어떤 Pod에서도 거래하지 않는 symbol만 필터 (overlap 제외)
    # 4. weight=0 SignalEvent 발행 → PM → OMS → 청산
```

**Edge Case**:

- 제거된 Pod의 symbol이 다른 활성 Pod에서 거래 중 → 청산 불필요 (활성 Pod가 관리)
- 거래소 미연결 상태 → 로그 경고, 다음 reconciliation 시 재시도
- 부분 체결 → OMS가 정상 처리, 다음 tick에서 잔여 청산

### 5.4 State Persistence (`state_persistence.py`)

`StrategyPod.to_dict()` 내 asset_selector 추가:

```python
"asset_selector": self._asset_selector.to_dict() if self._asset_selector else None,
```

`StrategyPod.restore_from_dict()` 내:

```python
selector_data = data.get("asset_selector")
if isinstance(selector_data, dict) and self._asset_selector is not None:
    self._asset_selector.restore_from_dict(selector_data)
```

기존 save/restore 구조 위에 자연스럽게 확장 (별도 DB 키 불필요).

---

## 6. 데이터 가용성 (에셋 확장)

Silver OHLCV 데이터 보유 현황 (21개 symbol):

| 단계 | 심볼 | 합계 |
|------|------|------|
| **현재** (Tier 1) | BTC, ETH, BNB, SOL, DOGE | 5 |
| **확장 1** | + ADA, AVAX, LINK, ATOM | 9 |
| **확장 2** | + LTC, DOT, XRP, UNI | 13 |
| **확장 3** | + FIL, MATIC, NEAR, POL | 17 |

각 단계에서:

1. `uv run mcbot ingest pipeline {SYMBOL} --year 2024 --year 2025` 으로 Silver 확인
2. 기존 ACTIVE 전략으로 단일 에셋 백테스트 (P4 기준)
3. Sharpe > 0.5인 에셋만 라이브 편입
4. AssetSelector가 라이브에서 추가 필터링

---

## 7. 구현 로드맵

### Phase 1: Config + Turnover Constraint (1~2일)

- `config.py`: `AssetSelectorConfig`, `TurnoverConstraintConfig` 추가
- `config.py`: `PodConfig.asset_selector`, `OrchestratorConfig.turnover_constraint` 필드 추가
- `orchestrator.py`: `_apply_turnover_constraints()` 구현
- 테스트: config validation, turnover clamp 로직

**파일**: `config.py`, `orchestrator.py`, `tests/orchestrator/test_config.py`, `tests/orchestrator/test_orchestrator.py`

### Phase 2: AssetSelector 코어 (2~3일)

- `asset_selector.py` 신규 생성 (AssetSelector 클래스)
- `models.py`에 `AssetLifecycleState` 추가
- FSM 전이, 점수 산출, hysteresis, ramp, hard exclusion, min_active 구현
- `to_dict()` / `restore_from_dict()` 직렬화
- 테스트: ~40 unit tests (각 전이, 경계 조건, 직렬화 round-trip)

**파일**: `asset_selector.py`, `models.py`, `tests/orchestrator/test_asset_selector.py`

### Phase 3: Pod 통합 (1~2일)

- `pod.py`: AssetSelector 초기화, `_apply_asset_weight()` 수정, 직렬화 확장
- IntraPodAllocator와 연동 (active symbols만 전달)
- 통합 테스트: Pod + Selector + Allocator 조합
- 하위 호환성 테스트: selector 없는 기존 config 정상 동작

**파일**: `pod.py`, `tests/orchestrator/test_pod.py`

### Phase 4: Orchestrator 통합 (1일)

- `orchestrator.py`: all-excluded → WARNING 로직
- Turnover constraint를 `_execute_rebalance()` 에 연결
- `state_persistence.py`: asset_selector 상태 save/restore 검증
- 통합 테스트: lifecycle 전이 + 상태 영속화

**파일**: `orchestrator.py`, `state_persistence.py`, `tests/orchestrator/test_orchestrator.py`

### Phase 5: 전략 제거 시 청산 (1일)

- `live_runner.py`: `_cleanup_orphaned_positions()` 구현
- Orphan 감지 → weight=0 SignalEvent 발행
- Edge case 처리 (overlap symbols, 거래소 미연결)
- 테스트: save → config 변경 → restore → 청산 확인

**파일**: `live_runner.py`, `tests/orchestrator/test_live_integration.py`

### Phase 6: 에셋 다각화 백테스트 (2~3일, Config only)

- `orchestrator-live.yaml` symbols 확장 (5 → 9 → 13)
- ACTIVE 전략 × 신규 에셋 단일 백테스트
- Per-asset Sharpe 검증
- AssetSelector 임계값 보정 (백테스트 데이터 기반)

**파일**: `config/orchestrator-live.yaml`

### Phase 7: Pod 배분 방식 전환 (Config only)

라이브 운영 실적 축적에 따라 config만 변경:

- 시작: `allocation_method: equal_weight`
- 90일 후: `allocation_method: inverse_volatility`
- 180일 후: `allocation_method: adaptive_kelly`

코드 변경 없음 (기존 `CapitalAllocator` 지원)

---

## 8. 테스트 전략

### Unit Tests

| 모듈 | 파일 | 예상 테스트 수 |
|------|------|--------------|
| AssetSelector FSM | `test_asset_selector.py` | ~40 |
| Config validation | `test_config.py` (확장) | ~5 |
| Turnover constraint | `test_orchestrator.py` (확장) | ~3 |
| Pod integration | `test_pod.py` (확장) | ~5 |
| State persistence | `test_state_persistence.py` (확장) | ~3 |
| Orphan cleanup | `test_live_integration.py` (확장) | ~2 |
| **합계** | | **~58** |

### 핵심 시나리오

1. **Whipsaw 방지**: Score 0.19 → 0.22 → 0.18 진동 시 상태 유지 (confirmation 미달)
2. **Ramp 정확성**: 3-step ramp down/up 시 multiplier 값 정확성
3. **Min active 안전망**: 5개 에셋 중 4개 제외 시도 → 최소 2개 유지
4. **Hard exclusion**: Sharpe -1.5, DD 20% → 즉시 COOLDOWN (ramp 무시)
5. **Turnover 클램프**: Pod 25% → 45% 시도 → 35%로 클램프 (10%p 제한)
6. **Orphan 청산**: Pod 제거 → 재시작 → weight=0 시그널 확인
7. **하위 호환성**: selector 없는 기존 config → 기존과 동일 동작

### 검증 명령어

```bash
# Phase 1~5 완료 후
uv run ruff check --fix . && uv run ruff format .
uv run pyright src/orchestrator/
uv run pytest tests/orchestrator/ -v

# Phase 6 백테스트
uv run mcbot orchestrate backtest config/orchestrator-live.yaml --report
```

---

## 9. 위험 요소 및 완화

| 위험 | 영향 | 완화 |
|------|------|------|
| 과적합 (최근 성과 추종) | 최근 승자 과집중 → 반전 시 손실 | Turnover 제한 + Shrinkage (TODO) |
| Whipsaw (빈번한 ON/OFF) | 거래 비용 증가 | Hysteresis + Confirmation + Cooldown |
| 전 에셋 탈락 | 전략 비활성화 | min_active_assets 안전망 |
| 상태 손실 | 재시작 시 제외 상태 리셋 | SQLite 영속화 (기존 인프라 활용) |
| 에셋 확장 시 성과 저하 | 약한 에셋이 포트폴리오 희석 | AssetSelector가 자동 필터링 |
