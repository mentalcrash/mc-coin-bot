# Orchestrator Allocation Enhancement

## 1. Context

Orchestrator는 3-level 자본 배분 구조(Portfolio - Pod - Asset)를 구현하고 있으나,
per-pod 레버리지 미적용, rolling 모니터링 부재, all-excluded pod 시그널 미차단 등
실전 운영에 필요한 안전장치와 팔로업 체계가 부족했습니다.

이 문서는 7개 Gap을 식별하고, 각각에 대한 Enhancement를 설계 및 구현한 결과를 정리합니다.

## 2. 현재 상태 (이미 동작하는 것)

| 컴포넌트 | 기능 |
|----------|------|
| CapitalAllocator | EW/InvVol/RP/Kelly + lifecycle clamps + turnover constraints |
| IntraPodAllocator | EW/InvVol/RP/Signal + min/max clamp |
| AssetSelector FSM | ACTIVE - UNDERPERFORMING - COOLDOWN - RE_ENTRY |
| LifecycleManager | INCUBATION - PRODUCTION - WARNING - PROBATION - RETIRED |
| Fill attribution | Pod별 비례 귀속 (netting.attribute_fill) |
| State persistence | SQLite 기반 상태 저장/복원 |
| Discord notifications | Lifecycle 전이, 리밸런스, 리스크 알림 |

## 3. 3-Level Architecture Vision

```text
Level 0: Portfolio (Orchestrator + RiskAggregator)
  - Pod 간 자본 배분, cross-pod netting, gross leverage cap
Level 1: Pod (StrategyPod + CapitalAllocator)
  - 자기 자본을 1.0으로 인식, per-pod max_leverage 적용
Level 2: Asset (AssetSelector + IntraPodAllocator)
  - WHO (eligibility) + HOW MUCH (allocation) 분리
```

## 4. Gap Analysis

| # | Gap | 영향 | 우선순위 |
|---|-----|------|---------|
| G1 | Per-pod leverage 미적용 | 단일 Pod 과도 레버리지 | P0 |
| G2 | All-excluded pod 시그널 미차단 | WARNING 상태 시그널 발행 | P0 |
| G3 | Rolling metrics 부재 | 라이브 성과 추적 불가 | P1 |
| G4 | Absolute eligibility 미지원 | 약세장 cross-sectional 한계 | P1 |
| G5 | Risk defense turnover 무시 | 긴급 축소 시 대량 주문 | P2 |
| G6 | Allocation follow-up 부재 | 배분 드리프트 추적 불가 | P2 |
| G7 | Retired pod 잔여 타겟 미정리 | 백테스트 phantom 포지션 | P2 |

## 5. Enhancement Details

### E1: 3-Layer Leverage Defense

**파일:** `pod.py`, `orchestrator.py`

```text
Strategy signal (unbounded) -> Pod cap (max_leverage) -> Portfolio cap (max_gross_leverage)
```

- **Layer 1** (pod.py:compute_signal): per-symbol strength cap
  - `strength = min(strength, self._config.max_leverage)`
- **Layer 2** (orchestrator.py:_apply_per_pod_leverage_cap): pod aggregate cap
  - `sum(|weight|) <= max_leverage * capital_fraction`
  - 초과 시 비례 축소
- **Layer 3** (orchestrator.py:_flush_net_signals): portfolio cap (기존)
  - `scale_weights_to_leverage(full_net, max_gross_leverage)`

### E2: All-Excluded Signal Suppression

**파일:** `pod.py`, `orchestrator.py`

- `should_emit_signals` property 추가 (pod.py)
  - `is_active AND NOT (asset_selector.all_excluded)`
- `_on_bar()`에서 `should_emit_signals` 체크 추가 (orchestrator.py)
  - `compute_signal()` 내부 상태 갱신은 유지 (AssetSelector 평가 지속)
  - 시그널 누적/전파만 차단

### E3: Rolling Metrics

**파일:** `pod.py`, `metrics.py`

- `_ROLLING_WINDOW = 30` 상수
- `rolling_sharpe` property: 최근 30일 annualized Sharpe
- `rolling_drawdown` property: 최근 30일 max drawdown
- Prometheus gauges: `mcbot_pod_rolling_sharpe_30d`, `mcbot_pod_rolling_drawdown_30d`

### E4: Absolute Eligibility Thresholds

**파일:** `config.py`, `asset_selector.py`

- `AssetSelectorConfig` 새 필드:
  - `absolute_min_sharpe: float | None` (None=비활성)
  - `absolute_max_drawdown: float | None` (None=비활성)
  - `max_cooldown_cycles: int | None` (초과 시 영구 제외)
- `_check_absolute_thresholds()`: cross-sectional과 독립적 평가
- `_check_permanent_exclusion()`: max_cooldown_cycles 초과 시 `permanently_excluded=True`
- `_AssetState` 새 필드: `cooldown_cycles`, `permanently_excluded`

### E5: Risk Defense Turnover Awareness

**파일:** `config.py`

- `risk_defense_bypass_turnover: bool = True` 설정 추가
- 현행 동작 명시화 (risk defense는 turnover 제약 우회)

### E6: Allocation Dashboard

**파일:** `dashboard.py` (신규)

- `AllocationDashboard` 클래스
  - `compute_drift()`: 현재 vs 목표 배분 드리프트
  - `get_timeline()`: 배분 히스토리 timeline
  - `get_pod_leverage_usage()`: Pod별 레버리지 사용 현황
- Data classes: `PodDrift`, `AllocationSnapshot`, `AllocationTimeline`

### E7: Retired Pod Target Cleanup

**파일:** `orchestrator.py`

- `_cleanup_retired_pod_targets()`: RETIRED pod의 `_last_pod_targets` zero-out
- `_execute_rebalance()`에서 `_evaluate_lifecycle()` 직후 호출

## 6. Leverage Model

```text
Strategy signal (unbounded)
  -> Layer 1: min(strength, pod.max_leverage)     [per-symbol]
  -> Layer 2: sum(|w|) <= max_leverage * cap_frac  [per-pod aggregate]
  -> Layer 3: scale_weights_to_leverage(max_gross) [portfolio]
  -> RM execution cap                              [외부]
```

Pod는 자기 자본을 1.0으로 인식. strength > 1.0 = 레버리지 요청.

## 7. Auto-Retire 2-Stage

- **Asset-level**: `max_cooldown_cycles` 초과 시 `permanently_excluded` (영구 제외)
- **Pod-level**: LifecycleManager (MDD >= 25%, 6연속 손실월 등) -> RETIRED

## 8. Verification

```bash
# Lint
uv run ruff check --fix . && uv run ruff format .

# Type check
uv run pyright src/orchestrator/

# Tests
uv run pytest tests/orchestrator/ -v
```

### Test Coverage

| Enhancement | Test Class | Tests |
|-------------|-----------|-------|
| E1 Layer 1 | TestE1PerSymbolLeverageCap | 2 |
| E1 Layer 2 | TestE1PerPodLeverageCap | 3 |
| E1 Integration | TestE1IntegrationLeverageLayers | 1 |
| E2 Property | TestE2ShouldEmitSignals | 6 |
| E2 Orchestrator | TestE2OrchestratorSignalSuppression | 1 |
| E3 | TestE3RollingMetrics | 7 |
| E4 Thresholds | TestE4AbsoluteThresholds | 3 |
| E4 Permanent | TestE4PermanentExclusion | 3 |
| E4 Config | TestE4ConfigFields | 6 |
| E5 | TestE5RiskDefenseTurnoverConfig | 2 |
| E6 | TestE6AllocationDashboard | 4 |
| E7 | TestE7RetiredPodTargetCleanup | 3 |
| **Total** | | **42** |

## 9. Implementation Roadmap

- Phase 1 (P0): E1+E2+E7 -- Core Safety
- Phase 2 (P1): E3+E4 -- Monitoring + Eligibility
- Phase 3 (P2): E5+E6 -- Polish

## 10. Critical Files

| File | Changes |
|------|---------|
| `src/orchestrator/pod.py` | Layer 1 leverage cap, `should_emit_signals`, rolling metrics |
| `src/orchestrator/orchestrator.py` | Layer 2 leverage cap, signal suppression, retired cleanup, `last_allocated_weights` |
| `src/orchestrator/asset_selector.py` | Absolute thresholds, permanent exclusion, cooldown_cycles |
| `src/orchestrator/config.py` | New config fields (absolute thresholds, risk_defense_bypass_turnover) |
| `src/orchestrator/metrics.py` | Rolling Sharpe/DD Prometheus gauges |
| `src/orchestrator/dashboard.py` | New -- allocation follow-up module |
| `tests/orchestrator/test_allocation_enhancements.py` | 42 new tests |
