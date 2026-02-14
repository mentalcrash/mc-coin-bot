# Strategy Orchestrator Pre-Live Audit Report

> **Date**: 2026-02-14
> **Scope**: `src/orchestrator/` 전체 + Runner 통합 + 테스트 커버리지
> **Method**: 3개 독립 감사 에이전트 병렬 실행 (Core Models, Lifecycle/Risk, Runner/Tests)
> **Verdict**: **BLOCK** — Critical 7건 해결 전 라이브 배포 불가

---

## Quality Gate 현황

| 항목 | 결과 | 비고 |
|------|------|------|
| Ruff Lint | **PASS** (0 errors) | |
| Pyright | **PASS** (0 errors, 0 warnings) | |
| Tests | **PASS** (293/293) | |
| Coverage | **90%** (1292 stmts, 90 missed) | |
| RuntimeWarning | **126건** | covariance NaN/Inf — allocator edge case |

### Coverage 상세

| File | Stmts | Miss | Cover | 주요 미커버 영역 |
|------|-------|------|-------|------------------|
| `allocator.py` | 185 | 24 | 82% | fallback paths, Kelly edge cases |
| `orchestrator.py` | 244 | 31 | 84% | risk check, notification, rebalance |
| `pod.py` | 177 | 17 | 88% | warmup detection, serialization |
| `state_persistence.py` | 112 | 6 | 92% | version mismatch, partial restore |
| `lifecycle.py` | 159 | 5 | 93% | timeout transitions, edge branches |
| `risk_aggregator.py` | 99 | 4 | 94% | correlation stress, PRC edge |
| `metrics.py` | 63 | 3 | 95% | portfolio metrics fallback |
| models, config 등 6파일 | — | — | 100% | — |

---

## CRITICAL — 라이브 배포 차단 (7건)

### C-1. Pod realized PnL 항상 0

**위치**: `src/orchestrator/pod.py:246-249`
**발견 에이전트**: 3/3 (전원 독립 확인)

```python
realized_delta = fill_qty * fill_price - fill_qty * (
    pos.notional_usd / max(abs(pos.notional_usd / fill_price), 1e-12)
)
```

**문제**: 수식 `pos.notional_usd / abs(pos.notional_usd / fill_price)`이 대수적으로 `fill_price`와 동일.
따라서 `realized_delta = qty * price - qty * price = 0.0` 항상 성립.

**근본 원인**: `PodPosition`에 평균 진입가(avg_entry_price) 추적 부재. `notional_usd`만으로는 진입가 역산 불가.

**추가 결함**: Short 포지션 커버(buy-closing-short) 경로 부재 — `if not is_buy and pos.notional_usd > 0` 조건만 존재.

**영향**: 모든 Pod P&L이 수수료만 기록. Lifecycle 졸업/퇴출, Risk 평가, 자본 배분 전체 체인 무효.

---

### C-2. PodPerformance 메트릭 미갱신

**위치**: `src/orchestrator/pod.py` 전역 (update_position, record_daily_return 등)
**발견 에이전트**: 2/3

`PodPerformance`의 13개 필드 중 실제 갱신되는 것은 `trade_count`와 `live_days` 뿐.
`sharpe_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `rolling_volatility`, `peak_equity`, `current_equity`, `current_drawdown`, `total_return` 모두 **영원히 0.0**.

**영향**:
- 졸업 조건 `sharpe >= 1.0`, `calmar >= 0.8` → **절대 충족 불가**
- Hard stop `max_drawdown >= 0.25` → **절대 트리거 안 됨**
- 전체 Lifecycle 상태 머신 사실상 비활성

---

### C-3. Lifecycle PH score vs Sharpe 단위 불일치

**위치**: `src/orchestrator/lifecycle.py:157`
**발견 에이전트**: 3/3

```python
if pod_ls.ph_detector.score < self._retirement.rolling_sharpe_floor:
    self._transition(pod, pod_ls, LifecycleState.PRODUCTION)
```

**문제**: PH score는 CUSUM 누적값(범위 0~50+), `rolling_sharpe_floor`은 Sharpe 비율(기본 0.3).
PH score가 0에서 시작하므로 거의 항상 < 0.3 → **WARNING 즉시 PRODUCTION 복귀**.

**영향**: WARNING 상태가 사실상 no-op. 열화 감지 후 방어 체계 무의미.

---

### C-4. Orchestrator 상태 LiveRunner에서 미영속

**위치**: `src/eda/live_runner.py` (run, _restore_state, _periodic_state_save)
**발견 에이전트**: 2/3

`OrchestratorStatePersistence` 클래스는 구현 완료되어 있으나, `LiveRunner.run()`에서 **한 번도 인스턴스화/호출되지 않음**.

- `_restore_state()` → PM/RM/OMS만 복원
- `_periodic_state_save()` → PM/RM/OMS만 저장
- shutdown → PM/RM/OMS만 저장

**영향**: 재시작 시 모든 Pod lifecycle state, capital_fraction, daily_returns, rebalance timestamp, fill attribution target 소실. PRODUCTION까지 승격된 Pod이 INCUBATION으로 리셋.

---

### C-5. Risk alert 무방어

**위치**: `src/orchestrator/orchestrator.py:401-405`
**발견 에이전트**: 2/3

```python
for alert in alerts:
    logger.warning("RiskAlert [{}]: {}", alert.severity, alert.message)

if alerts and self._notification is not None:
    self._fire_notification(self._notification.notify_risk_alerts(alerts))
```

`check_portfolio_limits()`가 critical alert(레버리지 초과, MDD 초과, 일간 손실 한도) 반환 시 **로그+Discord만**. 포지션 축소, 거래 중단, circuit breaker 없음.

**영향**: 포트폴리오 리스크 한도 초과 시에도 거래 계속 진행.

---

### C-6. daily_pnl_pct 하드코딩 0.0

**위치**: `src/orchestrator/orchestrator.py:395-400`
**발견 에이전트**: 2/3

`_check_risk_limits()`가 `check_portfolio_limits()`에 `daily_pnl_pct`를 전달하지 않아 기본값 0.0 사용.

**영향**: `daily_loss_limit=0.03` (3% 일간 손실 한도) 체크가 **영원히 미트리거** — 데드 코드.

---

### C-7. Risk check 빈 데이터 대상 실행

**위치**: `src/orchestrator/orchestrator.py:182,396`
**발견 에이전트**: 3/3

`_flush_net_signals()` (L141) → `_pending_net_weights` 클리어 + `_pending_bar_ts = None`
→ 직후 `_check_rebalance()` (L182) → `_execute_rebalance()` → `_check_risk_limits()`
→ `self._pending_net_weights` 읽기 (빈 dict)

**영향**: gross_leverage = 0으로 항상 계산 → 레버리지 한도 초과 미감지. 타임스탬프도 `None`으로 감사 기록 오류.

---

## HIGH — 잘못된 동작 유발 (8건)

### H-1. Fill attribution 방향 무시

**위치**: `src/orchestrator/netting.py:61-93`
**발견 에이전트**: 3/3

```python
for pod_id, target in pod_targets.items():
    share = abs(target) / total_abs
    result[pod_id] = (qty * share, price, fee * share)
```

Long Pod(+0.3)과 Short Pod(-0.3) 동일 심볼 → BUY fill 시 `abs(target)`로 양쪽에 동일 비율 귀속. Short pod에 매수가 잘못 배분.

**수정 방향**: fill 방향과 일치하는 pod에게만 귀속, 또는 net 기여도 비례 귀속.

---

### H-2. Kelly fraction 이중 적용

**위치**: `src/orchestrator/allocator.py:316+328`
**발견 에이전트**: 2/3

1. Raw Kelly에 `kelly_fraction` 곱셈 (L316: fractional Kelly)
2. Blend alpha에 또 `kelly_fraction` (L328: `confidence * kelly_fraction`)

실효 Kelly = `0.25 * 0.25 = 6.25%` of full Kelly at default. 의도와 다를 가능성 높음.

---

### H-3. Lifecycle 타임아웃 wall clock 사용

**위치**: `src/orchestrator/lifecycle.py:162,179`
**발견 에이전트**: 2/3

```python
days_in_warning = (datetime.now(UTC) - pod_ls.state_entered_at).days
```

백테스트에서 전체 실행이 수 초 → WARNING→PROBATION, PROBATION→RETIRED 타임아웃 전이 불가.
라이브에서도 bar 진행 속도와 무관하게 wall clock 기준.

**수정 방향**: `bar_timestamp`를 `evaluate()`에 전달, wall clock 대신 bar 시간 사용.

---

### H-4. PM stop-loss/trailing-stop 비활성 + Orchestrator 미방어

**위치**: `src/eda/orchestrated_runner.py:57-60`
**발견 에이전트**: 2/3

`_derive_pm_config()`: `system_stop_loss=None`, `use_trailing_stop=False` — "Orchestrator manages risk" 주석.
그러나 C-5/C-6/C-7에 의해 Orchestrator 리스크 관리 미작동.

**영향**: 3단계 방어(PM→RM→OMS)가 RM 단독으로 축소.

---

### H-5. Monthly return 산술합 사용

**위치**: `src/orchestrator/lifecycle.py:303`
**발견 에이전트**: 3/3

```python
monthly_return = sum(chunk)  # 틀림: prod(1+r)-1 사용해야 함
```

일 -5% × 30일 = -150% (불가능 값). Consecutive loss months 판정 오류 가능.

---

### H-6. State 저장 비원자적

**위치**: `src/orchestrator/state_persistence.py:79-83`
**발견 에이전트**: 2/3

`orchestrator_state`와 `daily_returns` 각각 별도 `commit()`. 중간 crash 시 `live_days`와 `daily_returns` 길이 불일치.

**수정 방향**: 두 키를 단일 트랜잭션으로 묶기.

---

### H-7. PageHinkley NaN 가드 부재

**위치**: `src/orchestrator/degradation.py:52-78`
**발견 에이전트**: 2/3

단일 NaN daily return → `_x_mean`, `_m_t`, `_m_min` 영구 오염 → 이후 모든 비교 False (NaN > lambda = False) → 감지기 영구 비활성화, 경고 없음.

---

### H-8. `_avg_live_days` 실제 데이터 미사용

**위치**: `src/orchestrator/allocator.py:352-369`
**발견 에이전트**: 2/3

`PodPerformance.live_days` 대신 state→일수 하드코딩 매핑 (INCUBATION=30, PRODUCTION=180). 89일 INCUBATION pod이 30일로 취급. Kelly confidence ramp 왜곡.

---

## MEDIUM — 설계 이슈 (8건)

| # | 위치 | 이슈 | 발견 에이전트 |
|---|------|------|:---:|
| M-1 | `pod.py:179-186` | 매 bar마다 전체 `strategy.run()` 호출 (`run_incremental` 미사용). 10 pod × 8 symbol = 80회/bar | 2/3 |
| M-2 | `allocator.py:193-199` | 이중 정규화로 `min_fraction` 이하로 축소 가능. Clamp 후 normalize → min 위반 | 2/3 |
| M-3 | `orchestrator.py:290-296` | Drift 기준이 `initial_fraction` (last allocated weight가 아님) | 2/3 |
| M-4 | `orchestrator.py:527-532` | Fill마다 O(N) pod lookup. dict[str, StrategyPod] 사용 필요 | 3/3 |
| M-5 | `risk_aggregator.py:133`, `lifecycle.py:348` | `abs(corr)` → 음의 상관(분산 이득)을 위험/불이익으로 오판 | 2/3 |
| M-6 | `degradation.py:33` | alpha=0.9999 → EWMA 반감기 ~19년 → running mean 사실상 고정 | 2/3 |
| M-7 | `cli/orchestrate.py:273` | `int(max_gross_leverage)` → 2.5x가 2x로 절삭 | 1/3 |
| M-8 | `config.py:388` | `max_fraction` 합계 검증 없음 (4 pod × 0.4 = 1.6 가능) | 2/3 |

---

## 누락 테스트 (8건)

| # | 영역 | 설명 | 관련 이슈 |
|---|------|------|-----------|
| T-1 | LiveRunner Orchestrator 영속성 | `OrchestratorStatePersistence`가 `run()` 내에서 호출되는지 검증 | C-4 |
| T-2 | `Pod.update_position()` PnL | realized PnL, notional 추적, short 포지션 처리 검증 | C-1 |
| T-3 | Multi-symbol 동일 timestamp batch | 여러 심볼 동시 도착 시 올바른 누적→flush 동작 | C-7 |
| T-4 | `daily_pnl_pct` 통합 경로 | Orchestrator가 non-zero `daily_pnl_pct`를 전달하는지 | C-6 |
| T-5 | WARNING/PROBATION timeout 전이 | wall clock 의존으로 백테스트에서 미테스트 | H-3 |
| T-6 | PRC sum == 1.0 invariant | 편향된 weight 분포에서 invariant 검증 | — |
| T-7 | All-pods-retired Orchestrator 동작 | 모든 Pod RETIRED 시 시그널 미발행, rebalance 안전 처리 | — |
| T-8 | Pod warmup 실패 격리 | 한 Pod 실패 시 다른 Pod 영향 없음 | — |

---

## 수정 우선순위

### P0 — 라이브 차단 (배포 전 필수)

| 순서 | 이슈 | 작업 요약 |
|------|------|-----------|
| 1 | C-1 | `PodPosition`에 `avg_entry_price`, `quantity` 추가. Long/Short 양방향 realized PnL 계산 구현 |
| 2 | C-2 | `_compute_pod_metrics()` 구현 — daily_returns에서 Sharpe, MDD, Calmar, total_return 등 계산 |
| 3 | C-3 | PH score 비교 대상을 별도 threshold로 분리, 또는 rolling Sharpe 직접 계산 후 비교 |
| 4 | C-4 | `LiveRunner.run()`에 `OrchestratorStatePersistence` 통합 (restore, periodic save, shutdown) |
| 5 | C-5 | Critical alert 시 defensive action 구현 (capital_fraction 강제 축소 또는 signal 차단) |
| 6 | C-6 | `_check_risk_limits()`에 실제 `daily_pnl_pct` 계산 + 전달 |
| 7 | C-7 | Risk check를 flush 전 또는 complete weights snapshot 기반으로 이동 |

### P1 — 배포 전 권장

| 순서 | 이슈 | 작업 요약 |
|------|------|-----------|
| 8 | H-1 | Fill attribution에 방향 필터 추가 |
| 9 | H-2 | Kelly fraction 적용 로직 정리 (한 곳에서만 적용) |
| 10 | H-3 | `evaluate()`에 `bar_timestamp` 파라미터 추가 |
| 11 | H-4 | Orchestrator-level system_stop_loss 구현 또는 PM stop 복원 |
| 12 | H-5 | `sum(chunk)` → `prod(1+r)-1` |
| 13 | H-6 | 두 키 단일 트랜잭션 저장 |
| 14 | H-7 | PH detector NaN guard 추가 |
| 15 | H-8 | `PodPerformance.live_days` 실제 값 사용 |

### P2 — 조기 수정 권장

| 순서 | 이슈 | 작업 요약 |
|------|------|-----------|
| 16 | M-1 | `run_incremental()` 사용으로 전환 |
| 17 | M-2 | 이중 정규화 제거 (한 곳에서만) |
| 18 | M-3 | Drift 기준을 last allocated weight로 변경 |
| 19 | M-4 | `_pod_map: dict[str, StrategyPod]` 추가 |
| 20 | M-5 | `abs(corr)` → signed correlation 사용 |
| 21 | M-6 | alpha 기본값 조정 (0.99 → 반감기 ~69일) |
| 22 | M-7 | `int()` → `float()` 레버리지 전달 |
| 23 | M-8 | `max_fraction` 합계 검증 추가 |

---

## Appendix: 감사 방법론

### 에이전트 구성

| 에이전트 | 대상 파일 | 역할 |
|----------|-----------|------|
| Agent 1 (Core) | models, config, allocator, pod, orchestrator | 수학적 정확성, 타입 안전성, invariant |
| Agent 2 (Risk) | lifecycle, degradation, netting, risk_aggregator, state/history persistence | 상태 머신, 감지기, 넷팅, SQL 안전성 |
| Agent 3 (Integration) | runner, live_runner, orchestrated_runner, CLI, config_loader, 전체 테스트 | 통합 흐름, 테스트 커버리지, 누락 검증 |
| Agent 4 (QA) | — | ruff, pyright, pytest, coverage 실행 |

### 신뢰도

- 3/3 에이전트 독립 확인: C-1, C-3, C-7, H-1, H-5, M-4
- 2/3 에이전트 독립 확인: C-2, C-4, C-5, C-6, H-2~H-8, M-1~M-3, M-5~M-8
- 1/3 에이전트 단독 발견: M-7

교차 검증률 높음 — Critical/High 이슈의 100%가 2개 이상 에이전트에서 독립적으로 발견됨.
