# Pod Lifecycle Reference

> **Pod의 생애주기 상태 전이, 조건, 자본 배분 규칙을 정리한 운영 레퍼런스.**
>
> 전체 Orchestrator 아키텍처는 [strategy-orchestrator.md](strategy-orchestrator.md) 참조.

---

## 1. 상태 요약

| State | 설명 | 자본 배분 규칙 | Active |
|-------|------|---------------|:------:|
| **INCUBATION** | 초기 관찰 기간 (신규 Pod) | `min(weight, initial_fraction)` | O |
| **PRODUCTION** | 본격 운용 (최적 상태) | `clip(weight, min_fraction, max_fraction)` | O |
| **WARNING** | 성과 악화 감지 | `weight * 0.5`, then `clip(min, max)` | O |
| **PROBATION** | 유예 기간 (퇴출 임박) | `min_fraction` 고정 | O |
| **RETIRED** | 운용 종료 / 퇴출 | `0.0` (포지션 청산) | X |

> 코드: `src/orchestrator/models.py` — `LifecycleState(StrEnum)`

---

## 2. State Machine

```
                    ┌──────────────────────────────────────────────┐
                    │            Lifecycle State Machine            │
                    │                                              │
                    │  INCUBATION ──── graduation ────► PRODUCTION │
                    │      │                              │  ▲     │
                    │      │ hard stop              PH    │  │     │
                    │      ▼                     degrade  │  │     │
                    │  RETIRED ◄── expire ── PROBATION    │  │     │
                    │      ▲                     ▲   │    ▼  │     │
                    │      │                     │   │  WARNING     │
                    │      │                     │   │    │         │
                    │      │                     │   │    │ recover │
                    │      └─── hard stop ───────┘   └────┘        │
                    └──────────────────────────────────────────────┘
```

### 전이 경로 일람

| From | To | Trigger | 조건 |
|------|----|---------|------|
| INCUBATION | PRODUCTION | Graduation | 6개 기준 ALL 충족 |
| PRODUCTION | WARNING | PH Detection | PH score > lambda |
| WARNING | PRODUCTION | Recovery | PH score < lambda*0.2 AND 5일+ 경과 |
| WARNING | PROBATION | Timeout | 30일 미회복 |
| PROBATION | PRODUCTION | Strong Recovery | Sharpe >= min_sharpe AND PH score <= 0 |
| PROBATION | RETIRED | Expired | probation_days(30) 경과 |
| ANY | RETIRED | Hard Stop | MDD >= 25% OR 6개월 연속 손실 |

> 코드: `src/orchestrator/lifecycle.py` — `LifecycleManager.evaluate()`

---

## 3. INCUBATION (초기 관찰)

### 진입 조건
- 신규 Pod 생성 시 기본 상태

### 자본 배분
- `min(allocated_weight, initial_fraction)`
- 예: allocator가 30%를 계산해도 `initial_fraction=10%`이면 10%로 제한

### 졸업 기준 (→ PRODUCTION)

**모든 조건을 동시에 충족**해야 승격:

| 기준 | 기본값 | 설명 |
|------|--------|------|
| `min_live_days` | 30 | 최소 실운용 일수 |
| `min_sharpe` | 0.5 | 연환산 Sharpe ratio (양의 edge 확인 수준) |
| `max_drawdown` | 20% | 최대 낙폭 한도 (retirement 25%와 5%p 버퍼) |
| `min_trade_count` | 5 | 최소 거래 횟수 (1D 전략 30일 기준 현실적) |
| `min_calmar` | 0.3 | Calmar ratio (짧은 윈도우 노이즈 고려) |
| `max_portfolio_correlation` | 0.65 | 기존 포트폴리오와의 상관계수 한도 |

> `max_backtest_live_gap` (30%)은 정의만 있고 평가 로직은 deferred 상태.

### 졸업 실패 시
- INCUBATION 유지, `initial_fraction`으로 고정 배분
- 리밸런스마다 재평가 — 기간 제한 없음 (Hard Stop에 걸리지 않는 한 영구 대기)

> 코드: `lifecycle.py:264` — `_check_graduation()`
> 설정: `config.py` — `GraduationCriteria`

---

## 4. PRODUCTION (본격 운용)

### 진입 조건
- INCUBATION에서 졸업 기준 ALL 충족
- WARNING/PROBATION에서 recovery

### 자본 배분
- `clip(allocated_weight, min_fraction, max_fraction)`
- Allocator가 계산한 동적 가중치를 min/max 범위로 제한
- 예: min=2%, max=40% → allocator 50% 계산 시 40%로 clamp

### 열화 감지 (→ WARNING)

**Page-Hinkley (PH) Detector**로 지속적 음의 drift를 실시간 감지:

```
입력: Pod의 일별 수익률 stream

x̄_t = α · x̄_{t-1} + (1-α) · x_t    (EWMA 평활, α=0.99)
m_t += x̄_t - x_t - δ                 (누적 편차, δ=0.005)
M_t = min(M_t, m_t)                   (최소값 추적)

Detection: m_t - M_t > λ              (λ=50.0)
```

| 파라미터 | 기본값 | 의미 |
|----------|--------|------|
| `alpha` | 0.99 | EWMA 가중치 (~69일 half-life) |
| `delta` | 0.005 | 최소 감지 가능 변화량 |
| `lambda_` | 50.0 | 감지 임계값 (높을수록 보수적) |

> PH score가 lambda(50)를 초과하면 WARNING으로 전이.
>
> 코드: `src/orchestrator/degradation.py` — `PageHinkleyDetector`

---

## 5. WARNING (성과 악화 경고)

### 진입 조건
- PRODUCTION에서 PH detector가 열화 감지

### 자본 배분
- `weight * 0.5` (즉시 50% 감축) → `clip(min_fraction, max_fraction)`
- 예: 30% 배분 → 15%로 축소

### 전이 경로

#### Recovery (→ PRODUCTION)
- **조건**: `PH_score < lambda * 0.2` AND `days_in_warning >= 5`
- 의미: PH score가 임계값의 20% 미만으로 하락 + 최소 5일 관찰
- PRODUCTION 복귀 시 **PH detector reset** (누적값 초기화)

#### Escalation (→ PROBATION)
- **조건**: `days_in_warning >= 30`
- 의미: 30일간 WARNING 상태에서 회복 실패

#### Hard Stop (→ RETIRED)
- MDD >= 25% 또는 6개월 연속 손실 시 즉시 퇴출

> 코드: `lifecycle.py:181` — `_evaluate_warning()`

---

## 6. PROBATION (유예 기간)

### 진입 조건
- WARNING에서 30일간 미회복

### 자본 배분
- `min_fraction` 고정 (기본값 2%)
- 최소한의 자본만 유지하며 최종 관찰

### 전이 경로

#### Strong Recovery (→ PRODUCTION)
- **조건**: `sharpe_ratio >= min_sharpe` AND `ph_detector.score <= 0.0`
- 의미: Sharpe가 졸업 기준(1.0) 이상이고 PH detector가 완전 안정
- PRODUCTION 복귀 시 **PH detector reset**

#### Expired (→ RETIRED)
- **조건**: `days_in_probation >= probation_days` (기본값 30)
- 의미: 30일 유예 기간 경과 후 미회복 → 영구 퇴출

#### Hard Stop (→ RETIRED)
- MDD >= 25% 또는 6개월 연속 손실 시 즉시 퇴출

> 코드: `lifecycle.py:204` — `_evaluate_probation()`

---

## 7. RETIRED (운용 종료)

### 진입 조건
- PROBATION 유예 만료
- Hard Stop 트리거 (어떤 상태에서든)

### 자본 배분
- `0.0` — 모든 자본 회수, 포지션 청산
- **Terminal state** — 복귀 불가

### Hard Stop 기준

| 기준 | 기본값 | 적용 범위 |
|------|--------|----------|
| `max_drawdown_breach` | 25% | MDD >= 25% → 즉시 퇴출 |
| `consecutive_loss_months` | 6 | 6개월 연속 손실 → 즉시 퇴출 |

**연속 손실 개월 계산:**
- 30일 청크 기반 (0~29일 = M1, 30~59일 = M2, ...)
- 각 월의 복리 수익률이 음수 → 카운터 +1
- 양수 월 출현 → 카운터 reset

> 코드: `lifecycle.py:225` — `_check_hard_stops()`
> 설정: `config.py` — `RetirementCriteria`

---

## 8. 자본 배분 Clamp 요약

리밸런스 실행 시 Allocator가 계산한 raw weight에 상태별 clamp 적용:

```
raw_weight (from allocator)
    │
    ▼
┌─────────────────────────────────────────┐
│          Lifecycle Clamp                 │
│                                         │
│  RETIRED     → 0.0                      │
│  INCUBATION  → min(w, initial_fraction) │
│  PRODUCTION  → clip(w, min, max)        │
│  WARNING     → clip(w * 0.5, min, max)  │
│  PROBATION   → min_fraction             │
└─────────────────────────────────────────┘
    │
    ▼
합계 > 1.0 이면 비례 축소 (proportional scale-down)
    │
    ▼
pod.capital_fraction 업데이트
```

### PodConfig 기본값

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `initial_fraction` | 10% | INCUBATION 상한 |
| `min_fraction` | 2% | PROBATION 고정값 / PRODUCTION 하한 |
| `max_fraction` | 40% | PRODUCTION 상한 |

> 코드: `src/orchestrator/allocator.py:393` — `_apply_lifecycle_clamps()`

---

## 9. 리밸런스와 생애주기 평가 시점

리밸런스가 트리거될 때 **생애주기 평가 → 자본 배분 → Clamp** 순서로 실행:

```
Rebalance Trigger
    │
    ├─ 1. LifecycleManager.evaluate(pod) — 각 Pod 상태 전이 체크
    │      ├─ 연속 손실 개월 업데이트
    │      ├─ Hard Stop 체크
    │      └─ 상태별 전이 평가 (graduation/degradation/recovery/expire)
    │
    ├─ 2. CapitalAllocator.compute_weights() — 배분 알고리즘 실행
    │
    ├─ 3. _apply_lifecycle_clamps() — 상태별 가중치 제한
    │
    ├─ 4. Pod capital_fraction 업데이트
    │
    └─ 5. Equity 연속성 보장 (adjust_base_equity_on_rebalance)
```

### 리밸런스 트리거 모드

| 모드 | 조건 | 기본값 |
|------|------|--------|
| **CALENDAR** | N일 간격 | 7일 |
| **THRESHOLD** | capital_fraction drift > threshold | 10% |
| **HYBRID** | CALENDAR OR THRESHOLD | 기본값 |

---

## 10. 부가 모니터링 (상태 전이에 미반영)

LifecycleManager는 PH Detector 외에 3가지 추가 모니터를 **관측 전용**으로 지원합니다.
현재 상태 전이에는 반영되지 않으며, 결과 조회만 가능합니다.

| Monitor | 목적 | 설정 메서드 |
|---------|------|------------|
| **GBM Drawdown** | 기하 브라운 운동 기반 낙폭 이상 감지 | `set_gbm_params()` |
| **Distribution Drift** | 백테스트 대비 수익률 분포 변화 감지 | `set_distribution_reference()` |
| **Conformal-RANSAC** | 수익률 감쇠 트렌드 감지 | `set_ransac_params()` |

---

## 11. 예시 시나리오

```
Day 1    Pod 생성 → INCUBATION (capital: 10%)
         │
Day 1-29 Sharpe 0.4, trade_count 3 — 졸업 기준 미충족
         INCUBATION 유지 (capital: 10%)
         │
Day 30   Sharpe 0.8, MDD 12%, trades 7, calmar 0.5, corr 0.4
         모든 졸업 기준 충족
         → PRODUCTION (capital: 25% by allocator)
         │
Day 90   안정적 수익 지속 (Sharpe 0.7)
         계속 PRODUCTION
         │
Day 120  시장 악화, 매일 -0.5% 손실 지속
         PH detector score > 50
         → WARNING (capital: 25% * 0.5 = 12.5%)
         │
Day 130  시장 회복, PH score < 10
         5일+ 경과, recovery 조건 충족
         → PRODUCTION (capital: allocator 재계산)
         │
Day 200  다시 악화, 30일간 미회복
         → PROBATION (capital: 2%)
         │
Day 230  Sharpe 0.3 — probation_days(30) 경과, 미회복
         → RETIRED (capital: 0%)
```

---

## 12. 설정 레퍼런스

```yaml
orchestrator:
  graduation:
    min_live_days: 30
    min_sharpe: 0.5
    max_drawdown: 0.20
    min_trade_count: 5
    min_calmar: 0.3
    max_portfolio_correlation: 0.65
    max_backtest_live_gap: 0.30        # deferred

  retirement:
    max_drawdown_breach: 0.25          # hard stop
    consecutive_loss_months: 6         # hard stop
    rolling_sharpe_floor: 0.3
    probation_days: 30

pods:
  - pod_id: pod-tsmom-major
    initial_fraction: 0.15             # INCUBATION 상한
    max_fraction: 0.40                 # PRODUCTION 상한
    min_fraction: 0.05                 # PROBATION 고정 / PRODUCTION 하한
```

---

## 13. 코드 맵

| 파일 | 역할 |
|------|------|
| `src/orchestrator/models.py` | `LifecycleState` enum, `PodPerformance` |
| `src/orchestrator/config.py` | `GraduationCriteria`, `RetirementCriteria`, `PodConfig` |
| `src/orchestrator/lifecycle.py` | `LifecycleManager` — 상태 전이 자동화 |
| `src/orchestrator/degradation.py` | `PageHinkleyDetector` — 성과 열화 감지 |
| `src/orchestrator/allocator.py` | `_apply_lifecycle_clamps()` — 상태별 가중치 제한 |
| `src/orchestrator/pod.py` | `StrategyPod` — 상태/성과/포지션 보유 |
