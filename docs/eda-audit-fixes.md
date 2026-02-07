# EDA Audit Fix Report

EDA 백테스팅 시스템 감사에서 발견된 HIGH 4건 + MEDIUM 5건 이슈 수정 결과입니다.

---

## Summary

| Severity | 수정 건수 | 파일 수 | 신규 테스트 |
|----------|----------|---------|------------|
| HIGH     | 4        | 4       | 5          |
| MEDIUM   | 5        | 3       | 6          |
| **합계** | **9**    | **7**   | **11**     |

- **기존 테스트**: 92개 → **103개** (모두 통과)
- **전체 테스트**: 335개 (모두 통과)
- **Lint (ruff)**: 0 errors
- **Type check (pyright)**: 0 errors

---

## HIGH Issues

### H-001: fill_timestamp → bar_timestamp
**파일**: `src/eda/executors.py`

**문제**: `BacktestExecutor.execute()`에서 `datetime.now(UTC)`를 fill_timestamp으로 사용.
백테스트에서 모든 체결이 현재 시각으로 기록되어 시간 기반 분석이 불가능.

**수정**:
- `_last_bar_timestamp: dict[str, datetime]` 딕셔너리 추가
- `on_bar()`에서 `bar.bar_timestamp` 캐싱
- `execute()`에서 캐시된 bar_timestamp 사용 (fallback: `datetime.now(UTC)`)

**테스트**: `test_fill_timestamp_uses_bar_time`, `test_fill_timestamp_updates_per_bar`

---

### H-002: 펀딩비 post-hoc 보정
**파일**: `src/eda/analytics.py`

**문제**: Perpetual Futures의 펀딩비(8h마다 ~0.01%)가 수익률 계산에 미반영.
연간 ~10% 비용 누적 가능.

**수정**:
- `compute_metrics()`에 `cost_model: CostModel | None` 파라미터 추가
- `cost_model.funding_rate_8h > 0`이면 equity-curve returns에서 `funding_drag` 차감
- `funding_drag = funding_rate_8h × (hours_per_bar / 8.0)`

**테스트**: `test_funding_adjustment_reduces_return`

---

### H-003: Parity 테스트 정량적 tolerance
**파일**: `tests/eda/test_parity.py`

**문제**: `eda_metrics.total_trades >= 0` (항상 pass), 수치 비교 부재.

**수정**:
- `total_trades >= 0` → `total_trades > 0` (EDA도 거래 발생 필수)
- 신규 `test_return_relative_tolerance`: 수익률 상대 오차 30% 이내
- 신규 `test_trade_count_similar`: 거래 횟수 비율 0.5~2.0 범위

---

### H-004: Strategy.run() 실패 카운터 강화
**파일**: `src/eda/strategy_engine.py`

**문제**: `strategy.run()` 실패 시 silent warning만 출력. 연속 실패 감지 불가.

**수정**:
- `_consecutive_failures: dict[str, int]` 딕셔너리 추가
- 실패 시 카운터 증가, 성공 시 리셋
- 연속 3회 이상 실패 시 `logger.error()` + `RiskAlertEvent(alert_level="WARNING")` 발행

**테스트**: `test_strategy_failure_emits_risk_alert`

---

## MEDIUM Issues

### M-001: CAGR/Sharpe timeframe 인식
**파일**: `src/eda/analytics.py`

**문제**: Sharpe에 `np.sqrt(365)`, CAGR에 `n_bars / 365` 하드코딩.
4h/1h 데이터 사용 시 연환산이 잘못됨.

**수정**:
- `_freq_to_hours(freq: str)` 헬퍼 추가 (1D→24, 4h→4, 1h→1, 15T→0.25)
- `_annualized_sharpe(returns, hours_per_bar)`: `np.sqrt(periods_per_year)`
- `_compute_cagr(equity_values, n_bars, hours_per_bar)`: `n_bars × hours / (365 × 24)`
- `compute_metrics(timeframe="1D")` 파라미터 추가

**테스트**: `test_freq_to_hours`, `test_cagr_4h_timeframe`, `test_sharpe_timeframe_aware`

---

### M-002: Analytics 거래 추적 가중평균 진입가
**파일**: `src/eda/analytics.py`

**문제**: 같은 방향 추가 매매 시 기존 OpenTrade를 덮어쓰기하여 진입가/수량 유실.

**수정**:
- `_on_fill()`: 같은 방향 추가 매매 감지
- 가중평균 진입가 계산: `(old_price × old_size + new_price × new_size) / total_size`
- 수량 누적, 수수료 합산

**테스트**: `test_additional_buy_accumulates_size`

---

### M-003: Equity curve bar 단위 정규화
**파일**: `src/eda/analytics.py`

**문제**: 한 bar 내 여러 BalanceUpdateEvent가 모두 equity curve에 기록.
Signal→Fill→BalanceUpdate 체인에서 중간 값이 noise 유발.

**수정**:
- `_last_equity_ts` 추적
- 같은 timestamp의 여러 업데이트 → 마지막 값으로 덮어쓰기

**테스트**: `test_equity_curve_one_point_per_bar`

---

### M-004: DataFeed 데이터 품질 검증
**파일**: `src/eda/data_feed.py`

**문제**: NaN/Inf, high < low 등 비정상 bar가 필터링 없이 발행.

**수정**:
- `_validate_bar()` 정적 메서드 추가
- NaN/Inf 체크 (open, high, low, close, volume)
- high >= low 체크
- 검증 실패 시 bar skip + logger.warning

**테스트**: `test_nan_bar_skipped`, `test_high_less_than_low_skipped`

---

### M-005: backtest_fill_delay_bars 제거
**파일**: `src/models/eda.py`, `tests/models/test_eda.py`

**문제**: `EDAConfig.backtest_fill_delay_bars` 필드가 정의만 있고 어떤 컴포넌트에서도 미사용.

**수정**:
- `EDAConfig`에서 `backtest_fill_delay_bars` 필드 삭제
- 관련 docstring 정리
- 테스트에서 해당 필드 참조 3건 제거

---

## Runner 수정

**파일**: `src/eda/runner.py`

`analytics.compute_metrics()`에 `timeframe`과 `cost_model` 파라미터 전달:
```python
metrics = analytics.compute_metrics(
    timeframe=self._data.timeframe,
    cost_model=self._config.cost_model,
)
```

---

## 수정된 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `src/eda/executors.py` | H-001: bar_timestamp 캐시 + fill_timestamp 사용 |
| `src/eda/strategy_engine.py` | H-004: 연속 실패 카운터 + RiskAlertEvent |
| `src/eda/analytics.py` | H-002, M-001, M-002, M-003: 펀딩비/timeframe/가중평균/equity정규화 |
| `src/eda/data_feed.py` | M-004: _validate_bar() 데이터 품질 검증 |
| `src/eda/runner.py` | compute_metrics에 timeframe + cost_model 전달 |
| `src/models/eda.py` | M-005: backtest_fill_delay_bars 제거 |
| `tests/eda/test_parity.py` | H-003: 정량적 tolerance 테스트 추가 |
| `tests/eda/test_executors.py` | H-001 테스트 2건 |
| `tests/eda/test_analytics.py` | H-002, M-001, M-002, M-003 테스트 6건 |
| `tests/eda/test_strategy_engine.py` | H-004 테스트 1건 |
| `tests/eda/test_data_feed.py` | M-004 테스트 2건 |
| `tests/models/test_eda.py` | M-005: backtest_fill_delay_bars 참조 제거 |
