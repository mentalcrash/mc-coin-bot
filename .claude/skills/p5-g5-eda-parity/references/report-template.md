# Gate 5 EDA Parity 리포트 출력 형식

모든 검증 완료 후 아래 형식으로 리포트를 출력한다:

```
============================================================
  GATE 5: EDA PARITY VERIFICATION REPORT
  전략: {display_name} ({registry_key})
  실행일: {YYYY-MM-DD}
  검증 기간: {start} ~ {end} ({period})
  심볼: {best_asset} ({best_tf})
  실행 모드: {fast/standard}
============================================================

  PARITY COMPARISON
  ┌──────────────┬─────────┬─────────┬──────────┬──────────┐
  │ 지표         │   VBT   │   EDA   │   편차   │   판정   │
  ├──────────────┼─────────┼─────────┼──────────┼──────────┤
  │ Sharpe       │  X.XX   │  X.XX   │ +XX.X%   │   —      │
  │ CAGR         │ +XX.X%  │ +XX.X%  │ +XX.X%p  │  PASS    │
  │ Total Return │ +XX.X%  │ +XX.X%  │  XX.X%   │  PASS    │
  │ MDD          │  XX.X%  │  XX.X%  │  XX.X%p  │   —      │
  │ Trades       │   N     │   N     │  X.Xx    │  PASS    │
  │ Win Rate     │  XX.X%  │  XX.X%  │  XX.X%p  │   —      │
  │ Profit Factor│  X.XX   │  X.XX   │  XX.X%   │   —      │
  └──────────────┴─────────┴─────────┴──────────┴──────────┘

  수익 부호: [일치 / 불일치]
  수익률 편차: XX.X% [< 20% PASS / >= 20% FAIL]
  거래 수 비율: X.Xx [0.5x~2.0x PASS / 구조적 사유 PASS / FAIL]

  Parity 판정: [PASS / FAIL]

------------------------------------------------------------
  DISCREPANCY ANALYSIS
------------------------------------------------------------

  1. {괴리 원인 1}: {설명}
  2. {괴리 원인 2}: {설명}

------------------------------------------------------------
  LIVE READINESS CHECK (7/7)
------------------------------------------------------------

  [L1] EventBus Flush       : [PASS / FAIL]
  [L2] Executor Handler 순서 : [PASS / FAIL]
  [L3] Deferred Execution   : [PASS / FAIL]
  [L4] PM Batch Mode        : [PASS / N/A]
  [L5] Position Reconciler  : [PASS / FAIL]
  [L6] Graceful Shutdown    : [PASS / FAIL]
  [L7] Circuit Breaker      : [PASS / FAIL]

  Live Readiness: [PASS / WARNING / FAIL]

------------------------------------------------------------
  CTREND COMPARISON
------------------------------------------------------------

  │ 항목           │ CTREND  │ 현재 전략 │ 해석              │
  │ Sharpe 편차    │ +37.6%  │  XX.X%    │ {해석}            │
  │ 거래 수 비율   │ 0.25x   │  X.Xx     │ {해석}            │
  │ MDD 변화       │ -28.5%  │  XX.X%    │ {해석}            │

============================================================
  GATE 5 SUMMARY
  Parity: [PASS / FAIL]
  Live Readiness: [PASS / WARNING / FAIL]
  최종 판정: [PASS / CONDITIONAL PASS / FAIL]
  다음 단계: [G6 Paper Trading / 코드 수정 후 G5 재시도 / 폐기]
  YAML:       strategies/{strategy_name}.yaml (갱신 완료)
  대시보드:   pipeline report (콘솔 출력, --output로 파일 저장)
============================================================
```
