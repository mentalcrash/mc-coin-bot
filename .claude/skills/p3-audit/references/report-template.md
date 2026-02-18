# Phase 3: 리포트 출력 형식

검증 완료 후 **반드시** 아래 형식으로 리포트를 출력한다:

```
============================================================
  PHASE 3: STRATEGY VERIFICATION REPORT
  전략: [전략명] (registry key)
  감사일: [날짜]
  대상 파일: [분석한 파일 목록]
============================================================

  판정: [PASS / FAIL]

------------------------------------------------------------
  Critical Checklist (C1-C7) — 1개라도 FAIL이면 Phase 3 FAIL
------------------------------------------------------------

  [C1] Look-Ahead Bias        : [PASS / FAIL]
       (세부 사항)

  [C2] Data Leakage            : [PASS / FAIL]
       (세부 사항)

  [C3] Survivorship Bias       : [PASS / FAIL]
       (세부 사항)

  [C4] Signal Vectorization    : [PASS / FAIL]
       (세부 사항)

  [C5] Position Sizing         : [PASS / FAIL]
       (세부 사항)

  [C6] Cost Model              : [PASS / FAIL]
       (세부 사항)

  [C7] Entry/Exit Logic        : [PASS / FAIL]
       (세부 사항)

------------------------------------------------------------
  결함 상세 (FAIL 항목만)
------------------------------------------------------------

  [C?-001] 제목
    위치: src/strategy/{name}/signal.py:45
    문제: (구체적 코드와 함께 설명)
    영향: (실전 결과를 금액/비율로 추정)
    수정: (구체적 코드 수정안)

------------------------------------------------------------
  Warning Checklist (W1-W7) — 기록용, FAIL 사유 아님
------------------------------------------------------------

  [W1] Warmup Period           : [OK / WARNING]
       (세부 사항)

  [W2] Parameter Count         : [OK / WARNING]
       (세부 사항)

  [W3] Regime Concentration    : [OK / WARNING]
       (세부 사항)

  [W4] Turnover                : [OK / WARNING]
       (세부 사항)

  [W5] Correlation             : [OK / WARNING]
       (세부 사항)

  [W6] Derivatives NaN         : [OK / WARNING / N/A]
       (세부 사항)

  [W7] Shared Indicators        : [OK / WARNING]
       (세부 사항)

------------------------------------------------------------
  검증 요약
------------------------------------------------------------

  Critical PASS: N/7
  Critical FAIL: N/7
  Warnings:      N/7
  총 결함:       N건 (CRITICAL: N, HIGH: N, MEDIUM: N)

------------------------------------------------------------
  권장 액션 (우선순위순)
------------------------------------------------------------

  1. [C?-001] 수정 방향 (필수 — Phase 3 통과 조건)
  2. [W?] 개선 권고 (선택)

============================================================
```
