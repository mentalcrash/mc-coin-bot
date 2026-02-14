# Gate Pipeline Report Template

모든 Gate 완료 후 (FAIL 또는 G4 PASS) 아래 형식으로 리포트를 출력한다.

```
============================================================
  GATE PIPELINE REPORT
  전략: {display_name} ({registry_key})
  실행일: {YYYY-MM-DD}
  범위: G{start} -> G{end}
  Best Asset: {symbol} (1D)
============================================================

  GATE 1: 단일에셋 백테스트
  Best Asset: {symbol} | Sharpe {X.XX} | CAGR {+XX.X%} | MDD {-XX.X%} | Trades {N}
  판정: [PASS / FAIL]
  {FAIL 사유 또는 주요 관찰}

------------------------------------------------------------

  GATE 2: IS/OOS 70/30
  OOS Sharpe: {X.XX} | Decay: {XX.X%} | OOS Trades: {N}
  판정: [PASS / FAIL]
  {해석}

------------------------------------------------------------

  GATE 3: 파라미터 안정성
  파라미터: {N}/{M} PASS
  {파라미터별 한줄 요약}
  판정: [PASS / FAIL]

------------------------------------------------------------

  GATE 4: 심층검증
  WFA OOS: {X.XX} | Decay: {XX.X%} | Consist: {XX%}
  PBO: {XX%} | DSR: {X.XX} | MC p: {X.XXX}
  판정: [PASS / FAIL]
  {해석}

============================================================
  PIPELINE SUMMARY
  최종 판정: [G4 PASS / G{N} FAIL]
  다음 단계: {G5 EDA Parity / 폐기}
  YAML:       strategies/{strategy_name}.yaml (갱신 완료)
  대시보드:   pipeline report (콘솔 출력, --output로 파일 저장)
============================================================
```
