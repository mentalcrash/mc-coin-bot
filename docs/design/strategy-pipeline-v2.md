# Strategy Pipeline v2 — Design Document

> **Status**: DRAFT
> **Date**: 2026-02-19
> **Author**: Claude (Senior Engineer) + User (Architect)

---

## 1. Motivation

현재 파이프라인(v1)의 한계:

| 문제 | 영향 |
|------|------|
| G0A/G0B/G1/G2H 같은 불투명한 네이밍 | 새 세션에서 진행 상황 파악 어려움 |
| 세션 간 컨텍스트 단절 | 각 스킬이 YAML에 충분한 정보를 남기지 않아 재작업 발생 |
| Discovery 단계가 정성적 | 75개 데이터셋, 53개 지표를 체계적으로 활용하지 못함 |
| 단순 전략 위주 | 레짐, 온체인, 매크로, 옵션 데이터 미활용 |
| 하위호환 부담 | 83개 RETIRED 전략의 v1 YAML이 스키마 진화를 제약 |

### 목표

1. **Phase 1~7 명확한 단계 체계** — 누구나 현재 위치를 즉시 파악
2. **세션 독립성** — YAML에 충분한 컨텍스트를 기록하여 새 세션에서 즉시 이어 작업
3. **데이터 기반 발굴** — 75개 데이터셋 분석을 통한 정량적 알파 발견
4. **고도화된 전략** — 레짐 적응형, 멀티팩터, 크로스에셋, 이벤트 기반 전략
5. **통계적 엄밀성** — 다중 테스트 보정, DSR, PBO를 전 과정에 통합

---

## 2. Phase Architecture

```
Phase 1         Phase 2           Phase 3          Phase 4
Alpha Research → Implementation → Code Audit     → Backtest
(발굴·설계)      (코드 구현)       (코드 검증)       (성과 검증)

     Phase 5           Phase 6              Phase 7
  → Robustness      → Deep Validation    → Live Readiness
    (강건성 검증)      (통계 검증)           (실전 준비)
```

### Phase-Status 매핑

| Phase | v1 Gate | Status 전환 | 스킬 이름 |
|-------|---------|------------|----------|
| Phase 1: Alpha Research | G0A | → CANDIDATE | `p1-research` |
| Phase 2: Implementation | p2-implement | → IMPLEMENTED | `p2-implement` |
| Phase 3: Code Audit | G0B | (변경 없음) | `p3-audit` |
| Phase 4: Backtest | G1 + G2 | → TESTING | `p4-backtest` |
| Phase 5: Robustness | G2H + G3 | (변경 없음) | `p5-robustness` |
| Phase 6: Deep Validation | G4 | (변경 없음) | `p6-validation` |
| Phase 7: Live Readiness | G5 | → ACTIVE | `p7-live` |

### Status Lifecycle

```
CANDIDATE ──[P2 완료]──→ IMPLEMENTED ──[P4 첫 PASS]──→ TESTING ──[P7 PASS]──→ ACTIVE
     │                       │                            │
     └── FAIL ──→ RETIRED    └── FAIL ──→ RETIRED         └── FAIL ──→ RETIRED
```

---

## 3. Phase Details

### Phase 1: Alpha Research (발굴·설계)

**목적**: 데이터 분석을 통한 알파 소스 발견 + 전략 설계

**v1 대비 변경점**:

- 정성적 6항목 스코어카드 → **정량적 데이터 분석 기반** 발굴
- 단일 지표 전략 → **멀티소스 복합 전략** 지원
- Opus 4.6이 직접 데이터 패턴 분석 수행

#### 3.1.1 입력

| 항목 | 소스 |
|------|------|
| Pipeline 현황 | `pipeline status` |
| 기존 실패 패턴 | `lessons/*.yaml` + RETIRED 전략 YAML |
| 사용 가능 데이터 | `catalog list` (75 datasets) |
| 사용 가능 지표 | `catalogs/indicators.yaml` (53 indicators) |
| 레짐 시스템 | `src/regime/` (5 detectors + ensemble) |
| 기존 ACTIVE 전략 | CTREND, Anchor-Mom 성과·상관 |

#### 3.1.2 데이터 기반 발굴 프레임워크

**Step A: 데이터 탐색 (Quantitative Screening)**

Opus 4.6이 수행하는 정량 분석:

```
1. 카탈로그 데이터 로드 — 관심 데이터셋 선정
2. 리턴 상관 분석 — 데이터 시그널 vs 미래 수익률 rank IC
3. 레짐 조건부 분석 — TRENDING/RANGING/VOLATILE별 IC 분해
4. 리드-래그 분석 — 최적 시차 탐색 (1D~30D)
5. 비선형 관계 탐색 — 분위별 수익률 (quintile spread)
6. 기존 전략 직교성 — ACTIVE 전략과의 상관 < 0.3 확인
```

**Step B: 알파 소스 분류 (6 Categories)**

| Category | 데이터 소스 | 전략 유형 예시 |
|----------|-----------|--------------|
| **Trend** | OHLCV, Indicators | 모멘텀, 트렌드 강도, 채널 브레이크아웃 |
| **Mean-Reversion** | OHLCV, Bollinger, Z-Score | 과매도/과매수 반전, 변동성 수축→확장 |
| **Sentiment** | Fear&Greed, LS Ratio, Funding | 군중 심리 역행, 과열/공포 시그널 |
| **Flow** | On-chain(MVRV, Stablecoin, TVL) | 자금 흐름, 스마트머니 추적 |
| **Macro** | FRED, yfinance, CoinGecko | 달러·금리·유동성 레짐, 크로스에셋 |
| **Derivatives** | OI, Liquidation, Options | 포지셔닝 불균형, 감마 스퀴즈, 텀스트럭처 |

**Step C: 전략 설계 문서**

Phase 1 완료 시 YAML에 기록해야 하는 설계 정보:

```yaml
phases:
  P1:
    status: PASS
    date: "2026-02-19"
    # ── 발굴 과정 ──
    data_analysis:
      datasets_screened:          # 분석한 데이터셋
        - fear_greed
        - funding_rate
        - stablecoin_total
      key_findings:               # 핵심 발견
        - "Fear&Greed < 25 후 7일 수익률 평균 +3.2% (p=0.02)"
        - "Stablecoin 유입 z-score > 2 시 30일 모멘텀 유의"
      rank_ic:                    # 시그널-리턴 상관
        signal_name: "composite_sentiment"
        ic_mean: 0.048
        ic_std: 0.021
        ic_ir: 2.29               # IC / IC_std
      regime_breakdown:           # 레짐별 IC
        TRENDING: 0.061
        RANGING: 0.032
        VOLATILE: 0.055
      active_correlation:         # 기존 ACTIVE와 상관
        ctrend: 0.18
        anchor-mom: 0.22
    # ── 스코어카드 (정량화) ──
    scorecard:
      total: 24
      max: 30
      items:
        economic_rationale: 4
        ic_verification: 5
        category_success_rate: 3
        regime_independence: 4
        ensemble_contribution: 5
        capacity: 3
    # ── 전략 설계 ──
    design:
      hypothesis: "극단 공포 + 스테이블코인 유입 = 기관 매집 시그널"
      alpha_category: "Sentiment + Flow"
      entry_logic: |
        1. Fear&Greed 7DMA < 30 (공포)
        2. Stablecoin 7D 변화율 z-score > 1.5 (자금 유입)
        3. Regime != VOLATILE (필터)
        → Long signal (strength = normalized composite)
      exit_logic: |
        1. Fear&Greed 7DMA > 65 (탐욕)
        2. Trailing Stop 3.0x ATR
        3. Regime 전환 시 포지션 축소
      short_mode: "HEDGE_ONLY"
      timeframe: "1D"
      target_assets:
        - "BTC/USDT"
        - "ETH/USDT"
        - "SOL/USDT"
      required_data:              # 필요 데이터셋
        - fear_greed
        - stablecoin_total
        - ohlcv_1m
      required_indicators:        # 필요 지표
        - atr
        - realized_volatility
      regime_adaptation: "conditional_filter"  # probability_weighted | conditional_filter | direction_weighted
      risk_params:
        vol_target: 0.40
        max_leverage: 2.0
        trailing_stop_atr: 3.0
        system_stop_loss: 0.10
      expected_characteristics:   # 사전 예상치
        sharpe_range: [0.8, 1.8]
        annual_trades: [40, 120]
        mdd_range: [15, 35]
        holding_period_days: [3, 15]
```

#### 3.1.3 고도화 전략 유형

Phase 1에서 발굴 가능한 전략 분류:

| 유형 | 복잡도 | 데이터 | 설명 |
|------|--------|--------|------|
| **Single-Factor** | Low | OHLCV | 단일 지표 기반 (기존 v1 수준) |
| **Multi-Factor Composite** | Medium | OHLCV + 2-3 보조 | 2~4개 팩터 가중 합성 |
| **Regime-Adaptive** | Medium | OHLCV + Regime | 레짐별 로직/파라미터 분기 |
| **Cross-Data** | High | OHLCV + On-chain/Macro | 이종 데이터 결합 (Flow + Price) |
| **Event-Driven** | High | Derivatives + On-chain | 임계값 이벤트 트리거 (청산 캐스케이드 등) |
| **Relative Value** | High | Multi-Asset OHLCV | 크로스에셋 스프레드/페어 |

#### 3.1.4 PASS 기준

| 항목 | 기준 | 비고 |
|------|------|------|
| Scorecard | >= 21/30 | v1과 동일 |
| Rank IC | \|IC\| > 0.02 | 정량 검증 필수 |
| Regime IC | 2/3 레짐에서 양수 | 레짐 독립성 |
| Active 상관 | < 0.5 | 앙상블 기여 |
| 설계 문서 | 모든 필드 작성 | 세션 연속성 보장 |

---

### Phase 2: Implementation (코드 구현)

**목적**: Phase 1 설계를 4-file 구조로 구현 + 테스트

**v1 대비 변경점**:

- Phase 1의 `design` 섹션을 읽어 **설계 의도를 정확히 반영**
- 레짐/파생상품/온체인 데이터 연동 패턴 표준화
- 앙상블 호환성 기본 내장

#### 입력 (YAML에서 읽기)

```
phases.P1.design.entry_logic      → signal.py 구현 가이드
phases.P1.design.exit_logic       → signal.py 청산 로직
phases.P1.design.required_data    → config.py required_columns
phases.P1.design.required_indicators → preprocessor.py
phases.P1.design.regime_adaptation → strategy.py regime 통합 방식
phases.P1.design.risk_params      → config.py 기본값
phases.P1.design.short_mode       → config.py ShortMode
```

#### 출력 (YAML에 기록)

```yaml
phases:
  P2:
    status: PASS
    date: "2026-02-19"
    implementation:
      files_created:
        - src/strategy/fear_flow/config.py
        - src/strategy/fear_flow/preprocessor.py
        - src/strategy/fear_flow/signal.py
        - src/strategy/fear_flow/strategy.py
        - src/strategy/fear_flow/__init__.py
      test_files:
        - tests/strategy/test_fear_flow_config.py
        - tests/strategy/test_fear_flow_preprocessor.py
        - tests/strategy/test_fear_flow_signal.py
        - tests/strategy/test_fear_flow_strategy.py
      test_count: 56
      test_pass: 56
      coverage_pct: 94.2
    design_decisions:              # 구현 중 내린 결정
      - "Fear&Greed 7DMA → EMA(7) 사용 (더 반응적)"
      - "Stablecoin z-score 룩백 90일 (분기 변동 제거)"
    deviations_from_design: []     # P1 설계와 달라진 점 (있으면)
```

#### PASS 기준

| 항목 | 기준 |
|------|------|
| 4-file 구조 완성 | 필수 5파일 존재 |
| 테스트 전체 통과 | pytest 0 failures |
| Ruff + Pyright | 0 errors |
| Registry 등록 | `@register("name")` 확인 |
| P1 설계 반영 | `deviations_from_design` 문서화 |

---

### Phase 3: Code Audit (코드 검증)

**목적**: 구현된 코드의 정합성 검증 (look-ahead bias, data leakage 등)

**v1 대비 변경점**:

- 자동화 스캔 강화 (`scripts/scan_strategy.sh` 확장)
- Warning 체크 W6(Derivatives NaN), W7(Shared Indicators) 추가

#### 입력

```
phases.P1.design.required_data    → C2/C3 검증 범위 결정
phases.P1.design.regime_adaptation → W3 레짐 집중도 검증 기준
phases.P2.implementation.files_created → 검사 대상 파일
```

#### 출력

```yaml
phases:
  P3:
    status: PASS
    date: "2026-02-19"
    critical_checks:
      C1_look_ahead: PASS
      C2_data_leakage: PASS
      C3_survivorship: PASS
      C4_vectorization: PASS
      C5_position_sizing: PASS
      C6_cost_model: PASS
      C7_entry_exit: PASS
    warnings:
      W1_warmup: "OK — warmup=90, max indicator lookback=90"
      W2_params: "OK — 4 params, est. trades 80+, ratio 20"
      W3_regime: "WARN — 70% profit in TRENDING regime"
      W4_turnover: "OK — est. 60 trades/year"
      W5_correlation: "OK — max corr with ACTIVE: 0.22"
    recommendations:              # 다음 단계 참고사항
      - "W3: Phase 4에서 레짐별 성과 분해 확인 필요"
```

#### PASS 기준

C1~C7 모든 Critical check = PASS (v1과 동일)

---

### Phase 4: Backtest (성과 검증)

**목적**: 단일에셋 백테스트 + IS/OOS 과적합 검증

**v1 대비 변경점**:

- G1 + G2를 하나의 Phase로 통합 (한 세션에서 연속 실행)
- 에셋별 레짐 분해 분석 추가
- 비용 민감도 분석 필수화

#### 입력

```
phases.P1.design.target_assets     → 백테스트 대상 에셋
phases.P1.design.timeframe         → 타임프레임
phases.P1.design.risk_params       → PM/RM 파라미터
phases.P1.design.expected_characteristics → 사전 예상치 대비 검증
```

#### 출력

```yaml
phases:
  P4:
    status: PASS
    date: "2026-02-19"
    # ── 단일에셋 백테스트 (구 G1) ──
    single_asset:
      test_period: "2020-01-01 ~ 2025-12-31"
      results:
        - symbol: SOL/USDT
          sharpe: 1.85
          cagr: 67.2
          mdd: 28.4
          trades: 92
          profit_factor: 1.72
          win_rate: 48.3
        - symbol: BTC/USDT
          sharpe: 1.42
          cagr: 45.1
          mdd: 22.8
          trades: 88
          # ...
      best_asset: SOL/USDT
      regime_breakdown:            # 레짐별 성과 분해
        SOL/USDT:
          TRENDING: { sharpe: 2.31, pct_profit: 65 }
          RANGING: { sharpe: 0.45, pct_profit: 15 }
          VOLATILE: { sharpe: 1.12, pct_profit: 20 }
      cost_sensitivity:            # 비용 민감도
        base_fee_bps: 4.0
        sharpe_at_6bps: 1.52       # 50% 비용 증가 시
        sharpe_at_8bps: 1.21       # 100% 비용 증가 시
        break_even_fee_bps: 18.5   # Sharpe = 0 되는 비용
      vs_expected:                 # P1 예상 대비
        sharpe_in_range: true
        trades_in_range: true
        mdd_in_range: true
    # ── IS/OOS 검증 (구 G2) ──
    is_oos:
      split_ratio: "70/30"
      is_sharpe: 2.12
      oos_sharpe: 1.45
      decay_pct: 31.6
      oos_trades: 28
      oos_total_return: 42.3
```

#### PASS 기준

| 항목 | 기준 | 출처 |
|------|------|------|
| Best Asset Sharpe | > 1.0 | 단일에셋 |
| Best Asset CAGR | > 20% | 단일에셋 |
| Best Asset MDD | < 40% | 단일에셋 |
| Best Asset Trades | > 50 | 단일에셋 |
| OOS Sharpe | >= 0.3 | IS/OOS |
| Sharpe Decay | < 50% | IS/OOS |
| OOS Trades | >= 15 | IS/OOS |
| Break-even Fee | > 8 bps | 비용 민감도 (신규) |

**Immediate Fail**: MDD > 50%, 모든 에셋 Sharpe < 0, Break-even Fee < 6 bps

---

### Phase 5: Robustness (강건성 검증)

**목적**: 파라미터 최적화 + 안정성 검증

**v1 대비 변경점**:

- G2H + G3를 하나의 Phase로 통합
- 최적화 결과가 자동으로 안정성 검증 범위를 결정

#### 입력

```
phases.P4.single_asset.best_asset  → 최적화 대상 에셋
phases.P1.design.risk_params       → 파라미터 범위 기준
phases.P4.is_oos.is_sharpe         → 최적화 목표 기준선
```

#### 출력

```yaml
phases:
  P5:
    status: PASS
    date: "2026-02-19"
    # ── 파라미터 최적화 (구 G2H) ──
    optimization:
      method: "Optuna TPE"
      n_trials: 200
      best_params:
        fear_threshold: 28
        stablecoin_zscore_lookback: 85
        greed_exit_threshold: 62
        ema_period: 7
      is_sharpe: 2.35
      oos_sharpe: 1.52
      improvement_vs_default: "+12.3%"
    # ── 파라미터 안정성 (구 G3) ──
    stability:
      sweep_ranges:                # 자동 생성된 탐색 범위
        fear_threshold: [20, 35]
        stablecoin_zscore_lookback: [60, 120]
      plateau_pct: 78.5            # Sharpe > 0 비율
      pm20_all_positive: true      # ±20% 모두 양수
      heatmap_summary:             # 파라미터 조합별 요약
        total_combos: 256
        sharpe_positive: 201
        sharpe_above_1: 124
```

#### PASS 기준

| 항목 | 기준 |
|------|------|
| 최적화 완료 | 정상 종료 |
| Plateau | >= 60% |
| ±20% 안정성 | 모든 핵심 파라미터에서 Sharpe > 0 |

---

### Phase 6: Deep Validation (통계 검증)

**목적**: 과적합 여부를 통계적으로 검증 (WFA + CPCV + PBO + DSR + MC)

**v1 대비 변경점**:

- 구조 동일하나 YAML 출력 표준화
- P1~P5 컨텍스트를 활용한 해석 가이드 제공

#### 입력

```
phases.P4.single_asset.best_asset  → 검증 대상
phases.P5.optimization.best_params → 최적 파라미터
phases.P1.design.hypothesis        → 결과 해석 컨텍스트
```

#### 출력

```yaml
phases:
  P6:
    status: PASS
    date: "2026-02-19"
    wfa:
      n_folds: 5
      oos_sharpe_mean: 0.92
      oos_sharpe_std: 0.31
      decay_pct: 28.4
      consistency_pct: 80.0        # 4/5 folds positive
      fold_details:
        - { fold: 1, is_sharpe: 1.8, oos_sharpe: 1.1, decay: 38.9 }
        - { fold: 2, is_sharpe: 2.1, oos_sharpe: 0.85, decay: 59.5 }
        - { fold: 3, is_sharpe: 1.9, oos_sharpe: 0.72, decay: 62.1 }
        - { fold: 4, is_sharpe: 2.3, oos_sharpe: 1.15, decay: 50.0 }
        - { fold: 5, is_sharpe: 1.7, oos_sharpe: 0.78, decay: 54.1 }
    cpcv:
      pbo: 22.0                    # Probability of Backtest Overfitting
      pbo_path: "A"                # A: pbo < 40% | B: dual-path
    dsr:
      value: 0.97
      n_trials: 87                 # Pipeline 전체 시도 수
    monte_carlo:
      n_simulations: 1000
      p_value: 0.003
      median_sharpe: 1.62
      ci_95: [0.91, 2.34]
    interpretation:                # 종합 해석
      overfitting_risk: "LOW"      # LOW | MEDIUM | HIGH
      key_concern: "Fold 2-3 decay > 50% — 특정 레짐 구간 취약"
      recommendation: "Phase 7 진행 권장, 레짐 모니터링 중점"
```

#### PASS 기준

| 항목 | 기준 |
|------|------|
| WFA OOS Sharpe | >= 0.5 |
| WFA Decay | < 40% |
| WFA Consistency | >= 60% |
| PBO | Path A: < 40% \| Path B: < 80% + 모든 fold OOS 양수 + MC p < 0.05 |
| DSR | > 0.95 |
| MC p-value | < 0.05 |

---

### Phase 7: Live Readiness (실전 준비)

**목적**: VBT vs EDA 정합성 + 라이브 인프라 검증

**v1 대비 변경점**:

- 라이브 배포 설정 생성 포함
- 모니터링 알림 룰 설정 포함

#### 입력

```
phases.P4.single_asset.best_asset   → EDA 테스트 에셋
phases.P5.optimization.best_params  → 최적 파라미터
phases.P1.design.risk_params        → PM/RM 설정
```

#### 출력

```yaml
phases:
  P7:
    status: PASS
    date: "2026-02-19"
    # ── EDA Parity ──
    parity:
      vbt_sharpe: 1.85
      eda_sharpe: 2.12
      return_deviation_pct: 8.3
      profit_sign_match: true
      trade_ratio: 0.92            # EDA trades / VBT trades
    # ── 라이브 준비 점검 ──
    live_readiness:
      L1_eventbus_flush: PASS
      L2_executor_order: PASS
      L3_deferred_execution: PASS
      L4_pm_batch_mode: N/A        # single-asset
      L5_position_reconciler: PASS
      L6_graceful_shutdown: PASS
      L7_circuit_breaker: PASS
    # ── 배포 설정 ──
    deployment:
      config_file: "config/live/fear-flow.yaml"
      initial_assets: ["SOL/USDT", "BTC/USDT"]
      initial_allocation: 0.10     # 포트폴리오의 10%
      monitoring_alerts:
        - "Sharpe 30D < 0 → WARN"
        - "MDD > 25% → ALERT"
        - "Daily PnL < -3% → CIRCUIT_BREAKER"
```

#### PASS 기준

| 항목 | 기준 |
|------|------|
| 수익 부호 일치 | VBT/EDA 동일 |
| 수익률 편차 | < 20% |
| 거래 수 비율 | 0.5x ~ 2.0x |
| L1/L2/L3/L6/L7 | 모든 Critical PASS |
| 배포 설정 | config YAML 생성 완료 |

---

## 4. YAML Schema v2

### 전체 구조

```yaml
# strategies/{name}.yaml — Strategy Pipeline v2
version: 2                          # 스키마 버전 (신규)

meta:
  name: "fear-flow"                 # kebab-case registry key
  display_name: "Fear & Flow"
  category: "Cross-Data"            # 전략 유형
  timeframe: "1D"
  short_mode: "HEDGE_ONLY"
  status: "TESTING"                 # CANDIDATE | IMPLEMENTED | TESTING | ACTIVE | RETIRED
  created_at: "2026-02-19"
  retired_at: null
  economic_rationale: "극단 공포 + 스테이블코인 유입 = 기관 매집 시그널"
  rationale_references:
    - type: paper
      title: "Fear and Greed in Crypto Markets"
      source: "SSRN"
      url: "https://..."
      relevance: "F&G 극단값의 예측력 실증"

parameters:                         # 최종 파라미터 (P5 최적화 후 갱신)
  fear_threshold: 28
  stablecoin_zscore_lookback: 85
  greed_exit_threshold: 62
  ema_period: 7

phases:                             # Phase 1~7 결과 (세션 간 컨텍스트)
  P1: { ... }                      # 상세는 Section 3 참조
  P2: { ... }
  P3: { ... }
  P4: { ... }
  P5: { ... }
  P6: { ... }
  P7: { ... }

asset_performance:                  # P4 완료 후 기록 (v1과 동일)
  - symbol: SOL/USDT
    sharpe: 1.85
    cagr: 67.2
    mdd: 28.4
    trades: 92
    profit_factor: 1.72
    win_rate: 48.3

decisions:                          # 주요 의사결정 이력 (v1과 동일)
  - date: "2026-02-19"
    phase: P1                       # gate → phase로 변경
    verdict: PASS
    rationale: "24/30, IC=0.048"
```

### v1 → v2 변경 사항 요약

| 항목 | v1 | v2 |
|------|----|----|
| 스키마 버전 | 없음 | `version: 2` |
| Gate 식별자 | `G0A`, `G0B`, `G1`... | `P1`, `P2`, `P3`... |
| Gate 결과 구조 | `gates: { G0A: { status, date, details } }` | `phases: { P1: { status, date, ...rich data } }` |
| Phase 세부 정보 | `details: dict` (비구조) | **Phase별 전용 스키마** (구조화) |
| Decision.gate | `GateId` | `PhaseId` |
| 설계 정보 | 없음 | `phases.P1.design` (전략 설계 전문) |
| 구현 정보 | 없음 | `phases.P2.implementation` (파일, 테스트) |
| 레짐 분해 | 없음 | `phases.P4.single_asset.regime_breakdown` |
| 비용 민감도 | 없음 | `phases.P4.single_asset.cost_sensitivity` |
| 배포 설정 | 없음 | `phases.P7.deployment` |

---

## 5. Model 변경 사항

### 5.1 PhaseId (GateId 대체)

```python
class PhaseId(StrEnum):
    P1 = "P1"  # Alpha Research
    P2 = "P2"  # Implementation
    P3 = "P3"  # Code Audit
    P4 = "P4"  # Backtest
    P5 = "P5"  # Robustness
    P6 = "P6"  # Deep Validation
    P7 = "P7"  # Live Readiness

PHASE_ORDER: list[PhaseId] = [
    PhaseId.P1, PhaseId.P2, PhaseId.P3, PhaseId.P4,
    PhaseId.P5, PhaseId.P6, PhaseId.P7,
]
```

### 5.2 PhaseResult (GateResult 대체)

```python
class PhaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: PhaseVerdict           # PASS | FAIL
    date: date
    details: dict[str, Any] = {}   # Phase별 상세 데이터 (자유 구조)
```

`details`는 Phase별로 다른 구조를 가지지만, 공통 인터페이스(`status`, `date`)를 유지.

### 5.3 StrategyRecord v2

```python
class StrategyRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 2
    meta: StrategyMeta
    parameters: dict[str, Any] = {}
    phases: dict[PhaseId, PhaseResult] = {}    # gates → phases
    asset_performance: list[AssetMetrics] = []
    decisions: list[Decision] = []
```

### 5.4 Gate Criteria 파일

`gates/criteria.yaml` → `gates/phase-criteria.yaml`로 리네임.

Gate ID를 Phase ID로 매핑:

| v1 `gate_id` | v2 `phase_id` |
|--------------|---------------|
| G0A | P1 |
| G0B | P3 |
| G1 + G2 | P4 |
| G2H + G3 | P5 |
| G4 | P6 |
| G5 | P7 |

P2(Implementation)는 criteria YAML이 아닌 스킬 내부에서 검증.

---

## 6. Migration Plan

### 6.1 전략 YAML 마이그레이션

**대상**: `strategies/*.yaml` (100+ 파일)

**규칙**:

| 조건 | 처리 |
|------|------|
| status=RETIRED + gates 없음 | 삭제 (데이터 없는 RETIRED) |
| status=RETIRED + gates 있음 | v2 변환 (학습 데이터 보존) |
| status=ACTIVE | v2 변환 (필수) |
| status=CANDIDATE/IMPLEMENTED/TESTING | v2 변환 |

**변환 로직**:

```python
def migrate_v1_to_v2(v1: dict) -> dict:
    """v1 YAML → v2 YAML 변환."""
    v2 = {
        "version": 2,
        "meta": v1["meta"],  # 구조 동일
        "parameters": v1.get("parameters", {}),
        "phases": {},
        "asset_performance": v1.get("asset_performance", []),
        "decisions": [],
    }
    # Gate → Phase 매핑
    gate_to_phase = {
        "G0A": "P1", "G0B": "P3",
        "G1": "P4", "G2": "P4",     # G1+G2 → P4
        "G2H": "P5", "G3": "P5",    # G2H+G3 → P5
        "G4": "P6", "G5": "P7",
    }
    for gate_id, result in v1.get("gates", {}).items():
        phase_id = gate_to_phase[gate_id]
        if phase_id not in v2["phases"]:
            v2["phases"][phase_id] = result
        else:
            # P4, P5는 두 gate가 합쳐지므로 details merge
            v2["phases"][phase_id]["details"].update(result.get("details", {}))

    # Decision 변환
    for d in v1.get("decisions", []):
        d["phase"] = gate_to_phase.get(d.pop("gate", ""), "P1")
        v2["decisions"].append(d)

    return v2
```

### 6.2 코드 마이그레이션

| 파일 | 변경 |
|------|------|
| `src/pipeline/models.py` | `GateId` → `PhaseId`, `GateResult` → `PhaseResult` |
| `src/pipeline/store.py` | `record_gate()` → `record_phase()`, YAML key `gates` → `phases` |
| `src/pipeline/gate_models.py` | gate_id → phase_id |
| `src/pipeline/gate_store.py` | 파일명 변경, phase_id 사용 |
| `gates/criteria.yaml` | → `gates/phase-criteria.yaml`, ID 변경 |
| CLI (`src/cli/pipeline_cmd.py`) | gate 서브커맨드 → phase 서브커맨드 |
| 모든 테스트 | gate → phase 용어 변경 |

### 6.3 스킬 마이그레이션

| v1 스킬 | v2 스킬 | 주요 변경 |
|---------|---------|----------|
| `p1-g0a-discover` | `p1-research` | 데이터 기반 발굴 프레임워크 전면 개편 |
| `p2-implement` | `p2-implement` | P1.design 입력 연동 강화 |
| `p3-g0b-verify` | `p3-audit` | 리네임 + YAML 출력 구조화 |
| `p4-g1g4-gate` | `p4-backtest` | G1+G2만 담당 (P5, P6 분리) |
| (p4 일부) | `p5-robustness` | G2H+G3 분리 독립 |
| (p4 일부) | `p6-validation` | G4 분리 독립 |
| `p5-g5-eda-parity` | `p7-live` | 리네임 + 배포 설정 생성 추가 |

---

## 7. Data-Driven Discovery Framework (Phase 1 심화)

### 7.1 Opus 4.6 분석 역할

Phase 1에서 Opus 4.6은 다음을 수행:

```
1. 가설 생성 (Hypothesis Generation)
   - 학술 문헌 + lessons + 실패 패턴 기반 아이디어
   - 데이터 카탈로그 조합 탐색

2. 정량 검증 (Quantitative Screening)
   - 백테스트 엔진으로 Rank IC 계산 코드 실행
   - 레짐별 IC 분해
   - 기존 ACTIVE와의 상관 계산

3. 패턴 탐색 (Pattern Discovery)
   - 온체인/매크로 데이터에서 리턴 예측력 있는 패턴 탐색
   - 임계값 최적화 (예: Fear&Greed < X일 때 최대 IC)
   - 조건부 관계 탐색 (레짐 A에서만 유효한 시그널)

4. 전략 설계 (Strategy Architecture)
   - 발견된 패턴을 트레이딩 규칙으로 구조화
   - 리스크 파라미터 초기값 설정
   - 앙상블 기여도 사전 평가
```

### 7.2 활용 가능한 데이터 조합 (예시)

| 전략 유형 | Primary Data | Secondary Data | Regime |
|----------|-------------|----------------|--------|
| Sentiment Reversal | Fear&Greed | Funding Rate Z-Score | VOLATILE 필터 |
| Flow Momentum | Stablecoin Supply | TVL Change | TRENDING 부스트 |
| Macro Rotation | DXY, VIX | BTC Dominance | 전체 |
| Derivatives Squeeze | OI Divergence | Liquidation Intensity | 전체 |
| Vol Structure | DVOL, RV Ratio | Term Structure Slope | RANGING 진입 |
| Smart Money | MVRV | Exchange Flow | TRENDING 부스트 |
| Cross-Asset | ETH/BTC Ratio | SOL/ETH Ratio | 전체 |
| Yield Curve | T10Y2Y, DGS10 | M2 Growth | 전체 (monthly lag) |

### 7.3 IC 기반 사전 검증 표준

```python
# Phase 1 IC 검증 기준
IC_THRESHOLDS = {
    "minimum": 0.02,       # |IC| > 0.02 필수
    "good": 0.05,          # |IC| > 0.05 → 고품질 시그널
    "excellent": 0.10,     # |IC| > 0.10 → 매우 강력 (의심 필요)
    "suspicious": 0.15,    # |IC| > 0.15 → look-ahead 의심
}

IC_IR_THRESHOLD = 0.5      # IC / IC_std > 0.5 → 안정적
```

### 7.4 기존 전략과의 차별화 매트릭스

신규 전략 후보는 기존 ACTIVE 전략과 다음을 비교:

| 비교 항목 | 기준 | 이유 |
|----------|------|------|
| 수익률 상관 | < 0.3 (이상적) < 0.5 (최대) | 앙상블 분산 효과 |
| 동일 데이터 사용 | 50% 미만 겹침 | 정보 다양성 |
| 레짐 프로파일 | 다른 레짐에서 강점 | 레짐 커버리지 |
| 포지션 방향 | 방향 상관 < 0.5 | 헤지 효과 |
| 보유 기간 | 다른 범위 | 시간 분산 |

---

## 8. Implementation Roadmap

### Phase A: 코드 마이그레이션 (Models + Store + CLI)

1. `src/pipeline/models.py` — `PhaseId`, `PhaseResult`, `StrategyRecord` v2
2. `src/pipeline/store.py` — `record_phase()`, v2 직렬화
3. `src/pipeline/gate_models.py` — phase_id 전환
4. `src/pipeline/gate_store.py` — 파일명 + ID 변경
5. `gates/phase-criteria.yaml` — 신규 criteria 파일
6. `src/cli/pipeline_cmd.py` — CLI 커맨드 갱신
7. 마이그레이션 스크립트 — v1 YAML → v2 YAML 일괄 변환
8. 기존 테스트 갱신 — gate → phase 용어

### Phase B: 스킬 재작성

1. `p1-research` — 데이터 기반 발굴 프레임워크 (가장 큰 변경)
2. `p2-implement` — P1.design 입력 연동
3. `p3-audit` — 리네임 + YAML 출력 구조화
4. `p4-backtest` — G1+G2 통합, 레짐 분해 추가
5. `p5-robustness` — G2H+G3 분리, 신규 스킬
6. `p6-validation` — G4 분리, 신규 스킬
7. `p7-live` — 배포 설정 생성 추가

### Phase C: 데이터 정리

1. RETIRED 전략 YAML 정리 (데이터 없는 것 삭제)
2. v1 → v2 마이그레이션 실행
3. v1 호환 코드 제거
4. 불필요한 `results/` 파일 정리

### 예상 작업 순서

```
Phase A (코드) → Phase C (마이그레이션) → Phase B (스킬)
```

코드를 먼저 변경하고, 데이터를 마이그레이션한 뒤, 스킬을 새로 작성.
스킬은 마이그레이션된 v2 YAML 위에서 동작해야 하므로 마지막.

---

## 9. 부록: 삭제 대상

### 불필요한 v1 아티팩트

| 항목 | 이유 |
|------|------|
| `strategies/` 내 gates 없는 RETIRED YAML | 학습 데이터 없음 |
| `results/gate2h_*.json` 중 RETIRED 전략 | P5로 통합 |
| v1 스킬의 `references/discarded-strategies.md` | lessons 시스템으로 대체 |
| `references/idea-sources.md` 일부 | 데이터 기반 발굴로 대체 |

### 보존 대상

| 항목 | 이유 |
|------|------|
| ACTIVE 전략 YAML (CTREND, Anchor-Mom) | 운영 중 |
| gates 있는 RETIRED YAML | 실패 학습 데이터 |
| `lessons/*.yaml` | 핵심 지식 자산 |
| `catalogs/` | 데이터/지표 카탈로그 |
