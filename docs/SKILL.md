# Strategy Optimization Guide

전략 빌드업을 위한 **Strategy Advisor & Tiered Validation** 시스템 활용 가이드입니다.

---

## 개요

이 시스템은 **반복적 전략 개발**을 위한 코치/가이드 역할을 합니다.

```
간단한 전략 → 백테스트 → Advisor 분석 → 개선 제안 적용 → 반복
```

### 핵심 철학

- **과적합 방지**: 매 반복마다 IS/OOS 검증으로 과적합 조기 감지
- **점진적 개선**: 한 번에 모든 것을 최적화하지 않고 단계별 개선
- **실전 준비도 평가**: Development → Testing → Production 3단계 평가

---

## Quick Start

### 1. 기본 백테스트 + Advisor 분석

```bash
python -m src.cli.backtest run BTC/USDT -y 2024 -y 2025 --advisor
```

### 2. Quick Validation (IS/OOS) + Advisor

```bash
python -m src.cli.backtest run BTC/USDT -y 2024 -y 2025 --validation quick --advisor
```

### 3. Milestone Validation (Walk-Forward) + Advisor

```bash
python -m src.cli.backtest run BTC/USDT -y 2024 -y 2025 --validation milestone --advisor
```

---

## Validation Levels

### Quick (IS/OOS Split)

| 특성 | 값 |
|------|-----|
| 비용 | ~2x (2회 백테스트) |
| 분할 | 70% Train / 30% Test |
| 용도 | 빠른 반복 개발 |

**사용 시점**: 새 아이디어 테스트, 파라미터 초기 탐색

```bash
--validation quick
```

### Milestone (Walk-Forward)

| 특성 | 값 |
|------|-----|
| 비용 | ~5x (5회 백테스트) |
| 분할 | 5 Expanding Folds |
| 용도 | 중간 점검, 안정성 확인 |

**사용 시점**: 전략 구조 확정 전, 주요 마일스톤

```bash
--validation milestone
```

### Final (CPCV + Monte Carlo)

| 특성 | 값 |
|------|-----|
| 비용 | ~20x |
| 분할 | CPCV + Bootstrap |
| 용도 | 최종 검증, 실전 배포 전 |

**사용 시점**: Production 배포 직전 (추후 구현 예정)

---

## Advisor Report 해석

### 1. Overall Score (0-100)

| 점수 | 의미 | 권장 액션 |
|------|------|----------|
| 0-54 | 개발 필요 | 기본 로직 재검토 |
| 55-74 | 테스트 가능 | 세부 튜닝 진행 |
| 75-100 | 배포 고려 | 최종 검증 수행 |

### 2. Readiness Level

```
DEVELOPMENT → 기본 전략 로직에 문제 있음
TESTING     → 튜닝 단계, 세부 조정 필요
PRODUCTION  → 실전 배포 가능 수준
```

### 3. Loss Concentration

손실이 특정 패턴에 집중되는지 분석합니다.

| 지표 | 해석 |
|------|------|
| Worst Hours | 손실이 집중되는 시간대 (UTC) |
| Max Consecutive Losses | 최대 연속 손실 횟수 |
| Large Loss Count | 임계값 초과 대규모 손실 횟수 |

**개선 방향**:
- 특정 시간대 손실 집중 → Session Filter 추가
- 연속 손실 과다 → 연속 손실 리미터 추가
- 대규모 손실 빈번 → Stop Loss 강화

### 4. Regime Profile

시장 레짐별 전략 성과입니다.

| 레짐 | 정의 |
|------|------|
| Bull | 20일 수익률 > 5% |
| Bear | 20일 수익률 < -5% |
| Sideways | 그 외 |

**개선 방향**:
- Bear에서 부진 → Short 전략 강화 또는 현금화
- Sideways에서 부진 → 레짐 필터 추가 (ADX 등)
- 레짐 간 편차 큼 → 레짐 적응형 파라미터

### 5. Signal Quality

시그널의 예측력과 효율성입니다.

| 지표 | 양호 기준 |
|------|----------|
| Hit Rate | > 50% |
| Risk/Reward | > 1.0 |
| Expectancy | > 0 |

**개선 방향**:
- Hit Rate 낮음 → 시그널 생성 로직 재검토
- Risk/Reward 낮음 → 익절/손절 비율 조정
- Expectancy 음수 → 전략 근본 재설계

### 6. Overfit Score

과적합 위험도입니다.

| 확률 | 위험 수준 | 액션 |
|------|----------|------|
| < 20% | LOW | 진행 |
| 20-40% | MODERATE | 모니터링 |
| 40-60% | HIGH | 단순화 필요 |
| > 60% | CRITICAL | 전면 재검토 |

**개선 방향**:
- Sharpe Decay 큼 → lookback 기간 증가
- 일관성 낮음 → 파라미터 수 감소
- OOS 성과 부진 → 더 많은 데이터로 검증

---

## 개선 제안 우선순위

### HIGH (즉시 조치)

전략의 근본적 문제를 나타냅니다.

```
- 손절 로직 강화 → system_stop_loss 파라미터 조정
- 과적합 위험 높음 → 전략 단순화, lookback 증가
- 횡보장 레짐 필터 추가 → ADX/BB Width 필터
```

### MEDIUM (개선 권장)

성과 향상을 위한 튜닝 포인트입니다.

```
- 연속 손실 리미터 → N회 연속 손실 시 일시 중단
- 레짐 적응형 파라미터 → 레짐별 다른 설정
- IS/OOS 성과 차이 큼 → Walk-Forward 최적화
```

### LOW (선택적 개선)

미세 조정 영역입니다.

```
- 거래 빈도 최적화 → rebalance_threshold 조정
- 시그널 효율 개선 → 확인 시그널 추가
```

---

## 실전 워크플로우

### Phase 1: 아이디어 검증 (1-2일)

```bash
# 기본 전략으로 시작
python -m src.cli.backtest run BTC/USDT -y 2024 --validation quick --advisor

# 결과 확인:
# - Overall Score > 40?
# - Sharpe > 0?
# - 치명적 문제 없음?
```

**통과 기준**: Overall Score > 40, Sharpe > 0

### Phase 2: 기본 튜닝 (3-5일)

```bash
# HIGH 우선순위 제안 적용 후 재검증
python -m src.cli.backtest run BTC/USDT -y 2024 -y 2025 --validation quick --advisor

# 반복:
# 1. HIGH 제안 1개 적용
# 2. quick validation 실행
# 3. Overfit Probability 모니터링
# 4. 개선되면 다음 제안으로
```

**통과 기준**: Overall Score > 55, Readiness = TESTING

### Phase 3: 안정성 검증 (1주)

```bash
# Milestone 검증으로 안정성 확인
python -m src.cli.backtest run BTC/USDT -y 2024 -y 2025 --validation milestone --advisor

# 체크포인트:
# - Consistency > 60%?
# - Sharpe Decay < 50%?
# - 모든 Fold에서 양수 Sharpe?
```

**통과 기준**: Consistency > 60%, Overall Score > 70

### Phase 4: 최종 검증 (배포 전)

```bash
# 다중 심볼 테스트
python -m src.cli.backtest run ETH/USDT -y 2024 -y 2025 --validation milestone --advisor
python -m src.cli.backtest run SOL/USDT -y 2024 -y 2025 --validation milestone --advisor

# 리포트 생성
python -m src.cli.backtest run BTC/USDT -y 2024 -y 2025 --validation milestone --advisor --report
```

**통과 기준**: 모든 심볼에서 Overall Score > 75, Readiness = PRODUCTION

---

## 자주 발생하는 문제와 해결책

### 1. Quick에서 통과, Milestone에서 실패

**원인**: 특정 기간에만 잘 작동하는 전략

**해결책**:
```bash
# 더 긴 기간으로 테스트
python -m src.cli.backtest run BTC/USDT -y 2023 -y 2024 -y 2025 --validation milestone --advisor
```

### 2. IS Sharpe 높음, OOS Sharpe 낮음

**원인**: 과적합

**해결책**:
- lookback 기간 증가
- 파라미터 수 감소
- 데이터 기간 확장

```python
# config 조정 예시
TSMOMConfig(
    lookback=48,  # 24 → 48
    vol_window=48,  # 30 → 48
)
```

### 3. Bear 레짐에서 대규모 손실

**원인**: Long-only 편향

**해결책**:
```bash
# Short 모드 활성화
python -m src.cli.backtest run BTC/USDT -y 2024 -y 2025 \
    --short-mode full \
    --validation quick --advisor
```

### 4. 연속 손실 과다

**원인**: 횡보장에서 whipsaw

**해결책**:
- 레짐 필터 추가
- 리밸런싱 임계값 증가

```python
PortfolioManagerConfig(
    rebalance_threshold=0.10,  # 0.05 → 0.10
)
```

---

## CLI 옵션 요약

```bash
python -m src.cli.backtest run [SYMBOL] [OPTIONS]

# Validation 옵션
--validation none      # 검증 없음 (기본값)
--validation quick     # IS/OOS 70/30
--validation milestone # Walk-Forward 5 folds

# Advisor 옵션
--advisor              # Advisor 분석 활성화
--no-advisor           # Advisor 분석 비활성화 (기본값)

# 전략 옵션
--strategy tsmom       # 전략 선택
--short-mode hedge     # Short 모드: disabled, hedge, full
--vol-target 0.40      # 변동성 타겟 (0.1-1.0)
--max-leverage 2.0     # 최대 레버리지

# 기타
--report               # QuantStats HTML 리포트 생성
--verbose              # 상세 로그 출력
```

---

## Best Practices

### DO

- 매 반복마다 `--validation quick --advisor` 사용
- HIGH 우선순위 제안부터 순차적으로 적용
- Overfit Probability를 항상 모니터링
- 마일스톤마다 `--validation milestone` 수행
- 다중 심볼로 일반화 테스트

### DON'T

- 한 번에 여러 파라미터 동시 변경
- OOS 데이터를 보고 전략 수정 (데이터 스누핑)
- Validation 없이 반복 최적화 (과적합 함정)
- HIGH 제안 무시하고 MEDIUM/LOW만 적용
- 단일 심볼만 테스트 후 배포

---

## 관련 파일

| 모듈 | 경로 | 설명 |
|------|------|------|
| Validation | `src/backtest/validation/` | Tiered Validation 시스템 |
| Advisor | `src/backtest/advisor/` | Strategy Advisor |
| CLI | `src/cli/backtest.py` | CLI 인터페이스 |
| Models | `src/backtest/advisor/models.py` | 분석 결과 모델 |
