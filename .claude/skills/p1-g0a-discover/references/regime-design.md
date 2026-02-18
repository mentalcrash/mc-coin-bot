# RegimeService 설계 가이드

## 사용 가능한 레짐 컬럼 (StrategyEngine 자동 주입)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `regime_label` | str | 현재 레짐: `"trending"`, `"ranging"`, `"volatile"` |
| `p_trending` | float | TRENDING 확률 (0~1) |
| `p_ranging` | float | RANGING 확률 (0~1) |
| `p_volatile` | float | VOLATILE 확률 (0~1) |
| `trend_direction` | int | 추세 방향: +1(상승), -1(하락), 0(중립) |
| `trend_strength` | float | 추세 강도 (0.0~1.0) |

## RegimeService 아키텍처

```
EnsembleRegimeDetector (Rule-Based + HMM + Vol-Structure + MSAR)
        |
    RegimeService.precompute()  (backtest: vectorized 사전계산)
    RegimeService._on_bar()     (live: BAR 이벤트 구독 -> 증분 업데이트)
        |
    StrategyEngine._enrich_with_regime()  (DataFrame에 6개 컬럼 자동 추가)
        |
    strategy.preprocess() / generate_signals()  (컬럼 읽어서 활용)
```

## 레짐 적응형 전략의 3가지 패턴

### 패턴 A: 확률 가중 파라미터 적응 (Probability-Weighted)

레짐 확률로 vol_target/threshold 등을 연속적으로 조절한다.

```python
adaptive_vol_target = p_trending * 0.40 + p_ranging * 0.15 + p_volatile * 0.10
```

- 부드러운 전환, 레짐 전환 시 whipsaw 최소
- 예시: regime-tsmom (기존 구현)

### 패턴 B: 레짐 조건부 필터 (Regime Conditional)

특정 레짐에서만 시그널 활성화/비활성화한다.

```python
# trending + trend_direction=+1 -> 롱 시그널만 허용
# volatile -> 포지션 축소 또는 시그널 억제
```

- 명확한 논리, 디버깅 용이

### 패턴 C: 방향 가중 시그널 (Direction-Weighted)

trend_direction/trend_strength로 시그널 방향/강도를 가중한다.

```python
strength *= trend_strength  # 추세 강도에 비례
# direction과 trend_direction 일치 시 conviction 부여
```

- 추세 방향 정보 활용

## 올바른 vs 잘못된 사용 -- 핵심 구분

### 올바른 사용: 레짐을 "오버레이/필터"로 사용

- 기존 alpha 소스(momentum, MR 등)가 독립적으로 존재
- 레짐은 포지션 사이징 조절, 시그널 강도 감쇄에만 사용
- `regime_service=None`이면 기본 동작 (backward compatible)

### 잘못된 사용: 레짐 감지 자체가 alpha 소스

- 레짐 전환을 매매 시그널로 직접 사용
- ADX/HMM/Hurst/AC/VR 등 7개 전략 전멸의 교훈 (안티패턴 #9)
- 레짐 정보만으로 시장 방향 예측 불가

## 설계 시 레짐 활용 결정 체크리스트

1. 기존 alpha 소스가 레짐과 독립적인가? -> 예: 진행
1. 레짐 없이도 시그널이 생성되는가? -> 예: 진행
1. 레짐은 "어떻게 거래할지"를 조절하는가? (not "무엇을 거래할지") -> 예: 진행
1. `regime_service=None` 시 graceful fallback이 있는가? -> 예: 진행

4개 모두 "예"이면 레짐 적응형 설계 적합.
