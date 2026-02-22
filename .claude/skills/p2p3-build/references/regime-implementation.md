# 레짐 적응형 전략 구현 가이드

> 발굴 단계에서 "레짐 활용 여부"가 지정된 경우 참조.

---

## 접근 A: RegimeService 자동 주입 (권장)

StrategyEngine이 DataFrame에 regime 컬럼을 자동 추가. 전략은 컬럼만 읽으면 됨.

### 사용 가능 컬럼 (StrategyEngine._enrich_with_regime() 자동 주입)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `regime_label` | str | `"trending"` / `"ranging"` / `"volatile"` |
| `p_trending` | float | TRENDING 확률 (0~1) |
| `p_ranging` | float | RANGING 확률 (0~1) |
| `p_volatile` | float | VOLATILE 확률 (0~1) |
| `trend_direction` | int | +1(상승) / -1(하락) / 0(중립) |
| `trend_strength` | float | 추세 강도 (0.0~1.0) |

### preprocessor.py 패턴

```python
# RegimeService 컬럼은 StrategyEngine이 주입하므로
# preprocessor에서는 regime 관련 코드 불필요!
# 단, regime 컬럼이 없는 경우(VBT 백테스트 등) graceful fallback 필요
```

### signal.py 패턴 (확률 가중 vol_target)

```python
# Regime 컬럼 존재 여부 체크 (backward compatible)
if "p_trending" in df.columns:
    p_trending = df["p_trending"].shift(1).fillna(1/3)
    p_ranging = df["p_ranging"].shift(1).fillna(1/3)
    p_volatile = df["p_volatile"].shift(1).fillna(1/3)
    adaptive_target = (
        p_trending * config.trending_vol_target
        + p_ranging * config.ranging_vol_target
        + p_volatile * config.volatile_vol_target
    )
else:
    adaptive_target = config.vol_target  # fallback
```

### signal.py 패턴 (조건부 필터)

```python
if "regime_label" in df.columns:
    regime = df["regime_label"].shift(1)
    suppress = regime == "volatile"
    direction = direction.where(~suppress, 0)
    strength = strength.where(~suppress, 0.0)
```

### config.py 추가 필드 (레짐별 파라미터)

```python
# 패턴 A: 확률 가중
trending_vol_target: float = Field(default=0.40, ge=0.05, le=1.0)
ranging_vol_target: float = Field(default=0.15, ge=0.0, le=1.0)
volatile_vol_target: float = Field(default=0.10, ge=0.0, le=1.0)
```

---

## 접근 B: 자체 레짐 감지 (기존 방식, regime-tsmom 참고)

전략이 직접 regime detector를 preprocessor에서 호출. RegimeService 불필요.

```python
# preprocessor.py
from src.regime.detector import add_regime_columns
result = add_regime_columns(df, config.regime)
```

> 접근 A와 B를 혼용하지 않는다. 하나만 선택.
> 접근 A가 EDA live와 호환성이 높으므로 신규 전략에서 권장.

---

## 주의사항

- regime 컬럼에도 **shift(1) 적용 필수** (signal.py에서)
- regime 컬럼이 없을 때 **graceful fallback** 구현 (backward compatible)
- 레짐 기반 파라미터 변경은 **연속적**(확률 가중)이 이산적(if-else)보다 안정
- `regime_service=None`이면 컬럼 미주입 → fallback 작동 확인 테스트 필수

### 테스트 패턴

```python
class TestRegimeAdaptation:
    def test_with_regime_columns(self):       # regime 컬럼 있을 때 정상 작동
    def test_without_regime_columns(self):    # regime 컬럼 없을 때 fallback 작동
    def test_trending_aggressive(self):       # trending에서 더 공격적
    def test_volatile_conservative(self):     # volatile에서 보수적
```
