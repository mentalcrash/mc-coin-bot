# Implementation Checklist — p2-implement

> 구현 완료 전 반드시 확인하는 체크리스트와 폐기 전략 패턴 회피 가이드.

---

## Pre-Implementation 체크리스트

### 1. 후보 정보 완전성

- [ ] 전략명 (registry key, kebab-case)
- [ ] 카테고리 (Momentum, MR, Regime, Behavioral 등)
- [ ] 타임프레임 + annualization_factor
- [ ] ShortMode (DISABLED / HEDGE_ONLY / FULL)
- [ ] Gate 0 점수 (>= 18/30)
- [ ] 핵심 가설 (경제적 논거)
- [ ] 사용 지표 (구체적 수식)
- [ ] 시그널 생성 로직 (조건 + 방향)
- [ ] 차별화 포인트 (기존 전략과 구분)
- [ ] 데이터 요구사항 확인 (OHLCV only / Derivatives)
- [ ] Derivatives 필요 시: Silver _deriv 파일 존재 확인 + 백테스팅 가용 데이터인지 확인

### 2. 폐기 전략 패턴 확인

후보의 시그널 로직이 아래 실패 패턴과 유사하지 않은지 확인.

---

## 폐기 전략 실패 패턴 (Anti-Pattern)

### Pattern 1: 단일 지표 의존 (12개 G2 FAIL)

```
❌ signal = SMA_fast > SMA_slow               (VW-TSMOM, TSMOM)
❌ signal = RSI > 70 or RSI < 30               (RSI Crossover)
❌ signal = BB_upper / BB_lower cross           (BB-RSI)
```

**교훈**: 단일 지표 → IS 과적합 → OOS Decay 50%+. 최소 2개 독립 시그널 결합 필수.

### Pattern 2: 과소 거래 (MTF-MACD, VPIN-Flow)

```
❌ significance_z > 1.96 → 연 2~5건 거래
❌ VPIN > 0.7 → 1D에서 도달 불가 (max 0.45)
```

**교훈**: 필터가 과도하면 거래 0건. threshold를 TF/에셋별 실제 분포 기반으로 설정.

### Pattern 3: 비용 구조 문제 (Larry-VB)

```
❌ 1-bar hold → 연 125건 × 0.1% = 12.5% drag
```

**교훈**: 거래 빈도 × 편도 비용 > 예상 alpha이면 구조적 손실.

### Pattern 4: 1H TF 크립토 전멸 (Tier 5 전량)

```
❌ FX session breakout → 크립토 24/7 무효
❌ Amihud 1H → 과빈번 전환 (연 1,700건)
❌ BVC/OFI 1H → flow 방향 예측 불가
❌ Hour seasonality → noise 과적합
```

**교훈**: 1H OHLCV 기반 microstructure/session 전략은 크립토에서 구조적 한계.

### Pattern 5: 밈코인 FULL Short (VWAP-Disposition)

```
❌ DOGE + ShortMode.FULL → MDD -622%
```

**교훈**: 밈코인 급등에 대한 FULL short = 구조적 자살. HEDGE_ONLY 권장.

### Pattern 6: Mom + MR 동일 TF 블렌딩 (Mom-MR Blend)

```
❌ momentum_signal * 0.5 + mr_signal * 0.5 → alpha 상쇄
```

**교훈**: 동일 TF에서 반대 방향 시그널 평균 = 서로 상쇄.

### Pattern 7: Rolling 통계의 과적합 (Hour-Season)

```
❌ 30일 rolling t-stat → noise를 유의 패턴으로 오인
```

**교훈**: 짧은 rolling window + 통계적 검정 = 위양성 폭증.

### Pattern 8: 학술 지표의 데이터 해상도 불일치 (VPIN, Amihud)

```
❌ VPIN (tick data 설계) → 1D OHLCV 적용 → max 0.45 (threshold 0.7 미도달)
❌ Amihud (일간 설계) → 1H 적용 → 과빈번 유동성 상태 전환
```

**교훈**: 학술 지표의 원래 설계 TF와 적용 TF가 일치해야 함.

---

## Implementation 체크리스트

### config.py

- [ ] `from __future__ import annotations`
- [ ] `BaseModel` + `ConfigDict(frozen=True)`
- [ ] `ShortMode(IntEnum)` 정의
- [ ] 전략별 파라미터 + `Field()` 검증
- [ ] `vol_target`, `vol_window`, `min_volatility`
- [ ] `annualization_factor` (TF에 맞는 정확한 값)
- [ ] `short_mode` + HEDGE_ONLY 파라미터
- [ ] `model_validator` 교차 검증
- [ ] `warmup_periods()` 메서드

### preprocessor.py

- [ ] `from __future__ import annotations`
- [ ] `preprocess(df, config)` 모듈 레벨 함수
- [ ] `df = df.copy()` (원본 불변)
- [ ] missing columns 검증
- [ ] `returns` (log return)
- [ ] `realized_vol` (rolling + annualized)
- [ ] `vol_scalar` (target / vol, clipped)
- [ ] 전략별 feature (벡터화)
- [ ] `drawdown` (HEDGE_ONLY용)
- [ ] `atr` (trailing stop용, 필요 시)
- [ ] NaN 방어 (0 나눗셈, edge case)
- [ ] 유틸리티 재사용 (중복 구현 금지)
- [ ] Derivatives 컬럼 사용 시: `_REQUIRED_COLUMNS`에 포함
- [ ] Derivatives 컬럼 NaN 처리: `ffill()` 적용 (merge_asof 후 첫 구간)

### signal.py

- [ ] `from __future__ import annotations`
- [ ] `generate_signals(df, config)` 모듈 레벨 함수
- [ ] **모든 feature에 `.shift(1)`** 적용
- [ ] ShortMode 3-way 분기 (DISABLED / HEDGE_ONLY / FULL)
- [ ] HEDGE_ONLY: drawdown + hedge_threshold + hedge_strength_ratio
- [ ] strength = direction *vol_scalar (* conviction)
- [ ] strength NaN → 0 처리
- [ ] entries/exits 생성 (direction 변경 기반)
- [ ] `StrategySignals` 반환
- [ ] 동시 long+short 불가 확인

### strategy.py

- [ ] `@register("{registry-key}")`
- [ ] `BaseStrategy` 상속
- [ ] `name`, `required_columns`, `config` property
- [ ] `preprocess()` → preprocessor 위임
- [ ] `generate_signals()` → signal 위임
- [ ] `recommended_config()` classmethod
- [ ] `from_params()` classmethod
- [ ] `get_startup_info()` 메서드

### __init__.py

- [ ] 전략 모듈 __init__.py (Config, Strategy, preprocess, generate_signals, ShortMode)
- [ ] `src/strategy/__init__.py`에 import 추가 (알파벳 순)

### Tests

- [ ] test_config.py: 기본값, frozen, 경계값, 교차 검증, warmup
- [ ] test_preprocessor.py: 컬럼, 길이, 불변, missing, feature 범위
- [ ] test_signal.py: 구조, shift(1), ShortMode 3종, 전략 고유 로직
- [ ] test_strategy.py: registry, properties, pipeline, from_params

### Quality

- [ ] `ruff check` — 0 errors
- [ ] `ruff format` — 적용
- [ ] `pyright src/strategy/{name}/` — 0 errors
- [ ] `pytest tests/strategy/{name}/` — 0 failures
- [ ] `pytest` (전체) — 기존 테스트 깨짐 없음

### Documentation

- [ ] `pipeline update-status` → IMPLEMENTED
- [ ] `pipeline report` → Dashboard 재생성

---

## Common Pyright Issues & Solutions

| Issue | Solution |
|-------|----------|
| `reportImplicitStringConcatenation` | f-string 연속 → 한 줄로 합치거나 변수 분리 |
| `100 * Series` → `int` 추론 | `pd.Series` 타입 어노테이션 추가 |
| `reportPrivateUsage` | `_method` → `method` (public으로 변경) |
| `reportUnusedImport` | `# pyright: reportUnusedImport=false` (파일 상단) |
| `reportIncompatibleVariableOverride` | flat model (상속 없이 독립 정의) |
| `TYPE_CHECKING` import | runtime 불필요 타입은 `if TYPE_CHECKING:` 블록 |

## Common Ruff Issues & Solutions

| Rule | Issue | Solution |
|------|-------|----------|
| TC002/TC003 | `from __future__` + import | TYPE_CHECKING 블록으로 이동 |
| PLR0912 | too many branches | 헬퍼 함수 추출 |
| S106 | 테스트의 하드코딩 비밀 | per-file-ignores 설정 |
| SLF001 | 테스트의 private 접근 | per-file-ignores 설정 |
| UP007 | `Optional[X]` → `X \| None` | 파이프 문법 사용 |
