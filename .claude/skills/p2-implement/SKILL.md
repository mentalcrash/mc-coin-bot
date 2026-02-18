---
name: p2-implement
description: >
  전략 후보를 4-file 구조로 구현 + 테스트 + Registry 등록하는 스킬.
  사용 시점: CANDIDATE 전략 구현, "implement" 요청, p1 완료 후.
context: fork
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
argument-hint: <strategy-name>
---

# p2-implement: 전략 코드 구현

## 역할

**시니어 퀀트 개발자**로서 행동한다.
판단 기준: **"이 코드가 실제 돈을 운용하는 프로덕션 시스템에 배포되어도 안전한가?"**

핵심 원칙:

- **정확성 > 속도**: shift(1) 누락이 수백만 원 차이를 만든다
- **벡터화 필수**: pandas/numpy 벡터 연산 (루프 절대 금지)
- **일관된 패턴**: 기존 전략과 동일한 코드 구조 유지
- **테스트 우선**: 구현 → 테스트 → lint/typecheck/test 통과 → 등록

---

## 인수

| 인수 | 필수 | 설명 | 예시 |
|------|:----:|------|------|
| `strategy_name` | O | registry key (kebab-case) | `accel-conv` |

> kebab-case (e.g., `accel-conv`), 디렉토리명은 snake_case (e.g., `accel_conv`).

---

## Step 0: 후보 정보 수집

### 0-1. YAML에서 후보 로드

```bash
cat strategies/{strategy_name}.yaml
# meta.status=CANDIDATE, gates.G0A.status=PASS 확인
# YAML 없으면: "p1-g0a-discover에서 pipeline create를 먼저 실행하세요"
```

추출 항목: 전략명, 카테고리, TF, ShortMode, Gate 0 점수, 핵심 가설, 사용 지표, 시그널 로직, 차별화 포인트, 데이터 요구사항.

### 0-2. annualization_factor 결정

| TF | factor | TF | factor |
|----|--------|----|--------|
| 1D | 365.0 | 4H | 2190.0 |
| 12H | 730.0 | 1H | 8760.0 |
| 8H | 1095.0 | 6H | 1460.0 |

> **Critical**: annualization_factor 오류는 vol_scalar를 왜곡하여 포지션 사이징에 직접 영향.

### 0-3. 기존 전략과 중복 확인

```bash
uv run python -c "from src.strategy import list_strategies; print(list_strategies())"
```

### 0-4. 폐기 전략 패턴 확인

[references/implementation-checklist.md](references/implementation-checklist.md)의 anti-pattern 목록 참조.

---

## Step 1: 디렉토리 구조 생성

```bash
mkdir -p src/strategy/{name_snake}/ tests/strategy/{name_snake}/
touch tests/strategy/{name_snake}/__init__.py
```

**필수 파일**: config.py, preprocessor.py, signal.py, strategy.py, **init**.py + 테스트 4개.

---

## Step 2: config.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Config Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `from __future__ import annotations` | 상단 필수 |
| `BaseModel` + `frozen=True` | 불변 설정 |
| `ShortMode(IntEnum)` | DISABLED=0, HEDGE_ONLY=1, FULL=2 |
| 전략 파라미터 `Field()` | `ge`, `le` 검증 |
| `vol_target`, `min_volatility` | Vol-target 파라미터 |
| `annualization_factor` | TF별 연환산 (Step 0-2) |
| HEDGE_ONLY 파라미터 | `hedge_threshold`, `hedge_strength_ratio` |
| `model_validator` | `vol_target >= min_volatility` |
| `warmup_periods()` | warmup bar 수 |

**주의**: `from __future__ import annotations` 사용 시 `TYPE_CHECKING` import 활용.

---

## Step 2.5: 레짐 적응형 전략 (해당 시)

레짐 활용이 지정된 경우: [references/regime-implementation.md](references/regime-implementation.md) 참조.

- 접근 A: RegimeService 자동 주입 (권장, EDA live 호환)
- 접근 B: 자체 레짐 감지 (regime-tsmom 참고)

---

## Step 2.6: Derivatives 데이터 전략 (해당 시)

OHLCV 외 Derivatives 데이터 필요 시: [references/derivatives-implementation.md](references/derivatives-implementation.md) 참조.

---

## Step 3: preprocessor.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Preprocessor Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `preprocess(df, config) -> pd.DataFrame` | 모듈 레벨 함수 |
| `df = df.copy()` | 원본 불변 (첫 줄) |
| missing columns 검증 | `required = {"open", "high", "low", "close", "volume"}` |
| `returns` = `np.log(close / close.shift(1))` | log return |
| `realized_vol` = `returns.rolling(N).std() * np.sqrt(factor)` | 연환산 변동성 |
| `vol_scalar` = `target / realized_vol.clip(lower=min_vol)` | 포지션 스케일러 |
| 전략별 feature | 벡터화 연산만 |

**공유 지표 재사용**: `src/market/indicators/` (53개 함수). 중복 구현 금지.
`from src.market.indicators import atr, rsi, drawdown, ...`

**NaN 처리**: Rolling warmup NaN 유지, 0 나눗셈 `.clip(lower=epsilon)`, feature 간 NaN 전파 허용.

**금지 패턴**: 전체 기간 `.mean()/.std()` (rolling 필수), `inplace=True`, `apply(axis=1)`.

---

## Step 4: signal.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Signal Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `generate_signals(df, config) -> StrategySignals` | 모듈 레벨 함수 |
| **Shift(1) Rule** | 모든 feature에 `.shift(1)` — preprocessor가 아닌 signal에서 적용 |
| ShortMode 3-way 분기 | DISABLED / HEDGE_ONLY / FULL |
| `StrategySignals` 반환 | `(entries, exits, direction, strength)` |
| NaN → 0 처리 | strength `.fillna(0.0)`, direction NaN → 0 |

**Shift(1) 핵심**: preprocessor에서 계산한 feature를 signal에서 shift(1). preprocessor에서는 shift하지 않음.

**금지**: `shift(-N)` (미래 참조), 동시 long+short 미처리, strength NaN에서 entries=True.

---

## Step 5: strategy.py 구현

템플릿: [references/code-templates.md](references/code-templates.md) Strategy Template 참조.

| 필수 요소 | 설명 |
|-----------|------|
| `@register("kebab-name")` | Registry 등록 |
| `BaseStrategy` 상속 | 추상 메서드 구현 |
| `name`, `required_columns`, `config` | property |
| `preprocess()`, `generate_signals()` | 위임 |
| `recommended_config()` | PM 설정 권장값 |
| `from_params()` | Config → 인스턴스 factory |
| `get_startup_info()` | CLI 표시용 핵심 파라미터 |

---

## Step 6: **init**.py + Registry 등록

전략 모듈 `__init__.py`: Config, Strategy, ShortMode, preprocess, generate_signals export.

`src/strategy/__init__.py`에 알파벳 순 import 추가:

```python
from src.strategy import {name_snake}  # noqa: F401 — registry side-effect
```

등록 확인:

```bash
uv run python -c "from src.strategy import list_strategies; print('{name}' in list_strategies())"
```

---

## Step 7: 테스트 작성

전략당 4개 파일, 총 **40~60개 테스트**. 상세 테스트 클래스 목록: [references/implementation-checklist.md](references/implementation-checklist.md) Tests 섹션 참조.

**핵심 테스트 그룹**:

- **test_config.py**: ShortMode enum, 기본값, frozen, 경계값, 교차 검증, warmup
- **test_preprocessor.py**: 컬럼 존재, 길이, 불변, missing, feature 범위
- **test_signal.py**: 구조, shift(1) 첫 bar 중립, ShortMode 3종, 전략 고유 로직
- **test_strategy.py**: registry, properties, pipeline, from_params, recommended_config

레짐 적응형 전략은 `TestRegimeAdaptation` 추가 (with/without regime 컬럼).

---

## Step 8: 품질 검증

```bash
# Lint + Format
uv run ruff check src/strategy/{name_snake}/ tests/strategy/{name_snake}/ --fix
uv run ruff format src/strategy/{name_snake}/ tests/strategy/{name_snake}/

# Type check
uv run pyright src/strategy/{name_snake}/

# Tests
uv run pytest tests/strategy/{name_snake}/ -v
uv run pytest --tb=short -q  # 전체
```

**0 error 필수**. 최종 체크리스트: [references/implementation-checklist.md](references/implementation-checklist.md)

---

## Step 8.5: 앙상블 호환성 확인

모든 `BaseStrategy` 구현체는 앙상블 자동 호환. 확인 사항:

1. `from_params()` 정상 작동
1. `warmup_periods()` 정의
1. direction 값 범위 {-1, 0, 1}

> 앙상블 편입 시 `config/ensemble-example.yaml`에 서브 전략 추가. 상관 < 0.4 + Sharpe > 0.3 기준.

---

## Step 9-10: YAML 갱신 + Dashboard

```bash
# status: CANDIDATE → IMPLEMENTED
uv run mcbot pipeline update-status {strategy_name} --status IMPLEMENTED

# Dashboard 재생성
uv run mcbot pipeline report
```

---

## Step 11: 완료 리포트

```
============================================================
  STRATEGY IMPLEMENTATION REPORT
  전략: {DisplayName} ({registry-key})
  구현일: {YYYY-MM-DD} | TF: {TF} | ShortMode: {mode}
============================================================
  파일: src/strategy/{name_snake}/ (5 files)
  테스트: tests/strategy/{name_snake}/ (4 files)
  Ruff: PASS | Pyright: PASS | Tests: PASS ({N} tests)
  Registry: {registry-key} ({total} strategies)
  다음: /p3-g0b-verify → /p4-g1g4-gate → 앙상블 편입 검토
============================================================
```

---

## Anti-Pattern 체크리스트

| # | Anti-Pattern |
|---|-------------|
| 1 | `shift(-N)` 사용 (미래 참조) |
| 2 | 전체 기간 `.mean()/.std()` (rolling 없이) |
| 3 | `for i in range(len(df))` |
| 4 | `iterrows()`, `itertuples()` |
| 5 | `apply(axis=1)` |
| 6 | `inplace=True` |
| 7 | `fillna(0)` 부적절 사용 |
| 8 | `except:` (광범위 예외) |
| 9 | 매직 넘버 (config 파라미터로 추출) |
| 10 | annualization_factor 하드코딩 |
| 11 | `# noqa`, `# type: ignore` 남용 |

---

## 참조 문서

| 문서 | 용도 |
|------|------|
| [references/code-templates.md](references/code-templates.md) | 4-file 코드 템플릿 |
| [references/implementation-checklist.md](references/implementation-checklist.md) | 구현 체크리스트 + 폐기 패턴 + 테스트 목록 |
| [references/regime-implementation.md](references/regime-implementation.md) | 레짐 적응형 구현 가이드 |
| [references/derivatives-implementation.md](references/derivatives-implementation.md) | Derivatives 데이터 구현 가이드 |
| [.claude/rules/strategy.md](../../rules/strategy.md) | 전략 개발 규칙 |
| [src/strategy/base.py](../../../src/strategy/base.py) | BaseStrategy ABC |
