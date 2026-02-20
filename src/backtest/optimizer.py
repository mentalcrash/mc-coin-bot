"""Optuna TPE-based parameter optimizer for strategy hyperparameter search.

Gate 2H에서 사용하는 핵심 모듈:
- extract_search_space: Pydantic Config에서 최적화 가능 파라미터 추출
- optimize_strategy: Optuna TPE로 IS 데이터에서 Sharpe 극대화 파라미터 탐색
- generate_g3_sweeps: 최적 파라미터 중심 Gate 3 안정성 검증용 sweep 생성
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pydantic import BaseModel

    from src.data.market_data import MarketDataSet
    from src.portfolio.portfolio import Portfolio

# Minimum margin for sweep range (avoid degenerate sweeps)
_MIN_MARGIN = 1e-9

# Minimum number of sweep points for integer parameters
_MIN_INT_SWEEP_POINTS = 3

# Fields to always skip during optimization
_SKIP_FIELDS: frozenset[str] = frozenset(
    {
        "annualization_factor",
        "use_log_returns",
        "min_volatility",
        "short_mode",
        "hedge_threshold",
        "hedge_strength_ratio",
        "use_adx_filter",
        "trending_position_scale",
    }
)

# Weight pair mapping: primary → complement (complement = 1.0 - primary)
WEIGHT_PAIRS: dict[str, str] = {
    "bb_weight": "rsi_weight",
    "rsi_weight": "bb_weight",
    "max_weight": "min_weight",
    "min_weight": "max_weight",
}

# Complement fields (the "dependent" side of weight pairs — skip these)
_COMPLEMENT_FIELDS: frozenset[str] = frozenset(
    {
        "rsi_weight",
        "min_weight",
    }
)


@dataclass(frozen=True)
class ParamSpec:
    """Optimizable parameter specification."""

    name: str
    param_type: str  # "int" | "float"
    low: float
    high: float
    default: Any


@dataclass(frozen=True)
class OptimizationResult:
    """Result of Optuna optimization run."""

    best_params: dict[str, Any]
    best_sharpe: float
    default_sharpe: float
    improvement_pct: float
    n_trials: int
    n_completed: int
    search_space: list[ParamSpec]
    top_trials: list[dict[str, Any]] = field(default_factory=list)


def _resolve_param_type(annotation: type[Any] | None) -> str | None:
    """Annotation에서 optimizable param_type을 결정. None이면 skip 대상."""
    if annotation is None or annotation is bool:
        return None
    if issubclass(annotation, IntEnum):
        return None
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return None


def _extract_bounds(field_info: Any) -> tuple[float | None, float | None]:
    """FieldInfo.metadata에서 ge/le 값을 추출."""
    ge_val: float | None = None
    le_val: float | None = None
    for meta in field_info.metadata:
        if hasattr(meta, "ge") and meta.ge is not None:
            ge_val = float(meta.ge)
        if hasattr(meta, "le") and meta.le is not None:
            le_val = float(meta.le)
    return ge_val, le_val


def extract_search_space(config_class: type[BaseModel]) -> list[ParamSpec]:
    """Pydantic V2 Config 클래스에서 최적화 가능한 파라미터를 추출.

    ge/le constraints가 모두 있는 int/float 필드만 추출합니다.
    bool, IntEnum, _SKIP_FIELDS, complement weight 필드는 제외됩니다.

    Args:
        config_class: Pydantic BaseModel 서브클래스

    Returns:
        최적화 가능 ParamSpec 목록
    """
    specs: list[ParamSpec] = []

    for name, field_info in config_class.model_fields.items():
        if name in _SKIP_FIELDS or name in _COMPLEMENT_FIELDS:
            continue

        param_type = _resolve_param_type(field_info.annotation)
        if param_type is None:
            continue

        ge_val, le_val = _extract_bounds(field_info)
        if ge_val is None or le_val is None:
            continue

        specs.append(
            ParamSpec(
                name=name,
                param_type=param_type,
                low=ge_val,
                high=le_val,
                default=field_info.default,
            )
        )

    return specs


def optimize_strategy(
    strategy_name: str,
    data_is: MarketDataSet,
    portfolio: Portfolio,
    *,
    n_trials: int = 100,
    seed: int = 42,
    fixed_params: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Optuna TPE로 IS 데이터에서 최적 파라미터를 탐색.

    Args:
        strategy_name: Registry에 등록된 전략 이름
        data_is: In-Sample 데이터
        portfolio: 백테스트용 포트폴리오
        n_trials: Optuna trial 수
        seed: 재현성을 위한 seed
        fixed_params: 고정할 파라미터 (최적화 대상에서 제외)

    Returns:
        OptimizationResult with best params and metrics
    """
    try:
        import optuna
    except ImportError as e:
        msg = "optuna is required for optimization. Install with: uv add --group research optuna"
        raise ImportError(msg) from e

    from src.backtest.engine import BacktestEngine
    from src.strategy import get_strategy

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    strategy_cls = get_strategy(strategy_name)
    config_class = get_config_class(strategy_cls)
    search_space = extract_search_space(config_class)
    engine = BacktestEngine()
    fixed = fixed_params or {}

    # Compute default Sharpe as baseline
    default_sharpe = _run_single_backtest(engine, strategy_cls, {}, data_is, portfolio)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = dict(fixed)

        for spec in search_space:
            if spec.name in fixed:
                continue
            if spec.param_type == "int":
                val = trial.suggest_int(spec.name, int(spec.low), int(spec.high))
            else:
                val = trial.suggest_float(spec.name, spec.low, spec.high)

            params[spec.name] = val

            # Weight pair complement
            if spec.name in WEIGHT_PAIRS:
                complement = WEIGHT_PAIRS[spec.name]
                if complement not in _COMPLEMENT_FIELDS:
                    continue
                params[complement] = round(1.0 - val, 6)

        sharpe = _run_single_backtest(engine, strategy_cls, params, data_is, portfolio)
        return sharpe

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract results
    best_params = dict(study.best_params)
    # Add complement weights
    for name, complement in WEIGHT_PAIRS.items():
        if name in best_params and complement in _COMPLEMENT_FIELDS:
            best_params[complement] = round(1.0 - best_params[name], 6)
    # Add fixed params
    best_params.update(fixed)

    best_sharpe = study.best_value
    improvement = (
        ((best_sharpe - default_sharpe) / abs(default_sharpe) * 100) if default_sharpe != 0 else 0.0
    )

    # Top 10 trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed, key=lambda t: t.value or float("-inf"), reverse=True)
    top_trials = [
        {"number": t.number, "sharpe": t.value, "params": dict(t.params)}
        for t in sorted_trials[:10]
    ]

    logger.info(
        f"Optimization complete: default={default_sharpe:.3f} → best={best_sharpe:.3f} "
        + f"({improvement:+.1f}%) [{len(completed)}/{n_trials} completed]"
    )

    return OptimizationResult(
        best_params=best_params,
        best_sharpe=best_sharpe,
        default_sharpe=default_sharpe,
        improvement_pct=round(improvement, 1),
        n_trials=n_trials,
        n_completed=len(completed),
        search_space=search_space,
        top_trials=top_trials,
    )


def _generate_int_sweep(
    spec: ParamSpec,
    best_val: float,
    raw_low: float,
    raw_high: float,
    n_points: int,
) -> list[int]:
    """정수 파라미터용 sweep 값을 생성 (최소 _MIN_INT_SWEEP_POINTS 보장)."""
    low_i = math.ceil(raw_low)
    high_i = max(math.floor(raw_high), low_i)
    # Generate int range
    if high_i - low_i + 1 <= n_points:
        values = list(range(low_i, high_i + 1))
    else:
        step = max(1, (high_i - low_i) // (n_points - 1))
        values = list(range(low_i, high_i + 1, step))
        if high_i not in values:
            values.append(high_i)
    # Ensure best value is included
    best_int = round(best_val)
    if best_int not in values and spec.low <= best_int <= spec.high:
        values.append(best_int)
        values.sort()
    # Expand small integer sweeps to minimum points
    if len(values) < _MIN_INT_SWEEP_POINTS:
        available = list(range(int(spec.low), int(spec.high) + 1))
        if len(available) >= _MIN_INT_SWEEP_POINTS:
            expanded = set(values)
            delta = 1
            while len(expanded) < _MIN_INT_SWEEP_POINTS:
                for candidate in (best_int - delta, best_int + delta):
                    if spec.low <= candidate <= spec.high:
                        expanded.add(candidate)
                delta += 1
            values = sorted(expanded)
        else:
            values = available
    return values


def generate_g3_sweeps(
    result: OptimizationResult,
    config_class: type[BaseModel],
    *,
    n_points: int = 10,
    margin_ratio: float = 0.3,
) -> dict[str, list[Any]]:
    """최적 파라미터 중심으로 Gate 3 안정성 검증용 sweep 값을 생성.

    Args:
        result: OptimizationResult
        config_class: 전략 Config 클래스 (bounds 참조)
        n_points: 파라미터당 sweep point 수
        margin_ratio: best 기준 ±margin 비율 (0.3 = ±30%)

    Returns:
        {param_name: [sweep_values]} dict
    """
    sweeps: dict[str, list[Any]] = {}

    for spec in result.search_space:
        best_val = result.best_params.get(spec.name, spec.default)
        if best_val is None:
            continue

        # Skip complement weights
        if spec.name in _COMPLEMENT_FIELDS:
            continue

        margin = abs(best_val) * margin_ratio
        if margin < _MIN_MARGIN:
            margin = (spec.high - spec.low) * margin_ratio

        raw_low = max(spec.low, best_val - margin)
        raw_high = min(spec.high, best_val + margin)

        if spec.param_type == "int":
            values: list[Any] = _generate_int_sweep(spec, best_val, raw_low, raw_high, n_points)
        else:
            step = (raw_high - raw_low) / max(n_points - 1, 1)
            values = [round(raw_low + i * step, 6) for i in range(n_points)]
            # Ensure best value is included
            best_f = round(float(best_val), 6)
            if best_f not in values:
                values.append(best_f)
                values.sort()

        sweeps[spec.name] = values

    return sweeps


# ─── Private helpers ────────────────────────────────────────────────


def get_config_class(strategy_cls: type[Any]) -> type[Any]:
    """전략 클래스에서 Config 클래스를 추출."""
    # Try instantiating with defaults and check .config
    try:
        instance = strategy_cls()
        config = instance.config
    except Exception:
        logger.debug(f"Could not instantiate {strategy_cls.__name__} for config extraction")
    else:
        if config is not None:
            return type(config)

    # Fallback: look for Config class attribute or module-level *Config
    import inspect

    module = inspect.getmodule(strategy_cls)
    if module is not None:
        for attr_name, attr_val in inspect.getmembers(module):
            if (
                inspect.isclass(attr_val)
                and attr_name.endswith("Config")
                and hasattr(attr_val, "model_fields")
            ):
                return attr_val

    msg = f"Could not find Config class for {strategy_cls.__name__}"
    raise ValueError(msg)


def _run_single_backtest(
    engine: Any,
    strategy_cls: type[Any],
    params: dict[str, Any],
    data: MarketDataSet,
    portfolio: Portfolio,
) -> float:
    """단일 파라미터 조합으로 백테스트 실행, Sharpe 반환."""
    from src.backtest.request import BacktestRequest

    try:
        strategy = strategy_cls.from_params(**params) if params else strategy_cls()
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        result = engine.run(request)
    except Exception:
        logger.debug(f"Trial failed with params: {params}")
        return float("-inf")

    sharpe = result.metrics.sharpe_ratio
    if math.isnan(sharpe) or math.isinf(sharpe):
        return float("-inf")
    return sharpe
