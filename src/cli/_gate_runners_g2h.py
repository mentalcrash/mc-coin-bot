"""Gate 2H runner logic: Optuna TPE parameter optimization.

IS 데이터에서 최적 파라미터를 탐색하고, OOS에서 검증합니다.
Always PASS — 정보 제공 목적이며, 과적합 방어는 G4에서 담당합니다.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.table import Table

from src.cli._gate_runners import resolve_timeframe

if TYPE_CHECKING:
    from rich.console import Console

_RESULTS_DIR = Path("results")
_DEFAULT_START = datetime(2020, 1, 1, tzinfo=UTC)
_DEFAULT_END = datetime(2025, 12, 31, tzinfo=UTC)
_DEFAULT_CAPITAL = Decimal(100_000)


def _create_portfolio(strategy_name: str, capital: Decimal) -> Any:
    """전략별 권장 설정으로 Portfolio 생성."""
    from src.portfolio import Portfolio, PortfolioManagerConfig
    from src.strategy import get_strategy

    strategy_cls = get_strategy(strategy_name)
    rec = strategy_cls.recommended_config()
    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    return Portfolio.create(initial_capital=capital, config=pm_config)


def run_gate2h(
    strategies: list[str],
    n_trials: int,
    seed: int,
    save_json: bool,
    console: Console,
) -> None:
    """Gate 2H 전체 실행: Optuna TPE 파라미터 최적화 + YAML 업데이트."""
    from src.backtest.optimizer import (
        generate_g3_sweeps,
        get_config_class,
        optimize_strategy,
    )
    from src.backtest.validation.splitters import split_is_oos
    from src.data.market_data import MarketDataRequest
    from src.data.service import MarketDataService
    from src.pipeline.store import StrategyStore
    from src.strategy import get_strategy

    _RESULTS_DIR.mkdir(exist_ok=True)
    store = StrategyStore()
    service = MarketDataService()
    t0 = time.perf_counter()

    for strategy_name in strategies:
        console.rule(f"[bold]G2H: {strategy_name}[/]")

        if not store.exists(strategy_name):
            console.print(f"[red]Strategy not found: {strategy_name}[/]")
            continue

        record = store.load(strategy_name)
        best_asset = record.best_asset
        if not best_asset:
            console.print(f"[red]{strategy_name}: No best_asset (run G1 first)[/]")
            continue

        timeframe = resolve_timeframe(strategy_name)
        console.print(f"  Asset: {best_asset}, TF: {timeframe}")

        # Load data and split IS/OOS
        data = service.get(
            MarketDataRequest(
                symbol=best_asset,
                timeframe=timeframe,
                start=_DEFAULT_START,
                end=_DEFAULT_END,
            )
        )
        data_is, data_oos = split_is_oos(data, ratio=0.7)
        console.print(f"  IS: {data_is.periods} bars, OOS: {data_oos.periods} bars")

        # Create portfolio
        portfolio = _create_portfolio(strategy_name, _DEFAULT_CAPITAL)

        # Run optimization on IS
        console.print(f"  Optimizing ({n_trials} trials, seed={seed})...")
        result = optimize_strategy(
            strategy_name,
            data_is,
            portfolio,
            n_trials=n_trials,
            seed=seed,
        )

        # OOS verification (info only)
        oos_sharpe = _run_oos_verification(strategy_name, result.best_params, data_oos, portfolio)

        # Generate G3 sweeps
        strategy_cls = get_strategy(strategy_name)
        config_class = get_config_class(strategy_cls)
        g3_sweeps = generate_g3_sweeps(result, config_class)

        # Print results
        _print_optimization_results(console, strategy_name, result, oos_sharpe)
        _print_top_trials(console, result)

        # Save JSON
        if save_json:
            _save_json_results(strategy_name, result, oos_sharpe, g3_sweeps, seed)
            console.print(f"  [dim]JSON saved: results/gate2h_{strategy_name}.json[/]")

        # Update YAML
        _update_yaml_g2h(strategy_name, result, oos_sharpe, store)
        console.print(f"  [green]YAML updated: {strategy_name} G2H PASS[/]")

    elapsed = time.perf_counter() - t0
    console.print(f"\n[dim]Total elapsed: {elapsed:.1f}s[/]")


def _run_oos_verification(
    strategy_name: str,
    best_params: dict[str, Any],
    data_oos: Any,
    portfolio: Any,
) -> float:
    """OOS 데이터로 최적 파라미터 검증 (정보 제공)."""
    from src.backtest.engine import BacktestEngine
    from src.backtest.request import BacktestRequest
    from src.strategy import get_strategy

    engine = BacktestEngine()
    strategy_cls = get_strategy(strategy_name)

    try:
        strategy = strategy_cls.from_params(**best_params)
        request = BacktestRequest(data=data_oos, strategy=strategy, portfolio=portfolio)
        result = engine.run(request)
    except Exception:
        logger.warning(f"OOS verification failed for {strategy_name}")
        return float("nan")

    return result.metrics.sharpe_ratio


def _print_optimization_results(
    console: Console,
    strategy_name: str,
    result: Any,
    oos_sharpe: float,
) -> None:
    """최적화 결과 Rich Table 출력."""
    table = Table(title=f"G2H Optimization: {strategy_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Default Sharpe (IS)", f"{result.default_sharpe:.3f}")
    table.add_row("Best Sharpe (IS)", f"{result.best_sharpe:.3f}")
    table.add_row("Improvement", f"{result.improvement_pct:+.1f}%")
    table.add_row("OOS Sharpe", f"{oos_sharpe:.3f}")
    table.add_row("Trials (completed/total)", f"{result.n_completed}/{result.n_trials}")
    table.add_row("Search Space", f"{len(result.search_space)} params")

    console.print(table)

    # Best params table
    params_table = Table(title="Best Parameters")
    params_table.add_column("Parameter", style="bold")
    params_table.add_column("Value", justify="right")

    for name, value in sorted(result.best_params.items()):
        if isinstance(value, float):
            params_table.add_row(name, f"{value:.6f}")
        else:
            params_table.add_row(name, str(value))

    console.print(params_table)


def _print_top_trials(console: Console, result: Any) -> None:
    """Top 10 trials 출력."""
    if not result.top_trials:
        return

    table = Table(title="Top 10 Trials")
    table.add_column("#", style="dim", width=4)
    table.add_column("Sharpe", justify="right", width=8)

    # Get param names from first trial
    if result.top_trials:
        param_names = sorted(result.top_trials[0].get("params", {}).keys())
        for pname in param_names:
            table.add_column(pname, justify="right", width=10)

        for trial in result.top_trials:
            sharpe_val = trial.get("sharpe")
            sharpe_str = f"{sharpe_val:.3f}" if sharpe_val is not None else "N/A"
            values = [_format_param_value(trial.get("params", {}).get(p)) for p in param_names]
            table.add_row(str(trial["number"]), sharpe_str, *values)

    console.print(table)


def _format_param_value(value: Any) -> str:
    """파라미터 값을 Rich 출력용으로 포맷."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _save_json_results(
    strategy_name: str,
    result: Any,
    oos_sharpe: float,
    g3_sweeps: dict[str, list[Any]],
    seed: int,
) -> None:
    """JSON 결과 파일 저장."""
    output_path = _RESULTS_DIR / f"gate2h_{strategy_name}.json"

    output: dict[str, Any] = {
        "meta": {
            "strategy": strategy_name,
            "run_date": datetime.now(tz=UTC).isoformat(),
            "n_trials": result.n_trials,
            "n_completed": result.n_completed,
            "seed": seed,
        },
        "optimization": {
            "best_params": result.best_params,
            "best_sharpe_is": result.best_sharpe,
            "default_sharpe_is": result.default_sharpe,
            "improvement_pct": result.improvement_pct,
            "oos_sharpe": oos_sharpe,
            "search_space": [
                {
                    "name": s.name,
                    "type": s.param_type,
                    "low": s.low,
                    "high": s.high,
                    "default": s.default,
                }
                for s in result.search_space
            ],
            "top_trials": result.top_trials,
        },
        "g3_sweeps": g3_sweeps,
    }

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return str(obj)

    output_path.write_text(
        json.dumps(output, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _update_yaml_g2h(
    strategy_name: str,
    result: Any,
    oos_sharpe: float,
    store: Any,
) -> None:
    """Gate 2H 결과를 YAML에 기록. Always PASS."""
    from src.pipeline.models import GateId, GateVerdict

    details: dict[str, Any] = {
        "best_sharpe_is": round(result.best_sharpe, 3),
        "default_sharpe_is": round(result.default_sharpe, 3),
        "improvement_pct": result.improvement_pct,
        "oos_sharpe": round(oos_sharpe, 3) if not math.isnan(oos_sharpe) else None,
        "n_trials": result.n_trials,
        "n_completed": result.n_completed,
    }

    rationale = (
        f"IS Sharpe {result.default_sharpe:.2f}→{result.best_sharpe:.2f} "
        f"({result.improvement_pct:+.1f}%), OOS={oos_sharpe:.2f}"
    )

    # Record gate (Always PASS)
    store.record_gate(
        strategy_name,
        GateId.G2H,
        GateVerdict.PASS,
        details=details,
        rationale=rationale,
    )

    # Update parameters with optimized values
    record = store.load(strategy_name)
    updated_params = dict(record.parameters)
    updated_params.update(result.best_params)
    updated_record = record.model_copy(update={"parameters": updated_params})
    store.save(updated_record)

    logger.info(f"  YAML updated: {strategy_name} G2H PASS")
