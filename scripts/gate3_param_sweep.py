#!/usr/bin/env python3
"""Gate 3: Parameter Stability Validation.

G2 통과 전략에 대해 핵심 파라미터를 ±20% 및 넓은 범위로 스윕하여
파라미터 안정성(고원 존재 + ±20% Sharpe 부호 유지)을 검증한다.

Usage:
    uv run python scripts/gate3_param_sweep.py                # 전체 전략
    uv run python scripts/gate3_param_sweep.py ctrend         # 특정 전략만
    uv run python scripts/gate3_param_sweep.py ctrend kama    # 복수 전략

Output:
    results/gate3_param_sweep.json  -- 전략별 파라미터 스윕 결과
    console                         -- 요약 테이블 + PASS/FAIL 판정
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy

# =============================================================================
# Constants
# =============================================================================

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100_000)
RESULTS_DIR = ROOT / "results"

console = Console()

# =============================================================================
# Strategy Definitions (G2 PASS strategies)
# =============================================================================

# 각 전략의 Best Asset, 기본 파라미터, 스윕할 파라미터 정의
STRATEGIES: dict[str, dict[str, Any]] = {
    "kama": {
        "best_asset": "DOGE/USDT",
        "baseline": {
            "er_lookback": 10,
            "fast_period": 2,
            "slow_period": 30,
            "atr_multiplier": 1.5,
            "vol_target": 0.30,
            "short_mode": 1,
        },
        "sweeps": {
            "er_lookback": [5, 7, 8, 9, 10, 11, 12, 14, 16, 20],
            "slow_period": [20, 24, 26, 28, 30, 32, 34, 36, 40, 50],
            "vol_target": [0.15, 0.20, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40, 0.50],
        },
    },
    "donchian-ensemble": {
        "best_asset": "ETH/USDT",
        "baseline": {
            "vol_target": 0.40,
            "atr_period": 14,
            "short_mode": 0,
        },
        "sweeps": {
            "vol_target": [0.20, 0.25, 0.30, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.55],
            "atr_period": [8, 10, 11, 12, 14, 16, 17, 18, 20, 25],
        },
    },
    "ttm-squeeze": {
        "best_asset": "BTC/USDT",
        "baseline": {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_mult": 1.5,
            "mom_period": 20,
            "exit_sma_period": 21,
            "vol_target": 0.40,
        },
        "sweeps": {
            "bb_period": [12, 14, 16, 18, 20, 22, 24, 26, 30],
            "kc_mult": [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0],
            "vol_target": [0.20, 0.25, 0.30, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.55],
        },
    },
    "max-min": {
        "best_asset": "DOGE/USDT",
        "baseline": {
            "lookback": 10,
            "max_weight": 0.5,
            "min_weight": 0.5,
            "vol_target": 0.30,
        },
        "sweeps": {
            "lookback": [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20],
            "vol_target": [0.15, 0.20, 0.24, 0.27, 0.30, 0.33, 0.36, 0.40, 0.50],
            # max_weight는 별도 처리 (min_weight = 1 - max_weight)
            "max_weight": [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7],
        },
    },
    "bb-rsi": {
        "best_asset": "SOL/USDT",
        "baseline": {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "bb_weight": 0.6,
            "rsi_weight": 0.4,
            "vol_target": 0.20,
        },
        "sweeps": {
            "bb_period": [12, 14, 16, 18, 20, 22, 24, 26, 30],
            "rsi_period": [8, 10, 11, 12, 14, 16, 17, 18, 20],
            "vol_target": [0.10, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.25, 0.30],
            # bb_weight는 별도 처리 (rsi_weight = 1 - bb_weight)
            "bb_weight": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
    },
    "ctrend": {
        "best_asset": "SOL/USDT",
        "baseline": {
            "training_window": 252,
            "prediction_horizon": 5,
            "alpha": 0.5,
            "vol_window": 30,
            "vol_target": 0.35,
            "short_mode": 2,
        },
        "sweeps": {
            # ML 핵심: training window (±20% = 202~302, 넓은 범위 126~400)
            "training_window": [126, 150, 175, 200, 225, 252, 280, 315, 350, 400],
            # 예측 기간 (±20% = 4~6, 넓은 범위 1~21)
            "prediction_horizon": [1, 2, 3, 4, 5, 6, 7, 10, 14, 21],
            # ElasticNet L1 ratio (±20% = 0.4~0.6, 전체 범위)
            "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            # 변동성 타겟 (±20% = 0.28~0.42, 넓은 범위)
            "vol_target": [0.15, 0.20, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.50, 0.60],
        },
    },
}

# 가중치 쌍 (한쪽을 스윕하면 다른 쪽은 1 - value)
WEIGHT_PAIRS: dict[str, dict[str, str]] = {
    "max-min": {"max_weight": "min_weight"},
    "bb-rsi": {"bb_weight": "rsi_weight"},
}


# =============================================================================
# Helpers
# =============================================================================


def load_data(service: MarketDataService, symbol: str) -> Any:
    """Best Asset 데이터 로드."""
    return service.get(
        MarketDataRequest(
            symbol=symbol,
            timeframe="1D",
            start=START,
            end=END,
        )
    )


def create_portfolio(strategy_name: str) -> Portfolio:
    """전략별 권장 설정으로 Portfolio 생성."""
    strategy_cls = get_strategy(strategy_name)
    rec = strategy_cls.recommended_config()
    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    return Portfolio.create(initial_capital=INITIAL_CAPITAL, config=pm_config)


def run_single_backtest(
    engine: BacktestEngine,
    strategy_name: str,
    params: dict[str, Any],
    data: Any,
    portfolio: Portfolio,
) -> dict[str, float]:
    """단일 파라미터 조합으로 백테스트 실행."""
    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls.from_params(**params)
    request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
    result = engine.run(request)
    return {
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "total_return": result.metrics.total_return,
        "max_drawdown": result.metrics.max_drawdown,
        "cagr": result.metrics.cagr,
        "total_trades": result.metrics.total_trades,
    }


def run_one_at_a_time_sweep(
    engine: BacktestEngine,
    strategy_name: str,
    baseline: dict[str, Any],
    param_name: str,
    param_values: list[Any],
    data: Any,
    portfolio: Portfolio,
) -> list[dict[str, Any]]:
    """한 파라미터만 변경하며 스윕 (나머지 고정).

    가중치 쌍 파라미터는 자동으로 보완값 설정.
    """
    results = []
    weight_pair = WEIGHT_PAIRS.get(strategy_name, {})

    for value in param_values:
        params = dict(baseline)
        params[param_name] = value

        # 가중치 쌍 처리
        if param_name in weight_pair:
            complement_name = weight_pair[param_name]
            params[complement_name] = round(1.0 - value, 6)

        try:
            metrics = run_single_backtest(engine, strategy_name, params, data, portfolio)
            results.append({"value": value, **metrics, "error": None})
        except Exception as e:
            logger.warning(f"  {param_name}={value}: {e}")
            results.append(
                {
                    "value": value,
                    "sharpe_ratio": float("nan"),
                    "total_return": float("nan"),
                    "max_drawdown": float("nan"),
                    "cagr": float("nan"),
                    "total_trades": 0,
                    "error": str(e),
                }
            )

    return results


def analyze_sweep(
    param_name: str,
    baseline_value: Any,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """스윕 결과 분석: 고원 + ±20% 안정성."""
    valid = [r for r in results if not math.isnan(r["sharpe_ratio"])]
    if not valid:
        return {
            "param": param_name,
            "baseline_value": baseline_value,
            "plateau_exists": False,
            "pm20_stable": False,
            "verdict": "FAIL",
            "reason": "No valid results",
        }

    sharpes = [r["sharpe_ratio"] for r in valid]
    values = [r["value"] for r in valid]
    best_sharpe = max(sharpes)

    # --- Plateau Detection ---
    # "고원" = best Sharpe의 80% 이상인 값이 3개 이상
    plateau_threshold = best_sharpe * 0.8 if best_sharpe > 0 else 0
    plateau_count = sum(1 for s in sharpes if s >= plateau_threshold)
    plateau_exists = plateau_count >= 3

    # 고원 범위 (plateau에 속하는 값들의 min~max)
    plateau_values = [v for v, s in zip(values, sharpes, strict=True) if s >= plateau_threshold]
    plateau_range = (min(plateau_values), max(plateau_values)) if plateau_values else (None, None)

    # --- ±20% Stability ---
    # 기본값의 ±20% 범위에서 Sharpe 부호 유지
    if isinstance(baseline_value, (int, float)):
        low = baseline_value * 0.8
        high = baseline_value * 1.2
        pm20_results = [r for r in valid if low <= r["value"] <= high]
        if pm20_results:
            pm20_sharpes = [r["sharpe_ratio"] for r in pm20_results]
            pm20_stable = all(s > 0 for s in pm20_sharpes)
            pm20_min_sharpe = min(pm20_sharpes)
            pm20_max_sharpe = max(pm20_sharpes)
        else:
            pm20_stable = False
            pm20_min_sharpe = None
            pm20_max_sharpe = None
    else:
        # 비숫자 파라미터는 ±20% 판정 생략
        pm20_stable = True
        pm20_min_sharpe = None
        pm20_max_sharpe = None

    # --- Baseline Sharpe ---
    baseline_result = next((r for r in valid if r["value"] == baseline_value), None)
    baseline_sharpe = baseline_result["sharpe_ratio"] if baseline_result else None

    # --- Verdict ---
    verdict = "PASS" if (plateau_exists and pm20_stable) else "FAIL"

    return {
        "param": param_name,
        "baseline_value": baseline_value,
        "baseline_sharpe": baseline_sharpe,
        "best_sharpe": best_sharpe,
        "plateau_exists": plateau_exists,
        "plateau_count": plateau_count,
        "plateau_threshold": round(plateau_threshold, 4),
        "plateau_range": plateau_range,
        "pm20_stable": pm20_stable,
        "pm20_min_sharpe": pm20_min_sharpe,
        "pm20_max_sharpe": pm20_max_sharpe,
        "verdict": verdict,
        "all_sharpes": list(zip(values, sharpes, strict=True)),
    }


def print_sweep_table(strategy_name: str, analyses: list[dict[str, Any]]) -> None:
    """스윕 결과 콘솔 출력."""
    table = Table(title=f"Gate 3: {strategy_name}")
    table.add_column("Param", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Base Sharpe", justify="right")
    table.add_column("Best Sharpe", justify="right")
    table.add_column("Plateau", justify="center")
    table.add_column("Plateau #", justify="right")
    table.add_column("Plateau Range", justify="center")
    table.add_column("±20% Stable", justify="center")
    table.add_column("±20% Sharpe", justify="center")
    table.add_column("Verdict", justify="center")

    for a in analyses:
        plateau_range = (
            f"{a['plateau_range'][0]}~{a['plateau_range'][1]}"
            if a["plateau_range"][0] is not None
            else "N/A"
        )
        pm20_sharpe = (
            f"{a['pm20_min_sharpe']:.2f}~{a['pm20_max_sharpe']:.2f}"
            if a["pm20_min_sharpe"] is not None
            else "N/A"
        )
        base_sharpe = f"{a['baseline_sharpe']:.2f}" if a["baseline_sharpe"] is not None else "N/A"

        verdict_style = "green bold" if a["verdict"] == "PASS" else "red bold"

        table.add_row(
            a["param"],
            str(a["baseline_value"]),
            base_sharpe,
            f"{a['best_sharpe']:.2f}",
            "[green]YES[/]" if a["plateau_exists"] else "[red]NO[/]",
            str(a["plateau_count"]),
            plateau_range,
            "[green]YES[/]" if a["pm20_stable"] else "[red]NO[/]",
            pm20_sharpe,
            f"[{verdict_style}]{a['verdict']}[/{verdict_style}]",
        )

    console.print(table)


def print_detail_table(param_name: str, results: list[dict[str, Any]], baseline_value: Any) -> None:
    """파라미터별 상세 Sharpe 테이블."""
    table = Table(title=f"  {param_name} Sweep Detail", show_lines=False)
    table.add_column("Value", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("", justify="center")

    for r in results:
        is_baseline = r["value"] == baseline_value
        marker = "[yellow]*[/]" if is_baseline else ""
        sharpe_str = f"{r['sharpe_ratio']:.2f}" if not math.isnan(r["sharpe_ratio"]) else "ERR"
        ret_str = f"{r['total_return']:.1f}%" if not math.isnan(r["total_return"]) else "ERR"
        mdd_str = f"-{r['max_drawdown']:.1f}%" if not math.isnan(r["max_drawdown"]) else "ERR"
        cagr_str = f"{r['cagr']:.1f}%" if not math.isnan(r["cagr"]) else "ERR"

        table.add_row(
            str(r["value"]),
            sharpe_str,
            ret_str,
            mdd_str,
            cagr_str,
            str(r["total_trades"]),
            marker,
        )

    console.print(table)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # CLI 인수로 특정 전략만 실행 가능
    filter_strategies = sys.argv[1:] if len(sys.argv) > 1 else None
    strategies_to_run = {
        k: v for k, v in STRATEGIES.items() if filter_strategies is None or k in filter_strategies
    }
    if not strategies_to_run:
        console.print(f"[red]No matching strategies. Available: {list(STRATEGIES.keys())}[/]")
        sys.exit(1)

    engine = BacktestEngine()
    service = MarketDataService()
    t0 = time.perf_counter()

    all_results: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    total_runs = 0

    for strategy_name, config in strategies_to_run.items():
        console.rule(f"[bold]{strategy_name}[/] ({config['best_asset']})")

        # Load data
        data = load_data(service, config["best_asset"])
        portfolio = create_portfolio(strategy_name)
        baseline = config["baseline"]
        sweeps = config["sweeps"]

        strategy_results: dict[str, Any] = {
            "best_asset": config["best_asset"],
            "baseline": baseline,
            "analyses": {},
            "raw_sweeps": {},
        }
        analyses = []

        for param_name, param_values in sweeps.items():
            logger.info(f"  Sweeping {param_name}: {param_values}")

            # One-at-a-time sweep
            sweep_results = run_one_at_a_time_sweep(
                engine=engine,
                strategy_name=strategy_name,
                baseline=baseline,
                param_name=param_name,
                param_values=param_values,
                data=data,
                portfolio=portfolio,
            )
            total_runs += len(param_values)

            # Analyze
            baseline_val = baseline.get(param_name)
            analysis = analyze_sweep(param_name, baseline_val, sweep_results)
            analyses.append(analysis)

            # Print detail
            print_detail_table(param_name, sweep_results, baseline_val)

            # Store
            strategy_results["analyses"][param_name] = analysis
            strategy_results["raw_sweeps"][param_name] = sweep_results

        # Print summary table
        print_sweep_table(strategy_name, analyses)

        # Overall verdict: ALL params must PASS
        all_pass = all(a["verdict"] == "PASS" for a in analyses)
        overall = "PASS" if all_pass else "FAIL"
        fail_params = [a["param"] for a in analyses if a["verdict"] != "PASS"]

        strategy_results["overall_verdict"] = overall
        strategy_results["fail_params"] = fail_params
        all_results[strategy_name] = strategy_results

        verdict_style = "[green bold]PASS[/]" if overall == "PASS" else "[red bold]FAIL[/]"
        console.print(f"\n  Gate 3 Overall: {verdict_style}")
        if fail_params:
            console.print(f"  Failed params: {', '.join(fail_params)}")
        console.print()

        summary.append(
            {
                "strategy": strategy_name,
                "best_asset": config["best_asset"],
                "verdict": overall,
                "fail_params": fail_params,
                "param_verdicts": {a["param"]: a["verdict"] for a in analyses},
            }
        )

    elapsed = time.perf_counter() - t0

    # Final summary
    console.rule("[bold]Gate 3 Summary[/]")
    final_table = Table(title="Gate 3 Parameter Stability Results")
    final_table.add_column("Strategy", style="cyan")
    final_table.add_column("Best Asset")
    final_table.add_column("Params Tested", justify="right")
    final_table.add_column("Verdict", justify="center")
    final_table.add_column("Failed Params")

    for s in summary:
        verdict_style = "green bold" if s["verdict"] == "PASS" else "red bold"
        n_params = len(s["param_verdicts"])
        n_pass = sum(1 for v in s["param_verdicts"].values() if v == "PASS")
        final_table.add_row(
            s["strategy"],
            s["best_asset"],
            f"{n_pass}/{n_params}",
            f"[{verdict_style}]{s['verdict']}[/{verdict_style}]",
            ", ".join(s["fail_params"]) if s["fail_params"] else "-",
        )

    console.print(final_table)
    console.print(f"\nTotal: {total_runs} backtests in {elapsed:.1f}s")

    # Save results
    output_path = RESULTS_DIR / "gate3_param_sweep.json"

    def json_default(obj: Any) -> Any:
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return str(obj)

    output = {
        "meta": {
            "run_date": datetime.now(tz=UTC).isoformat(),
            "period": f"{START.date()} ~ {END.date()}",
            "initial_capital": str(INITIAL_CAPITAL),
            "total_runs": total_runs,
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": summary,
        "details": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=json_default), encoding="utf-8")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
