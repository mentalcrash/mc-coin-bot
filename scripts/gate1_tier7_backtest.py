#!/usr/bin/env python3
"""Gate 1: Tier 7 전략 (6H/12H TF) 5-coin 백테스트.

Usage:
    uv run python scripts/gate1_tier7_backtest.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

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

STRATEGIES: dict[str, str] = {
    "accel-conv": "6h",
    "anchor-mom": "12h",
    "qd-mom": "6h",
    "accel-skew": "12h",
}

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
]

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100_000)
RESULTS_DIR = ROOT / "results"

console = Console()


def create_portfolio(strategy_name: str) -> Portfolio:
    strategy_cls = get_strategy(strategy_name)
    rec = strategy_cls.recommended_config()
    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    return Portfolio.create(initial_capital=INITIAL_CAPITAL, config=pm_config)


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    return {
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
    }


def benchmark_to_dict(bench: Any) -> dict[str, Any]:
    return {
        "alpha": bench.alpha,
        "beta": bench.beta,
    }


def run_single(
    engine: BacktestEngine,
    service: MarketDataService,
    strategy_name: str,
    symbol: str,
    timeframe: str,
) -> dict[str, Any] | None:
    try:
        data = service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe=timeframe,
                start=START,
                end=END,
            )
        )
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
        portfolio = create_portfolio(strategy_name)

        request = BacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result = engine.run(request)

        entry: dict[str, Any] = {
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "metrics": metrics_to_dict(result.metrics),
        }
        if result.benchmark is not None:
            entry["benchmark"] = benchmark_to_dict(result.benchmark)
        return entry

    except Exception:
        logger.exception(f"FAIL: {strategy_name} / {symbol}")
        return None


def print_strategy_results(strategy_name: str, results: list[dict[str, Any]]) -> None:
    table = Table(title=f"Gate 1: {strategy_name}")
    table.add_column("Asset", style="cyan")
    table.add_column("Sharpe", justify="right")
    table.add_column("CAGR", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("WR", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Beta", justify="right")

    for r in sorted(results, key=lambda x: x["metrics"]["sharpe_ratio"], reverse=True):
        m = r["metrics"]
        b = r.get("benchmark", {})
        sharpe_style = "green" if m["sharpe_ratio"] > 1.0 else ("yellow" if m["sharpe_ratio"] > 0 else "red")
        table.add_row(
            r["symbol"],
            f"[{sharpe_style}]{m['sharpe_ratio']:.2f}[/]",
            f"{m['cagr']:+.1f}%",
            f"-{m['max_drawdown']:.1f}%",
            str(m["total_trades"]),
            f"{m['profit_factor']:.2f}" if m["profit_factor"] else "N/A",
            f"{m['win_rate']:.1f}%",
            f"{m['sortino_ratio']:.2f}" if m["sortino_ratio"] else "N/A",
            f"{m['calmar_ratio']:.2f}" if m["calmar_ratio"] else "N/A",
            f"{b.get('alpha', 0):+.1f}%" if b else "N/A",
            f"{b.get('beta', 0):.2f}" if b else "N/A",
        )

    console.print(table)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    engine = BacktestEngine()
    service = MarketDataService()
    t0 = time.perf_counter()

    all_results: dict[str, list[dict[str, Any]]] = {}
    total = len(STRATEGIES) * len(SYMBOLS)
    done = 0

    for strategy_name, tf in STRATEGIES.items():
        console.rule(f"[bold]{strategy_name}[/] ({tf})")
        strategy_results: list[dict[str, Any]] = []

        for symbol in SYMBOLS:
            done += 1
            logger.info(f"[{done}/{total}] {strategy_name} / {symbol} ({tf})")

            entry = run_single(engine, service, strategy_name, symbol, tf)
            if entry is not None:
                m = entry["metrics"]
                logger.info(
                    f"  -> Sharpe={m['sharpe_ratio']:.2f}, CAGR={m['cagr']:+.1f}%, "
                    f"MDD=-{m['max_drawdown']:.1f}%, Trades={m['total_trades']}"
                )
                strategy_results.append(entry)
            else:
                logger.error(f"  -> FAILED")

        all_results[strategy_name] = strategy_results
        print_strategy_results(strategy_name, strategy_results)

    elapsed = time.perf_counter() - t0

    # Save
    output_path = RESULTS_DIR / "gate1_tier7_results.json"
    output = {
        "meta": {
            "run_date": datetime.now(tz=UTC).isoformat(),
            "period": f"{START.date()} ~ {END.date()}",
            "initial_capital": str(INITIAL_CAPITAL),
            "symbols": SYMBOLS,
            "strategies": dict(STRATEGIES),
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info(f"Results saved to {output_path}")

    # Summary
    console.rule("[bold]Gate 1 Summary[/]")
    summary_table = Table(title="Tier 7 Gate 1 Results")
    summary_table.add_column("Strategy", style="cyan")
    summary_table.add_column("TF")
    summary_table.add_column("Best Asset")
    summary_table.add_column("Best Sharpe", justify="right")
    summary_table.add_column("Best CAGR", justify="right")
    summary_table.add_column("Best MDD", justify="right")
    summary_table.add_column("Trades", justify="right")
    summary_table.add_column("Verdict", justify="center")

    for strategy_name, results in all_results.items():
        if not results:
            summary_table.add_row(strategy_name, STRATEGIES[strategy_name], "-", "-", "-", "-", "-", "[red]ERROR[/]")
            continue
        best = max(results, key=lambda x: x["metrics"]["sharpe_ratio"])
        m = best["metrics"]
        is_pass = (
            m["sharpe_ratio"] > 1.0
            and m["cagr"] > 20
            and m["max_drawdown"] < 40
            and m["total_trades"] > 50
        )
        verdict = "[green bold]PASS[/]" if is_pass else "[red bold]FAIL[/]"
        summary_table.add_row(
            strategy_name,
            STRATEGIES[strategy_name],
            best["symbol"],
            f"{m['sharpe_ratio']:.2f}",
            f"{m['cagr']:+.1f}%",
            f"-{m['max_drawdown']:.1f}%",
            str(m["total_trades"]),
            verdict,
        )

    console.print(summary_table)
    console.print(f"\nTotal: {done} backtests in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
