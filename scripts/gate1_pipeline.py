#!/usr/bin/env python3
"""Gate 1: 5-coin x 6-year single-asset backtest for gate pipeline candidates.

Usage:
    uv run python scripts/gate1_pipeline.py entropy-switch kalman-trend vwap-disposition
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy

SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100_000)
RESULTS_DIR = ROOT / "results"

# 비-1D timeframe 전략 목록
HOURLY_STRATEGIES = {"session-breakout", "liq-momentum", "flow-imbalance", "hour-season"}
FOUR_HOURLY_STRATEGIES = {"perm-entropy-mom", "candle-reject", "vol-climax", "ou-meanrev"}


def create_portfolio(strategy_name: str) -> Portfolio:
    strategy_cls = get_strategy(strategy_name)
    rec = strategy_cls.recommended_config()
    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    return Portfolio.create(initial_capital=INITIAL_CAPITAL, config=pm_config)


def run_single(
    engine: BacktestEngine,
    service: MarketDataService,
    strategy_name: str,
    symbol: str,
) -> dict[str, Any] | None:
    try:
        if strategy_name in HOURLY_STRATEGIES:
            timeframe = "1h"
        elif strategy_name in FOUR_HOURLY_STRATEGIES:
            timeframe = "4h"
        else:
            timeframe = "1D"
        data = service.get(
            MarketDataRequest(symbol=symbol, timeframe=timeframe, start=START, end=END)
        )
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
        portfolio = create_portfolio(strategy_name)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        result = engine.run(request)

        m = result.metrics
        entry: dict[str, Any] = {
            "strategy": strategy_name,
            "symbol": symbol,
            "sharpe_ratio": m.sharpe_ratio,
            "cagr": m.cagr,
            "max_drawdown": m.max_drawdown,
            "total_trades": m.total_trades,
            "total_return": m.total_return,
            "profit_factor": m.profit_factor,
            "win_rate": m.win_rate,
            "sortino_ratio": m.sortino_ratio,
            "calmar_ratio": m.calmar_ratio,
            "volatility": m.volatility,
        }
        if result.benchmark is not None:
            entry["alpha"] = result.benchmark.alpha
            entry["beta"] = result.benchmark.beta
        return entry
    except Exception:
        logger.exception(f"FAIL: {strategy_name} / {symbol}")
        return None


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    strategies = sys.argv[1:] if len(sys.argv) > 1 else []
    if not strategies:
        print("Usage: uv run python scripts/gate1_pipeline.py strategy1 strategy2 ...")
        sys.exit(1)

    engine = BacktestEngine()
    service = MarketDataService()
    t0 = time.perf_counter()
    all_results: dict[str, list[dict[str, Any]]] = {}

    for sname in strategies:
        logger.info(f"=== {sname} ===")
        results = []
        for sym in SYMBOLS:
            logger.info(f"  {sname} / {sym}")
            entry = run_single(engine, service, sname, sym)
            if entry:
                results.append(entry)
                logger.info(
                    f"    Sharpe={entry['sharpe_ratio']:.2f} CAGR={entry['cagr']:.1f}% "
                    f"MDD={entry['max_drawdown']:.1f}% Trades={entry['total_trades']}"
                )
        all_results[sname] = results

    elapsed = time.perf_counter() - t0

    output_path = RESULTS_DIR / "gate1_pipeline_results.json"
    output = {
        "meta": {
            "run_date": datetime.now(tz=UTC).isoformat(),
            "period": f"{START.date()} ~ {END.date()}",
            "initial_capital": str(INITIAL_CAPITAL),
            "symbols": SYMBOLS,
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": all_results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info(f"Results saved to {output_path} ({elapsed:.1f}s)")

    # Update YAML
    for sname, results in all_results.items():
        _update_yaml(sname, results)

    # Print summary
    for sname, results in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"  {sname}")
        print(f"{'=' * 60}")
        print(
            f"  {'Symbol':<12} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Trades':>8} {'PF':>8} {'Alpha':>10} {'Beta':>6}"
        )
        print(
            f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 6}"
        )
        for r in sorted(results, key=lambda x: x.get("sharpe_ratio") or 0, reverse=True):

            def fmt(val: Any, spec: str = ".2f") -> str:
                return f"{val:{spec}}" if val is not None else "N/A"

            alpha_val = r.get("alpha")
            beta_val = r.get("beta")
            alpha = f"{alpha_val:.1f}%" if alpha_val is not None else "N/A"
            beta = fmt(beta_val)
            pf = fmt(r.get("profit_factor"))
            sharpe = fmt(r.get("sharpe_ratio"))
            cagr = fmt(r.get("cagr"), ".1f")
            mdd = fmt(-r["max_drawdown"] if r.get("max_drawdown") is not None else None, ".1f")
            print(
                f"  {r['symbol']:<12} {sharpe:>8} "
                f"{cagr:>7}% {mdd:>7}% "
                f"{r['total_trades']:>8} {pf:>8} "
                f"{alpha:>10} {beta:>6}"
            )


def _update_yaml(strategy_name: str, results: list[dict[str, Any]]) -> None:
    """Gate 1 결과를 YAML에 자동 기록."""
    from src.pipeline.models import AssetMetrics, GateId, GateVerdict, StrategyStatus
    from src.pipeline.store import StrategyStore

    store = StrategyStore()
    if not store.exists(strategy_name):
        return

    sorted_results = sorted(results, key=lambda x: x.get("sharpe_ratio") or 0, reverse=True)
    best = sorted_results[0]
    best_sharpe = best.get("sharpe_ratio") or 0
    best_cagr = (best.get("cagr") or 0) * 100
    best_mdd = abs(best.get("max_drawdown") or 0)
    best_trades = best.get("total_trades") or 0

    verdict = GateVerdict.PASS if (
        best_sharpe > 1.0 and best_cagr > 20.0 and best_mdd < 40.0 and best_trades > 50
    ) else GateVerdict.FAIL

    details = {
        "best_asset": best["symbol"],
        "sharpe": round(best_sharpe, 2),
        "cagr": round(best_cagr, 1),
        "mdd": round(best_mdd, 1),
        "trades": best_trades,
    }
    rationale = (
        f"{best['symbol']} Sharpe {best_sharpe:.2f}, "
        f"CAGR {best_cagr:+.1f}%, MDD -{best_mdd:.1f}%"
    )

    store.record_gate(strategy_name, GateId.G1, verdict, details=details, rationale=rationale)

    metrics = [
        AssetMetrics(
            symbol=r["symbol"],
            sharpe=round(r.get("sharpe_ratio") or 0, 2),
            cagr=round((r.get("cagr") or 0) * 100, 1),
            mdd=round(abs(r.get("max_drawdown") or 0), 1),
            trades=r.get("total_trades") or 0,
            profit_factor=round(r["profit_factor"], 2) if r.get("profit_factor") else None,
            win_rate=round(r["win_rate"], 1) if r.get("win_rate") else None,
            sortino=round(r["sortino_ratio"], 2) if r.get("sortino_ratio") else None,
            calmar=round(r["calmar_ratio"], 2) if r.get("calmar_ratio") else None,
            alpha=round(r["alpha"], 1) if r.get("alpha") else None,
            beta=round(r["beta"], 2) if r.get("beta") else None,
        )
        for r in results
    ]
    store.set_asset_performance(strategy_name, metrics)

    if verdict == GateVerdict.FAIL:
        store.update_status(strategy_name, StrategyStatus.RETIRED)

    logger.info(f"  YAML updated: {strategy_name} G1 {verdict}")


if __name__ == "__main__":
    main()
