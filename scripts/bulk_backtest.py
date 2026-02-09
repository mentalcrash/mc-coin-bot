#!/usr/bin/env python3
"""Bulk Backtest: 23개 전략 × 5개 자산 일괄 백테스트.

tsmom을 제외한 모든 등록 전략에 대해 Tier 1 자산(5개) × 2020-2025(6년)
단일에셋 백테스트를 실행하고, 결과를 JSON으로 저장한다.

Usage:
    uv run python scripts/bulk_backtest.py

Output:
    results/bulk_backtest_results.json  — 전략별 5개 자산 메트릭
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

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy, list_strategies

# =============================================================================
# Constants
# =============================================================================

EXCLUDE = {"tsmom"}  # 이미 스코어카드 있음

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
]

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)

INITIAL_CAPITAL = Decimal(100000)

RESULTS_DIR = ROOT / "results"


# =============================================================================
# Helpers
# =============================================================================


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


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """PerformanceMetrics → JSON-serializable dict."""
    return {
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown": metrics.max_drawdown,
        "avg_drawdown": metrics.avg_drawdown,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
        "winning_trades": metrics.winning_trades,
        "losing_trades": metrics.losing_trades,
        "volatility": metrics.volatility,
        "skewness": metrics.skewness,
        "kurtosis": metrics.kurtosis,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
    }


def benchmark_to_dict(bench: Any) -> dict[str, Any]:
    """BenchmarkComparison → JSON-serializable dict."""
    return {
        "benchmark_name": bench.benchmark_name,
        "benchmark_return": bench.benchmark_return,
        "alpha": bench.alpha,
        "beta": bench.beta,
        "correlation": bench.correlation,
        "information_ratio": bench.information_ratio,
        "tracking_error": bench.tracking_error,
    }


def run_single(
    engine: BacktestEngine,
    service: MarketDataService,
    strategy_name: str,
    symbol: str,
) -> dict[str, Any] | None:
    """단일 전략 × 단일 자산 백테스트."""
    try:
        data = service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe="1D",
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
            "metrics": metrics_to_dict(result.metrics),
        }
        if result.benchmark is not None:
            entry["benchmark"] = benchmark_to_dict(result.benchmark)
        return entry

    except Exception:
        logger.exception(f"FAIL: {strategy_name} / {symbol}")
        return None


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    strategies = sorted(s for s in list_strategies() if s not in EXCLUDE)
    total = len(strategies) * len(SYMBOLS)

    logger.info(
        f"Bulk Backtest: {len(strategies)} strategies × {len(SYMBOLS)} symbols = {total} runs"
    )
    logger.info(f"Period: {START.date()} ~ {END.date()}, Capital: ${INITIAL_CAPITAL}")

    engine = BacktestEngine()
    service = MarketDataService()

    results: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    done = 0
    t0 = time.perf_counter()

    for strategy_name in strategies:
        strategy_results: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            done += 1
            logger.info(f"[{done}/{total}] {strategy_name} / {symbol}")

            entry = run_single(engine, service, strategy_name, symbol)
            if entry is not None:
                strategy_results.append(entry)
            else:
                errors.append(f"{strategy_name}/{symbol}")

        results[strategy_name] = strategy_results

    elapsed = time.perf_counter() - t0

    # Save results
    output_path = RESULTS_DIR / "bulk_backtest_results.json"
    output = {
        "meta": {
            "run_date": datetime.now(tz=UTC).isoformat(),
            "period": f"{START.date()} ~ {END.date()}",
            "initial_capital": str(INITIAL_CAPITAL),
            "symbols": SYMBOLS,
            "total_runs": total,
            "successful": total - len(errors),
            "failed": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "errors": errors,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    logger.info(f"Done in {elapsed:.1f}s — {total - len(errors)}/{total} succeeded")
    logger.info(f"Results saved to {output_path}")

    if errors:
        logger.warning(f"Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
