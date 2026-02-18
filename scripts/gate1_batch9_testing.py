#!/usr/bin/env python3
"""Gate 1: TESTING 전략 6종 단일에셋 백테스트.

Batch 9: adaptive-fr-carry, fear-divergence, liq-cascade-rev,
         multi-domain-score, onchain-bias-4h, vol-squeeze-deriv

모두 4H TF. Derivatives(funding_rate) / On-chain 데이터 포함.

Usage:
    uv run python scripts/gate1_batch9_testing.py
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
from src.data.market_data import MarketDataRequest, MarketDataSet
from src.data.onchain.service import OnchainDataService
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy

# =============================================================================
# Constants
# =============================================================================

# Group A: OHLCV + funding_rate (derivatives)
DERIV_STRATEGIES = [
    "adaptive-fr-carry",
    "liq-cascade-rev",
    "multi-domain-score",
    "vol-squeeze-deriv",
]

# Group B: OHLCV + on-chain (oc_fear_greed)
ONCHAIN_LIGHT_STRATEGIES = ["fear-divergence"]

# Group C: OHLCV + on-chain (mvrv, flow, stablecoin)
ONCHAIN_HEAVY_STRATEGIES = ["onchain-bias-4h"]

ALL_STRATEGIES = DERIV_STRATEGIES + ONCHAIN_LIGHT_STRATEGIES + ONCHAIN_HEAVY_STRATEGIES

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
TIMEFRAME = "4H"

RESULTS_DIR = ROOT / "results"

# On-chain rename mapping
ONCHAIN_RENAME = {
    "oc_stablecoin_total_usd": "oc_stablecoin_total_circulating_usd",
}


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


def enrich_with_onchain(data: MarketDataSet, symbol: str) -> MarketDataSet:
    """On-chain 데이터를 OHLCV에 병합."""
    import pandas as pd

    onchain_service = OnchainDataService()
    onchain_df = onchain_service.precompute(
        symbol=symbol,
        ohlcv_index=data.ohlcv.index,
    )

    if onchain_df.empty:
        logger.warning(f"No on-chain data for {symbol}")
        return data

    merged = pd.merge_asof(
        data.ohlcv,
        onchain_df,
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # Rename: oc_stablecoin_total_usd -> oc_stablecoin_total_circulating_usd
    for old_name, new_name in ONCHAIN_RENAME.items():
        if old_name in merged.columns:
            merged = merged.rename(columns={old_name: new_name})

    return MarketDataSet(
        symbol=data.symbol,
        timeframe=data.timeframe,
        start=data.start,
        end=data.end,
        ohlcv=merged,
    )


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """PerformanceMetrics -> JSON dict."""
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
    """BenchmarkComparison -> JSON dict."""
    return {
        "benchmark_name": bench.benchmark_name,
        "benchmark_return": bench.benchmark_return,
        "alpha": bench.alpha,
        "beta": bench.beta,
        "correlation": bench.correlation,
        "information_ratio": bench.information_ratio,
        "tracking_error": bench.tracking_error,
    }


def needs_derivatives(strategy_name: str) -> bool:
    """전략이 funding_rate 등 derivatives 컬럼을 필요로 하는지."""
    return strategy_name in DERIV_STRATEGIES


def needs_onchain(strategy_name: str) -> bool:
    """전략이 on-chain 데이터를 필요로 하는지."""
    return strategy_name in ONCHAIN_LIGHT_STRATEGIES + ONCHAIN_HEAVY_STRATEGIES


def run_single(
    engine: BacktestEngine,
    data_service: MarketDataService,
    strategy_name: str,
    symbol: str,
) -> dict[str, Any] | None:
    """단일 전략 x 단일 자산 백테스트."""
    try:
        # 1. OHLCV (+derivatives if needed) 로드
        data = data_service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe=TIMEFRAME,
                start=START,
                end=END,
            ),
            include_derivatives=needs_derivatives(strategy_name),
        )
        logger.info(
            f"  Loaded {symbol}: {data.periods} bars "
            f"({data.start.date()} ~ {data.end.date()})"
        )

        # 2. On-chain enrichment (if needed)
        if needs_onchain(strategy_name):
            data = enrich_with_onchain(data, symbol)

        # 3. 전략 + 포트폴리오 생성
        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
        portfolio = create_portfolio(strategy_name)

        # 4. 백테스트 실행
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

    total = len(ALL_STRATEGIES) * len(SYMBOLS)
    logger.info(
        f"Gate 1 Batch 9 (TESTING): {len(ALL_STRATEGIES)} strategies x "
        f"{len(SYMBOLS)} symbols = {total} runs"
    )
    logger.info(f"Period: {START.date()} ~ {END.date()}, TF: {TIMEFRAME}, Capital: ${INITIAL_CAPITAL}")

    engine = BacktestEngine()
    data_service = MarketDataService()

    results: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    done = 0
    t0 = time.perf_counter()

    for strategy_name in ALL_STRATEGIES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"{'='*60}")

        strategy_results: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            done += 1
            logger.info(f"[{done}/{total}] {strategy_name} / {symbol}")

            entry = run_single(engine, data_service, strategy_name, symbol)
            if entry is not None:
                strategy_results.append(entry)
            else:
                errors.append(f"{strategy_name}/{symbol}")

        results[strategy_name] = strategy_results

    elapsed = time.perf_counter() - t0

    # Save results
    output_path = RESULTS_DIR / "gate1_batch9_testing.json"
    output = {
        "meta": {
            "run_date": datetime.now(tz=UTC).isoformat(),
            "period": f"{START.date()} ~ {END.date()}",
            "timeframe": TIMEFRAME,
            "initial_capital": str(INITIAL_CAPITAL),
            "symbols": SYMBOLS,
            "strategies": ALL_STRATEGIES,
            "total_runs": total,
            "successful": total - len(errors),
            "failed": len(errors),
            "elapsed_seconds": round(elapsed, 1),
            "errors": errors,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    # Print summary
    logger.info(f"\nDone in {elapsed:.1f}s — {total - len(errors)}/{total} succeeded")
    logger.info(f"Results saved to {output_path}")

    for strat_name, strat_results in results.items():
        logger.info(f"\n--- {strat_name} ---")
        for r in strat_results:
            m = r["metrics"]
            logger.info(
                f"  {r['symbol']}: Sharpe={m['sharpe_ratio']:.2f}, "
                f"CAGR={m['cagr']:.1%}, MDD={m['max_drawdown']:.1%}, "
                f"Trades={m['total_trades']}"
            )
        if not strat_results:
            logger.info("  (no results)")

    if errors:
        logger.warning(f"Errors ({len(errors)}): {errors}")


if __name__ == "__main__":
    main()
