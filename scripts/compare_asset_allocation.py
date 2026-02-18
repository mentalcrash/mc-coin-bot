#!/usr/bin/env python3
"""8-Asset CTREND Allocation Comparison Script.

CTREND 전략 + 8개 에셋에서 EW / IV / RP / SW 4가지 배분 방법 성과 비교.

Usage:
    uv run python scripts/compare_asset_allocation.py

Output:
    results/asset_allocation_comparison.csv  — 4 methods × metrics
    results/asset_allocation_per_symbol.csv  — method × symbol × contribution
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.backtest.request import MultiAssetBacktestRequest
from src.data.service import MarketDataService
from src.orchestrator.asset_allocator import AssetAllocationConfig
from src.orchestrator.models import AssetAllocationMethod
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy

if TYPE_CHECKING:
    from src.data.market_data import MultiSymbolData
    from src.models.backtest import MultiAssetBacktestResult, PerformanceMetrics

# =============================================================================
# Constants
# =============================================================================

SYMBOLS_8 = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]

SYMBOLS_5_FALLBACK = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
]

STRATEGY_NAME = "ctrend"
FULL_START = datetime(2022, 1, 1, tzinfo=UTC)
FULL_END = datetime(2025, 12, 31, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100_000)

ALLOCATION_METHODS = [
    AssetAllocationMethod.EQUAL_WEIGHT,
    AssetAllocationMethod.INVERSE_VOLATILITY,
    AssetAllocationMethod.RISK_PARITY,
    AssetAllocationMethod.SIGNAL_WEIGHTED,
]

RESULTS_DIR = ROOT / "results"


# =============================================================================
# Helpers
# =============================================================================


def metrics_to_dict(
    metrics: PerformanceMetrics,
    method_name: str,
) -> dict[str, Any]:
    """PerformanceMetrics -> flat dict for CSV."""
    return {
        "method": method_name,
        "sharpe_ratio": metrics.sharpe_ratio,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "total_return": metrics.total_return,
        "win_rate": metrics.win_rate,
        "total_trades": metrics.total_trades,
        "profit_factor": metrics.profit_factor,
        "volatility": metrics.volatility,
    }


def per_symbol_to_rows(
    result: MultiAssetBacktestResult,
    method_name: str,
) -> list[dict[str, Any]]:
    """심볼별 기여도를 flat dict 리스트로 변환."""
    rows: list[dict[str, Any]] = []
    for symbol in result.config.symbols:
        weight = result.config.asset_weights.get(symbol, 0.0)
        contribution = result.contribution.get(symbol, 0.0)
        rows.append(
            {
                "method": method_name,
                "symbol": symbol,
                "final_weight": round(weight, 4),
                "contribution_pct": contribution,
            }
        )
    return rows


def run_allocation_backtest(
    engine: BacktestEngine,
    multi_data: MultiSymbolData,
    strategy_name: str,
    portfolio: Portfolio,
    method: AssetAllocationMethod,
) -> MultiAssetBacktestResult:
    """특정 배분 방법으로 멀티에셋 백테스트 실행."""
    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls()

    config = AssetAllocationConfig(
        method=method,
        vol_lookback=60,
        rebalance_bars=5,
        min_weight=0.05,
        max_weight=0.60,
    )

    request = MultiAssetBacktestRequest(
        data=multi_data,
        strategy=strategy,
        portfolio=portfolio,
        asset_allocation=config,
    )
    return engine.run_multi(request)


def run_ew_baseline(
    engine: BacktestEngine,
    multi_data: MultiSymbolData,
    strategy_name: str,
    portfolio: Portfolio,
) -> MultiAssetBacktestResult:
    """EW baseline (asset_allocation=None, 기존 정적 weight 경로)."""
    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls()

    request = MultiAssetBacktestRequest(
        data=multi_data,
        strategy=strategy,
        portfolio=portfolio,
    )
    return engine.run_multi(request)


def load_multi_data(service: MarketDataService) -> MultiSymbolData:
    """데이터 로드 (8종 시도 → 실패 시 5종 fallback)."""
    try:
        return service.get_multi(
            symbols=SYMBOLS_8,
            timeframe="1D",
            start=FULL_START,
            end=FULL_END,
        )
    except Exception:
        logger.warning("8-symbol load failed, falling back to 5 symbols")
        return service.get_multi(
            symbols=SYMBOLS_5_FALLBACK,
            timeframe="1D",
            start=FULL_START,
            end=FULL_END,
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """메인 실행 함수."""
    logger.info("=" * 70)
    logger.info("Asset Allocation Comparison: CTREND")
    logger.info(f"Methods: {len(ALLOCATION_METHODS)}")
    logger.info(f"Period: {FULL_START.date()} ~ {FULL_END.date()}")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    service = MarketDataService()
    engine = BacktestEngine()

    # 1. 데이터 1회 로드
    logger.info("[Step 1/3] Loading market data...")
    multi_data = load_multi_data(service)
    n_symbols = len(multi_data.symbols)
    logger.info(f"  Loaded {n_symbols} symbols: {multi_data.symbols}")

    # 전략 권장 설정으로 Portfolio 생성
    strategy_cls = get_strategy(STRATEGY_NAME)
    rec = strategy_cls.recommended_config()
    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    portfolio = Portfolio.create(initial_capital=INITIAL_CAPITAL, config=pm_config)

    # 2. 4가지 method별 백테스트
    logger.info(f"\n[Step 2/3] Running {len(ALLOCATION_METHODS)} allocation backtests...")
    all_metrics: list[dict[str, Any]] = []
    all_per_symbol: list[dict[str, Any]] = []
    total_start = time.time()

    for i, method in enumerate(ALLOCATION_METHODS, 1):
        logger.info(f"  [{i}/{len(ALLOCATION_METHODS)}] {method.value}...")
        t0 = time.time()
        result = run_allocation_backtest(engine, multi_data, STRATEGY_NAME, portfolio, method)
        elapsed = time.time() - t0

        row = metrics_to_dict(result.portfolio_metrics, method.value)
        all_metrics.append(row)
        all_per_symbol.extend(per_symbol_to_rows(result, method.value))

        logger.info(
            f"    Sharpe={row['sharpe_ratio']:.2f}, "
            f"CAGR={row['cagr']:.1f}%, "
            f"MDD={row['max_drawdown']:.1f}% "
            f"({elapsed:.1f}s)"
        )

    # 3. EW baseline (asset_allocation=None) 교차 검증
    logger.info("\n[Step 3/3] EW Baseline (static weights)...")
    baseline_result = run_ew_baseline(engine, multi_data, STRATEGY_NAME, portfolio)
    baseline_row = metrics_to_dict(baseline_result.portfolio_metrics, "ew_baseline_static")
    all_metrics.append(baseline_row)
    all_per_symbol.extend(per_symbol_to_rows(baseline_result, "ew_baseline_static"))

    total_elapsed = time.time() - total_start

    # 4. CSV 저장
    metrics_df = pd.DataFrame(all_metrics)
    per_symbol_df = pd.DataFrame(all_per_symbol)

    metrics_path = RESULTS_DIR / "asset_allocation_comparison.csv"
    per_symbol_path = RESULTS_DIR / "asset_allocation_per_symbol.csv"

    metrics_df.to_csv(metrics_path, index=False)
    per_symbol_df.to_csv(per_symbol_path, index=False)

    logger.info("\nResults saved:")
    logger.info(f"  {metrics_path} ({len(metrics_df)} rows)")
    logger.info(f"  {per_symbol_path} ({len(per_symbol_df)} rows)")
    logger.info(f"Total time: {total_elapsed:.1f}s")

    # 5. 콘솔 요약
    _print_summary(metrics_df, per_symbol_df)


def _print_summary(
    metrics_df: pd.DataFrame,
    per_symbol_df: pd.DataFrame,
) -> None:
    """콘솔 요약 테이블 출력."""
    print("\n" + "=" * 90)
    print("ASSET ALLOCATION COMPARISON — CTREND")
    print("=" * 90)

    # 메인 비교 테이블
    print(
        f"\n{'Method':<25} {'Sharpe':>8} {'CAGR%':>8} {'MDD%':>8} "
        f"{'Sortino':>8} {'Calmar':>8} {'WinRate%':>8} {'Trades':>8}"
    )
    print("-" * 90)

    # Sharpe 기준 best method 찾기
    non_baseline = metrics_df[metrics_df["method"] != "ew_baseline_static"]
    best_sharpe = non_baseline["sharpe_ratio"].max() if not non_baseline.empty else 0.0

    for _, row in metrics_df.iterrows():
        method = str(row["method"])
        sharpe = float(row.get("sharpe_ratio") or 0)
        cagr = float(row.get("cagr") or 0)
        mdd = float(row.get("max_drawdown") or 0)
        sortino = float(row.get("sortino_ratio") or 0)
        calmar = float(row.get("calmar_ratio") or 0)
        wr = float(row.get("win_rate") or 0)
        trades = int(row.get("total_trades") or 0)

        marker = " ***" if sharpe == best_sharpe and method != "ew_baseline_static" else ""
        if method == "ew_baseline_static":
            method = "ew_baseline (static)"

        print(
            f"{method:<25} {sharpe:>8.2f} {cagr:>8.1f} {mdd:>8.1f} "
            f"{sortino:>8.2f} {calmar:>8.2f} {wr:>8.1f} {trades:>8d}{marker}"
        )

    # Per-symbol weight 테이블
    methods = per_symbol_df["method"].unique()
    symbols = per_symbol_df["symbol"].unique()

    print(f"\n{'Method':<25}", end="")
    for s in symbols:
        short = s.replace("/USDT", "")
        print(f" {short:>8}", end="")
    print()
    print("-" * (25 + 9 * len(symbols)))

    for method in methods:
        method_rows = per_symbol_df[per_symbol_df["method"] == method]
        label = method if method != "ew_baseline_static" else "ew_baseline (static)"
        print(f"{label:<25}", end="")
        for s in symbols:
            symbol_row = method_rows[method_rows["symbol"] == s]
            if not symbol_row.empty:
                w = float(symbol_row["final_weight"].iloc[0])
                print(f" {w:>8.3f}", end="")
            else:
                print(f" {'—':>8}", end="")
        print()

    print("\n" + "=" * 90)
    print("*** = Best Sharpe Ratio")
    print("=" * 90)


if __name__ == "__main__":
    main()
