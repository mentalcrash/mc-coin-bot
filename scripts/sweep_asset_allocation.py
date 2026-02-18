#!/usr/bin/env python3
"""Asset Allocation Parameter Sensitivity Sweep.

IV method 기준 vol_lookback × rebalance_bars + min/max_weight 그리드 스윕.

Stage 1: vol_lookback × rebalance_bars (30 조합)
Stage 2: min_weight × max_weight (6 조합, Stage 1 best로 고정)

Usage:
    uv run python scripts/sweep_asset_allocation.py

Output:
    results/sweep_vol_rebal.csv       — 30 rows
    results/sweep_weight_bounds.csv   — 6 rows
"""

from __future__ import annotations

import itertools
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
    from src.models.backtest import PerformanceMetrics

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

# Stage 1 grids
VOL_LOOKBACKS = [10, 20, 30, 60, 90, 120]
REBALANCE_BARS_LIST = [1, 3, 5, 10, 20]

# Stage 2 grids
MIN_WEIGHTS = [0.05, 0.10]
MAX_WEIGHTS = [0.40, 0.60, 0.80]

# Stage 1 defaults
DEFAULT_MIN_WEIGHT = 0.05
DEFAULT_MAX_WEIGHT = 0.60

RESULTS_DIR = ROOT / "results"


# =============================================================================
# Helpers
# =============================================================================


def run_single_config(
    engine: BacktestEngine,
    multi_data: MultiSymbolData,
    strategy_name: str,
    portfolio: Portfolio,
    vol_lookback: int,
    rebalance_bars: int,
    min_weight: float,
    max_weight: float,
) -> dict[str, Any]:
    """단일 파라미터 조합으로 백테스트 실행 → flat dict."""
    strategy_cls = get_strategy(strategy_name)
    strategy = strategy_cls()

    config = AssetAllocationConfig(
        method=AssetAllocationMethod.INVERSE_VOLATILITY,
        vol_lookback=vol_lookback,
        rebalance_bars=rebalance_bars,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    request = MultiAssetBacktestRequest(
        data=multi_data,
        strategy=strategy,
        portfolio=portfolio,
        asset_allocation=config,
    )

    result = engine.run_multi(request)
    m: PerformanceMetrics = result.portfolio_metrics

    return {
        "vol_lookback": vol_lookback,
        "rebalance_bars": rebalance_bars,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "sharpe_ratio": m.sharpe_ratio,
        "cagr": m.cagr,
        "max_drawdown": m.max_drawdown,
        "sortino_ratio": m.sortino_ratio,
        "calmar_ratio": m.calmar_ratio,
        "total_return": m.total_return,
        "win_rate": m.win_rate,
        "total_trades": m.total_trades,
    }


def sweep_vol_rebal(
    engine: BacktestEngine,
    multi_data: MultiSymbolData,
    portfolio: Portfolio,
) -> pd.DataFrame:
    """Stage 1: vol_lookback × rebalance_bars 30 조합 스윕."""
    combos = list(itertools.product(VOL_LOOKBACKS, REBALANCE_BARS_LIST))
    logger.info(f"[Stage 1] Sweeping {len(combos)} vol_lookback × rebalance_bars combos...")

    results: list[dict[str, Any]] = []
    for i, (vol_lb, rebal) in enumerate(combos, 1):
        logger.info(f"  [{i}/{len(combos)}] vol_lookback={vol_lb}, rebalance_bars={rebal}")
        t0 = time.time()
        row = run_single_config(
            engine,
            multi_data,
            STRATEGY_NAME,
            portfolio,
            vol_lookback=vol_lb,
            rebalance_bars=rebal,
            min_weight=DEFAULT_MIN_WEIGHT,
            max_weight=DEFAULT_MAX_WEIGHT,
        )
        elapsed = time.time() - t0
        results.append(row)
        logger.info(
            f"    Sharpe={row['sharpe_ratio']:.3f}, "
            f"CAGR={row['cagr']:.1f}%, "
            f"MDD={row['max_drawdown']:.1f}% ({elapsed:.1f}s)"
        )

    return pd.DataFrame(results)


def sweep_weight_bounds(
    engine: BacktestEngine,
    multi_data: MultiSymbolData,
    portfolio: Portfolio,
    best_vol_lookback: int,
    best_rebalance_bars: int,
) -> pd.DataFrame:
    """Stage 2: min_weight × max_weight 6 조합 스윕."""
    combos = list(itertools.product(MIN_WEIGHTS, MAX_WEIGHTS))
    logger.info(
        f"[Stage 2] Sweeping {len(combos)} min_weight × max_weight combos "
        f"(vol_lookback={best_vol_lookback}, rebalance_bars={best_rebalance_bars})..."
    )

    results: list[dict[str, Any]] = []
    for i, (min_w, max_w) in enumerate(combos, 1):
        logger.info(f"  [{i}/{len(combos)}] min_weight={min_w}, max_weight={max_w}")
        t0 = time.time()
        row = run_single_config(
            engine,
            multi_data,
            STRATEGY_NAME,
            portfolio,
            vol_lookback=best_vol_lookback,
            rebalance_bars=best_rebalance_bars,
            min_weight=min_w,
            max_weight=max_w,
        )
        elapsed = time.time() - t0
        results.append(row)
        logger.info(
            f"    Sharpe={row['sharpe_ratio']:.3f}, "
            f"CAGR={row['cagr']:.1f}%, "
            f"MDD={row['max_drawdown']:.1f}% ({elapsed:.1f}s)"
        )

    return pd.DataFrame(results)


def _print_pivot_table(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str,
) -> None:
    """콘솔 pivot table 출력 (heatmap 스타일)."""
    if df.empty:
        return

    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="first")

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    # Header
    col_values = sorted(pivot.columns)
    header = f"{row_col:>12}"
    for c in col_values:
        header += f" {c:>10}"
    print(header)
    print("-" * (12 + 11 * len(col_values)))

    # Rows — best value 마킹
    all_values = pivot.to_numpy().flatten()
    all_values = all_values[~pd.isna(all_values)]
    best_val = float(max(all_values)) if len(all_values) > 0 else 0.0

    for row_val in sorted(pivot.index):
        line = f"{row_val:>12}"
        for c in col_values:
            val = pivot.loc[row_val, c]
            if pd.isna(val):
                line += f" {'—':>10}"
            else:
                marker = " *" if abs(float(val) - best_val) < 1e-6 else "  "
                line += f" {float(val):>8.3f}{marker}"
        print(line)

    print(f"\n  * = Best {value_col}")


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
    logger.info("Asset Allocation Parameter Sensitivity Sweep")
    logger.info(f"Strategy: {STRATEGY_NAME}, Method: inverse_volatility")
    logger.info(f"Period: {FULL_START.date()} ~ {FULL_END.date()}")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    service = MarketDataService()
    engine = BacktestEngine()

    # 1. 데이터 1회 로드
    logger.info("[Step 1/3] Loading market data...")
    multi_data = load_multi_data(service)
    logger.info(f"  Loaded {len(multi_data.symbols)} symbols: {multi_data.symbols}")

    # Portfolio 생성
    strategy_cls = get_strategy(STRATEGY_NAME)
    rec = strategy_cls.recommended_config()
    pm_config = PortfolioManagerConfig(
        max_leverage_cap=float(rec.get("max_leverage_cap", 2.0)),
        system_stop_loss=float(rec.get("system_stop_loss", 0.10)),
        rebalance_threshold=float(rec.get("rebalance_threshold", 0.05)),
    )
    portfolio = Portfolio.create(initial_capital=INITIAL_CAPITAL, config=pm_config)

    total_start = time.time()

    # 2. Stage 1: vol_lookback × rebalance_bars
    stage1_df = sweep_vol_rebal(engine, multi_data, portfolio)

    # Best combo 추출
    best_idx = stage1_df["sharpe_ratio"].idxmax()
    best_vol = int(stage1_df.loc[best_idx, "vol_lookback"])
    best_rebal = int(stage1_df.loc[best_idx, "rebalance_bars"])
    best_sharpe = float(stage1_df.loc[best_idx, "sharpe_ratio"])

    logger.info(
        f"\n  Stage 1 Best: vol_lookback={best_vol}, "
        f"rebalance_bars={best_rebal}, Sharpe={best_sharpe:.3f}"
    )

    # 3. Stage 2: min_weight × max_weight
    stage2_df = sweep_weight_bounds(engine, multi_data, portfolio, best_vol, best_rebal)

    total_elapsed = time.time() - total_start

    # 4. CSV 저장
    stage1_path = RESULTS_DIR / "sweep_vol_rebal.csv"
    stage2_path = RESULTS_DIR / "sweep_weight_bounds.csv"

    stage1_df.to_csv(stage1_path, index=False)
    stage2_df.to_csv(stage2_path, index=False)

    logger.info("\nResults saved:")
    logger.info(f"  {stage1_path} ({len(stage1_df)} rows)")
    logger.info(f"  {stage2_path} ({len(stage2_df)} rows)")
    logger.info(f"Total time: {total_elapsed:.1f}s")

    # 5. 콘솔 요약
    _print_pivot_table(
        stage1_df,
        row_col="vol_lookback",
        col_col="rebalance_bars",
        value_col="sharpe_ratio",
        title="Stage 1: vol_lookback × rebalance_bars → Sharpe",
    )

    _print_pivot_table(
        stage2_df,
        row_col="min_weight",
        col_col="max_weight",
        value_col="sharpe_ratio",
        title="Stage 2: min_weight × max_weight → Sharpe",
    )

    # Stage 2 best
    if not stage2_df.empty:
        best2_idx = stage2_df["sharpe_ratio"].idxmax()
        best2_min = float(stage2_df.loc[best2_idx, "min_weight"])
        best2_max = float(stage2_df.loc[best2_idx, "max_weight"])
        best2_sharpe = float(stage2_df.loc[best2_idx, "sharpe_ratio"])
        print(
            f"\n  Final Best: vol_lookback={best_vol}, rebalance_bars={best_rebal}, "
            f"min_weight={best2_min}, max_weight={best2_max}, Sharpe={best2_sharpe:.3f}"
        )


if __name__ == "__main__":
    main()
