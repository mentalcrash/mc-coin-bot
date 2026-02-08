#!/usr/bin/env python3
"""Tier 2 Multi-Coin Strategy Comparison Script.

6개 Tier 2 전략을 5개 코인에 대해 2022-2025 백테스트하고,
레짐별 분석까지 수행하는 종합 비교 스크립트.

Usage:
    uv run python scripts/tier2_multi_coin_comparison.py

Output:
    results/tier2_single_coin_metrics.csv  — 30개 개별 결과
    results/tier2_multi_asset_metrics.csv  — 6개 포트폴리오 결과
    results/tier2_regime_analysis.csv      — 레짐별 결과
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
from src.backtest.request import BacktestRequest, MultiAssetBacktestRequest
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio import Portfolio, PortfolioManagerConfig
from src.strategy import get_strategy

if TYPE_CHECKING:
    from src.models.backtest import PerformanceMetrics

# =============================================================================
# Constants
# =============================================================================

STRATEGIES = [
    "donchian-ensemble",
    "larry-vb",
    "mtf-macd",
    "stoch-mom",
    "mom-mr-blend",
    "ttm-squeeze",
]

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
]

FULL_START = datetime(2022, 1, 1, tzinfo=UTC)
FULL_END = datetime(2025, 12, 31, tzinfo=UTC)

# Regime definitions
REGIMES: dict[str, tuple[datetime, datetime]] = {
    "Bear": (datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 12, 31, tzinfo=UTC)),
    "Recovery": (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)),
    "Bull": (datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
}

INITIAL_CAPITAL = Decimal(100000)

RESULTS_DIR = ROOT / "results"


# =============================================================================
# Helpers
# =============================================================================


def metrics_to_dict(
    metrics: PerformanceMetrics,
    strategy_name: str,
    symbol: str = "",
    regime: str = "Full",
) -> dict[str, Any]:
    """PerformanceMetrics → flat dict for CSV."""
    return {
        "strategy": strategy_name,
        "symbol": symbol,
        "regime": regime,
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "calmar_ratio": metrics.calmar_ratio,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
        "winning_trades": metrics.winning_trades,
        "losing_trades": metrics.losing_trades,
        "volatility": metrics.volatility,
    }


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


def safe_backtest_single(
    engine: BacktestEngine,
    service: MarketDataService,
    strategy_name: str,
    symbol: str,
    start: datetime,
    end: datetime,
    regime: str = "Full",
) -> dict[str, Any] | None:
    """단일 백테스트 실행 (에러 시 None 반환)."""
    try:
        data = service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe="1D",
                start=start,
                end=end,
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
        return metrics_to_dict(result.metrics, strategy_name, symbol, regime)
    except Exception as e:
        logger.warning(f"Failed: {strategy_name} / {symbol} / {regime}: {e}")
        return None


def safe_backtest_multi(
    engine: BacktestEngine,
    service: MarketDataService,
    strategy_name: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, Any] | None:
    """멀티에셋 EW 포트폴리오 백테스트 (에러 시 None 반환)."""
    try:
        multi_data = service.get_multi(
            symbols=symbols,
            timeframe="1D",
            start=start,
            end=end,
        )

        strategy_cls = get_strategy(strategy_name)
        strategy = strategy_cls()
        portfolio = create_portfolio(strategy_name)

        request = MultiAssetBacktestRequest(
            data=multi_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        result = engine.run_multi(request)
        return metrics_to_dict(
            result.portfolio_metrics,
            strategy_name,
            "EW Portfolio",
            "Full",
        )
    except Exception as e:
        logger.warning(f"Failed multi: {strategy_name}: {e}")
        return None


def compute_buy_hold(
    service: MarketDataService,
    symbol: str,
    start: datetime,
    end: datetime,
) -> dict[str, Any] | None:
    """Buy & Hold 벤치마크 계산."""
    try:
        data = service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe="1D",
                start=start,
                end=end,
            )
        )
        close = data.ohlcv["close"].astype(float)
        total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
        years = len(close) / 365.0
        cagr = ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        # Max Drawdown
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        max_dd = abs(float(drawdown.min()) * 100)

        return {
            "strategy": "Buy & Hold",
            "symbol": symbol,
            "regime": "Full",
            "total_return": round(total_return, 2),
            "cagr": round(cagr, 2),
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "calmar_ratio": round(cagr / max_dd, 2) if max_dd > 0 else None,
            "max_drawdown": round(max_dd, 2),
            "win_rate": None,
            "profit_factor": None,
            "total_trades": 1,
            "winning_trades": 1 if total_return > 0 else 0,
            "losing_trades": 0 if total_return > 0 else 1,
            "volatility": None,
        }
    except Exception as e:
        logger.warning(f"Buy & Hold failed for {symbol}: {e}")
        return None


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """메인 실행 함수."""
    logger.info("=" * 70)
    logger.info("Tier 2 Multi-Coin Strategy Comparison")
    logger.info(f"Strategies: {len(STRATEGIES)}, Coins: {len(SYMBOLS)}")
    logger.info(f"Period: {FULL_START.date()} ~ {FULL_END.date()}")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    service = MarketDataService()
    engine = BacktestEngine()

    all_single: list[dict[str, Any]] = []
    all_multi: list[dict[str, Any]] = []
    all_regime: list[dict[str, Any]] = []

    total_start = time.time()

    # =========================================================================
    # Step 1: 개별 코인 백테스트 (6 strategies x 5 coins = 30)
    # =========================================================================
    logger.info("\n[Step 1/4] Individual Coin Backtests (30 runs)")
    for i, strategy_name in enumerate(STRATEGIES, 1):
        for j, symbol in enumerate(SYMBOLS, 1):
            idx = (i - 1) * len(SYMBOLS) + j
            logger.info(f"  [{idx}/30] {strategy_name} / {symbol}")
            row = safe_backtest_single(engine, service, strategy_name, symbol, FULL_START, FULL_END)
            if row:
                all_single.append(row)

    # =========================================================================
    # Step 2: 5코인 EW 포트폴리오 (6 strategies)
    # =========================================================================
    logger.info("\n[Step 2/4] Multi-Asset EW Portfolios (6 runs)")
    for i, strategy_name in enumerate(STRATEGIES, 1):
        logger.info(f"  [{i}/6] {strategy_name} / EW Portfolio")
        row = safe_backtest_multi(engine, service, strategy_name, SYMBOLS, FULL_START, FULL_END)
        if row:
            all_multi.append(row)

    # =========================================================================
    # Step 3: Buy & Hold 벤치마크 (5 coins)
    # =========================================================================
    logger.info("\n[Step 3/4] Buy & Hold Benchmarks (5 runs)")
    for i, symbol in enumerate(SYMBOLS, 1):
        logger.info(f"  [{i}/5] Buy & Hold / {symbol}")
        row = compute_buy_hold(service, symbol, FULL_START, FULL_END)
        if row:
            all_single.append(row)

    # =========================================================================
    # Step 4: 레짐별 분석 (3 regimes x 6 strategies x 5 coins = 90)
    # =========================================================================
    logger.info("\n[Step 4/4] Regime Analysis (90 runs)")
    run_idx = 0
    for regime_name, (regime_start, regime_end) in REGIMES.items():
        for strategy_name in STRATEGIES:
            for symbol in SYMBOLS:
                run_idx += 1
                logger.info(f"  [{run_idx}/90] {strategy_name} / {symbol} / {regime_name}")
                row = safe_backtest_single(
                    engine,
                    service,
                    strategy_name,
                    symbol,
                    regime_start,
                    regime_end,
                    regime_name,
                )
                if row:
                    all_regime.append(row)

    # =========================================================================
    # Save results
    # =========================================================================
    elapsed = time.time() - total_start

    single_df = pd.DataFrame(all_single)
    multi_df = pd.DataFrame(all_multi)
    regime_df = pd.DataFrame(all_regime)

    single_path = RESULTS_DIR / "tier2_single_coin_metrics.csv"
    multi_path = RESULTS_DIR / "tier2_multi_asset_metrics.csv"
    regime_path = RESULTS_DIR / "tier2_regime_analysis.csv"

    single_df.to_csv(single_path, index=False)
    multi_df.to_csv(multi_path, index=False)
    regime_df.to_csv(regime_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("Results Saved:")
    logger.info(f"  {single_path} ({len(single_df)} rows)")
    logger.info(f"  {multi_path} ({len(multi_df)} rows)")
    logger.info(f"  {regime_path} ({len(regime_df)} rows)")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 70)

    # =========================================================================
    # Console Summary
    # =========================================================================
    _print_summary(single_df, multi_df, regime_df)


def _print_summary(
    single_df: pd.DataFrame,
    multi_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> None:
    """콘솔 요약 출력."""
    print("\n" + "=" * 80)
    print("TIER 2 STRATEGY COMPARISON SUMMARY")
    print("=" * 80)

    # Single coin summary (strategy averages)
    strategy_rows = single_df[single_df["strategy"] != "Buy & Hold"]
    if not strategy_rows.empty:
        print("\n--- Individual Coin Averages (across 5 coins) ---")
        print(
            f"{'Strategy':<20} {'Sharpe':>8} {'CAGR%':>8} {'MDD%':>8} {'Trades':>8} {'WinRate%':>8}"
        )
        print("-" * 60)

        for name in STRATEGIES:
            rows = strategy_rows[strategy_rows["strategy"] == name]
            if rows.empty:
                continue
            sharpe = rows["sharpe_ratio"].mean()
            cagr = rows["cagr"].mean()
            mdd = rows["max_drawdown"].mean()
            trades = rows["total_trades"].mean()
            wr = rows["win_rate"].mean()
            print(f"{name:<20} {sharpe:>8.2f} {cagr:>8.1f} {mdd:>8.1f} {trades:>8.0f} {wr:>8.1f}")

    # Buy & Hold
    bh_rows = single_df[single_df["strategy"] == "Buy & Hold"]
    if not bh_rows.empty:
        print("\n--- Buy & Hold Benchmarks ---")
        print(f"{'Symbol':<20} {'Return%':>10} {'CAGR%':>10} {'MDD%':>10}")
        print("-" * 50)
        for _, row in bh_rows.iterrows():
            print(
                f"{row['symbol']:<20} {row['total_return']:>10.1f} {row['cagr']:>10.1f} {row['max_drawdown']:>10.1f}"
            )

    # Multi-asset
    if not multi_df.empty:
        print("\n--- Multi-Asset EW Portfolio ---")
        print(f"{'Strategy':<20} {'Sharpe':>8} {'CAGR%':>8} {'MDD%':>8} {'Trades':>8}")
        print("-" * 50)
        for _, row in multi_df.iterrows():
            sharpe = row.get("sharpe_ratio", 0) or 0
            cagr = row.get("cagr", 0) or 0
            mdd = row.get("max_drawdown", 0) or 0
            trades = row.get("total_trades", 0) or 0
            print(f"{row['strategy']:<20} {sharpe:>8.2f} {cagr:>8.1f} {mdd:>8.1f} {trades:>8.0f}")

    # Regime summary
    if not regime_df.empty:
        print("\n--- Regime Analysis (avg across coins) ---")
        print(f"{'Strategy':<20} {'Bear Sharpe':>12} {'Recv Sharpe':>12} {'Bull Sharpe':>12}")
        print("-" * 56)
        for name in STRATEGIES:
            vals: list[str] = []
            for regime in ["Bear", "Recovery", "Bull"]:
                rows = regime_df[(regime_df["strategy"] == name) & (regime_df["regime"] == regime)]
                if rows.empty:
                    vals.append(f"{'N/A':>12}")
                else:
                    avg = rows["sharpe_ratio"].mean()
                    vals.append(f"{avg:>12.2f}")
            print(f"{name:<20} {''.join(vals)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
