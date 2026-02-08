"""Multi-Asset TSMOM vol_target × max_leverage_cap Parameter Sweep.

8-asset Equal-Weight TSMOM 포트폴리오에서
vol_target과 max_leverage_cap 조합별 성과를 비교합니다.

Usage:
    uv run python scripts/multi_asset_vol_leverage_sweep.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime
from itertools import product

import numpy as np
import pandas as pd
from loguru import logger

# 로거 설정: 핵심 진행 상황만 표시
logger.remove()
logger.add(sys.stderr, level="WARNING")

warnings.filterwarnings("ignore", category=FutureWarning)

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.data.market_data import MarketDataRequest, MarketDataSet
from src.data.service import MarketDataService
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.portfolio import Portfolio
from src.strategy.tsmom import TSMOMStrategy
from src.strategy.tsmom.config import ShortMode, TSMOMConfig

# =============================================================================
# Configuration
# =============================================================================

ASSETS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]

# Common period (all assets have data from 2020)
START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)

# Parameter grid
VOL_TARGETS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
LEVERAGE_CAPS = [1.0, 1.5, 2.0, 2.5, 3.0]

# Fixed TSMOM parameters (validated optimal from docs/strategy-evaluation/)
FIXED_TSMOM_PARAMS = {
    "lookback": 30,
    "vol_window": 30,
    "short_mode": ShortMode.HEDGE_ONLY,
    "hedge_threshold": -0.07,
    "hedge_strength_ratio": 0.8,
}

INITIAL_CAPITAL = 10000
ANNUALIZATION = 365  # daily data


# =============================================================================
# Helper Functions
# =============================================================================


def compute_portfolio_metrics(
    daily_returns: pd.Series,  # type: ignore[type-arg]
) -> dict[str, float]:
    """EW 포트폴리오 일간 수익률에서 핵심 성과 지표를 계산합니다."""
    returns = daily_returns.dropna()
    if len(returns) < 30:
        return {
            "sharpe": np.nan,
            "cagr": np.nan,
            "total_return": np.nan,
            "mdd": np.nan,
            "ann_vol": np.nan,
            "calmar": np.nan,
            "sortino": np.nan,
        }

    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1)

    # CAGR
    n_days = (returns.index[-1] - returns.index[0]).days
    if n_days <= 0:
        n_days = len(returns)
    cagr = float((1 + total_return) ** (365.0 / n_days) - 1)

    # Annualized volatility
    ann_vol = float(returns.std() * np.sqrt(ANNUALIZATION))

    # Sharpe (risk-free = 0 for crypto)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    # Max Drawdown
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    mdd = float(drawdown.min())

    # Calmar
    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0.0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(ANNUALIZATION)) if len(downside) > 0 else 0.0
    sortino = cagr / downside_std if downside_std > 0 else 0.0

    return {
        "sharpe": round(sharpe, 2),
        "cagr": round(cagr * 100, 1),
        "total_return": round(total_return * 100, 1),
        "mdd": round(mdd * 100, 1),
        "ann_vol": round(ann_vol * 100, 1),
        "calmar": round(calmar, 2),
        "sortino": round(sortino, 2),
    }


def load_all_data(
    service: MarketDataService,
) -> dict[str, MarketDataSet]:
    """8개 에셋의 일봉 데이터를 로드합니다."""
    data_map: dict[str, MarketDataSet] = {}
    for symbol in ASSETS:
        print(f"  Loading {symbol}...", end=" ", flush=True)
        try:
            data = service.get(
                MarketDataRequest(
                    symbol=symbol,
                    timeframe="1D",
                    start=START,
                    end=END,
                )
            )
            data_map[symbol] = data
            print(f"{data.periods} candles")
        except Exception as e:
            print(f"FAILED: {e}")
    return data_map


def run_single_backtest(
    engine: BacktestEngine,
    data: MarketDataSet,
    vol_target: float,
    max_leverage_cap: float,
) -> pd.Series | None:  # type: ignore[type-arg]
    """단일 에셋 TSMOM 백테스트 실행, 일간 수익률 반환."""
    try:
        strategy = TSMOMStrategy(
            TSMOMConfig(
                vol_target=vol_target,
                **FIXED_TSMOM_PARAMS,
            )
        )
        portfolio = Portfolio.create(
            initial_capital=INITIAL_CAPITAL,
            config=PortfolioManagerConfig(
                max_leverage_cap=max_leverage_cap,
                execution_mode="orders",
            ),
        )
        request = BacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
        )
        _, strat_returns, _ = engine.run_with_returns(request)
        return strat_returns
    except Exception as e:
        print(f"    ERROR ({data.symbol}): {e}")
        return None


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 80)
    print("Multi-Asset TSMOM: vol_target x max_leverage_cap Parameter Sweep")
    print(f"Assets: {len(ASSETS)} | Period: {START.year}-{END.year}")
    print(
        f"Grid: {len(VOL_TARGETS)} vol_targets x {len(LEVERAGE_CAPS)} leverage_caps = {len(VOL_TARGETS) * len(LEVERAGE_CAPS)} combos"
    )
    print(f"Total backtests: {len(VOL_TARGETS) * len(LEVERAGE_CAPS) * len(ASSETS)}")
    print("=" * 80)

    # 1. Load data
    print("\n[1/3] Loading market data...")
    service = MarketDataService()
    data_map = load_all_data(service)

    if len(data_map) < len(ASSETS):
        missing = set(ASSETS) - set(data_map.keys())
        print(f"\nWARNING: Missing data for: {missing}")
        print("Proceeding with available assets...")

    n_assets = len(data_map)

    # 2. Run parameter sweep
    print(
        f"\n[2/3] Running parameter sweep ({len(VOL_TARGETS) * len(LEVERAGE_CAPS)} combinations)..."
    )
    engine = BacktestEngine()
    results: list[dict[str, float | str]] = []
    total_combos = len(VOL_TARGETS) * len(LEVERAGE_CAPS)

    for idx, (vt, lc) in enumerate(product(VOL_TARGETS, LEVERAGE_CAPS), 1):
        print(
            f"  [{idx:3d}/{total_combos}] vol_target={vt:.2f}, leverage_cap={lc:.1f}x ...",
            end=" ",
            flush=True,
        )

        # Run backtest for each asset
        all_returns: list[pd.Series] = []  # type: ignore[type-arg]
        asset_metrics: dict[str, dict[str, float]] = {}

        for symbol, data in data_map.items():
            ret = run_single_backtest(engine, data, vt, lc)
            if ret is not None:
                all_returns.append(ret)

                # Individual asset metrics too
                asset_m = compute_portfolio_metrics(ret)
                asset_metrics[symbol] = asset_m

        if len(all_returns) < n_assets:
            print(f"SKIP (only {len(all_returns)}/{n_assets} assets)")
            continue

        # Align all return series to common dates and compute EW portfolio
        returns_df = pd.DataFrame({f"asset_{i}": r for i, r in enumerate(all_returns)})
        returns_df = returns_df.dropna()
        ew_returns: pd.Series = returns_df.mean(axis=1)  # type: ignore[assignment]

        # Compute portfolio metrics
        metrics = compute_portfolio_metrics(ew_returns)

        # Compute per-asset average Sharpe for reference
        avg_asset_sharpe = np.mean([m["sharpe"] for m in asset_metrics.values()])

        row: dict[str, float | str] = {
            "vol_target": vt,
            "leverage_cap": lc,
            **metrics,
            "avg_asset_sharpe": round(float(avg_asset_sharpe), 2),
        }
        results.append(row)

        print(
            f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%"
        )

    # 3. Results
    print("\n[3/3] Results Summary")
    print("=" * 80)

    df = pd.DataFrame(results)

    # Sort by Sharpe descending
    df = df.sort_values("sharpe", ascending=False)

    # Pivot table: vol_target (rows) x leverage_cap (cols) for Sharpe
    print("\n### Sharpe Ratio (vol_target x leverage_cap)")
    sharpe_pivot = df.pivot_table(
        values="sharpe",
        index="vol_target",
        columns="leverage_cap",
        aggfunc="first",
    )
    print(sharpe_pivot.to_string(float_format="{:.2f}".format))

    print("\n### CAGR % (vol_target x leverage_cap)")
    cagr_pivot = df.pivot_table(
        values="cagr",
        index="vol_target",
        columns="leverage_cap",
        aggfunc="first",
    )
    print(cagr_pivot.to_string(float_format="{:.1f}".format))

    print("\n### MDD % (vol_target x leverage_cap)")
    mdd_pivot = df.pivot_table(
        values="mdd",
        index="vol_target",
        columns="leverage_cap",
        aggfunc="first",
    )
    print(mdd_pivot.to_string(float_format="{:.1f}".format))

    print("\n### AnnVol % (vol_target x leverage_cap)")
    vol_pivot = df.pivot_table(
        values="ann_vol",
        index="vol_target",
        columns="leverage_cap",
        aggfunc="first",
    )
    print(vol_pivot.to_string(float_format="{:.1f}".format))

    print("\n### Calmar Ratio (vol_target x leverage_cap)")
    calmar_pivot = df.pivot_table(
        values="calmar",
        index="vol_target",
        columns="leverage_cap",
        aggfunc="first",
    )
    print(calmar_pivot.to_string(float_format="{:.2f}".format))

    print("\n### Sortino Ratio (vol_target x leverage_cap)")
    sortino_pivot = df.pivot_table(
        values="sortino",
        index="vol_target",
        columns="leverage_cap",
        aggfunc="first",
    )
    print(sortino_pivot.to_string(float_format="{:.2f}".format))

    # Top 10 by Sharpe
    print("\n### Top 10 Combinations (by Sharpe)")
    print("-" * 100)
    header = f"{'vol_target':>10} {'lev_cap':>8} {'Sharpe':>7} {'CAGR%':>7} {'MDD%':>7} {'AnnVol%':>8} {'Calmar':>7} {'Sortino':>8} {'Return%':>8}"
    print(header)
    print("-" * 100)
    for _, row in df.head(10).iterrows():
        print(
            f"{row['vol_target']:>10.2f} {row['leverage_cap']:>8.1f} "
            f"{row['sharpe']:>7.2f} {row['cagr']:>7.1f} {row['mdd']:>7.1f} "
            f"{row['ann_vol']:>8.1f} {row['calmar']:>7.2f} {row['sortino']:>8.2f} "
            f"{row['total_return']:>8.1f}"
        )

    # Top 10 by Calmar
    df_calmar = df.sort_values("calmar", ascending=False)
    print("\n### Top 10 Combinations (by Calmar)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    for _, row in df_calmar.head(10).iterrows():
        print(
            f"{row['vol_target']:>10.2f} {row['leverage_cap']:>8.1f} "
            f"{row['sharpe']:>7.2f} {row['cagr']:>7.1f} {row['mdd']:>7.1f} "
            f"{row['ann_vol']:>8.1f} {row['calmar']:>7.2f} {row['sortino']:>8.2f} "
            f"{row['total_return']:>8.1f}"
        )

    # Baseline comparison
    print("\n### Baseline Comparison (vol_target=0.30, leverage_cap=2.0)")
    baseline = df[(df["vol_target"] == 0.30) & (df["leverage_cap"] == 2.0)]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(
            f"  Sharpe={b['sharpe']:.2f}, CAGR={b['cagr']:.1f}%, MDD={b['mdd']:.1f}%, AnnVol={b['ann_vol']:.1f}%, Calmar={b['calmar']:.2f}"
        )

    best = df.iloc[0]
    print("\n### Best Combination (by Sharpe)")
    print(f"  vol_target={best['vol_target']:.2f}, leverage_cap={best['leverage_cap']:.1f}x")
    print(
        f"  Sharpe={best['sharpe']:.2f}, CAGR={best['cagr']:.1f}%, MDD={best['mdd']:.1f}%, AnnVol={best['ann_vol']:.1f}%, Calmar={best['calmar']:.2f}"
    )

    # Save full results to CSV
    csv_path = "data/multi_asset_vol_leverage_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    main()
