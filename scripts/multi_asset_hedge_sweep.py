"""Multi-Asset TSMOM hedge_threshold × hedge_strength_ratio Parameter Sweep.

8-asset Equal-Weight TSMOM 포트폴리오에서
hedge_threshold과 hedge_strength_ratio 조합별 성과를 비교합니다.

베이스라인 비교:
- ShortMode.DISABLED (Long-Only)
- ShortMode.FULL (Full Long/Short)
- ShortMode.HEDGE_ONLY (파라미터 그리드)

Usage:
    uv run python scripts/multi_asset_hedge_sweep.py
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

# Parameter grid — hedge parameters
HEDGE_THRESHOLDS = [-0.05, -0.07, -0.10, -0.12, -0.15, -0.20, -0.25, -0.30]
HEDGE_STRENGTH_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Fixed TSMOM parameters (confirmed optimal)
FIXED_PARAMS = {
    "lookback": 30,
    "vol_window": 30,
    "vol_target": 0.35,
}

# Fixed portfolio parameters (confirmed optimal)
MAX_LEVERAGE_CAP = 2.0
INITIAL_CAPITAL = 10000
ANNUALIZATION = 365


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
    short_mode: ShortMode,
    hedge_threshold: float = -0.07,
    hedge_strength_ratio: float = 0.8,
) -> pd.Series | None:  # type: ignore[type-arg]
    """단일 에셋 TSMOM 백테스트 실행, 일간 수익률 반환."""
    try:
        strategy = TSMOMStrategy(
            TSMOMConfig(
                **FIXED_PARAMS,
                short_mode=short_mode,
                hedge_threshold=hedge_threshold,
                hedge_strength_ratio=hedge_strength_ratio,
            )
        )
        portfolio = Portfolio.create(
            initial_capital=INITIAL_CAPITAL,
            config=PortfolioManagerConfig(
                max_leverage_cap=MAX_LEVERAGE_CAP,
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


def run_multi_asset_backtest(
    engine: BacktestEngine,
    data_map: dict[str, MarketDataSet],
    short_mode: ShortMode,
    hedge_threshold: float = -0.07,
    hedge_strength_ratio: float = 0.8,
) -> dict[str, float] | None:
    """8-asset EW 포트폴리오 백테스트 실행, 성과 지표 반환."""
    n_assets = len(data_map)
    all_returns: list[pd.Series] = []  # type: ignore[type-arg]

    for symbol, data in data_map.items():
        ret = run_single_backtest(
            engine, data, short_mode, hedge_threshold, hedge_strength_ratio
        )
        if ret is not None:
            all_returns.append(ret)

    if len(all_returns) < n_assets:
        return None

    # Align and compute EW portfolio
    returns_df = pd.DataFrame({f"asset_{i}": r for i, r in enumerate(all_returns)})
    returns_df = returns_df.dropna()
    ew_returns: pd.Series = returns_df.mean(axis=1)  # type: ignore[assignment]

    return compute_portfolio_metrics(ew_returns)


def print_pivot_table(
    df: pd.DataFrame,
    metric: str,
    title: str,
    fmt: str = "{:.2f}",
) -> None:
    """Pivot table 출력."""
    print(f"\n### {title}")
    pivot = df.pivot_table(
        values=metric,
        index="hedge_threshold",
        columns="hedge_strength_ratio",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=fmt.format))


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    total_combos = len(HEDGE_THRESHOLDS) * len(HEDGE_STRENGTH_RATIOS)
    total_backtests = (total_combos + 2) * len(ASSETS)  # +2 for baselines

    print("=" * 80)
    print("Multi-Asset TSMOM: hedge_threshold x hedge_strength_ratio Parameter Sweep")
    print(f"Assets: {len(ASSETS)} | Period: {START.year}-{END.year}")
    print(f"Grid: {len(HEDGE_THRESHOLDS)} thresholds x {len(HEDGE_STRENGTH_RATIOS)} ratios = {total_combos} combos")
    print(f"Baselines: DISABLED (Long-Only) + FULL (Long/Short)")
    print(f"Total backtests: {total_backtests}")
    print(f"Fixed: vol_target=0.35, lookback=30, leverage_cap={MAX_LEVERAGE_CAP}x")
    print("=" * 80)

    # 1. Load data
    print("\n[1/4] Loading market data...")
    service = MarketDataService()
    data_map = load_all_data(service)

    if len(data_map) < len(ASSETS):
        missing = set(ASSETS) - set(data_map.keys())
        print(f"\nWARNING: Missing data for: {missing}")
        print("Proceeding with available assets...")

    engine = BacktestEngine()

    # 2. Run baselines
    print("\n[2/4] Running baseline comparisons...")

    print("  DISABLED (Long-Only)...", end=" ", flush=True)
    baseline_disabled = run_multi_asset_backtest(engine, data_map, ShortMode.DISABLED)
    if baseline_disabled:
        print(f"Sharpe={baseline_disabled['sharpe']:.2f}, CAGR={baseline_disabled['cagr']:.1f}%, MDD={baseline_disabled['mdd']:.1f}%")

    print("  FULL (Long/Short)...", end=" ", flush=True)
    baseline_full = run_multi_asset_backtest(engine, data_map, ShortMode.FULL)
    if baseline_full:
        print(f"Sharpe={baseline_full['sharpe']:.2f}, CAGR={baseline_full['cagr']:.1f}%, MDD={baseline_full['mdd']:.1f}%")

    # 3. Run hedge parameter sweep
    print(f"\n[3/4] Running hedge parameter sweep ({total_combos} combinations)...")
    results: list[dict[str, float | str]] = []

    for idx, (ht, hsr) in enumerate(product(HEDGE_THRESHOLDS, HEDGE_STRENGTH_RATIOS), 1):
        print(
            f"  [{idx:3d}/{total_combos}] threshold={ht:.2f}, strength={hsr:.1f} ...",
            end=" ",
            flush=True,
        )

        metrics = run_multi_asset_backtest(
            engine, data_map, ShortMode.HEDGE_ONLY, ht, hsr
        )
        if metrics is None:
            print("SKIP")
            continue

        row: dict[str, float | str] = {
            "hedge_threshold": ht,
            "hedge_strength_ratio": hsr,
            **metrics,
        }
        results.append(row)

        print(
            f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%"
        )

    # 4. Results
    print(f"\n[4/4] Results Summary")
    print("=" * 80)

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    # Baseline comparison
    print("\n### Baseline Comparison")
    print("-" * 80)
    if baseline_disabled:
        print(
            f"  DISABLED (Long-Only):  Sharpe={baseline_disabled['sharpe']:.2f}, "
            f"CAGR={baseline_disabled['cagr']:.1f}%, MDD={baseline_disabled['mdd']:.1f}%, "
            f"AnnVol={baseline_disabled['ann_vol']:.1f}%, Calmar={baseline_disabled['calmar']:.2f}, "
            f"Sortino={baseline_disabled['sortino']:.2f}"
        )
    if baseline_full:
        print(
            f"  FULL (Long/Short):     Sharpe={baseline_full['sharpe']:.2f}, "
            f"CAGR={baseline_full['cagr']:.1f}%, MDD={baseline_full['mdd']:.1f}%, "
            f"AnnVol={baseline_full['ann_vol']:.1f}%, Calmar={baseline_full['calmar']:.2f}, "
            f"Sortino={baseline_full['sortino']:.2f}"
        )

    # Pivot tables
    print_pivot_table(df, "sharpe", "Sharpe Ratio (threshold x strength_ratio)", "{:.2f}")
    print_pivot_table(df, "cagr", "CAGR % (threshold x strength_ratio)", "{:.1f}")
    print_pivot_table(df, "mdd", "MDD % (threshold x strength_ratio)", "{:.1f}")
    print_pivot_table(df, "ann_vol", "AnnVol % (threshold x strength_ratio)", "{:.1f}")
    print_pivot_table(df, "calmar", "Calmar Ratio (threshold x strength_ratio)", "{:.2f}")
    print_pivot_table(df, "sortino", "Sortino Ratio (threshold x strength_ratio)", "{:.2f}")

    # Top 10 by Sharpe
    header = (
        f"{'threshold':>10} {'strength':>9} {'Sharpe':>7} {'CAGR%':>7} "
        f"{'MDD%':>7} {'AnnVol%':>8} {'Calmar':>7} {'Sortino':>8} {'Return%':>8}"
    )
    print("\n### Top 10 Combinations (by Sharpe)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    for _, row in df.head(10).iterrows():
        print(
            f"{row['hedge_threshold']:>10.2f} {row['hedge_strength_ratio']:>9.1f} "
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
            f"{row['hedge_threshold']:>10.2f} {row['hedge_strength_ratio']:>9.1f} "
            f"{row['sharpe']:>7.2f} {row['cagr']:>7.1f} {row['mdd']:>7.1f} "
            f"{row['ann_vol']:>8.1f} {row['calmar']:>7.2f} {row['sortino']:>8.2f} "
            f"{row['total_return']:>8.1f}"
        )

    # Top 10 by Sortino
    df_sortino = df.sort_values("sortino", ascending=False)
    print("\n### Top 10 Combinations (by Sortino)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    for _, row in df_sortino.head(10).iterrows():
        print(
            f"{row['hedge_threshold']:>10.2f} {row['hedge_strength_ratio']:>9.1f} "
            f"{row['sharpe']:>7.2f} {row['cagr']:>7.1f} {row['mdd']:>7.1f} "
            f"{row['ann_vol']:>8.1f} {row['calmar']:>7.2f} {row['sortino']:>8.2f} "
            f"{row['total_return']:>8.1f}"
        )

    # Current default comparison
    print("\n### Current Default (threshold=-0.07, strength=0.8)")
    current = df[
        (df["hedge_threshold"] == -0.07) & (df["hedge_strength_ratio"] == 0.8)
    ]
    if not current.empty:
        c = current.iloc[0]
        print(
            f"  Sharpe={c['sharpe']:.2f}, CAGR={c['cagr']:.1f}%, MDD={c['mdd']:.1f}%, "
            f"AnnVol={c['ann_vol']:.1f}%, Calmar={c['calmar']:.2f}, Sortino={c['sortino']:.2f}"
        )

    best = df.iloc[0]
    print(f"\n### Best Combination (by Sharpe)")
    print(
        f"  threshold={best['hedge_threshold']:.2f}, strength_ratio={best['hedge_strength_ratio']:.1f}"
    )
    print(
        f"  Sharpe={best['sharpe']:.2f}, CAGR={best['cagr']:.1f}%, MDD={best['mdd']:.1f}%, "
        f"AnnVol={best['ann_vol']:.1f}%, Calmar={best['calmar']:.2f}, Sortino={best['sortino']:.2f}"
    )

    # Improvement over baselines
    print("\n### Improvement vs Baselines")
    if baseline_disabled:
        delta_s = best["sharpe"] - baseline_disabled["sharpe"]
        delta_m = best["mdd"] - baseline_disabled["mdd"]
        print(
            f"  vs DISABLED: Sharpe {delta_s:+.2f}, MDD {delta_m:+.1f}pp"
        )
    if baseline_full:
        delta_s = best["sharpe"] - baseline_full["sharpe"]
        delta_m = best["mdd"] - baseline_full["mdd"]
        print(
            f"  vs FULL:     Sharpe {delta_s:+.2f}, MDD {delta_m:+.1f}pp"
        )

    # Save full results to CSV
    csv_path = "data/multi_asset_hedge_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")

    # Save baselines for reference
    if baseline_disabled or baseline_full:
        baselines_data = []
        if baseline_disabled:
            baselines_data.append({"mode": "DISABLED", **baseline_disabled})
        if baseline_full:
            baselines_data.append({"mode": "FULL", **baseline_full})
        pd.DataFrame(baselines_data).to_csv(
            "data/multi_asset_hedge_baselines.csv", index=False
        )
        print("Baselines saved to: data/multi_asset_hedge_baselines.csv")


if __name__ == "__main__":
    main()
