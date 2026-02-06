"""Multi-Timeframe TSMOM Analysis.

8-asset EW TSMOM 포트폴리오에서 타임프레임(4h, 8h, 12h, 1D)별
최적 lookback을 찾고, 어떤 타임프레임이 최고 성과를 내는지 검증.

Usage:
    uv run python scripts/multi_timeframe_sweep.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from loguru import logger

# 로거: WARNING만 표시
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
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "DOGE/USDT", "LINK/USDT", "ADA/USDT", "AVAX/USDT",
]

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2025, 12, 31, tzinfo=UTC)

# Timeframe definitions: (name, annualization_factor, candles_per_day)
TIMEFRAMES: list[tuple[str, float, int]] = [
    ("4h", 2190.0, 6),
    ("8h", 1095.0, 3),
    ("12h", 730.0, 2),
    ("1D", 365.0, 1),
]

# Lookback grid: time horizons in days
TIME_HORIZONS_DAYS = [7, 14, 21, 30, 45, 60]

# Fixed parameters (confirmed optimal)
VOL_TARGET = 0.35
MAX_LEVERAGE_CAP = 2.0
INITIAL_CAPITAL = 10000
ANNUALIZATION_FOR_METRICS = 365  # portfolio metrics always daily


def candles_for_horizon(horizon_days: int, candles_per_day: int) -> int:
    """시간 수평선(일)을 캔들 수로 변환."""
    return horizon_days * candles_per_day


# =============================================================================
# Helpers
# =============================================================================


def compute_portfolio_metrics(daily_returns: pd.Series) -> dict[str, float]:  # type: ignore[type-arg]
    """EW 포트폴리오 일간 수익률에서 성과 지표를 계산."""
    returns = daily_returns.dropna()
    if len(returns) < 30:
        return dict.fromkeys(["sharpe", "cagr", "total_return", "mdd", "ann_vol", "calmar", "sortino"], np.nan)

    cum_returns = (1 + returns).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1)
    n_days = max((returns.index[-1] - returns.index[0]).days, len(returns))
    cagr = float((1 + total_return) ** (365.0 / n_days) - 1)
    ann_vol = float(returns.std() * np.sqrt(ANNUALIZATION_FOR_METRICS))
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    mdd = float(drawdown.min())
    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = float(downside.std() * np.sqrt(ANNUALIZATION_FOR_METRICS)) if len(downside) > 0 else 0.0
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


def resample_returns_to_daily(returns: pd.Series, tf_name: str) -> pd.Series:  # type: ignore[type-arg]
    """서브데일리 수익률을 일간으로 합성 (compound)."""
    if tf_name == "1D":
        return returns
    # Compound sub-daily returns to daily
    daily = (1 + returns).resample("1D").prod() - 1
    return daily.dropna()


def run_single_backtest(
    engine: BacktestEngine,
    data: MarketDataSet,
    lookback: int,
    annualization_factor: float,
) -> pd.Series | None:  # type: ignore[type-arg]
    """단일 에셋 TSMOM 백테스트 → 수익률 시리즈 반환."""
    try:
        strategy = TSMOMStrategy(
            TSMOMConfig(
                lookback=lookback,
                vol_window=lookback,
                vol_target=VOL_TARGET,
                annualization_factor=annualization_factor,
                short_mode=ShortMode.HEDGE_ONLY,
                hedge_threshold=-0.07,
                hedge_strength_ratio=0.8,
            )
        )
        portfolio = Portfolio.create(
            initial_capital=INITIAL_CAPITAL,
            config=PortfolioManagerConfig(max_leverage_cap=MAX_LEVERAGE_CAP),
        )
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        _, strat_returns, _ = engine.run_with_returns(request)
        return strat_returns
    except Exception as e:
        print(f"      ERROR ({data.symbol}): {e}")
        return None


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    total_combos = len(TIMEFRAMES) * len(TIME_HORIZONS_DAYS)
    total_backtests = total_combos * len(ASSETS)

    print("=" * 90)
    print("Multi-Timeframe TSMOM Analysis (8-asset EW Portfolio)")
    print(f"Timeframes: {[tf[0] for tf in TIMEFRAMES]}")
    print(f"Time Horizons: {TIME_HORIZONS_DAYS} days")
    print(f"Assets: {len(ASSETS)} | Period: {START.year}-{END.year}")
    print(f"Fixed: vol_target={VOL_TARGET}, leverage_cap={MAX_LEVERAGE_CAP}x")
    print(f"Grid: {len(TIMEFRAMES)} TF × {len(TIME_HORIZONS_DAYS)} lookbacks = {total_combos} combos")
    print(f"Total backtests: {total_backtests}")
    print("=" * 90)

    service = MarketDataService()
    engine = BacktestEngine()

    # =========================================================================
    # 1. Load data for all timeframes
    # =========================================================================
    print("\n[1/4] Loading market data for all timeframes...")
    data_cache: dict[tuple[str, str], MarketDataSet] = {}

    for tf_name, _, _ in TIMEFRAMES:
        print(f"\n  --- {tf_name} ---")
        for symbol in ASSETS:
            print(f"    {symbol}...", end=" ", flush=True)
            try:
                data = service.get(MarketDataRequest(
                    symbol=symbol, timeframe=tf_name, start=START, end=END,
                ))
                data_cache[(tf_name, symbol)] = data
                print(f"{data.periods:,} candles")
            except Exception as e:
                print(f"FAILED: {e}")

    # =========================================================================
    # 2. Run backtests
    # =========================================================================
    print(f"\n[2/4] Running {total_backtests} backtests...")
    results: list[dict[str, float | str | int]] = []
    asset_level_results: list[dict[str, float | str | int]] = []
    combo_idx = 0

    for tf_name, ann_factor, cpd in TIMEFRAMES:
        for horizon_days in TIME_HORIZONS_DAYS:
            lookback = candles_for_horizon(horizon_days, cpd)
            combo_idx += 1
            print(f"  [{combo_idx:2d}/{total_combos}] {tf_name} lookback={lookback} ({horizon_days}d)...", end=" ", flush=True)

            all_returns: list[pd.Series] = []  # type: ignore[type-arg]
            per_asset: dict[str, dict[str, float]] = {}

            for symbol in ASSETS:
                key = (tf_name, symbol)
                if key not in data_cache:
                    continue

                ret = run_single_backtest(engine, data_cache[key], lookback, ann_factor)
                if ret is not None:
                    # Resample to daily for portfolio combination
                    daily_ret = resample_returns_to_daily(ret, tf_name)
                    all_returns.append(daily_ret)

                    # Per-asset metrics
                    asset_m = compute_portfolio_metrics(daily_ret)
                    per_asset[symbol] = asset_m

                    # Store asset-level result
                    asset_level_results.append({
                        "timeframe": tf_name,
                        "horizon_days": horizon_days,
                        "lookback": lookback,
                        "symbol": symbol,
                        **asset_m,
                    })

            if len(all_returns) < len(ASSETS):
                print(f"SKIP (only {len(all_returns)}/{len(ASSETS)} assets)")
                continue

            # EW portfolio combination
            returns_df = pd.DataFrame({f"a{i}": r for i, r in enumerate(all_returns)})
            returns_df = returns_df.dropna()
            ew_returns: pd.Series = returns_df.mean(axis=1)  # type: ignore[assignment]

            metrics = compute_portfolio_metrics(ew_returns)
            avg_asset_sharpe = np.mean([m["sharpe"] for m in per_asset.values()])

            results.append({
                "timeframe": tf_name,
                "horizon_days": horizon_days,
                "lookback": lookback,
                **metrics,
                "avg_asset_sharpe": round(float(avg_asset_sharpe), 2),
                "n_days_data": len(returns_df),
            })

            print(f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%")

    # =========================================================================
    # 3. Analysis
    # =========================================================================
    print("\n[3/4] Results Analysis")
    print("=" * 90)

    df = pd.DataFrame(results)
    asset_df = pd.DataFrame(asset_level_results)

    # --- 3.1: Sharpe Heatmap (TF × Horizon) ---
    print("\n### 3.1 Sharpe Ratio Heatmap (Timeframe × Time Horizon)")
    sharpe_pivot = df.pivot_table(values="sharpe", index="timeframe", columns="horizon_days", aggfunc="first")
    sharpe_pivot = sharpe_pivot.reindex(index=[tf[0] for tf in TIMEFRAMES])
    print(sharpe_pivot.to_string(float_format="{:.2f}".format))

    # --- 3.2: CAGR Heatmap ---
    print("\n### 3.2 CAGR % Heatmap (Timeframe × Time Horizon)")
    cagr_pivot = df.pivot_table(values="cagr", index="timeframe", columns="horizon_days", aggfunc="first")
    cagr_pivot = cagr_pivot.reindex(index=[tf[0] for tf in TIMEFRAMES])
    print(cagr_pivot.to_string(float_format="{:.1f}".format))

    # --- 3.3: MDD Heatmap ---
    print("\n### 3.3 MDD % Heatmap (Timeframe × Time Horizon)")
    mdd_pivot = df.pivot_table(values="mdd", index="timeframe", columns="horizon_days", aggfunc="first")
    mdd_pivot = mdd_pivot.reindex(index=[tf[0] for tf in TIMEFRAMES])
    print(mdd_pivot.to_string(float_format="{:.1f}".format))

    # --- 3.4: Calmar Heatmap ---
    print("\n### 3.4 Calmar Ratio Heatmap (Timeframe × Time Horizon)")
    calmar_pivot = df.pivot_table(values="calmar", index="timeframe", columns="horizon_days", aggfunc="first")
    calmar_pivot = calmar_pivot.reindex(index=[tf[0] for tf in TIMEFRAMES])
    print(calmar_pivot.to_string(float_format="{:.2f}".format))

    # --- 3.5: Sortino Heatmap ---
    print("\n### 3.5 Sortino Ratio Heatmap (Timeframe × Time Horizon)")
    sortino_pivot = df.pivot_table(values="sortino", index="timeframe", columns="horizon_days", aggfunc="first")
    sortino_pivot = sortino_pivot.reindex(index=[tf[0] for tf in TIMEFRAMES])
    print(sortino_pivot.to_string(float_format="{:.2f}".format))

    # --- 3.6: AnnVol Heatmap ---
    print("\n### 3.6 AnnVol % Heatmap (Timeframe × Time Horizon)")
    vol_pivot = df.pivot_table(values="ann_vol", index="timeframe", columns="horizon_days", aggfunc="first")
    vol_pivot = vol_pivot.reindex(index=[tf[0] for tf in TIMEFRAMES])
    print(vol_pivot.to_string(float_format="{:.1f}".format))

    # --- 3.7: Best lookback per timeframe ---
    print("\n### 3.7 Best Lookback per Timeframe (by Sharpe)")
    print("-" * 100)
    header = f"{'TF':>4} {'Horizon':>8} {'Lookback':>8} {'Sharpe':>7} {'CAGR%':>7} {'MDD%':>7} {'AnnVol%':>8} {'Calmar':>7} {'Sortino':>8} {'Return%':>8}"
    print(header)
    print("-" * 100)

    best_per_tf: dict[str, pd.Series] = {}  # type: ignore[type-arg]
    for tf_name, _, _ in TIMEFRAMES:
        tf_df = df[df["timeframe"] == tf_name]
        if tf_df.empty:
            continue
        best_idx = tf_df["sharpe"].idxmax()
        best = tf_df.loc[best_idx]
        best_per_tf[tf_name] = best
        print(
            f"{best['timeframe']:>4} {int(best['horizon_days']):>5}d {int(best['lookback']):>8} "
            f"{best['sharpe']:>7.2f} {best['cagr']:>7.1f} {best['mdd']:>7.1f} "
            f"{best['ann_vol']:>8.1f} {best['calmar']:>7.2f} {best['sortino']:>8.2f} "
            f"{best['total_return']:>8.1f}"
        )

    # --- 3.8: Cross-Timeframe Comparison (best of each) ---
    print("\n### 3.8 Cross-Timeframe Comparison (Best Settings)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    for tf_name in [tf[0] for tf in TIMEFRAMES]:
        if tf_name not in best_per_tf:
            continue
        b = best_per_tf[tf_name]
        marker = " ★" if b["sharpe"] == max(bb["sharpe"] for bb in best_per_tf.values()) else ""
        print(
            f"{b['timeframe']:>4} {int(b['horizon_days']):>5}d {int(b['lookback']):>8} "
            f"{b['sharpe']:>7.2f} {b['cagr']:>7.1f} {b['mdd']:>7.1f} "
            f"{b['ann_vol']:>8.1f} {b['calmar']:>7.2f} {b['sortino']:>8.2f} "
            f"{b['total_return']:>8.1f}{marker}"
        )

    # --- 3.9: Global Top 10 ---
    print("\n### 3.9 Global Top 10 (by Sharpe)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    df_sorted = df.sort_values("sharpe", ascending=False)
    for _, row in df_sorted.head(10).iterrows():
        print(
            f"{row['timeframe']:>4} {int(row['horizon_days']):>5}d {int(row['lookback']):>8} "
            f"{row['sharpe']:>7.2f} {row['cagr']:>7.1f} {row['mdd']:>7.1f} "
            f"{row['ann_vol']:>8.1f} {row['calmar']:>7.2f} {row['sortino']:>8.2f} "
            f"{row['total_return']:>8.1f}"
        )

    # --- 3.10: Global Top 10 by Calmar ---
    print("\n### 3.10 Global Top 10 (by Calmar)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    df_calmar = df.sort_values("calmar", ascending=False)
    for _, row in df_calmar.head(10).iterrows():
        print(
            f"{row['timeframe']:>4} {int(row['horizon_days']):>5}d {int(row['lookback']):>8} "
            f"{row['sharpe']:>7.2f} {row['cagr']:>7.1f} {row['mdd']:>7.1f} "
            f"{row['ann_vol']:>8.1f} {row['calmar']:>7.2f} {row['sortino']:>8.2f} "
            f"{row['total_return']:>8.1f}"
        )

    # --- 3.11: Per-Asset Analysis (best TF per asset) ---
    print("\n### 3.11 Per-Asset Best Timeframe (by Sharpe)")
    print("-" * 80)
    print(f"{'Asset':>12} {'Best TF':>8} {'Horizon':>8} {'Sharpe':>7} {'CAGR%':>7} {'MDD%':>7}")
    print("-" * 80)
    if not asset_df.empty:
        for symbol in ASSETS:
            sym_df = asset_df[asset_df["symbol"] == symbol]
            if sym_df.empty:
                continue
            best_idx = sym_df["sharpe"].idxmax()
            b = sym_df.loc[best_idx]
            print(f"{symbol:>12} {b['timeframe']:>8} {int(b['horizon_days']):>5}d {b['sharpe']:>7.2f} {b['cagr']:>7.1f} {b['mdd']:>7.1f}")

    # --- 3.12: Same Time Horizon Across Timeframes ---
    print("\n### 3.12 Same Time Horizon (30d) Across Timeframes")
    print("-" * 100)
    print(header)
    print("-" * 100)
    horizon_30 = df[df["horizon_days"] == 30]
    for _, row in horizon_30.iterrows():
        print(
            f"{row['timeframe']:>4} {int(row['horizon_days']):>5}d {int(row['lookback']):>8} "
            f"{row['sharpe']:>7.2f} {row['cagr']:>7.1f} {row['mdd']:>7.1f} "
            f"{row['ann_vol']:>8.1f} {row['calmar']:>7.2f} {row['sortino']:>8.2f} "
            f"{row['total_return']:>8.1f}"
        )

    # --- 3.13: Baseline Sanity Check ---
    print("\n### 3.13 Baseline Sanity Check (1D, lookback=30)")
    baseline = df[(df["timeframe"] == "1D") & (df["lookback"] == 30)]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(f"  Sharpe={b['sharpe']:.2f}, CAGR={b['cagr']:.1f}%, MDD={b['mdd']:.1f}%")
        print("  Expected: Sharpe≈2.06, CAGR≈48.8%, MDD≈-23.5% (from vol_target sweep)")

    # =========================================================================
    # 4. Save results
    # =========================================================================
    print("\n[4/4] Saving results...")
    csv_path = "data/multi_timeframe_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Portfolio results: {csv_path}")

    asset_csv_path = "data/multi_timeframe_sweep_assets.csv"
    asset_df.to_csv(asset_csv_path, index=False)
    print(f"  Asset-level results: {asset_csv_path}")

    # Final summary
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    overall_best = df_sorted.iloc[0]
    print(f"\n  Overall Best: {overall_best['timeframe']} lookback={int(overall_best['lookback'])} ({int(overall_best['horizon_days'])}d)")
    print(f"  Sharpe={overall_best['sharpe']:.2f}, CAGR={overall_best['cagr']:.1f}%, MDD={overall_best['mdd']:.1f}%, Calmar={overall_best['calmar']:.2f}, Sortino={overall_best['sortino']:.2f}")

    # Compare vs 1D baseline
    if not baseline.empty:
        b = baseline.iloc[0]
        delta_sharpe = overall_best["sharpe"] - b["sharpe"]
        delta_cagr = overall_best["cagr"] - b["cagr"]
        print(f"\n  vs 1D/30d Baseline: Sharpe {delta_sharpe:+.2f}, CAGR {delta_cagr:+.1f}%")


if __name__ == "__main__":
    main()
