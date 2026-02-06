"""Multi-Asset TSMOM Risk Parameter Sweep.

8-asset Equal-Weight TSMOM 포트폴리오에서
system_stop_loss, trailing_stop, rebalance_threshold 조합별 성과를 비교합니다.

Phase A: Stop-Loss 단독 스윕
Phase B: Trailing Stop 단독 스윕
Phase C: Stop-Loss × Trailing Stop 결합 스윕
Phase D: Rebalance Threshold 스윕

Usage:
    uv run python scripts/multi_asset_risk_sweep.py
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

# Fixed TSMOM parameters (confirmed optimal from previous sweeps)
FIXED_TSMOM = {
    "lookback": 30,
    "vol_window": 30,
    "vol_target": 0.35,
    "short_mode": ShortMode.HEDGE_ONLY,
    "hedge_threshold": -0.07,
    "hedge_strength_ratio": 0.3,
}

INITIAL_CAPITAL = 10000
MAX_LEVERAGE_CAP = 2.0
ANNUALIZATION = 365

# =============================================================================
# Parameter Grids
# =============================================================================

# Phase A: Stop-Loss sweep
STOP_LOSS_VALUES: list[float | None] = [None, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]

# Phase B: Trailing Stop sweep (ATR multiplier)
TRAILING_STOP_MULTIPLIERS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Phase C: Combined (stop_loss × trailing_stop)
COMBINED_STOP_LOSS: list[float | None] = [None, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
COMBINED_TRAILING: list[tuple[bool, float]] = [
    (False, 2.0),  # use_trailing_stop=False, multiplier is ignored but must be >= 0.5
    (True, 2.0),
    (True, 3.0),
    (True, 4.0),
    (True, 5.0),
]

# Phase D: Rebalance Threshold sweep
REBALANCE_VALUES = [0.02, 0.03, 0.05, 0.07, 0.10]


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

    cum_returns = (1 + returns).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1)

    n_days = (returns.index[-1] - returns.index[0]).days
    if n_days <= 0:
        n_days = len(returns)
    cagr = float((1 + total_return) ** (365.0 / n_days) - 1)

    ann_vol = float(returns.std() * np.sqrt(ANNUALIZATION))
    sharpe = cagr / ann_vol if ann_vol > 0 else 0.0

    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    mdd = float(drawdown.min())

    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0.0

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
    system_stop_loss: float | None = 0.10,
    use_trailing_stop: bool = False,
    trailing_stop_atr_multiplier: float = 2.0,
    rebalance_threshold: float = 0.05,
) -> pd.Series | None:  # type: ignore[type-arg]
    """단일 에셋 TSMOM 백테스트 실행, 일간 수익률 반환."""
    try:
        strategy = TSMOMStrategy(TSMOMConfig(**FIXED_TSMOM))
        portfolio = Portfolio.create(
            initial_capital=INITIAL_CAPITAL,
            config=PortfolioManagerConfig(
                max_leverage_cap=MAX_LEVERAGE_CAP,
                execution_mode="orders",
                system_stop_loss=system_stop_loss,
                use_trailing_stop=use_trailing_stop,
                trailing_stop_atr_multiplier=trailing_stop_atr_multiplier,
                rebalance_threshold=rebalance_threshold,
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
    system_stop_loss: float | None = 0.10,
    use_trailing_stop: bool = False,
    trailing_stop_atr_multiplier: float = 2.0,
    rebalance_threshold: float = 0.05,
) -> dict[str, float] | None:
    """8-asset EW 포트폴리오 백테스트 실행, 성과 지표 반환."""
    n_assets = len(data_map)
    all_returns: list[pd.Series] = []  # type: ignore[type-arg]

    for data in data_map.values():
        ret = run_single_backtest(
            engine,
            data,
            system_stop_loss=system_stop_loss,
            use_trailing_stop=use_trailing_stop,
            trailing_stop_atr_multiplier=trailing_stop_atr_multiplier,
            rebalance_threshold=rebalance_threshold,
        )
        if ret is not None:
            all_returns.append(ret)

    if len(all_returns) < n_assets:
        return None

    returns_df = pd.DataFrame({f"asset_{i}": r for i, r in enumerate(all_returns)})
    returns_df = returns_df.dropna()
    ew_returns: pd.Series = returns_df.mean(axis=1)  # type: ignore[assignment]

    return compute_portfolio_metrics(ew_returns)


def fmt_sl(val: float | None) -> str:
    """Stop-loss 값을 문자열로 포맷."""
    return "None" if val is None else f"{val:.0%}"


def fmt_ts(use: bool, mult: float) -> str:
    """Trailing stop 설정을 문자열로 포맷."""
    return f"{mult:.1f}x ATR" if use else "OFF"


def print_results_table(
    results: list[dict[str, object]],
    title: str,
    sort_by: str = "sharpe",
    top_n: int = 0,
) -> None:
    """결과 테이블을 출력합니다."""
    df = pd.DataFrame(results)
    df = df.sort_values(sort_by, ascending=False)
    if top_n > 0:
        df = df.head(top_n)

    print(f"\n### {title} (sorted by {sort_by})")
    print("-" * 110)
    # Determine column set from first row
    cols = [
        c
        for c in df.columns
        if c not in ("sharpe", "cagr", "mdd", "ann_vol", "calmar", "sortino", "total_return")
    ]
    header = "".join(f"{c:>14}" for c in cols)
    header += f"{'Sharpe':>8}{'CAGR%':>8}{'MDD%':>8}{'AnnVol%':>9}{'Calmar':>8}{'Sortino':>9}{'Return%':>9}"
    print(header)
    print("-" * 110)

    for _, row in df.iterrows():
        line = "".join(f"{row[c]!s:>14}" for c in cols)
        line += (
            f"{row['sharpe']:>8.2f}{row['cagr']:>8.1f}{row['mdd']:>8.1f}"
            f"{row['ann_vol']:>9.1f}{row['calmar']:>8.2f}{row['sortino']:>9.2f}"
            f"{row['total_return']:>9.1f}"
        )
        print(line)


# =============================================================================
# Sweep Phases
# =============================================================================


def phase_a_stop_loss(
    engine: BacktestEngine,
    data_map: dict[str, MarketDataSet],
) -> list[dict[str, object]]:
    """Phase A: System Stop-Loss 단독 스윕."""
    total = len(STOP_LOSS_VALUES)
    results: list[dict[str, object]] = []

    for idx, sl in enumerate(STOP_LOSS_VALUES, 1):
        label = fmt_sl(sl)
        print(f"  [{idx:2d}/{total}] stop_loss={label} ...", end=" ", flush=True)

        metrics = run_multi_asset_backtest(
            engine,
            data_map,
            system_stop_loss=sl,
        )
        if metrics is None:
            print("SKIP")
            continue

        row: dict[str, object] = {"stop_loss": label, **metrics}
        results.append(row)
        print(
            f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%"
        )

    return results


def phase_b_trailing_stop(
    engine: BacktestEngine,
    data_map: dict[str, MarketDataSet],
) -> list[dict[str, object]]:
    """Phase B: Trailing Stop (ATR 배수) 단독 스윕."""
    total = len(TRAILING_STOP_MULTIPLIERS)
    results: list[dict[str, object]] = []

    for idx, mult in enumerate(TRAILING_STOP_MULTIPLIERS, 1):
        print(f"  [{idx:2d}/{total}] trailing_stop={mult:.1f}x ATR ...", end=" ", flush=True)

        metrics = run_multi_asset_backtest(
            engine,
            data_map,
            system_stop_loss=None,
            use_trailing_stop=True,
            trailing_stop_atr_multiplier=mult,
        )
        if metrics is None:
            print("SKIP")
            continue

        row: dict[str, object] = {"trailing_atr_mult": f"{mult:.1f}x", **metrics}
        results.append(row)
        print(
            f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%"
        )

    return results


def phase_c_combined(
    engine: BacktestEngine,
    data_map: dict[str, MarketDataSet],
) -> list[dict[str, object]]:
    """Phase C: Stop-Loss × Trailing Stop 결합 스윕."""
    combos = list(product(COMBINED_STOP_LOSS, COMBINED_TRAILING))
    total = len(combos)
    results: list[dict[str, object]] = []

    for idx, (sl, (use_ts, ts_mult)) in enumerate(combos, 1):
        sl_label = fmt_sl(sl)
        ts_label = fmt_ts(use_ts, ts_mult)
        print(
            f"  [{idx:3d}/{total}] stop_loss={sl_label}, trailing={ts_label} ...",
            end=" ",
            flush=True,
        )

        metrics = run_multi_asset_backtest(
            engine,
            data_map,
            system_stop_loss=sl,
            use_trailing_stop=use_ts,
            trailing_stop_atr_multiplier=ts_mult,
        )
        if metrics is None:
            print("SKIP")
            continue

        row: dict[str, object] = {
            "stop_loss": sl_label,
            "trailing_stop": ts_label,
            **metrics,
        }
        results.append(row)
        print(
            f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%"
        )

    return results


def phase_d_rebalance(
    engine: BacktestEngine,
    data_map: dict[str, MarketDataSet],
    best_sl: float | None,
    best_use_ts: bool,
    best_ts_mult: float,
) -> list[dict[str, object]]:
    """Phase D: Rebalance Threshold 스윕 (최적 stop/trailing 고정)."""
    total = len(REBALANCE_VALUES)
    results: list[dict[str, object]] = []

    for idx, rebal in enumerate(REBALANCE_VALUES, 1):
        print(
            f"  [{idx:2d}/{total}] rebalance_threshold={rebal:.0%} ...",
            end=" ",
            flush=True,
        )

        metrics = run_multi_asset_backtest(
            engine,
            data_map,
            system_stop_loss=best_sl,
            use_trailing_stop=best_use_ts,
            trailing_stop_atr_multiplier=best_ts_mult,
            rebalance_threshold=rebal,
        )
        if metrics is None:
            print("SKIP")
            continue

        row: dict[str, object] = {"rebalance_threshold": f"{rebal:.0%}", **metrics}
        results.append(row)
        print(
            f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, MDD={metrics['mdd']:.1f}%"
        )

    return results


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    n_phase_a = len(STOP_LOSS_VALUES) * len(ASSETS)
    n_phase_b = len(TRAILING_STOP_MULTIPLIERS) * len(ASSETS)
    n_phase_c = len(COMBINED_STOP_LOSS) * len(COMBINED_TRAILING) * len(ASSETS)
    n_phase_d = len(REBALANCE_VALUES) * len(ASSETS)
    total_backtests = n_phase_a + n_phase_b + n_phase_c + n_phase_d

    print("=" * 80)
    print("Multi-Asset TSMOM: Risk Parameter Sweep")
    print(f"Assets: {len(ASSETS)} | Period: {START.year}-{END.year}")
    print(f"Phase A: Stop-Loss ({len(STOP_LOSS_VALUES)} values, {n_phase_a} backtests)")
    print(
        f"Phase B: Trailing Stop ({len(TRAILING_STOP_MULTIPLIERS)} values, {n_phase_b} backtests)"
    )
    print(
        f"Phase C: Combined ({len(COMBINED_STOP_LOSS)}×{len(COMBINED_TRAILING)} = {len(COMBINED_STOP_LOSS) * len(COMBINED_TRAILING)} combos, {n_phase_c} backtests)"
    )
    print(f"Phase D: Rebalance ({len(REBALANCE_VALUES)} values, {n_phase_d} backtests)")
    print(f"Total backtests: {total_backtests}")
    print(f"Fixed TSMOM: {FIXED_TSMOM}")
    print(f"Fixed: leverage_cap={MAX_LEVERAGE_CAP}x")
    print("=" * 80)

    # 1. Load data
    print("\n[1/6] Loading market data...")
    service = MarketDataService()
    data_map = load_all_data(service)

    if len(data_map) < len(ASSETS):
        missing = set(ASSETS) - set(data_map.keys())
        print(f"\nWARNING: Missing data for: {missing}")
        print("Proceeding with available assets...")

    engine = BacktestEngine()

    # 2. Baseline
    print("\n[2/6] Running baseline (current config: stop_loss=10%, no trailing, rebal=5%)...")
    baseline = run_multi_asset_backtest(
        engine,
        data_map,
        system_stop_loss=0.10,
    )
    if baseline:
        print(
            f"  Baseline: Sharpe={baseline['sharpe']:.2f}, CAGR={baseline['cagr']:.1f}%, "
            f"MDD={baseline['mdd']:.1f}%, AnnVol={baseline['ann_vol']:.1f}%, "
            f"Calmar={baseline['calmar']:.2f}, Sortino={baseline['sortino']:.2f}"
        )

    baseline_none = run_multi_asset_backtest(
        engine,
        data_map,
        system_stop_loss=None,
    )
    if baseline_none:
        print(
            f"  No Risk:  Sharpe={baseline_none['sharpe']:.2f}, CAGR={baseline_none['cagr']:.1f}%, "
            f"MDD={baseline_none['mdd']:.1f}%, AnnVol={baseline_none['ann_vol']:.1f}%, "
            f"Calmar={baseline_none['calmar']:.2f}, Sortino={baseline_none['sortino']:.2f}"
        )

    # 3. Phase A: Stop-Loss sweep
    print(f"\n[3/6] Phase A: Stop-Loss Sweep ({len(STOP_LOSS_VALUES)} values)...")
    results_a = phase_a_stop_loss(engine, data_map)
    print_results_table(results_a, "Phase A: Stop-Loss Results")

    # Find best stop_loss from Phase A
    best_a = max(results_a, key=lambda r: float(r.get("sharpe", 0)))
    best_sl_label = str(best_a["stop_loss"])
    print(f"\n  >> Best Stop-Loss: {best_sl_label} (Sharpe={best_a['sharpe']:.2f})")

    # 4. Phase B: Trailing Stop sweep
    print(f"\n[4/6] Phase B: Trailing Stop Sweep ({len(TRAILING_STOP_MULTIPLIERS)} values)...")
    results_b = phase_b_trailing_stop(engine, data_map)
    print_results_table(results_b, "Phase B: Trailing Stop Results")

    # Find best trailing_stop from Phase B
    best_b = max(results_b, key=lambda r: float(r.get("sharpe", 0)))
    best_ts_label = str(best_b["trailing_atr_mult"])
    print(f"\n  >> Best Trailing Stop: {best_ts_label} (Sharpe={best_b['sharpe']:.2f})")

    # 5. Phase C: Combined sweep
    total_c = len(COMBINED_STOP_LOSS) * len(COMBINED_TRAILING)
    print(f"\n[5/6] Phase C: Combined Sweep ({total_c} combinations)...")
    results_c = phase_c_combined(engine, data_map)
    print_results_table(
        results_c, "Phase C: Combined (Stop-Loss x Trailing Stop) — Top 15 by Sharpe", top_n=15
    )
    print_results_table(
        results_c, "Phase C: Combined — Top 10 by Calmar", sort_by="calmar", top_n=10
    )
    print_results_table(
        results_c, "Phase C: Combined — Top 10 by Sortino", sort_by="sortino", top_n=10
    )

    # Find best combination from Phase C
    best_c = max(results_c, key=lambda r: float(r.get("sharpe", 0)))
    best_c_sl_label = str(best_c["stop_loss"])
    best_c_ts_label = str(best_c["trailing_stop"])
    best_c_sl: float | None = (
        None if best_c_sl_label == "None" else float(best_c_sl_label.strip("%")) / 100
    )
    best_c_use_ts = best_c_ts_label != "OFF"
    best_c_ts_mult = float(best_c_ts_label.replace("x ATR", "")) if best_c_use_ts else 2.0

    print(
        f"\n  >> Best Combined: stop_loss={best_c_sl_label}, trailing={best_c_ts_label} "
        f"(Sharpe={best_c['sharpe']:.2f})"
    )

    # 6. Phase D: Rebalance sweep
    print(f"\n[6/6] Phase D: Rebalance Threshold Sweep ({len(REBALANCE_VALUES)} values)...")
    print(f"  Using best from Phase C: stop_loss={best_c_sl_label}, trailing={best_c_ts_label}")
    results_d = phase_d_rebalance(engine, data_map, best_c_sl, best_c_use_ts, best_c_ts_mult)
    print_results_table(results_d, "Phase D: Rebalance Threshold Results")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\n### Baseline Comparison")
    if baseline:
        print(
            f"  Current (SL=10%, no TS, rebal=5%): Sharpe={baseline['sharpe']:.2f}, "
            f"CAGR={baseline['cagr']:.1f}%, MDD={baseline['mdd']:.1f}%"
        )
    if baseline_none:
        print(
            f"  No Risk Management:                Sharpe={baseline_none['sharpe']:.2f}, "
            f"CAGR={baseline_none['cagr']:.1f}%, MDD={baseline_none['mdd']:.1f}%"
        )

    print("\n### Phase A Best (Stop-Loss only)")
    print(f"  {best_a}")

    print("\n### Phase B Best (Trailing Stop only)")
    print(f"  {best_b}")

    print("\n### Phase C Best (Combined)")
    print(f"  {best_c}")

    if results_d:
        best_d = max(results_d, key=lambda r: float(r.get("sharpe", 0)))
        print("\n### Phase D Best (Rebalance)")
        print(f"  {best_d}")

    # Save all results
    for name, results in [
        ("stop_loss", results_a),
        ("trailing_stop", results_b),
        ("combined", results_c),
        ("rebalance", results_d),
    ]:
        if results:
            path = f"data/multi_asset_risk_{name}.csv"
            pd.DataFrame(results).to_csv(path, index=False)
            print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
